import csv
import io
import json

import pytest
import torch
from torch import nn

from kfrag.diagnostics.channel_sanity import (
    _evaluate,
    _reject_sensitive,
    capacity_mask,
    circular_payload_shuffle,
    generate_payload_bank,
    fresh_random_payloads,
    gradient_checks,
    masked_bce_with_logits,
    next_stage,
    payload_sensitivity,
    payload_diversity,
    recovery_metrics,
    payload_splits_are_disjoint,
    sensitivity_checks,
    should_advance_capacity,
    stage_d_allowed,
    validate_config,
)


def configuration():
    return {
        "experiment": {"seed": 2026, "device": "cpu"},
        "data": {"coco_directory": "unused", "image_size": 256, "num_natural_images": 8},
        "payloads": {"train_payloads": 4, "heldout_payloads": 2,
                     "grid_size": 4, "packet_bits": 44},
        "training": {"batch_size": 2, "learning_rate": .001, "weight_decay": 0,
                     "steps_per_capacity_level": 1, "evaluate_every": 1, "alpha": .05},
        "thresholds": {"active_field_accuracy": .95, "tag_accuracy": .95,
                       "exact_active_region_accuracy": .8,
                       "maximum_original_packet_accuracy": .01,
                       "minimum_payload_sensitivity": 1e-6,
                       "minimum_logit_sensitivity": 1e-6,
                       "minimum_gradient_norm": 1e-10},
    }


def test_payload_splits_are_authenticated_shape_and_disjoint():
    training, heldout = generate_payload_bank(8, 3, seed=12)
    assert training.shape == (8, 44, 4, 4)
    assert heldout.shape == (3, 44, 4, 4)
    assert payload_splits_are_disjoint(training, heldout)


@pytest.mark.parametrize("level,count", [(1, 8), (2, 16), (3, 24), (4, 40)])
def test_capacity_masks(level, count):
    mask = capacity_mask(level)
    assert mask.shape == (1, 44, 4, 4)
    assert int(mask.sum()) == count * 16
    assert not mask[:, :4].any()
    assert mask[:, 4:12].all()


def test_masked_bce_selects_exact_ranges():
    logits = torch.zeros(2, 44, 4, 4)
    targets = torch.zeros_like(logits)
    logits[:, 4:12] = 2
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 4:12], targets[:, 4:12]
    )
    assert torch.equal(masked_bce_with_logits(logits, targets, capacity_mask(1)), expected)


def test_region_index_does_not_control_progression():
    threshold = configuration()["thresholds"]
    metrics = {"region_index_accuracy": 0.0, "active_field_accuracy": 1.0,
               "authentication_tag_accuracy": 1.0, "exact_active_region_accuracy": 1.0}
    original = {"exact_regional_packet_accuracy": 0.0}
    assert should_advance_capacity(metrics, original, 4, threshold)


def test_only_active_tag_bits_control_intermediate_progression():
    threshold = configuration()["thresholds"]
    metrics = {"active_field_accuracy": 1.0, "authentication_tag_accuracy": .6,
               "active_authentication_tag_accuracy": 1.0, "exact_active_region_accuracy": 1.0}
    assert should_advance_capacity(metrics, {"exact_regional_packet_accuracy": 0.0}, 2, threshold)


def test_circular_shuffle_has_no_fixed_positions():
    values = torch.arange(5)
    shuffled = circular_payload_shuffle(values)
    assert not torch.eq(values, shuffled).any()


class TinyChannel(nn.Module):
    def __init__(self, sensitive=True):
        super().__init__()
        self.sensitive = sensitive

    def forward(self, image, payload):
        signal = payload.mean((1, 2, 3), keepdim=True) if self.sensitive else payload.new_zeros((len(payload), 1, 1, 1))
        residual = signal.expand(-1, 3, 256, 256) * .01
        return {"residual": residual, "watermarked_image": image + residual,
                "payload_logits": payload * (2 if self.sensitive else 0)}


def test_payload_independent_encoder_fails_sensitivity():
    carrier = torch.zeros(1, 3, 256, 256)
    first, second = torch.zeros(1, 44, 4, 4), torch.ones(1, 44, 4, 4)
    values = payload_sensitivity(TinyChannel(False), carrier, first, second)
    assert len(sensitivity_checks(values, configuration()["thresholds"])) == 2


def test_payload_sensitive_encoder_passes_sensitivity():
    carrier = torch.zeros(1, 3, 256, 256)
    first, second = torch.zeros(1, 44, 4, 4), torch.ones(1, 44, 4, 4)
    values = payload_sensitivity(TinyChannel(True), carrier, first, second)
    assert sensitivity_checks(values, configuration()["thresholds"]) == []


def test_zero_projector_gradients_are_detected():
    values = {"payload_projector_gradient_norm": 0.0, "decoder_gradient_norm": 1.0}
    assert "payload projector" in gradient_checks(values, 1e-10)[0]


def test_stage_gating_stops_after_failure():
    assert next_stage("A", {"A": {"passed": False}}) is None
    assert next_stage("A", {"A": {"passed": True}}) == "B"


def test_stage_d_requires_successful_stage_c():
    assert not stage_d_allowed({"C": {"passed": False}})
    assert stage_d_allowed({"C": {"passed": True}})
    assert next_stage("C", {"C": {"passed": False}}) is None


class BypassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_calls = 0
        self.decoder = self.Decoder()

    class Decoder(nn.Module):
        def forward(self, image):
            return torch.zeros(len(image), 44, 4, 4, device=image.device)

    def forward(self, image, payload):
        self.encoder_calls += 1
        return {"payload_logits": torch.zeros_like(payload), "watermarked_image": image,
                "residual": torch.zeros_like(image)}


def test_original_images_bypass_encoder():
    model = BypassModel()
    images = torch.zeros(1, 3, 256, 256)
    payloads = torch.zeros(2, 44, 4, 4)
    _evaluate(model, images, payloads, capacity_mask(1), 2, .05)
    assert model.encoder_calls == 1


def test_key_like_material_is_rejected_from_all_serializable_outputs():
    for value in ({"hmac_key": "sentinel"}, {"nested": {"secret": "sentinel"}},
                  {"authentication_key": b"sentinel"}):
        with pytest.raises(ValueError, match="forbidden key-like"):
            _reject_sensitive(value)
    safe = {"metrics": {"active": .5}, "payload_tensors": [0, 1]}
    text = json.dumps(safe)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=["metrics"])
    writer.writeheader(); writer.writerow({"metrics": .5})
    assert "sentinel" not in text + stream.getvalue() + repr(safe)


def test_malformed_configuration_has_clear_error():
    bad = configuration(); bad["payloads"]["packet_bits"] = 43
    with pytest.raises(ValueError, match="malformed configuration.*packet_bits"):
        validate_config(bad)


def test_fresh_payloads_do_not_overlap_stored_active_patterns_and_vary():
    training, heldout = generate_payload_bank(12, 6, seed=91)
    mask = capacity_mask(1)
    fresh = fresh_random_payloads(16, mask, (training, heldout), torch.Generator().manual_seed(7))
    diversity = payload_diversity({"training": training, "heldout": heldout, "fresh": fresh}, mask)
    assert diversity["overlap"]["training__fresh"] == 0
    assert diversity["overlap"]["heldout__fresh"] == 0
    assert diversity["groups"]["fresh"]["every_target_active_bit_varies"]
    assert diversity["groups"]["heldout"]["target_active_bits_vary"]


def test_shuffled_targets_score_substantially_worse_synthetically():
    mask = capacity_mask(1)
    targets = torch.zeros(4, 44, 4, 4)
    targets[1, 4:12] = 1
    targets[2, 4:12, :, ::2] = 1
    targets[3, 4:12, :, 1::2] = 1
    logits = (targets * 2 - 1) * 10
    correct = recovery_metrics(logits, targets, mask)
    shuffled = recovery_metrics(logits, circular_payload_shuffle(targets), mask)
    assert correct["active_bit_accuracy"] - shuffled["active_bit_accuracy"] > .4
    assert shuffled["active_bce_loss"] - correct["active_bce_loss"] > 4


def test_payload_change_changes_encoder_residual_and_inactive_tag_is_na():
    carrier = torch.zeros(1, 3, 256, 256)
    values = payload_sensitivity(TinyChannel(True), carrier, torch.zeros(1, 44, 4, 4),
                                 torch.ones(1, 44, 4, 4))
    assert values["encoder_residual_pairwise_distance"] > 0
    metrics = recovery_metrics(torch.zeros(2, 44, 4, 4), torch.zeros(2, 44, 4, 4), capacity_mask(1))
    assert metrics["active_authentication_tag_accuracy"] is None
    assert metrics["inactive_authentication_tag_accuracy"] == "not applicable"
