import inspect

import torch

from kfrag.diagnostics.channel_repair import (decoder_reconstruction_test,
    carrier_change_metrics, single_bit_residual_causality, stage_a_recovery_metrics)
from kfrag.diagnostics.channel_sanity import capacity_mask
from kfrag.models import RegionalCarrierBank, StructuredChannelSystem, StructuredRegionalDecoder


def random_payload(batch=4):
    value = torch.zeros(batch, 44, 4, 4)
    value[:, 4:12] = torch.randint(0, 2, (batch, 8, 4, 4)).float()
    return value


def test_single_bit_residual_causality_and_region_locality():
    result = single_bit_residual_causality(StructuredChannelSystem(), bit=3, region=(2, 1))
    assert result["passed"]
    assert result["inside_change"] > 0
    assert result["outside_max_change"] == 0


def test_carrier_normalization_and_orthogonality():
    bank = RegionalCarrierBank(mode="learnable")
    flat = bank.normalized_carriers().flatten(1)
    assert torch.allclose(flat.norm(dim=1), torch.ones(8), atol=1e-5)
    assert torch.allclose(flat @ flat.T, torch.eye(8), atol=1e-5)
    assert torch.allclose(flat.mean(1), torch.zeros(8), atol=1e-6)


def test_payload_target_channel_alignment_and_analytical_decoding():
    model = StructuredChannelSystem()
    payload = random_payload()
    image = torch.full((len(payload), 3, 256, 256), .5)
    logits = model(image, payload)["payload_logits"]
    assert (logits[:, 4:12] >= 0).eq(payload[:, 4:12].bool()).all()
    assert decoder_reconstruction_test(count=8)["heldout_bit_accuracy"] == 1


def test_decoder_is_blind_to_payload_and_expected_coefficients():
    assert list(inspect.signature(StructuredRegionalDecoder.forward).parameters) == ["self", "image"]


def test_fresh_payloads_change_and_shuffle_discriminates():
    model = StructuredChannelSystem()
    payload = random_payload(8)
    assert torch.unique(payload[:, 4:12].flatten(1), dim=0).shape[0] > 1
    image = torch.full((8, 3, 256, 256), .5)
    logits = model(image, payload)["payload_logits"][:, 4:12]
    correct = (logits >= 0).eq(payload[:, 4:12].bool()).float().mean()
    shuffled = (logits >= 0).eq(torch.roll(payload[:, 4:12], 1, 0).bool()).float().mean()
    assert correct - shuffled >= .4


def test_initialization_snapshot_does_not_alias_live_carrier():
    bank = RegionalCarrierBank(mode="learnable")
    snapshot = bank.carriers.detach().clone()
    expected = snapshot.clone()
    with torch.no_grad():
        bank.carriers.add_(1)
    assert torch.equal(snapshot, expected)
    assert not torch.equal(snapshot, bank.carriers)


def test_known_carrier_change_metrics():
    initial = torch.tensor([3.0, 4.0])
    learned = torch.tensor([0.0, 4.0])
    metrics = carrier_change_metrics(initial, learned, epsilon=1e-9, changed_tolerance=1e-6)
    assert metrics["mean_absolute_change_from_initialization"] == 1.5
    assert metrics["l1_change_from_initialization"] == 3.0
    assert metrics["l2_change_from_initialization"] == 3.0
    assert metrics["relative_l2_change_from_initialization"] == 0.6
    assert metrics["cosine_similarity_with_initialization"] == 0.8
    assert metrics["maximum_absolute_change_from_initialization"] == 3.0
    assert metrics["number_of_changed_parameters"] == 1
    assert metrics["total_number_of_carrier_parameters"] == 2


def test_stage_a_inactive_packet_fields_are_unavailable_but_active_metric_is_numeric():
    targets = random_payload(2)
    logits = (targets * 2 - 1) * 10
    metrics = stage_a_recovery_metrics(logits, targets, capacity_mask(1))
    for name in ("authentication_tag_accuracy", "active_authentication_tag_accuracy",
                 "exact_regional_packet_accuracy", "number_exact_regional_packets",
                 "exact_image_payload_accuracy"):
        assert metrics[name] is None
    assert isinstance(metrics["active_bit_accuracy"], float)
    assert isinstance(metrics["exact_active_region_accuracy"], float)
    assert metrics["exact_active_region_accuracy"] == 1.0
