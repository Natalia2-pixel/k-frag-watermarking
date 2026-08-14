import json

import pytest
import torch

from kfrag.evaluation.anti_memorization import (
    FIELD_SLICES,
    _write_outputs,
    circular_shift_targets,
    evaluate_anti_memorization,
    evaluate_checkpoint,
    field_metrics,
)


class MockSystem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_payloads = []
        self.decoder_calls = 0

        class Decoder(torch.nn.Module):
            def __init__(inner, owner):
                super().__init__(); object.__setattr__(inner, "owner", owner)
            def forward(inner, images):
                inner.owner.decoder_calls += 1
                return torch.full((len(images), 44, 4, 4), -10.0, device=images.device)
        self.decoder = Decoder(self)

    def forward(self, images, payloads):
        self.forward_payloads.append(payloads.detach().cpu().clone())
        residual = torch.zeros_like(images)
        return {"payload_logits": (payloads * 2 - 1) * 10, "watermarked_image": images,
                "residual": residual}


def test_circular_shuffle_is_a_derangement():
    values = torch.arange(8).view(8, 1, 1, 1).expand(8, 44, 4, 4)
    shifted = circular_shift_targets(values)
    assert not torch.any((shifted == values).flatten(1).all(1))


def test_field_slices_and_metrics():
    target = torch.zeros(1, 44, 4, 4)
    predicted = target.clone()
    predicted[:, 0:4] = 1
    logits = (predicted * 2 - 1) * 10
    metrics = field_metrics(logits, target)
    assert FIELD_SLICES == {"index": slice(0, 4), "coded_symbol": slice(4, 12),
                            "authentication_tag": slice(12, 44), "non_index": slice(4, 44)}
    assert metrics["index_bit_accuracy"] == 0
    assert metrics["coded_symbol_accuracy"] == 1
    assert metrics["authentication_tag_accuracy"] == 1
    assert metrics["non_index_accuracy"] == 1
    assert metrics["exact_regional_packets"] == 0


def test_original_bypasses_encoder_and_unseen_payloads_are_new():
    model = MockSystem()
    images = torch.zeros(8, 3, 256, 256)
    training = torch.zeros(8, 44, 4, 4)
    results = evaluate_anti_memorization(model, images, training)
    assert len(model.forward_payloads) == 2  # A and D only; B uses decoder, C reuses A logits
    assert model.decoder_calls == 1
    unseen = model.forward_payloads[1]
    assert all(not torch.equal(unseen[i], training[i]) for i in range(8))
    assert set(results) == {"correct_watermarked", "original_unwatermarked", "shuffled_targets", "unseen_payloads"}
    assert results["correct_watermarked"]["exact_regional_packets"] == 128


def test_outputs_contain_no_key(tmp_path):
    row = {"overall_bit_accuracy": .5, "index_bit_accuracy": 1.0, "coded_symbol_accuracy": .5,
           "authentication_tag_accuracy": .5, "non_index_accuracy": .5,
           "regional_packet_accuracy": 0.0, "image_payload_accuracy": 0.0,
           "mean_decoder_confidence": .2, "exact_regional_packets": 0}
    _write_outputs({name: dict(row) for name in ("correct_watermarked", "original_unwatermarked",
                   "shuffled_targets", "unseen_payloads")}, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"results.json", "results.csv", "comparison.png"}
    text = (tmp_path / "results.json").read_text() + (tmp_path / "results.csv").read_text()
    assert "key" not in text.lower() and "secret" not in text.lower()
    assert json.loads((tmp_path / "results.json").read_text())


def test_malformed_checkpoint_has_clear_error(tmp_path):
    path = tmp_path / "bad.pt"; torch.save({"wrong": True}, path)
    with pytest.raises(ValueError, match="malformed checkpoint.*model_state"):
        evaluate_checkpoint({}, path)


def test_missing_bundle_requires_rerun(tmp_path):
    path = tmp_path / "legacy.pt"; torch.save({"model_state": {}}, path)
    with pytest.raises(ValueError, match="missing a valid evaluation_bundle.*rerun tiny overfit"):
        evaluate_checkpoint({"coco_directory": tmp_path}, path)
