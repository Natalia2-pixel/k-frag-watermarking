import inspect
import torch

from kfrag.diagnostics.stage_b_natural import (decoder_accepts_only_questioned_image,
    fresh_disjoint_payloads, payload_fingerprints, random_symbol_payloads, symbol_metrics)
from kfrag.models import StructuredChannelSystem, StructuredRegionalDecoder


def test_stage_b_decoder_api_accepts_only_questioned_image():
    assert decoder_accepts_only_questioned_image()
    assert list(inspect.signature(StructuredRegionalDecoder.forward).parameters) == ["self", "image"]


def test_changing_only_payload_changes_residual():
    generator = torch.Generator().manual_seed(4)
    payloads = random_symbol_payloads(2, generator)
    model = StructuredChannelSystem(mode="learnable")
    image = torch.rand(1, 3, 256, 256)
    a = model(image, payloads[:1])["residual"]
    b = model(image, payloads[1:])["residual"]
    assert not torch.equal(a, b)
    assert (a - b).abs().mean() > 0


def test_fresh_evaluation_payloads_do_not_overlap_stored_sets():
    generator = torch.Generator().manual_seed(8)
    training = random_symbol_payloads(10, generator)
    heldout = fresh_disjoint_payloads(10, generator, [training])
    fresh = fresh_disjoint_payloads(10, generator, [training, heldout])
    assert payload_fingerprints(fresh).isdisjoint(payload_fingerprints(training))
    assert payload_fingerprints(fresh).isdisjoint(payload_fingerprints(heldout))


def test_symbol_metrics_make_inactive_protocol_metrics_unavailable():
    generator = torch.Generator().manual_seed(12)
    targets = random_symbol_payloads(3, generator)
    logits = torch.zeros_like(targets); logits[:, 4:12] = (targets[:, 4:12] * 2 - 1) * 10
    result = symbol_metrics(logits, targets)
    assert result["active_bit_accuracy"] == 1
    assert result["exact_8bit_regional_symbol_accuracy"] == 1
    for name in ("region_index_accuracy", "authentication_tag_accuracy",
                 "exact_44bit_regional_packet_accuracy", "number_of_exact_complete_packets",
                 "exact_image_payload_accuracy", "rs_identity_reconstruction_accuracy"):
        assert result[name] is None


def test_at_least_100_distinct_fresh_payload_tensors_can_be_proven():
    payloads = random_symbol_payloads(100, torch.Generator().manual_seed(99))
    assert len(payload_fingerprints(payloads)) == 100


def test_shuffled_target_metrics_expose_the_same_complete_metric_schema():
    generator = torch.Generator().manual_seed(101)
    targets = random_symbol_payloads(4, generator)
    metrics = symbol_metrics(torch.zeros_like(targets), torch.roll(targets, 1, 0))
    for key in ("active_bit_accuracy", "exact_8bit_regional_symbol_accuracy",
                "active_bce_loss", "mean_decoder_confidence", "per_bit_accuracy",
                "target_entropy", "prediction_entropy", "predicted_one_frequency"):
        assert key in metrics
