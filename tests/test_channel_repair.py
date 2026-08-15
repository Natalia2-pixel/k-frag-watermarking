import inspect

import torch

from kfrag.diagnostics.channel_repair import (decoder_reconstruction_test,
    single_bit_residual_causality)
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
