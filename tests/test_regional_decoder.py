import inspect

import pytest
import torch

from kfrag.models import RegionalDecoder


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_regional_decoder_returns_logits(batch_size: int) -> None:
    decoder = RegionalDecoder(base_channels=4)
    logits = decoder(torch.rand(batch_size, 3, 256, 256))
    assert logits.shape == (batch_size, 44, 4, 4)
    assert torch.isfinite(logits).all()


def test_decoder_api_accepts_only_questioned_image() -> None:
    parameters = list(inspect.signature(RegionalDecoder.forward).parameters)
    assert parameters == ["self", "image"]
