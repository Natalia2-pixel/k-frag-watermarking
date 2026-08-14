import pytest
import torch

from kfrag.models import CleanWatermarkEncoder, MessageProjector


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_clean_encoder_output_is_bounded_and_does_not_modify_input(batch_size: int) -> None:
    image = torch.rand(batch_size, 3, 256, 256)
    original = image.clone()
    messages = MessageProjector(message_channels=4)(torch.rand(batch_size, 44, 4, 4))
    encoder = CleanWatermarkEncoder(base_channels=4, message_channels=4, residual_alpha=0.02)
    watermarked, residual = encoder(image, messages)
    assert watermarked.shape == image.shape
    assert residual.shape == image.shape
    assert watermarked.min() >= 0 and watermarked.max() <= 1
    assert residual.abs().max() <= 0.02 + 1e-7
    assert torch.isfinite(watermarked).all() and torch.isfinite(residual).all()
    assert torch.equal(image, original)


def test_clean_encoder_validates_image_shape() -> None:
    with pytest.raises(ValueError, match="image must have shape"):
        CleanWatermarkEncoder(base_channels=4, message_channels=4)(torch.zeros(1, 3, 128, 128), {})
