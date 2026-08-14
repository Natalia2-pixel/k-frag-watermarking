import pytest
import torch

from kfrag.models import CleanWatermarkSystem
from kfrag.models.losses import (
    bit_accuracy,
    clean_watermark_loss,
    image_payload_accuracy,
    mean_absolute_residual,
    psnr,
    regional_packet_accuracy,
)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_clean_system_forward_shapes(batch_size: int) -> None:
    system = CleanWatermarkSystem(base_channels=4, message_channels=4)
    image = torch.rand(batch_size, 3, 256, 256)
    payload = torch.randint(0, 2, (batch_size, 44, 4, 4)).float()
    result = system(image, payload)
    assert result["watermarked_image"].shape == (batch_size, 3, 256, 256)
    assert result["residual"].shape == (batch_size, 3, 256, 256)
    assert result["payload_logits"].shape == (batch_size, 44, 4, 4)


def test_loss_metrics_and_backward_smoke() -> None:
    system = CleanWatermarkSystem(base_channels=4, message_channels=4)
    image = torch.rand(1, 3, 256, 256)
    payload = torch.randint(0, 2, (1, 44, 4, 4)).float()
    result = system(image, payload)
    losses = clean_watermark_loss(result["payload_logits"], payload, image,
                                  result["watermarked_image"], result["residual"])
    assert all(torch.isfinite(value) for value in losses.values())
    for metric in (bit_accuracy, regional_packet_accuracy, image_payload_accuracy):
        value = metric(result["payload_logits"], payload)
        assert torch.isfinite(value) and 0 <= value <= 1
    assert torch.isfinite(mean_absolute_residual(result["residual"]))
    assert torch.isfinite(psnr(image, result["watermarked_image"]))
    losses["total_loss"].backward()
    assert any(parameter.grad is not None for parameter in system.encoder.parameters())
    assert any(parameter.grad is not None for parameter in system.decoder.parameters())


def test_clean_system_batch_mismatch_is_clear() -> None:
    with pytest.raises(ValueError, match="same batch size"):
        CleanWatermarkSystem(base_channels=4, message_channels=4)(
            torch.rand(2, 3, 256, 256), torch.rand(1, 44, 4, 4)
        )
