"""End-to-end clean watermark baseline."""

from __future__ import annotations

import torch
from torch import nn

from .clean_encoder import CleanWatermarkEncoder
from .message_projector import MessageProjector
from .regional_decoder import RegionalDecoder


class CleanWatermarkSystem(nn.Module):
    """Connect message projection, residual embedding and blind decoding."""

    def __init__(self, payload_channels: int = 44, base_channels: int = 32,
                 message_channels: int = 32, residual_alpha: float = 0.02) -> None:
        super().__init__()
        self.projector = MessageProjector(payload_channels, message_channels)
        self.encoder = CleanWatermarkEncoder(base_channels, message_channels, residual_alpha)
        self.decoder = RegionalDecoder(payload_channels, base_channels)

    def forward(self, image: torch.Tensor, regional_payload: torch.Tensor) -> dict[str, torch.Tensor]:
        if not isinstance(image, torch.Tensor) or not isinstance(regional_payload, torch.Tensor):
            raise ValueError("image and regional_payload must be torch.Tensor instances")
        if image.ndim < 1 or regional_payload.ndim < 1 or image.shape[0] != regional_payload.shape[0]:
            raise ValueError("image and regional_payload must have the same batch size")
        messages = self.projector(regional_payload)
        watermarked, residual = self.encoder(image, messages)
        logits = self.decoder(watermarked)
        return {"watermarked_image": watermarked, "residual": residual, "payload_logits": logits}
