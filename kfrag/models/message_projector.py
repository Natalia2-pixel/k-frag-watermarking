"""Learned compact multi-scale representation of regional packet bits."""

from __future__ import annotations

import torch
from torch import nn


class MessageProjector(nn.Module):
    """Project ``[B, payload_channels, 4, 4]`` into five compact feature maps.

    The returned dictionary is keyed by spatial size (4, 8, 16, 32 and 64).
    Learned upsampling avoids ever materialising a raw 44-channel payload at
    image resolution.
    """

    scales = (4, 8, 16, 32, 64)

    def __init__(self, payload_channels: int = 44, message_channels: int = 32) -> None:
        super().__init__()
        if payload_channels <= 0 or message_channels <= 0:
            raise ValueError("payload_channels and message_channels must be positive")
        self.payload_channels = payload_channels
        self.message_channels = message_channels
        self.input_projection = nn.Sequential(
            nn.Conv2d(payload_channels, message_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(message_channels, message_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.upsamplers = nn.ModuleList(
            nn.Sequential(
                nn.ConvTranspose2d(message_channels, message_channels, 4, stride=2, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(message_channels, message_channels, 3, padding=1),
                nn.SiLU(inplace=True),
            )
            for _ in range(4)
        )

    def forward(self, payload: torch.Tensor) -> dict[int, torch.Tensor]:
        if not isinstance(payload, torch.Tensor):
            raise ValueError("payload must be a torch.Tensor with shape [B, C, 4, 4]")
        if payload.ndim != 4 or payload.shape[0] < 1 or tuple(payload.shape[1:]) != (
            self.payload_channels,
            4,
            4,
        ):
            raise ValueError(
                f"payload must have shape [B, {self.payload_channels}, 4, 4], "
                f"got {list(payload.shape)}"
            )
        feature = self.input_projection(payload.float())
        outputs = {4: feature}
        for scale, upsampler in zip(self.scales[1:], self.upsamplers):
            feature = upsampler(feature)
            outputs[scale] = feature
        return outputs
