"""Blind, fully convolutional decoder for regional packet logits."""

from __future__ import annotations

import torch
from torch import nn


class _DecodeBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(1, output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )


class RegionalDecoder(nn.Module):
    """Decode only a questioned image into ``[B, 44, 4, 4]`` logits."""

    def __init__(self, payload_channels: int = 44, base_channels: int = 32) -> None:
        super().__init__()
        if payload_channels <= 0 or base_channels <= 0:
            raise ValueError("payload_channels and base_channels must be positive")
        channels = (base_channels, base_channels, base_channels * 2, base_channels * 2,
                    base_channels * 3, base_channels * 4)
        blocks: list[nn.Module] = [
            nn.Sequential(nn.Conv2d(3, channels[0], 3, stride=2, padding=1), nn.SiLU(inplace=True))
        ]
        blocks.extend(_DecodeBlock(a, b) for a, b in zip(channels, channels[1:]))
        self.features = nn.Sequential(*blocks)  # 256 -> 4 after six stride-2 operations
        self.logits = nn.Conv2d(channels[-1], payload_channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1 or tuple(image.shape[1:]) != (3, 256, 256):
            actual = None if not isinstance(image, torch.Tensor) else list(image.shape)
            raise ValueError(f"image must have shape [B, 3, 256, 256], got {actual}")
        return self.logits(self.features(image))
