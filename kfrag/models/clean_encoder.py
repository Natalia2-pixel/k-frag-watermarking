"""Lightweight residual U-Net encoder conditioned by projected messages."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class _FiLM(nn.Module):
    """Feature-wise affine modulation derived from a message feature map."""

    def __init__(self, message_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.affine = nn.Conv2d(message_channels, feature_channels * 2, 1)

    def forward(self, image_feature: torch.Tensor, message_feature: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(message_feature).chunk(2, dim=1)
        return image_feature * (1.0 + gamma) + beta


class CleanWatermarkEncoder(nn.Module):
    """Predict a bounded RGB residual from an image and multi-scale messages."""

    required_scales = (4, 8, 16, 32, 64)

    def __init__(
        self, base_channels: int = 32, message_channels: int = 32, residual_alpha: float = 0.02
    ) -> None:
        super().__init__()
        if base_channels <= 0 or message_channels <= 0:
            raise ValueError("base_channels and message_channels must be positive")
        if residual_alpha <= 0:
            raise ValueError("residual_alpha must be positive")
        self.message_channels = message_channels
        self.residual_alpha = float(residual_alpha)
        # The 256 and 128 stages stay narrow; capacity grows only after 64x64.
        channels = {256: base_channels, 128: base_channels, 64: base_channels,
                    32: base_channels * 2, 16: base_channels * 2,
                    8: base_channels * 3, 4: base_channels * 4}
        self.stem = nn.Sequential(nn.Conv2d(3, channels[256], 3, padding=1), nn.SiLU(inplace=True))
        self.down = nn.ModuleDict()
        for source, target in zip((256, 128, 64, 32, 16, 8), (128, 64, 32, 16, 8, 4)):
            self.down[str(target)] = nn.Sequential(
                nn.Conv2d(channels[source], channels[target], 3, stride=2, padding=1),
                nn.GroupNorm(1, channels[target]), nn.SiLU(inplace=True),
                _ResidualBlock(channels[target]),
            )
        self.film_down = nn.ModuleDict(
            {str(s): _FiLM(message_channels, channels[s]) for s in self.required_scales}
        )
        self.up = nn.ModuleDict()
        for source, target in zip((4, 8, 16, 32, 64, 128), (8, 16, 32, 64, 128, 256)):
            self.up[str(target)] = nn.Sequential(
                nn.Conv2d(channels[source] + channels[target], channels[target], 3, padding=1),
                nn.GroupNorm(1, channels[target]), nn.SiLU(inplace=True),
                _ResidualBlock(channels[target]),
            )
        self.output = nn.Conv2d(channels[256], 3, 3, padding=1)

    def _validate_messages(self, messages: dict[int, torch.Tensor], batch: int) -> None:
        if not isinstance(messages, dict):
            raise ValueError("message_features must be a dictionary keyed by spatial scale")
        for scale in self.required_scales:
            feature = messages.get(scale)
            expected = (batch, self.message_channels, scale, scale)
            if not isinstance(feature, torch.Tensor) or tuple(feature.shape) != expected:
                actual = None if not isinstance(feature, torch.Tensor) else list(feature.shape)
                raise ValueError(f"message feature at scale {scale} must have shape {list(expected)}, got {actual}")

    def forward(
        self, image: torch.Tensor, message_features: dict[int, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1 or tuple(image.shape[1:]) != (3, 256, 256):
            actual = None if not isinstance(image, torch.Tensor) else list(image.shape)
            raise ValueError(f"image must have shape [B, 3, 256, 256], got {actual}")
        self._validate_messages(message_features, image.shape[0])
        features: dict[int, torch.Tensor] = {256: self.stem(image)}
        for scale in (128, 64, 32, 16, 8, 4):
            features[scale] = self.down[str(scale)](features[scale * 2])
            if scale <= 64:
                features[scale] = self.film_down[str(scale)](features[scale], message_features[scale])
        x = features[4]
        for scale in (8, 16, 32, 64, 128, 256):
            x = F.interpolate(x, size=(scale, scale), mode="bilinear", align_corners=False)
            x = self.up[str(scale)](torch.cat((x, features[scale]), dim=1))
        residual = self.residual_alpha * torch.tanh(self.output(x))
        watermarked = torch.clamp(image + residual, 0.0, 1.0)
        return watermarked, residual
