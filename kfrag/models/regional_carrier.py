"""Structured region-local communication channel for repaired Stage A."""

from __future__ import annotations

import math
import torch
from torch import nn


ACTIVE_START = 4
ACTIVE_BITS = 8
GRID_SIZE = 4


def _orthogonal_carriers(region_size: int, bits: int = ACTIVE_BITS) -> torch.Tensor:
    """Deterministic zero-mean orthonormal 2-D Walsh carriers."""
    if region_size < 4 or region_size & (region_size - 1):
        raise ValueError("region_size must be a power of two of at least 4")
    coordinates = torch.arange(region_size)
    patterns = []
    # Non-DC separable Walsh functions. Products remain exactly orthogonal.
    for index in range(1, bits + 1):
        row_code = index
        row = ((coordinates[:, None].bitwise_and(row_code)).ne(0).sum(1) % 2) * 2 - 1
        col_code = index * 2 + 1
        col = ((coordinates[:, None].bitwise_and(col_code)).ne(0).sum(1) % 2) * 2 - 1
        pattern = row.float()[:, None] * col.float()[None, :]
        pattern -= pattern.mean()
        # Gram-Schmidt makes the construction robust to code collisions.
        for previous in patterns:
            pattern -= (pattern * previous).sum() * previous
        norm = pattern.norm()
        if norm < 1e-6:
            # Fall back to a 1-D Walsh row reshaped into the region.
            flat = torch.arange(region_size * region_size)
            pattern = (((flat.bitwise_and(index)).ne(0)).float() * 2 - 1).reshape(region_size, region_size)
            pattern -= pattern.mean()
            for previous in patterns:
                pattern -= (pattern * previous).sum() * previous
            norm = pattern.norm()
        patterns.append(pattern / norm.clamp_min(1e-12))
    return torch.stack(patterns)


class RegionalCarrierBank(nn.Module):
    """Map eight bipolar symbol bits in every region to local RGB residuals."""

    def __init__(self, image_size: int = 256, alpha: float = .05,
                 mode: str = "fixed", payload_channels: int = 44) -> None:
        super().__init__()
        if image_size % GRID_SIZE:
            raise ValueError("image_size must be divisible by four")
        if mode not in {"fixed", "learnable"}:
            raise ValueError("carrier mode must be 'fixed' or 'learnable'")
        self.image_size, self.region_size = image_size, image_size // GRID_SIZE
        self.alpha, self.mode, self.payload_channels = float(alpha), mode, payload_channels
        initial = _orthogonal_carriers(self.region_size)
        if mode == "learnable":
            self.carriers = nn.Parameter(initial)
        else:
            self.register_buffer("carriers", initial)

    def normalized_carriers(self) -> torch.Tensor:
        carrier = self.carriers - self.carriers.mean((-2, -1), keepdim=True)
        flat = carrier.flatten(1)
        # Orthogonalize learnable carriers differentiably and normalize each bit.
        q, _ = torch.linalg.qr(flat.T, mode="reduced")
        return q.T.reshape_as(carrier)

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        expected = (self.payload_channels, GRID_SIZE, GRID_SIZE)
        if payload.ndim != 4 or tuple(payload.shape[1:]) != expected:
            raise ValueError(f"payload must have shape [B, {self.payload_channels}, 4, 4]")
        signed = payload[:, ACTIVE_START:ACTIVE_START + ACTIVE_BITS].float().mul(2).sub(1)
        carriers = self.normalized_carriers()
        # [B, row, col, h, w], with constant pre-alpha energy for every payload.
        patches = torch.einsum("bkrc,khw->brchw", signed, carriers) / math.sqrt(ACTIVE_BITS)
        canvas = patches.permute(0, 1, 3, 2, 4).reshape(len(payload), 1, self.image_size, self.image_size)
        # Bound only after energy normalization. No image clamp is used.
        return self.alpha * torch.tanh(canvas).expand(-1, 3, -1, -1)


class StructuredRegionalDecoder(nn.Module):
    """Blind decoder whose forward API accepts only the questioned image."""

    def __init__(self, carriers: torch.Tensor, image_size: int = 256,
                 payload_channels: int = 44, gain: float = 400.0) -> None:
        super().__init__()
        self.image_size, self.region_size = image_size, image_size // GRID_SIZE
        self.payload_channels = payload_channels
        self.register_buffer("carriers", carriers.detach().clone())
        self.log_gain = nn.Parameter(torch.tensor(math.log(gain)))
        self.active_bias = nn.Parameter(torch.zeros(ACTIVE_BITS))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f"image must have shape [B, 3, {self.image_size}, {self.image_size}]")
        grey = image.mean(1)
        patches = grey.reshape(len(image), GRID_SIZE, self.region_size, GRID_SIZE,
                               self.region_size).permute(0, 1, 3, 2, 4)
        patches = patches - patches.mean((-2, -1), keepdim=True)
        scores = torch.einsum("brchw,khw->bkrc", patches, self.carriers)
        scores = scores * self.log_gain.exp() + self.active_bias[None, :, None, None]
        logits = image.new_full((len(image), self.payload_channels, GRID_SIZE, GRID_SIZE), -20.)
        logits[:, ACTIVE_START:ACTIVE_START + ACTIVE_BITS] = scores
        return logits


class StructuredChannelSystem(nn.Module):
    """Image-independent Stage-A encoder and blind regional decoder."""

    def __init__(self, mode: str = "fixed", alpha: float = .05,
                 image_size: int = 256) -> None:
        super().__init__()
        self.carrier_bank = RegionalCarrierBank(image_size=image_size, alpha=alpha, mode=mode)
        self.decoder = StructuredRegionalDecoder(
            self.carrier_bank.normalized_carriers(), image_size=image_size
        )

    def sync_decoder_carriers(self) -> None:
        self.decoder.carriers.copy_(self.carrier_bank.normalized_carriers().detach())

    def forward(self, image: torch.Tensor, payload: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = self.carrier_bank(payload)
        questioned = image + residual
        return {"residual": residual, "watermarked_image": questioned,
                "payload_logits": self.decoder(questioned)}
