"""Losses and metrics for clean watermark training and evaluation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def clean_watermark_loss(payload_logits: torch.Tensor, target_payload: torch.Tensor,
                         original_image: torch.Tensor, watermarked_image: torch.Tensor,
                         residual: torch.Tensor, lambda_payload: float = 1.0,
                         lambda_image: float = 1.0, lambda_residual: float = 0.1) -> dict[str, torch.Tensor]:
    """Return the weighted clean objective and each of its three components."""
    if payload_logits.shape != target_payload.shape:
        raise ValueError("payload_logits and target_payload must have identical shapes")
    if original_image.shape != watermarked_image.shape or original_image.shape != residual.shape:
        raise ValueError("original_image, watermarked_image and residual must have identical shapes")
    payload_loss = F.binary_cross_entropy_with_logits(payload_logits, target_payload.float())
    image_loss = F.l1_loss(watermarked_image, original_image)
    residual_regularization = residual.abs().mean()
    total = lambda_payload * payload_loss + lambda_image * image_loss + lambda_residual * residual_regularization
    return {"total_loss": total, "payload_loss": payload_loss, "image_loss": image_loss,
            "residual_regularization": residual_regularization}


def _correct_bits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.ndim != 4 or tuple(logits.shape[1:]) != (44, 4, 4):
        raise ValueError("logits and target must both have shape [B, 44, 4, 4]")
    return (torch.sigmoid(logits) >= 0.5).eq(target >= 0.5)


def bit_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fraction of all 44 x 16 bits decoded correctly."""
    return _correct_bits(logits, target).float().mean()


def regional_packet_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fraction of regions where all 44 bits are correct."""
    return _correct_bits(logits, target).all(dim=1).float().mean()


def image_payload_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fraction of images where all 704 bits are correct."""
    return _correct_bits(logits, target).flatten(1).all(dim=1).float().mean()


def mean_absolute_residual(residual: torch.Tensor) -> torch.Tensor:
    """Mean absolute value of the RGB embedding residual."""
    return residual.abs().mean()


def psnr(original: torch.Tensor, watermarked: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in decibels, averaged over the batch."""
    if original.shape != watermarked.shape:
        raise ValueError("original and watermarked must have identical shapes")
    mse = (original - watermarked).square().flatten(1).mean(1)
    peak = torch.as_tensor(max_value, dtype=mse.dtype, device=mse.device)
    return (10.0 * torch.log10(peak.square() / mse.clamp_min(torch.finfo(mse.dtype).tiny))).mean()
