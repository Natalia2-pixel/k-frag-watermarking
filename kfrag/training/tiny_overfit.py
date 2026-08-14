"""Deterministic clean memorization experiment on eight fixed image/payload pairs."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from kfrag.crypto import ProvenanceToken, create_packets
from kfrag.data import CocoImageDataset
from kfrag.models import CleanWatermarkSystem
from kfrag.models.losses import (
    bit_accuracy,
    clean_watermark_loss,
    image_payload_accuracy,
    psnr,
    regional_packet_accuracy,
)
from kfrag.payload import batch_packets_to_grid

METRIC_NAMES = (
    "total_loss", "payload_bce_loss", "image_fidelity_loss", "bit_accuracy",
    "regional_packet_accuracy", "image_payload_accuracy", "psnr",
    "max_absolute_residual",
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and all available PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fixed_coco_images(image_directory: str | Path, num_images: int, seed: int) -> torch.Tensor:
    """Select a seeded, stable subset from the existing COCO image dataset."""
    dataset = CocoImageDataset(image_directory)
    if num_images != 8:
        raise ValueError("the tiny-overfit experiment requires exactly 8 images")
    if len(dataset) < num_images:
        raise ValueError(f"COCO directory contains {len(dataset)} images; 8 are required")
    indices = sorted(random.Random(seed).sample(range(len(dataset)), num_images))
    return torch.stack([dataset[index]["image"] for index in indices])


def _fixed_coco_selection(image_directory: str | Path, num_images: int, seed: int) -> tuple[torch.Tensor, list[str]]:
    """Load the fixed images and retain stable paths for evaluation replay."""
    dataset = CocoImageDataset(image_directory)
    if num_images != 8 or len(dataset) < num_images:
        raise ValueError("the tiny-overfit experiment requires exactly 8 available images")
    indices = sorted(random.Random(seed).sample(range(len(dataset)), num_images))
    samples = [dataset[index] for index in indices]
    return torch.stack([sample["image"] for sample in samples]), [sample["relative_path"] for sample in samples]


def _experiment_key(seed: int) -> bytes:
    # Kept only in memory. It is deliberately absent from configs, logs and checkpoints.
    return hashlib.sha256(f"kfrag-tiny-overfit-v1:{seed}".encode()).digest()


def build_fixed_payloads(num_images: int, seed: int) -> torch.Tensor:
    """Build unique deterministic tokens and their authenticated packet grids."""
    if num_images != 8:
        raise ValueError("the tiny-overfit experiment requires exactly 8 images")
    rng = random.Random(seed)
    asset_ids: set[int] = set()
    tokens: list[ProvenanceToken] = []
    while len(tokens) < num_images:
        asset_id = rng.getrandbits(64)
        if asset_id not in asset_ids:
            asset_ids.add(asset_id)
            tokens.append(ProvenanceToken(issuer_id=1, asset_id=asset_id, version=1))
    key = _experiment_key(seed)
    return batch_packets_to_grid([create_packets(token, key) for token in tokens])


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _metrics(losses: Mapping[str, torch.Tensor], result: Mapping[str, torch.Tensor],
             images: torch.Tensor, payloads: torch.Tensor) -> dict[str, float]:
    logits, residual = result["payload_logits"], result["residual"]
    return {
        "total_loss": losses["total_loss"].item(),
        "payload_bce_loss": losses["payload_loss"].item(),
        "image_fidelity_loss": losses["image_loss"].item(),
        "bit_accuracy": bit_accuracy(logits, payloads).item(),
        "regional_packet_accuracy": regional_packet_accuracy(logits, payloads).item(),
        "image_payload_accuracy": image_payload_accuracy(logits, payloads).item(),
        "psnr": psnr(images, result["watermarked_image"]).item(),
        "max_absolute_residual": residual.abs().max().item(),
    }


def run_tiny_overfit(config: Mapping[str, Any], images: torch.Tensor | None = None,
                     payloads: torch.Tensor | None = None,
                     output_directory: str | Path | None = None) -> dict[str, Any]:
    """Train the existing clean system on one fixed full batch."""
    cfg = dict(config)
    seed = int(cfg.get("seed", 2026))
    seed_everything(seed)
    if int(cfg.get("num_images", 8)) != 8 or int(cfg.get("image_size", 256)) != 256:
        raise ValueError("this experiment requires num_images=8 and image_size=256")
    if images is None:
        images, image_identifiers = _fixed_coco_selection(cfg["coco_directory"], 8, seed)
    else:
        image_identifiers = list(range(8))
    if payloads is None:
        payloads = build_fixed_payloads(8, seed)
    if tuple(images.shape) != (8, 3, 256, 256):
        raise ValueError("images must have shape [8, 3, 256, 256]")
    if tuple(payloads.shape) != (8, 44, 4, 4):
        raise ValueError("payloads must have shape [8, 44, 4, 4]")

    device = _device(str(cfg.get("device", "auto")))
    images = images.detach().clone().to(device)
    payloads = payloads.detach().clone().to(device)
    model = CleanWatermarkSystem(
        base_channels=int(cfg.get("base_channels", 32)),
        message_channels=int(cfg.get("message_channels", 32)),
        residual_alpha=float(cfg.get("residual_alpha", 0.02)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    out = Path(output_directory or cfg.get("output_directory", "outputs/tiny_overfit"))
    out.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_bits = -1.0
    last_metrics: dict[str, float] = {}
    stopped_early = False

    for step in range(1, int(cfg["steps"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = model(images, payloads)
        losses = clean_watermark_loss(
            result["payload_logits"], payloads, images, result["watermarked_image"],
            result["residual"], lambda_payload=float(cfg.get("lambda_payload", 1.0)),
            lambda_image=float(cfg.get("lambda_image", 1.0)),
            lambda_residual=float(cfg.get("lambda_residual", 0.1)),
        )
        losses["total_loss"].backward()
        optimizer.step()
        with torch.no_grad():
            last_metrics = _metrics(losses, result, images, payloads)
        row: dict[str, float | int] = {"step": step, **last_metrics}
        history.append(row)
        evaluation_bundle = {
            "image_identifiers": image_identifiers,
            "payloads": payloads.detach().cpu(),
        }
        checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "step": step, "metrics": last_metrics, "configuration": cfg,
                      "evaluation_bundle": evaluation_bundle}
        if last_metrics["bit_accuracy"] > best_bits:
            best_bits = last_metrics["bit_accuracy"]
            torch.save(checkpoint, out / "best.pt")
        if step == 1 or step % int(cfg["log_every"]) == 0:
            print("step=" + str(step) + " " + " ".join(f"{k}={v:.6f}" for k, v in last_metrics.items()))
        if last_metrics["bit_accuracy"] >= 0.995 and last_metrics["regional_packet_accuracy"] >= 0.95:
            stopped_early = True
            break

    torch.save(checkpoint, out / "last.pt")
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", *METRIC_NAMES))
        writer.writeheader()
        writer.writerows(history)
    summary = {"steps_completed": len(history), "stopped_early": stopped_early,
               "device": str(device), "final_metrics": last_metrics}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
