"""Variable-payload clean communication experiment.

The authentication key exists only while payload banks are being constructed.
Only the resulting authenticated bit tensors are retained by the experiment.
"""

from __future__ import annotations

import csv
import json
import random
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

from kfrag.crypto import ProvenanceToken, create_packets
from kfrag.evaluation.anti_memorization import field_metrics
from kfrag.models import CleanWatermarkSystem
from kfrag.models.losses import clean_watermark_loss, psnr
from kfrag.payload import batch_packets_to_grid
from kfrag.training.tiny_overfit import _device, _fixed_coco_selection, seed_everything


def generate_payload_splits(train_count: int = 256, heldout_count: int = 64,
                            seed: int = 2026) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate disjoint authenticated RS(16,12) payload banks.

    The random experiment key is deliberately local and is neither returned nor
    made deterministic from the public seed.
    """
    if train_count < 1 or heldout_count < 1:
        raise ValueError("payload split sizes must be positive")
    rng = random.Random(seed)
    asset_ids: set[int] = set()
    tokens: list[ProvenanceToken] = []
    while len(tokens) < train_count + heldout_count:
        asset_id = rng.getrandbits(64)
        if asset_id not in asset_ids:
            asset_ids.add(asset_id)
            tokens.append(ProvenanceToken(issuer_id=2, asset_id=asset_id, version=1))
    authentication_material = secrets.token_bytes(32)
    grids = batch_packets_to_grid([create_packets(token, authentication_material) for token in tokens])
    return grids[:train_count], grids[train_count:]


def payload_splits_are_disjoint(training: torch.Tensor, heldout: torch.Tensor) -> bool:
    """Return whether no complete payload occurs in both banks."""
    if training.ndim != 4 or heldout.ndim != 4 or tuple(training.shape[1:]) != (44, 4, 4) \
            or tuple(heldout.shape[1:]) != (44, 4, 4):
        raise ValueError("payload banks must have shape [N, 44, 4, 4]")
    train_rows = {bytes(row.to(torch.uint8).flatten().tolist()) for row in training.cpu()}
    return all(bytes(row.to(torch.uint8).flatten().tolist()) not in train_rows for row in heldout.cpu())


@dataclass
class PairingTracker:
    """Count image/payload assignments and verify that associations vary."""

    num_images: int
    num_payloads: int
    counts: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.counts = torch.zeros(self.num_images, self.num_payloads, dtype=torch.int64)

    def update(self, image_indices: torch.Tensor, payload_indices: torch.Tensor) -> None:
        if image_indices.shape != payload_indices.shape:
            raise ValueError("image and payload index batches must have identical shapes")
        for image_index, payload_index in zip(image_indices.cpu().tolist(), payload_indices.cpu().tolist()):
            self.counts[image_index, payload_index] += 1

    def assignments_vary(self) -> bool:
        """Require every image to see two payloads and every payload two images."""
        return bool(((self.counts > 0).sum(1) >= 2).all() and ((self.counts > 0).sum(0) >= 2).all())

    def assert_assignments_vary(self) -> None:
        if not self.assignments_vary():
            raise AssertionError("image-payload assignments have not varied sufficiently")


def fidelity_weight(step: int, total_steps: int, phase1_fraction: float,
                    phase1_weight: float, phase2_weight: float) -> float:
    """Low phase-one weight followed by a linear fidelity ramp."""
    boundary = max(1, int(total_steps * phase1_fraction))
    if step <= boundary:
        return phase1_weight
    progress = min(1.0, (step - boundary) / max(1, total_steps - boundary))
    return phase1_weight + progress * (phase2_weight - phase1_weight)


def should_early_stop(training_metrics: Mapping[str, float], heldout_metrics: Mapping[str, float],
                      original_metrics: Mapping[str, float], pairing_valid: bool = True,
                      non_index_threshold: float = .99, regional_threshold: float = .90,
                      false_positive_threshold: float = .01) -> bool:
    """Apply stopping thresholds exclusively to held-out and negative controls."""
    del training_metrics
    return bool(pairing_valid
        and heldout_metrics["non_index_accuracy"] >= non_index_threshold
        and heldout_metrics["regional_packet_accuracy"] >= regional_threshold
        and original_metrics["regional_packet_accuracy"] <= false_positive_threshold)


def _evaluate(model: torch.nn.Module, images: torch.Tensor, payloads: torch.Tensor,
              batch_size: int, generator: torch.Generator, original: bool = False) -> dict[str, float | int]:
    all_logits, all_targets, all_watermarked, all_originals, maxima = [], [], [], [], []
    order = torch.randperm(len(payloads), generator=generator)
    image_order = torch.randint(len(images), (len(payloads),), generator=generator)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(payloads), batch_size):
            chosen_payloads = payloads[order[start:start + batch_size]].to(images.device)
            chosen_images = images[image_order[start:start + batch_size].to(images.device)]
            if original:
                logits = model.decoder(chosen_images)
                watermarked, residual = chosen_images, torch.zeros_like(chosen_images)
            else:
                result = model(chosen_images, chosen_payloads)
                logits, watermarked, residual = result["payload_logits"], result["watermarked_image"], result["residual"]
            all_logits.append(logits.cpu()); all_targets.append(chosen_payloads.cpu())
            all_watermarked.append(watermarked.cpu()); all_originals.append(chosen_images.cpu())
            maxima.append(float(residual.abs().max().item()))
    metrics = field_metrics(torch.cat(all_logits), torch.cat(all_targets))
    metrics.pop("index_bit_accuracy"); metrics.pop("mean_decoder_confidence"); metrics.pop("exact_regional_packets")
    metrics["tag_accuracy"] = metrics["authentication_tag_accuracy"]
    metrics["psnr"] = float(psnr(torch.cat(all_originals), torch.cat(all_watermarked)).item())
    metrics["maximum_residual"] = max(maxima)
    return metrics


def run_variable_payload(config: Mapping[str, Any], images: torch.Tensor | None = None,
                         payload_splits: tuple[torch.Tensor, torch.Tensor] | None = None,
                         output_directory: str | Path | None = None) -> dict[str, Any]:
    """Jointly train the clean encoder/decoder with independently sampled pairs."""
    cfg = dict(config)
    sensitive_names = [name for name in cfg if "secret" in str(name).lower() or str(name).lower().endswith("key")]
    if sensitive_names:
        raise ValueError("authentication material must not be supplied in configuration")
    seed = int(cfg.get("seed", 2026)); seed_everything(seed)
    if int(cfg.get("num_images", 8)) != 8:
        raise ValueError("the variable-payload experiment requires exactly 8 images")
    if images is None:
        images, identifiers = _fixed_coco_selection(cfg["coco_directory"], 8, seed)
    else:
        identifiers = list(range(8))
    if tuple(images.shape) != (8, 3, 256, 256):
        raise ValueError("images must have shape [8, 3, 256, 256]")
    train_count, heldout_count = int(cfg.get("train_payloads", 256)), int(cfg.get("heldout_payloads", 64))
    training, heldout = payload_splits or generate_payload_splits(train_count, heldout_count, seed)
    if tuple(training.shape) != (train_count, 44, 4, 4) or tuple(heldout.shape) != (heldout_count, 44, 4, 4):
        raise ValueError("payload banks have incorrect shapes")
    if not payload_splits_are_disjoint(training, heldout):
        raise ValueError("held-out payloads overlap training payloads")

    device = _device(str(cfg.get("device", "auto"))); images = images.to(device)
    model = CleanWatermarkSystem(base_channels=int(cfg.get("base_channels", 32)),
        message_channels=int(cfg.get("message_channels", 32)), residual_alpha=float(cfg.get("residual_alpha", .02))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", .0002)),
                                  weight_decay=float(cfg.get("weight_decay", .0001)))
    out = Path(output_directory or cfg.get("output_directory", "outputs/variable_payload")); out.mkdir(parents=True, exist_ok=True)
    steps, batch_size = int(cfg.get("steps", 5000)), int(cfg.get("batch_size", 8))
    sample_rng = torch.Generator().manual_seed(seed); eval_rng = torch.Generator().manual_seed(seed + 1)
    tracker = PairingTracker(8, train_count); history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0); stopped = False; last_evaluation: dict[str, Any] = {}

    for step in range(1, steps + 1):
        image_indices = torch.randint(8, (batch_size,), generator=sample_rng)
        payload_indices = torch.randint(train_count, (batch_size,), generator=sample_rng)
        tracker.update(image_indices, payload_indices)
        batch_images = images[image_indices.to(device)]; batch_payloads = training[payload_indices].to(device)
        image_weight = fidelity_weight(step, steps, float(cfg.get("phase1_fraction", .4)),
            float(cfg.get("phase1_image_weight", .05)), float(cfg.get("phase2_image_weight", 1.0)))
        model.train(); optimizer.zero_grad(set_to_none=True); result = model(batch_images, batch_payloads)
        losses = clean_watermark_loss(result["payload_logits"], batch_payloads, batch_images,
            result["watermarked_image"], result["residual"], lambda_payload=float(cfg.get("lambda_payload", 1.0)),
            lambda_image=image_weight, lambda_residual=float(cfg.get("lambda_residual", .1)))
        losses["total_loss"].backward(); optimizer.step()

        if step % int(cfg.get("evaluation_every", 100)) == 0 or step == steps:
            train_metrics = _evaluate(model, images, training, batch_size, eval_rng)
            heldout_metrics = _evaluate(model, images, heldout, batch_size, eval_rng)
            original_metrics = _evaluate(model, images, heldout, batch_size, eval_rng, original=True)
            last_evaluation = {"training_payloads": train_metrics, "heldout_payloads": heldout_metrics,
                               "original_unwatermarked": original_metrics}
            row = {"step": step, "image_fidelity_weight": image_weight,
                   **{f"{condition}.{name}": value for condition, metrics in last_evaluation.items() for name, value in metrics.items()}}
            history.append(row)
            checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "step": step, "metrics": last_evaluation, "configuration": cfg,
                "evaluation_bundle": {"image_identifiers": identifiers, "training_payloads": training,
                                      "heldout_payloads": heldout}, "pairing_counts": tracker.counts}
            score = (float(heldout_metrics["regional_packet_accuracy"]), float(heldout_metrics["non_index_accuracy"]))
            if score > best_score:
                best_score = score; torch.save(checkpoint, out / "best.pt")
            torch.save(checkpoint, out / "last.pt")
            if should_early_stop(train_metrics, heldout_metrics, original_metrics, tracker.assignments_vary(),
                    float(cfg.get("early_stop_non_index", .99)), float(cfg.get("early_stop_regional", .90)),
                    float(cfg.get("early_stop_original_regional", .01))):
                stopped = True; break
        if step == 1 or step % int(cfg.get("log_every", 50)) == 0:
            print(f"step={step} total_loss={losses['total_loss'].item():.6f} image_fidelity_weight={image_weight:.6f}")

    tracker.assert_assignments_vary()
    if history:
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    summary = {"steps_completed": step, "stopped_early": stopped, "device": str(device),
               "pairing_assignments_vary": True, "final_evaluation": last_evaluation}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
