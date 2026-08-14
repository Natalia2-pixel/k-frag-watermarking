"""Four-condition anti-memorization evaluation for the clean watermark model."""

from __future__ import annotations

import csv
import json
import secrets
from pathlib import Path
from typing import Any, Mapping

import torch

from kfrag.crypto import ProvenanceToken, create_packets
from kfrag.data import CocoImageDataset
from kfrag.models import CleanWatermarkSystem
from kfrag.models.losses import psnr
from kfrag.payload import batch_packets_to_grid

FIELD_SLICES = {"index": slice(0, 4), "coded_symbol": slice(4, 12),
                "authentication_tag": slice(12, 44), "non_index": slice(4, 44)}


def circular_shift_targets(payloads: torch.Tensor) -> torch.Tensor:
    """Shift targets by one image, a derangement for batches larger than one."""
    if payloads.ndim != 4 or payloads.shape[0] < 2:
        raise ValueError("payloads must have shape [B, 44, 4, 4] with B >= 2")
    return torch.roll(payloads, shifts=1, dims=0)


def _validate(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape or logits.ndim != 4 or tuple(logits.shape[1:]) != (44, 4, 4):
        raise ValueError("logits and targets must both have shape [B, 44, 4, 4]")


def field_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    """Compute protocol-field and exact-match metrics from decoder logits."""
    _validate(logits, targets)
    correct = (torch.sigmoid(logits) >= 0.5).eq(targets >= 0.5)
    exact_regions = correct.all(dim=1)
    probabilities = torch.sigmoid(logits)
    confidence = torch.maximum(probabilities, 1 - probabilities).mean()
    return {
        "overall_bit_accuracy": correct.float().mean().item(),
        "index_bit_accuracy": correct[:, FIELD_SLICES["index"]].float().mean().item(),
        "coded_symbol_accuracy": correct[:, FIELD_SLICES["coded_symbol"]].float().mean().item(),
        "authentication_tag_accuracy": correct[:, FIELD_SLICES["authentication_tag"]].float().mean().item(),
        "non_index_accuracy": correct[:, FIELD_SLICES["non_index"]].float().mean().item(),
        "regional_packet_accuracy": exact_regions.float().mean().item(),
        "image_payload_accuracy": correct.flatten(1).all(1).float().mean().item(),
        "mean_decoder_confidence": confidence.item(),
        "exact_regional_packets": int(exact_regions.sum().item()),
    }


def _new_payloads(training_payloads: torch.Tensor) -> torch.Tensor:
    """Create authenticated evaluation payloads with an ephemeral, unreturned key."""
    key = secrets.token_bytes(32)
    seen: set[int] = set()
    grids: list[torch.Tensor] = []
    for original in training_payloads.cpu():
        while True:
            asset_id = secrets.randbits(64)
            if asset_id not in seen:
                seen.add(asset_id)
                grid = batch_packets_to_grid([[*create_packets(ProvenanceToken(2, asset_id, 1), key)]])[0]
                if not torch.equal(grid, original):
                    grids.append(grid)
                    break
    return torch.stack(grids)


def evaluate_anti_memorization(model: torch.nn.Module, images: torch.Tensor,
                                training_payloads: torch.Tensor) -> dict[str, dict[str, float | int]]:
    """Run all conditions. The original-image condition calls only the decoder."""
    if tuple(images.shape[1:]) != (3, 256, 256) or tuple(training_payloads.shape) != (len(images), 44, 4, 4):
        raise ValueError("expected images [B, 3, 256, 256] and payloads [B, 44, 4, 4]")
    device = images.device
    training_payloads = training_payloads.to(device)
    unseen = _new_payloads(training_payloads).to(device)
    model.eval()
    with torch.no_grad():
        correct = model(images, training_payloads)
        original_logits = model.decoder(images)
        unseen_result = model(images, unseen)
    results = {
        "correct_watermarked": field_metrics(correct["payload_logits"], training_payloads),
        "original_unwatermarked": field_metrics(original_logits, training_payloads),
        "shuffled_targets": field_metrics(correct["payload_logits"], circular_shift_targets(training_payloads)),
        "unseen_payloads": field_metrics(unseen_result["payload_logits"], unseen),
    }
    for name, output in (("correct_watermarked", correct), ("unseen_payloads", unseen_result)):
        results[name]["psnr"] = psnr(images, output["watermarked_image"]).item()
        results[name]["maximum_absolute_residual"] = output["residual"].abs().max().item()
    return results


def _load_bundle(checkpoint: Mapping[str, Any], coco_directory: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    bundle = checkpoint.get("evaluation_bundle")
    message = ("checkpoint is missing a valid evaluation_bundle containing the original eight payloads "
               "and image identifiers; rerun tiny overfit once to create an evaluable checkpoint")
    if not isinstance(bundle, Mapping) or set(("payloads", "image_identifiers")) - set(bundle):
        raise ValueError(message)
    payloads, identifiers = bundle["payloads"], bundle["image_identifiers"]
    if not isinstance(payloads, torch.Tensor) or tuple(payloads.shape) != (8, 44, 4, 4):
        raise ValueError(message)
    if not isinstance(identifiers, list) or len(identifiers) != 8 or not all(isinstance(x, str) for x in identifiers):
        raise ValueError(message)
    dataset = CocoImageDataset(coco_directory)
    by_path = {sample: index for index, sample in enumerate(
        [path.relative_to(dataset.image_directory).as_posix() for path in dataset.image_paths])}
    try:
        images = torch.stack([dataset[by_path[name]]["image"] for name in identifiers])
    except KeyError as exc:
        raise FileNotFoundError(f"evaluation image from checkpoint was not found: {exc.args[0]}") from exc
    return images, payloads


def _write_outputs(results: Mapping[str, Mapping[str, float | int]], output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    all_keys = ["condition", *dict.fromkeys(k for row in results.values() for k in row)]
    with (output_directory / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for condition, metrics in results.items(): writer.writerow({"condition": condition, **metrics})
    import matplotlib.pyplot as plt
    names = list(results)
    metrics = ["non_index_accuracy", "authentication_tag_accuracy", "regional_packet_accuracy", "image_payload_accuracy"]
    x = torch.arange(len(names)).numpy(); width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics): ax.bar(x + (i - 1.5) * width, [results[n][metric] for n in names], width, label=metric)
    ax.set_xticks(x, names, rotation=15); ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.legend(); fig.tight_layout()
    fig.savefig(output_directory / "comparison.png", dpi=160); plt.close(fig)


def evaluate_checkpoint(config: Mapping[str, Any], checkpoint_path: str | Path) -> dict[str, dict[str, float | int]]:
    """Load a trained checkpoint, replay its exact training inputs, and save results."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("model_state"), Mapping):
        raise ValueError("malformed checkpoint: expected a mapping containing model_state")
    images, payloads = _load_bundle(checkpoint, config["coco_directory"])
    saved = checkpoint.get("configuration", {})
    model = CleanWatermarkSystem(base_channels=int(saved.get("base_channels", 32)),
        message_channels=int(saved.get("message_channels", 32)), residual_alpha=float(saved.get("residual_alpha", .02)))
    try: model.load_state_dict(checkpoint["model_state"])
    except (RuntimeError, TypeError) as exc: raise ValueError(f"malformed checkpoint model_state: {exc}") from exc
    device_name = str(config.get("device", "auto")); device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    model.to(device); results = evaluate_anti_memorization(model, images.to(device), payloads.to(device))
    _write_outputs(results, Path(config.get("output_directory", "outputs/anti_memorization")))
    return results
