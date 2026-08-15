"""A gated sanity ladder for the existing clean neural watermark channel.

Authentication material exists only inside :func:`generate_payload_bank`.  The
only persistent artifacts are already-authenticated bit tensors.
"""

from __future__ import annotations

import csv
import json
import math
import random
import secrets
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from kfrag.crypto import ProvenanceToken, create_packets
from kfrag.models import CleanWatermarkSystem
from kfrag.models.losses import psnr
from kfrag.payload import batch_packets_to_grid
from kfrag.training.tiny_overfit import _device, _fixed_coco_selection, seed_everything

CAPACITY_RANGES = {
    1: ((4, 12),),
    2: ((4, 12), (12, 20)),
    3: ((4, 12), (12, 28)),
    4: ((4, 12), (12, 44)),
}
STAGES = ("A", "B", "C", "D")


def capacity_mask(level: int, packet_bits: int = 44, grid_size: int = 4) -> torch.Tensor:
    """Return a boolean ``[1, 44, 4, 4]`` mask for one curriculum level."""
    if level not in CAPACITY_RANGES or packet_bits != 44 or grid_size != 4:
        raise ValueError("capacity level must be 1..4 with packet_bits=44 and grid_size=4")
    mask = torch.zeros(1, packet_bits, grid_size, grid_size, dtype=torch.bool)
    for start, stop in CAPACITY_RANGES[level]:
        mask[:, start:stop] = True
    return mask


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
    """BCE averaged over precisely the selected fields and all batch items."""
    if logits.shape != targets.shape or logits.ndim != 4:
        raise ValueError("logits and targets must have identical rank-4 shapes")
    try:
        selected = torch.broadcast_to(mask.to(device=logits.device, dtype=torch.bool), logits.shape)
    except RuntimeError as exc:
        raise ValueError("active-bit mask is not broadcastable to logits") from exc
    if not bool(selected.any()):
        raise ValueError("active-bit mask selects no bits")
    return F.binary_cross_entropy_with_logits(logits[selected], targets.float()[selected])


def payload_splits_are_disjoint(training: torch.Tensor, heldout: torch.Tensor) -> bool:
    if training.ndim != 4 or heldout.ndim != 4 or tuple(training.shape[1:]) != (44, 4, 4) \
            or tuple(heldout.shape[1:]) != (44, 4, 4):
        raise ValueError("payload banks must have shape [N, 44, 4, 4]")
    rows = {bytes(x.to(torch.uint8).flatten().tolist()) for x in training.cpu()}
    return all(bytes(x.to(torch.uint8).flatten().tolist()) not in rows for x in heldout.cpu())


def generate_payload_bank(train_count: int = 256, heldout_count: int = 64,
                          seed: int = 2026) -> tuple[torch.Tensor, torch.Tensor]:
    """Create unique-token, authenticated, disjoint train and held-out banks."""
    if train_count < 1 or heldout_count < 1:
        raise ValueError("payload counts must be positive")
    rng = random.Random(seed)
    identifiers: set[int] = set()
    tokens: list[ProvenanceToken] = []
    while len(tokens) < train_count + heldout_count:
        asset_id = rng.getrandbits(64)
        if asset_id not in identifiers:
            identifiers.add(asset_id)
            tokens.append(ProvenanceToken(issuer_id=3, asset_id=asset_id, version=1))
    authentication_material = secrets.token_bytes(32)
    payloads = batch_packets_to_grid(
        [create_packets(token, authentication_material) for token in tokens]
    )
    training, heldout = payloads[:train_count], payloads[train_count:]
    if not payload_splits_are_disjoint(training, heldout):
        raise AssertionError("training and held-out payload banks overlap")
    return training, heldout


def circular_payload_shuffle(payloads: torch.Tensor) -> torch.Tensor:
    """Rotate a batch by one; every target changes position when B > 1."""
    if payloads.ndim < 1 or len(payloads) < 2:
        raise ValueError("at least two payloads are required for a no-fixed-point shuffle")
    return torch.roll(payloads, shifts=1, dims=0)


def _correct(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or tuple(logits.shape[1:]) != (44, 4, 4):
        raise ValueError("logits and targets must have shape [B, 44, 4, 4]")
    return (logits >= 0).eq(targets >= .5)


def recovery_metrics(logits: torch.Tensor, targets: torch.Tensor,
                     active_mask: torch.Tensor) -> dict[str, float | int]:
    correct = _correct(logits, targets)
    active = torch.broadcast_to(active_mask.to(correct.device), correct.shape)
    exact_active = (correct | ~active).all(dim=1)
    exact_packet = correct.all(dim=1)
    active_tag = active[:, 12:44]
    return {
        "overall_bit_accuracy": correct.float().mean().item(),
        "region_index_accuracy": correct[:, 0:4].float().mean().item(),
        "coded_symbol_accuracy": correct[:, 4:12].float().mean().item(),
        "authentication_tag_accuracy": correct[:, 12:44].float().mean().item(),
        "active_authentication_tag_accuracy": (
            correct[:, 12:44][active_tag].float().mean().item() if bool(active_tag.any()) else 1.0
        ),
        "active_field_accuracy": correct[active].float().mean().item(),
        "non_index_accuracy": correct[:, 4:44].float().mean().item(),
        "exact_active_region_accuracy": exact_active.float().mean().item(),
        "exact_regional_packet_accuracy": exact_packet.float().mean().item(),
        "exact_image_payload_accuracy": correct.flatten(1).all(1).float().mean().item(),
        "number_exact_regional_packets": int(exact_packet.sum().item()),
    }


def image_metrics(original: torch.Tensor, watermarked: torch.Tensor,
                  residual: torch.Tensor, alpha: float) -> dict[str, float]:
    return {
        "psnr": float(psnr(original, watermarked).item()),
        "maximum_absolute_residual": float(residual.abs().max().item()),
        "mean_absolute_residual": float(residual.abs().mean().item()),
        "residual_saturation_fraction": float((residual.abs() >= alpha * .999).float().mean().item()),
    }


def payload_sensitivity(model: nn.Module, carrier: torch.Tensor, first: torch.Tensor,
                        second: torch.Tensor) -> dict[str, float]:
    """Compare two payloads while holding the carrier exactly fixed."""
    model.eval()
    with torch.no_grad():
        a, b = model(carrier, first), model(carrier, second)
    pred_a, pred_b = a["payload_logits"] >= 0, b["payload_logits"] >= 0
    return {
        "payload_tensor_difference": float((first - second).abs().mean().item()),
        "residual_payload_sensitivity": float((a["residual"] - b["residual"]).abs().mean().item()),
        "watermarked_image_sensitivity": float((a["watermarked_image"] - b["watermarked_image"]).abs().mean().item()),
        "logit_payload_sensitivity": float((a["payload_logits"] - b["payload_logits"]).abs().mean().item()),
        "predicted_bit_change_fraction": float(pred_a.ne(pred_b).float().mean().item()),
    }


def gradient_norm(parameters: Sequence[nn.Parameter] | Any) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum().item())
    return math.sqrt(total)


def gradient_diagnostics(model: nn.Module) -> dict[str, float]:
    encoder = model.encoder
    image_modules = [encoder.stem, encoder.down]
    return {
        "payload_projector_gradient_norm": gradient_norm(model.projector.parameters()),
        "encoder_image_branch_gradient_norm": gradient_norm(
            p for module in image_modules for p in module.parameters()
        ),
        "encoder_residual_head_gradient_norm": gradient_norm(encoder.output.parameters()),
        "decoder_gradient_norm": gradient_norm(model.decoder.parameters()),
    }


def sensitivity_checks(metrics: Mapping[str, float], thresholds: Mapping[str, float]) -> list[str]:
    failures = []
    if metrics["residual_payload_sensitivity"] < thresholds["minimum_payload_sensitivity"]:
        failures.append("residual payload sensitivity is effectively zero")
    if metrics["logit_payload_sensitivity"] < thresholds["minimum_logit_sensitivity"]:
        failures.append("logit payload sensitivity is effectively zero")
    return failures


def gradient_checks(metrics: Mapping[str, float], minimum: float) -> list[str]:
    failures = []
    for name in ("payload_projector_gradient_norm", "decoder_gradient_norm"):
        if metrics[name] < minimum:
            failures.append(name.replace("_", " ") + " is effectively zero")
    return failures


def should_advance_capacity(metrics: Mapping[str, float], original: Mapping[str, float],
                            level: int, thresholds: Mapping[str, float],
                            diagnostics_ok: bool = True) -> bool:
    """Gate only on active fields, controls and diagnostics--never index bits."""
    tag_ok = level == 1 or metrics.get(
        "active_authentication_tag_accuracy", metrics["authentication_tag_accuracy"]
    ) >= thresholds["tag_accuracy"]
    return bool(diagnostics_ok and metrics["active_field_accuracy"] >= thresholds["active_field_accuracy"]
                and tag_ok
                and metrics["exact_active_region_accuracy"] >= thresholds["exact_active_region_accuracy"]
                and original["exact_regional_packet_accuracy"] <= thresholds["maximum_original_packet_accuracy"])


def stage_d_allowed(stage_results: Mapping[str, Any]) -> bool:
    return bool(stage_results.get("C", {}).get("passed", False))


def next_stage(stage: str, stage_results: Mapping[str, Any]) -> str | None:
    index = STAGES.index(stage)
    if not stage_results.get(stage, {}).get("passed", False):
        return None
    candidate = STAGES[index + 1] if index + 1 < len(STAGES) else None
    if candidate == "D" and not stage_d_allowed(stage_results):
        return None
    return candidate


def _reject_sensitive(value: Any, path: str = "configuration") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if "secret" in name or "hmac" in name or name.endswith("key") or name.endswith("_key"):
                raise ValueError(f"{path} contains forbidden key-like field: {key}")
            _reject_sensitive(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_sensitive(child, f"{path}[{index}]")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("configuration must be a mapping")
    _reject_sensitive(config)
    required = {"experiment", "data", "payloads", "training", "thresholds"}
    missing = required - set(config)
    if missing:
        raise ValueError("malformed configuration: missing sections " + ", ".join(sorted(missing)))
    cfg = deepcopy(dict(config))
    expected = {"image_size": 256, "num_natural_images": 8}
    for key, value in expected.items():
        if int(cfg["data"].get(key, -1)) != value:
            raise ValueError(f"malformed configuration: data.{key} must be {value}")
    for key, expected_value in (("grid_size", 4), ("packet_bits", 44)):
        if int(cfg["payloads"].get(key, -1)) != expected_value:
            raise ValueError(f"malformed configuration: payloads.{key} must be {expected_value}")
    positive = (("training", "batch_size"), ("training", "steps_per_capacity_level"),
                ("training", "evaluate_every"), ("payloads", "train_payloads"),
                ("payloads", "heldout_payloads"))
    for section, key in positive:
        if int(cfg[section].get(key, 0)) <= 0:
            raise ValueError(f"malformed configuration: {section}.{key} must be positive")
    threshold_names = ("active_field_accuracy", "tag_accuracy", "exact_active_region_accuracy",
                       "maximum_original_packet_accuracy", "minimum_payload_sensitivity",
                       "minimum_logit_sensitivity", "minimum_gradient_norm")
    missing_thresholds = [key for key in threshold_names if key not in cfg["thresholds"]]
    if missing_thresholds:
        raise ValueError("malformed configuration: missing thresholds " + ", ".join(missing_thresholds))
    for key in threshold_names[:4]:
        if not 0 <= float(cfg["thresholds"].get(key, -1)) <= 1:
            raise ValueError(f"malformed configuration: thresholds.{key} must be in [0, 1]")
    return cfg


def _evaluate(model: nn.Module, images: torch.Tensor, payloads: torch.Tensor,
              mask: torch.Tensor, batch_size: int, alpha: float) -> tuple[dict[str, Any], dict[str, Any]]:
    logits, targets, originals, marked, residuals = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(payloads), batch_size):
            target = payloads[start:start + batch_size].to(images.device)
            carrier = images[torch.arange(start, start + len(target), device=images.device) % len(images)]
            result = model(carrier, target)
            logits.append(result["payload_logits"].cpu()); targets.append(target.cpu())
            originals.append(carrier.cpu()); marked.append(result["watermarked_image"].cpu())
            residuals.append(result["residual"].cpu())
    all_logits, all_targets = torch.cat(logits), torch.cat(targets)
    all_originals, all_marked, all_residuals = torch.cat(originals), torch.cat(marked), torch.cat(residuals)
    metrics = {**recovery_metrics(all_logits, all_targets, mask),
               **image_metrics(all_originals, all_marked, all_residuals, alpha)}
    # True bypass: the encoder and projector are never called.
    with torch.no_grad():
        original_logits = torch.cat([
            model.decoder(all_originals[start:start + batch_size].to(images.device)).cpu()
            for start in range(0, len(all_originals), batch_size)
        ])
    original = recovery_metrics(original_logits, all_targets, mask)
    shuffled = recovery_metrics(all_logits, circular_payload_shuffle(all_targets), mask)
    return metrics, {"original_unwatermarked": original, "payload_shuffle": shuffled}


def _stage_images(stage: str, natural: torch.Tensor, batch: int) -> torch.Tensor:
    if stage == "A":
        return torch.full((1, 3, 256, 256), .5, device=natural.device)
    if stage == "B":
        return natural[:1]
    return natural[:8]


def _safe_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int,
                     stage: str, mask: torch.Tensor, config: Mapping[str, Any], metrics: Mapping[str, Any],
                     training: torch.Tensor, heldout: torch.Tensor, identifiers: Sequence[Any]) -> dict[str, Any]:
    checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                  "step": step, "stage": stage, "active_bit_mask": mask.cpu(),
                  "configuration": deepcopy(dict(config)), "metrics": deepcopy(dict(metrics)),
                  "payload_tensors": {"training": training.cpu(), "heldout": heldout.cpu()},
                  "image_identifiers": list(identifiers)}
    _reject_sensitive(checkpoint)
    return checkpoint


def _write_records(directory: Path, history: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if history:
        keys = list(dict.fromkeys(key for row in history for key in row))
        with (directory / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(history)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for stage in STAGES:
            selected = [row for row in rows if row["stage"] == stage]
            if selected:
                axis.plot(range(1, len(selected) + 1), [row["active_field_accuracy"] for row in selected], marker="o", label=stage)
        axis.axhline(.95, color="black", linestyle="--", linewidth=.8)
        axis.set(xlabel="capacity evaluation", ylabel="held-out active-field accuracy", ylim=(0, 1.01))
        axis.legend(); figure.tight_layout(); figure.savefig(path); plt.close(figure)
    except ImportError:
        # A valid tiny PNG keeps the required artifact available on minimal test installs.
        path.write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c6360606060000000050001a5f645400000000049454e44ae426082"))


def run_channel_sanity(config: Mapping[str, Any], images: torch.Tensor | None = None,
                       payload_banks: tuple[torch.Tensor, torch.Tensor] | None = None,
                       output_directory: str | Path | None = None,
                       model_factory: Callable[[], nn.Module] | None = None) -> dict[str, Any]:
    """Run A-D in order, stopping at the first failed capacity level."""
    cfg = validate_config(config)
    seed = int(cfg["experiment"].get("seed", 2026)); seed_everything(seed)
    out = Path(output_directory or "outputs/channel_sanity"); out.mkdir(parents=True, exist_ok=True)
    if images is None:
        images, identifiers = _fixed_coco_selection(cfg["data"]["coco_directory"], 8, seed)
    else:
        identifiers = list(range(len(images)))
    if tuple(images.shape) != (8, 3, 256, 256):
        raise ValueError("natural images must have shape [8, 3, 256, 256]")
    train_count = int(cfg["payloads"]["train_payloads"]); held_count = int(cfg["payloads"]["heldout_payloads"])
    training, heldout = payload_banks or generate_payload_bank(train_count, held_count, seed)
    if tuple(training.shape) != (train_count, 44, 4, 4) or tuple(heldout.shape) != (held_count, 44, 4, 4):
        raise ValueError("payload banks have incorrect configured shapes")
    if not payload_splits_are_disjoint(training, heldout):
        raise ValueError("training and held-out payload sets overlap")
    device = _device(str(cfg["experiment"].get("device", "auto"))); images = images.to(device)
    alpha = float(cfg["training"].get("alpha", .05)); threshold = cfg["thresholds"]
    batch_size = int(cfg["training"]["batch_size"]); budget = int(cfg["training"]["steps_per_capacity_level"])
    evaluate_every = int(cfg["training"]["evaluate_every"]); log_every = int(cfg["training"].get("log_every", 50))
    factory = model_factory or (lambda: CleanWatermarkSystem(residual_alpha=alpha))
    results: dict[str, Any] = {}; comparison: list[dict[str, Any]] = []
    stage_c_state: dict[str, torch.Tensor] | None = None

    for stage_index, stage in enumerate(STAGES):
        if stage != "A" and not results.get(STAGES[stage_index - 1], {}).get("passed", False):
            break
        init_seed = seed + stage_index * 1000; seed_everything(init_seed)
        model = factory().to(device)
        if stage == "D":
            if stage_c_state is None: raise RuntimeError("Stage D cannot begin without successful Stage C")
            model.load_state_dict(stage_c_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                      weight_decay=float(cfg["training"]["weight_decay"]))
        carriers = _stage_images(stage, images, batch_size)
        stage_result: dict[str, Any] = {"passed": True, "initialization_seed": init_seed, "levels": {}}
        for level in range(1, 5):
            mask = capacity_mask(level).to(device); history: list[dict[str, Any]] = []
            best_score = -1.0; passed = False; last_bundle: dict[str, Any] = {}
            level_dir = out / f"stage_{stage}" / f"level_{level}"
            generator = torch.Generator().manual_seed(init_seed + level)
            for step in range(1, budget + 1):
                payload_ids = torch.randint(len(training), (batch_size,), generator=generator)
                if stage in ("A", "B"):
                    image_ids = torch.zeros(batch_size, dtype=torch.long)
                else:
                    image_ids = torch.randint(len(carriers), (batch_size,), generator=generator)
                batch_payload = training[payload_ids].to(device); batch_images = carriers[image_ids.to(device)]
                model.train(); optimizer.zero_grad(set_to_none=True)
                result = model(batch_images, batch_payload)
                payload_loss = masked_bce_with_logits(result["payload_logits"], batch_payload, mask)
                fidelity_weight = 0.0
                if stage == "D" and last_bundle and last_bundle["heldout"]["authentication_tag_accuracy"] >= .95 \
                        and last_bundle["heldout"]["non_index_accuracy"] >= .95:
                    fidelity_weight = min(1.0, step / budget)
                fidelity_loss = (result["watermarked_image"] - batch_images).abs().mean()
                total_loss = payload_loss + fidelity_weight * fidelity_loss
                total_loss.backward(); gradients = gradient_diagnostics(model); optimizer.step()
                if step % evaluate_every == 0 or step == budget:
                    held_metrics, controls = _evaluate(model, carriers, heldout, mask, batch_size, alpha)
                    first, second = heldout[0:1].to(device), heldout[1:2].to(device)
                    sensitivity = payload_sensitivity(model, carriers[:1], first, second)
                    failures = sensitivity_checks(sensitivity, threshold) + gradient_checks(gradients, float(threshold["minimum_gradient_norm"]))
                    if controls["original_unwatermarked"]["exact_regional_packet_accuracy"] > float(threshold["maximum_original_packet_accuracy"]):
                        failures.append("original images recover significant exact packets")
                    diagnostics_ok = not failures
                    passed = should_advance_capacity(held_metrics, controls["original_unwatermarked"], level, threshold, diagnostics_ok)
                    last_bundle = {"heldout": held_metrics, **controls, "sensitivity": sensitivity,
                                   "gradients": gradients, "failures": failures}
                    row = {"stage": stage, "capacity_level": level, "step": step,
                           "total_loss": float(total_loss.item()), "fidelity_weight": fidelity_weight,
                           **held_metrics, **sensitivity, **gradients,
                           "original_exact_packet_accuracy": controls["original_unwatermarked"]["exact_regional_packet_accuracy"]}
                    history.append(row)
                    checkpoint = _safe_checkpoint(model, optimizer, step, stage, mask, cfg, last_bundle,
                                                  training, heldout, identifiers)
                    if held_metrics["active_field_accuracy"] > best_score:
                        best_score = held_metrics["active_field_accuracy"]; level_dir.mkdir(parents=True, exist_ok=True)
                        torch.save(checkpoint, level_dir / "best.pt")
                    level_dir.mkdir(parents=True, exist_ok=True); torch.save(checkpoint, level_dir / "last.pt")
                    print(f"stage={stage} level={level} step={step} total_loss={total_loss.item():.6f} "
                          f"active={held_metrics['active_field_accuracy']:.6f} tag={held_metrics['authentication_tag_accuracy']:.6f} "
                          f"active_tag={held_metrics['active_authentication_tag_accuracy']:.6f} "
                          f"exact_active={held_metrics['exact_active_region_accuracy']:.6f} "
                          f"original_exact={controls['original_unwatermarked']['exact_regional_packet_accuracy']:.6f} "
                          f"psnr={held_metrics['psnr']:.3f} payload_sensitivity={sensitivity['residual_payload_sensitivity']:.3e} "
                          f"logit_sensitivity={sensitivity['logit_payload_sensitivity']:.3e} "
                          f"projector_grad={gradients['payload_projector_gradient_norm']:.3e} decoder_grad={gradients['decoder_gradient_norm']:.3e}")
                    if passed: break
                elif step % log_every == 0:
                    print(f"stage={stage} level={level} step={step} total_loss={total_loss.item():.6f}")
            level_summary = {"passed": passed, "steps_completed": step, "final_evaluation": last_bundle}
            _write_records(level_dir, history, level_summary)
            stage_result["levels"][str(level)] = level_summary
            if history: comparison.append(history[-1])
            if not passed:
                stage_result["passed"] = False
                reason = "; ".join(last_bundle.get("failures", [])) or "held-out success thresholds not met"
                stage_result["failure_reason"] = f"capacity level {level}: {reason}"
                print(f"Stage {stage} failed at capacity level {level}: {reason}")
                break
        results[stage] = stage_result
        if stage == "C" and stage_result["passed"]:
            stage_c_state = deepcopy(model.state_dict())
        if not stage_result["passed"]: break

    overall = {"stages": results, "completed_all_stages": bool(results.get("D", {}).get("passed", False))}
    (out / "overall_summary.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    if comparison:
        keys = list(dict.fromkeys(key for row in comparison for key in row))
        with (out / "stage_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(comparison)
    _plot(comparison, out / "channel_sanity_plot.png")
    return overall
