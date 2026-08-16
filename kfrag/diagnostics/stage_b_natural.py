"""Stage B: blind regional-symbol communication on one natural image.

This module deliberately contains no authentication, erasure coding, attack,
crop, synchronization, or provenance-state logic.
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from kfrag.data import CocoImageDataset
from kfrag.diagnostics.channel_repair import carrier_change_metrics
from kfrag.diagnostics.channel_sanity import capacity_mask, circular_payload_shuffle
from kfrag.models import StructuredChannelSystem, StructuredRegionalDecoder
from kfrag.models.losses import psnr

ACTIVE = slice(4, 12)
INACTIVE_METRICS = {
    "region_index_accuracy": None,
    "authentication_tag_accuracy": None,
    "exact_44bit_regional_packet_accuracy": None,
    "number_of_exact_complete_packets": None,
    "exact_image_payload_accuracy": None,
    "rs_identity_reconstruction_accuracy": None,
}


def random_symbol_payloads(count: int, generator: torch.Generator) -> torch.Tensor:
    """Draw independent uniform symbols; inactive channels remain exactly zero."""
    if count < 1:
        raise ValueError("payload count must be positive")
    result = torch.zeros(count, 44, 4, 4)
    result[:, ACTIVE] = torch.randint(0, 2, (count, 8, 4, 4), generator=generator).float()
    return result


def payload_fingerprints(payloads: torch.Tensor) -> set[bytes]:
    return {bytes(row.to(torch.uint8).flatten().tolist()) for row in payloads[:, ACTIVE].cpu()}


def fresh_disjoint_payloads(count: int, generator: torch.Generator,
                            forbidden: Sequence[torch.Tensor]) -> torch.Tensor:
    blocked = set().union(*(payload_fingerprints(x) for x in forbidden))
    rows: list[torch.Tensor] = []
    while len(rows) < count:
        candidate = random_symbol_payloads(1, generator)
        key = next(iter(payload_fingerprints(candidate)))
        if key not in blocked:
            blocked.add(key); rows.append(candidate[0])
    return torch.stack(rows)


def decoder_accepts_only_questioned_image() -> bool:
    return list(inspect.signature(StructuredRegionalDecoder.forward).parameters) == ["self", "image"]


def _entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * p.log2() + (1 - p) * (1 - p).log2())


def symbol_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    active_logits, active_targets = logits[:, ACTIVE], targets[:, ACTIVE]
    predictions = active_logits.ge(0)
    correct = predictions.eq(active_targets.bool())
    target_p = active_targets.mean((0, 2, 3))
    prediction_p = predictions.float().mean((0, 2, 3))
    per_bit = []
    for bit in range(8):
        per_bit.append({
            "active_bit_index": bit,
            "channel": bit + 4,
            "accuracy": float(correct[:, bit].float().mean()),
            "target_entropy": float(_entropy(target_p[bit])),
            "prediction_entropy": float(_entropy(prediction_p[bit])),
            "predicted_one_frequency": float(prediction_p[bit]),
        })
    return {
        "active_bit_accuracy": float(correct.float().mean()),
        "exact_8bit_regional_symbol_accuracy": float(correct.all(1).float().mean()),
        "active_bce_loss": float(F.binary_cross_entropy_with_logits(active_logits, active_targets)),
        "mean_decoder_confidence": float(torch.sigmoid(active_logits).sub(.5).abs().mul(2).mean()),
        "per_bit_accuracy": [x["accuracy"] for x in per_bit],
        "target_entropy": float(_entropy(target_p).mean()),
        "prediction_entropy": float(_entropy(prediction_p).mean()),
        "predicted_one_frequency": float(prediction_p.mean()),
        "per_bit": per_bit,
        **INACTIVE_METRICS,
    }


def _ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Dependency-free global SSIM, reported as such rather than windowed SSIM."""
    dims = (1, 2, 3); c1, c2 = .01 ** 2, .03 ** 2
    mx, my = x.mean(dims), y.mean(dims)
    vx, vy = x.var(dims, unbiased=False), y.var(dims, unbiased=False)
    cov = ((x - mx[:, None, None, None]) * (y - my[:, None, None, None])).mean(dims)
    return float((((2*mx*my+c1)*(2*cov+c2))/((mx.square()+my.square()+c1)*(vx+vy+c2))).mean())


def evaluate(model: StructuredChannelSystem, image: torch.Tensor,
             payloads: torch.Tensor, alpha: float) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    images = image.expand(len(payloads), -1, -1, -1)
    model.eval()
    with torch.no_grad():
        out = model(images, payloads); logits = out["payload_logits"]
    metrics = symbol_metrics(logits, payloads)
    shuffled = symbol_metrics(logits, circular_payload_shuffle(payloads))
    residual = out["residual"]
    metrics.update({
        "shuffled_target_active_bit_accuracy": shuffled["active_bit_accuracy"],
        "correct_minus_shuffled_active_bit_accuracy": metrics["active_bit_accuracy"] - shuffled["active_bit_accuracy"],
        "psnr": float(psnr(images, out["watermarked_image"])),
        "ssim": _ssim(images, out["watermarked_image"]),
        "ssim_variant": "global_channel_aggregated",
        "maximum_absolute_residual": float(residual.abs().max()),
        "mean_absolute_residual": float(residual.abs().mean()),
        "residual_saturation_fraction": float(residual.abs().ge(alpha * .999).float().mean()),
    })
    return metrics, out


def _grad_norm(parameters: Any) -> float:
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None))


def _sensitivity(model: nn.Module, image: torch.Tensor, a: torch.Tensor,
                 b: torch.Tensor) -> dict[str, float]:
    with torch.no_grad(): oa, ob = model(image, a), model(image, b)
    return {
        "residual_payload_sensitivity": float((oa["residual"]-ob["residual"]).abs().mean()),
        "watermarked_image_payload_sensitivity": float((oa["watermarked_image"]-ob["watermarked_image"]).abs().mean()),
        "decoder_logit_payload_sensitivity": float((oa["payload_logits"]-ob["payload_logits"]).abs().mean()),
        "predicted_bit_change_fraction": float(oa["payload_logits"][:, ACTIVE].ge(0).ne(ob["payload_logits"][:, ACTIVE].ge(0)).float().mean()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _visualize(path: Path, image: torch.Tensor, outputs: list[dict[str, torch.Tensor]],
               payloads: torch.Tensor, metrics: Mapping[str, Any], amplification: float) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    rgb = image[0].permute(1, 2, 0).cpu().numpy()
    axes[0,0].imshow(rgb); axes[0,0].set_title("Original natural image")
    for i, label in enumerate(("A", "B")):
        axes[0,i+1].imshow(outputs[i]["watermarked_image"][0].clamp(0,1).permute(1,2,0).detach().cpu())
        axes[0,i+1].set_title(f"Watermarked — payload {label}")
        axes[1,i].imshow((outputs[i]["residual"][0].mean(0)*amplification).detach().cpu(), cmap="seismic")
        axes[1,i].set_title(f"Signed residual {label} (×{amplification:g})")
    difference = outputs[0]["residual"][0].mean(0)-outputs[1]["residual"][0].mean(0)
    axes[0,3].imshow((difference*amplification).detach().cpu(), cmap="seismic"); axes[0,3].set_title("Residual A − B")
    for i in range(2):
        correct = outputs[i]["payload_logits"][0, ACTIVE].ge(0).eq(payloads[i, ACTIVE].bool()).all(0).cpu()
        axes[1,i+2].imshow(correct, vmin=0, vmax=1, cmap="RdYlGn")
        axes[1,i+2].set_title(f"Payload {'AB'[i]}\n8-bit regional-symbol decoding correctness")
    fig.suptitle("Stage B — One Natural Image, Fresh Regional Symbols")
    fig.text(.5, .01, f"fresh bit={metrics['active_bit_accuracy']:.4f}  exact={metrics['exact_8bit_regional_symbol_accuracy']:.4f}  PSNR={metrics['psnr']:.2f} dB", ha="center")
    for ax in axes.flat: ax.axis("off")
    fig.tight_layout(rect=(0,.03,1,.95)); fig.savefig(path, dpi=150); plt.close(fig)


def run_stage_b(config: Mapping[str, Any]) -> dict[str, Any]:
    """Train and evaluate Stage B. Fresh evaluation is generated after training."""
    seed = int(config.get("seed", 2026)); random.seed(seed); torch.manual_seed(seed)
    output = Path(config.get("output_directory", "outputs/stage_b_natural"))
    if output.as_posix().rstrip("/").split("/")[-1] != "stage_b_natural":
        raise ValueError("Stage-B artifacts must be isolated in outputs/stage_b_natural")
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(config.get("data_root", "data/raw/coco_val2017_100"))
    dataset = CocoImageDataset(dataset_root)
    index = int(config.get("image_index", 0)); sample = dataset[index]
    image = sample["image"].unsqueeze(0)
    relative_identifier = sample["relative_path"]
    alpha = float(config.get("alpha", .05)); model = StructuredChannelSystem(mode="learnable", alpha=alpha)
    initial = model.carrier_bank.carriers.detach().clone()
    # No Stage-A checkpoint exists in this repository; this constructor is the
    # exact deterministic successful analytical initialization.
    decoder_initialization = {"checkpoint": None, "method": "deterministic_successful_stage_a_analytical_initialization"}
    optimizer = torch.optim.Adam([
        {"params": [model.carrier_bank.carriers], "lr": float(config.get("carrier_learning_rate", 1e-3))},
        {"params": model.decoder.parameters(), "lr": float(config.get("decoder_learning_rate", 1e-3))},
    ])
    gen = torch.Generator().manual_seed(seed + 1)
    fixed_training = random_symbol_payloads(int(config.get("fixed_training_payloads", 32)), gen)
    heldout = fresh_disjoint_payloads(int(config.get("heldout_payloads", 64)), gen, [fixed_training])
    batch_size, steps = int(config.get("batch_size", 8)), int(config.get("steps", 500))
    evaluate_every = int(config.get("evaluate_every", 50)); history: list[dict[str, Any]] = []
    stream_keys: set[bytes] = set(); last_batch = fixed_training[:batch_size]
    best_score = -1.; last_gradients = {"carrier_gradient_norm": 0., "decoder_gradient_norm": 0.}
    for step in range(1, steps + 1):
        last_batch = random_symbol_payloads(batch_size, gen); stream_keys.update(payload_fingerprints(last_batch))
        model.train(); optimizer.zero_grad(set_to_none=True)
        out = model(image.expand(batch_size,-1,-1,-1), last_batch)
        loss = F.binary_cross_entropy_with_logits(out["payload_logits"][:, ACTIVE], last_batch[:, ACTIVE])
        loss.backward()
        last_gradients = {"carrier_gradient_norm": _grad_norm([model.carrier_bank.carriers]),
                          "decoder_gradient_norm": _grad_norm(model.decoder.parameters())}
        optimizer.step(); model.sync_decoder_carriers()
        if step % evaluate_every == 0 or step == steps:
            current, _ = evaluate(model, image, heldout, alpha)
            row = {"step": step, "training_loss": float(loss), **last_gradients,
                   **{k:v for k,v in current.items() if isinstance(v,(int,float))}}
            row.update({key: "N/A" for key in INACTIVE_METRICS})
            history.append(row)
            state = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                     "step": step, "config": dict(config), "image_identifier": relative_identifier,
                     "decoder_initialization": decoder_initialization}
            torch.save(state, output / "last.pt")
            if current["active_bit_accuracy"] > best_score:
                best_score = current["active_bit_accuracy"]; torch.save(state, output / "best.pt")
    fresh = fresh_disjoint_payloads(int(config.get("fresh_payloads", 64)), gen, [fixed_training, heldout])
    groups = {"current_training_batch": last_batch, "fixed_training_payloads": fixed_training,
              "fixed_heldout_payloads": heldout, "fresh_on_the_fly_payloads": fresh}
    evaluations: dict[str, Any] = {}; cached: dict[str, dict[str, torch.Tensor]] = {}
    for name, values in groups.items(): evaluations[name], cached[name] = evaluate(model, image, values, alpha)
    # Evaluation E: the same fresh decoded logits, scored against a no-fixed-point
    # target permutation. It receives the full A--E communication metric schema.
    shuffled_targets = circular_payload_shuffle(fresh)
    shuffled_metrics = symbol_metrics(cached["fresh_on_the_fly_payloads"]["payload_logits"], shuffled_targets)
    for key in ("psnr", "ssim", "ssim_variant", "maximum_absolute_residual",
                "mean_absolute_residual", "residual_saturation_fraction"):
        shuffled_metrics[key] = evaluations["fresh_on_the_fly_payloads"][key]
    shuffled_metrics["shuffled_target_active_bit_accuracy"] = shuffled_metrics["active_bit_accuracy"]
    shuffled_metrics["correct_minus_shuffled_active_bit_accuracy"] = (
        evaluations["fresh_on_the_fly_payloads"]["active_bit_accuracy"] - shuffled_metrics["active_bit_accuracy"])
    evaluations["shuffled_targets"] = shuffled_metrics
    with torch.no_grad(): original_logits = model.decoder(image)
    original_targets = fresh_disjoint_payloads(int(config.get("original_control_targets", 256)), gen, [fixed_training, heldout, fresh])
    original_predictions = original_logits[:, ACTIVE].ge(0).expand(len(original_targets),-1,-1,-1)
    original_rate = float(original_predictions.eq(original_targets[:, ACTIVE].bool()).all(1).float().mean())
    sensitivity = _sensitivity(model, image, fresh[:1], fresh[1:2])
    threshold = float(config.get("effectively_zero_threshold", 1e-12))
    gate_groups = [evaluations["fixed_heldout_payloads"], evaluations["fresh_on_the_fly_payloads"]]
    variation = torch.cat([last_batch, fixed_training, heldout, fresh])[:, ACTIVE].float().mean((0,2,3))
    finite_gradients = all(math.isfinite(x) for x in last_gradients.values())
    passed = all(m["active_bit_accuracy"] >= .995 and m["exact_8bit_regional_symbol_accuracy"] >= .96 and
                 m["correct_minus_shuffled_active_bit_accuracy"] >= .45 for m in gate_groups)
    passed = bool(passed and original_rate <= .01 and math.isfinite(sensitivity["residual_payload_sensitivity"])
                  and sensitivity["residual_payload_sensitivity"] > threshold
                  and math.isfinite(sensitivity["decoder_logit_payload_sensitivity"])
                  and sensitivity["decoder_logit_payload_sensitivity"] > threshold
                  and finite_gradients and ((variation > 0) & (variation < 1)).all() and len(stream_keys) >= 100)
    summary = {
        "schema_version": "1.0", "stage": "B", "passed": passed,
        "scope": "natural_image_regional_symbol_communication_only", "stage_c_automatic_progression": False,
        "image": {"relative_identifier": relative_identifier, "deterministic_dataset_index": index,
                  "transform": "CocoImageDataset deterministic RGB resize to 256x256"},
        "alpha": alpha, "active_channels": list(range(4,12)), "decoder_input": "questioned_image_only",
        "decoder_initialization": decoder_initialization, "warm_up": None,
        "optimization": {"joint_carrier_and_decoder": True, "fidelity_curriculum": False, **last_gradients},
        "evaluations": evaluations, "original_random_target_exact_match_rate": original_rate,
        "sensitivity": sensitivity, "carrier_change_from_initialization": carrier_change_metrics(initial, model.carrier_bank.carriers),
        "anti_memorization": {"distinct_fresh_training_payload_tensors": len(stream_keys),
             "same_image_received_many_payloads": len(stream_keys) >= 100,
             "fresh_evaluation_generated_after_training": True,
             "fresh_overlap_fixed_training": len(payload_fingerprints(fresh)&payload_fingerprints(fixed_training)),
             "fresh_overlap_heldout": len(payload_fingerprints(fresh)&payload_fingerprints(heldout)),
             "every_active_target_bit_varies": bool(((variation>0)&(variation<1)).all())},
        "effectively_zero_threshold": threshold, "inactive_metrics": dict(INACTIVE_METRICS),
    }
    _write_csv(output / "history.csv", history)
    per_rows = [{"evaluation": name, **row} for name,m in evaluations.items() for row in m["per_bit"]]
    _write_csv(output / "per_bit_metrics.csv", per_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    vis_payloads = fresh[:2]; vis_outputs = [model(image, vis_payloads[i:i+1]) for i in range(2)]
    _visualize(output / "stage_b_visualization.png", image, vis_outputs, vis_payloads,
               evaluations["fresh_on_the_fly_payloads"], float(config.get("visualization_amplification", 10)))
    return summary
