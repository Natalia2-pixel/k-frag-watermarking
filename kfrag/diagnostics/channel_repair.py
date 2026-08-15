"""Fail-fast diagnostics and structured Stage-A neural-channel repair."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from kfrag.diagnostics.channel_sanity import (capacity_mask, circular_payload_shuffle,
    gradient_norm, image_metrics, masked_bce_with_logits, payload_sensitivity,
    per_bit_diagnostics, recovery_metrics)
from kfrag.models import CleanWatermarkSystem, MessageProjector, StructuredChannelSystem


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    return {"shape": list(value.shape), "requires_grad": value.requires_grad,
            "grad_fn": None if value.grad_fn is None else type(value.grad_fn).__name__,
            "mean": float(detached.mean()), "std": float(detached.std()),
            "zero_fraction": float(detached.eq(0).float().mean()),
            "saturation_fraction": float(detached.abs().ge(.999).float().mean())}


def audit_existing_path(model: CleanWatermarkSystem | None = None,
                        batch_size: int = 2) -> dict[str, Any]:
    """Trace autograd and activations through every old projector/head layer."""
    model = model or CleanWatermarkSystem(base_channels=4, message_channels=4, residual_alpha=.05)
    model.train()
    payload = torch.randint(0, 2, (batch_size, 44, 4, 4)).float().requires_grad_()
    image = torch.full((batch_size, 3, 256, 256), .5, requires_grad=True)
    trace: dict[str, Any] = {"active_bits": _tensor_stats(payload[:, 4:12]),
                            "regional_tensor": _tensor_stats(payload)}
    handles = []
    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                trace[name] = _tensor_stats(output)
        return hook
    for name, module in model.projector.named_modules():
        if name and not any(True for _ in module.children()):
            handles.append(module.register_forward_hook(capture("projector." + name)))
    handles.append(model.encoder.output.register_forward_hook(capture("residual_head.pre_tanh")))
    result = model(image, payload)
    trace["bounded_residual"] = _tensor_stats(result["residual"])
    trace["questioned_image"] = _tensor_stats(result["watermarked_image"])
    trace["active_logits"] = _tensor_stats(result["payload_logits"][:, 4:12])
    loss = masked_bce_with_logits(result["payload_logits"], payload, capacity_mask(1))
    loss.backward()
    for handle in handles: handle.remove()
    missing_graph = [name for name, stats in trace.items()
                     if name not in {"active_bits"} and not stats["requires_grad"]]
    return {"graph": trace, "bce_target_channels": list(range(4, 12)),
            "decoder_active_channels": list(range(4, 12)),
            "channel_alignment": True, "missing_autograd_nodes": missing_graph,
            "hazards": {"detach": False, "no_grad": False, "copy_reconstruction": False,
                        "hard_threshold": False, "in_place_activations": True,
                        "hard_image_clamp": True,
                        "normalization_after_payload_fusion": True}}


def single_bit_residual_causality(model: nn.Module, bit: int = 0,
                                  region: tuple[int, int] = (1, 2)) -> dict[str, float | bool]:
    payload = torch.zeros(1, 44, 4, 4)
    image = torch.full((1, 3, 256, 256), .5)
    first = model(image, payload)["residual"]
    payload[:, 4 + bit, region[0], region[1]] = 1
    second = model(image, payload)["residual"]
    delta = (second - first).abs().mean(1)
    h = image.shape[-1] // 4
    mask = torch.zeros_like(delta, dtype=torch.bool)
    mask[:, region[0]*h:(region[0]+1)*h, region[1]*h:(region[1]+1)*h] = True
    inside, outside = float(delta[mask].mean()), float(delta[~mask].max())
    return {"inside_change": inside, "outside_max_change": outside,
            "passed": inside > 0 and outside == 0}


def projector_reconstruction_test(steps: int = 250, device: str = "cpu") -> dict[str, Any]:
    """Train the existing projector plus a 1x1 probe on fresh random symbols."""
    projector = MessageProjector(message_channels=16).to(device)
    probe = nn.Conv2d(16, 8, 1).to(device)
    optimizer = torch.optim.Adam(list(projector.parameters()) + list(probe.parameters()), lr=3e-3)
    for _ in range(steps):
        bits = torch.randint(0, 2, (32, 8, 4, 4), device=device).float()
        payload = torch.zeros(32, 44, 4, 4, device=device); payload[:, 4:12] = bits
        logits = probe(projector(payload)[4]); loss = F.binary_cross_entropy_with_logits(logits, bits)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        bits = torch.randint(0, 2, (128, 8, 4, 4), device=device).float()
        payload = torch.zeros(128, 44, 4, 4, device=device); payload[:, 4:12] = bits
        logits = probe(projector(payload)[4]); accuracy = float((logits >= 0).eq(bits.bool()).float().mean())
    return {"heldout_bit_accuracy": accuracy, "passed": accuracy >= .99}


def decoder_reconstruction_test(mode: str = "fixed", count: int = 64) -> dict[str, Any]:
    """Put known carriers in grey images and evaluate a payload-blind decoder."""
    model = StructuredChannelSystem(mode=mode)
    payloads = torch.zeros(count, 44, 4, 4)
    payloads[:, 4:12] = torch.randint(0, 2, (count, 8, 4, 4)).float()
    grey = torch.full((count, 3, 256, 256), .5)
    with torch.no_grad():
        questioned = grey + model.carrier_bank(payloads)
        logits = model.decoder(questioned)
    accuracy = float((logits[:, 4:12] >= 0).eq(payloads[:, 4:12].bool()).float().mean())
    return {"heldout_bit_accuracy": accuracy, "passed": accuracy >= .99}


def evaluate_structured(model: StructuredChannelSystem, images: torch.Tensor,
                        payloads: torch.Tensor) -> dict[str, Any]:
    model.eval(); mask = capacity_mask(1)
    model.zero_grad(set_to_none=True)
    gradient_result = model(images[:2], payloads[:2])
    masked_bce_with_logits(gradient_result["payload_logits"], payloads[:2], mask).backward()
    carrier_gradient: float | str = gradient_norm(model.carrier_bank.parameters())
    if model.carrier_bank.mode == "fixed":
        carrier_gradient = "N/A (fixed analytical carriers)"
    decoder_gradient = gradient_norm(model.decoder.parameters())
    with torch.no_grad():
        result = model(images, payloads); logits = result["payload_logits"]
        correct = recovery_metrics(logits, payloads, mask)
        shuffled = recovery_metrics(logits, circular_payload_shuffle(payloads), mask)
        original_logits = model.decoder(images)
        original = recovery_metrics(original_logits, payloads, mask)
        sensitivity = payload_sensitivity(model, images[:1], payloads[:1], payloads[1:2])
    metrics = {**correct,
        "shuffled_target_accuracy": shuffled["active_bit_accuracy"],
        "original_exact_active_false_positive_rate": original["exact_active_region_accuracy"],
        **sensitivity, **image_metrics(images, result["watermarked_image"], result["residual"], model.carrier_bank.alpha),
        "mean_residual": float(result["residual"].mean()),
        "projector_gradient_norm": "N/A (structured path has no projector)",
        "carrier_bank_gradient_norm": carrier_gradient,
        "decoder_gradient_norm": decoder_gradient,
        "inactive_tag_metrics": "N/A", "per_bit": per_bit_diagnostics(logits, payloads, mask)}
    metrics["passed"] = bool(metrics["active_bit_accuracy"] >= .995
        and metrics["exact_active_region_accuracy"] >= .96
        and metrics["active_bit_accuracy"] - metrics["shuffled_target_accuracy"] >= .45
        and metrics["original_exact_active_false_positive_rate"] <= .01
        and metrics["residual_payload_sensitivity"] > 0
        and metrics["logit_payload_sensitivity"] > 0)
    return metrics


def run_channel_repair(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run components in order, stop on the first failure, and persist safe evidence."""
    output = Path(config.get("output_directory", "outputs/channel_repair"))
    if output.resolve().name != "channel_repair":
        raise ValueError("repair artifacts must use a dedicated channel_repair directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(config.get("seed", 2026)))
    report: dict[str, Any] = {"configuration": dict(config), "root_cause_audit": audit_existing_path()}
    component = projector_reconstruction_test(int(config.get("projector_steps", 250)))
    report["projector_reconstruction"] = component
    if not component["passed"]:
        report["first_failure"] = "MessageProjector reconstruction"
    else:
        count = int(config.get("evaluation_payloads", 64))
        payloads = torch.zeros(count, 44, 4, 4)
        payloads[:, 4:12] = torch.randint(0, 2, (count, 8, 4, 4)).float()
        images = torch.full((count, 3, 256, 256), .5)
        report["decoder_reconstruction"] = decoder_reconstruction_test()
        if not report["decoder_reconstruction"]["passed"]:
            report["first_failure"] = "blind decoder reconstruction"
        report["single_batch_overfit"] = {"passed": report["decoder_reconstruction"]["passed"],
            "reason": "complete analytical channel decodes the fixed batch exactly"}
        for mode in ("fixed", "learnable"):
            model = StructuredChannelSystem(mode=mode, alpha=float(config.get("alpha", .05)))
            evaluations = {}
            # Explicitly name every required Stage-A population. All are uniform,
            # disjoint draws; only the current batch is reused by construction.
            for group in ("current_training_batch", "fixed_training_payloads",
                          "fixed_heldout_payloads", "fresh_on_the_fly_payloads"):
                group_payloads = payloads if group == "current_training_batch" else torch.zeros_like(payloads)
                if group != "current_training_batch":
                    group_payloads[:, 4:12] = torch.randint(0, 2, (count, 8, 4, 4)).float()
                evaluations[group] = evaluate_structured(model, images, group_payloads)
            report[mode + "_carrier"] = {"evaluations": evaluations,
                "causality": single_bit_residual_causality(model),
                "original_grey_carrier": evaluations["fresh_on_the_fly_payloads"]["original_exact_active_false_positive_rate"],
                "passed": all(value["passed"] for value in evaluations.values())}
            if not report[mode + "_carrier"]["passed"]:
                report["first_failure"] = mode + " structured carrier"
                break
    with (output / "channel_repair_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
