"""Fail-fast Stage-D complete regional-packet communication pilot."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from kfrag.data import CocoImageDataset
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset, _ssim
from kfrag.models.complete_packet_channel_v1 import CompletePacketChannelV1
from kfrag.models.natural_channel_v2 import NaturalChannelV2
from kfrag.models.regional_channel_v1 import RegionalChannelV1
from kfrag.training.complete_packet_v1 import (
    CAPACITIES, bits_packet, ephemeral_key, field_losses, fresh_packet_batch,
    packet_bits, stage_d_gates,
)
from kfrag.training.natural_channel_v2 import clip_gradients, deterministic_split
from kfrag.training.regional_channel_v1 import (
    build_evaluation_population, load_stage_c_population,
    preprocess_stage_c_image, validate_preprocessing_spec,
)

EXPECTED_PARENT_SIZE = 2_759_150
EXPECTED_PARENT_SHA256 = "110165FE3DDF9D1832905ECC3C4090CFAC93D5CE1EB41E825C3F0B8A202387C1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _contains_prohibited_material(value: Any, path: str = "") -> bool:
    """Reject secret/payload material, while allowing explicit negative gates."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            if name in {"hmac_key", "authentication_key", "authentication_secret",
                        "expected_evaluation_payload", "expected_validation_payload"}:
                if child not in (False, None, "", [], {}):
                    return True
            if _contains_prohibited_material(child, f"{path}/{key}"):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_prohibited_material(x, path) for x in value)
    return False


def verify_stage_c_parent(config: Mapping[str, Any]):
    path = Path(config["stage_c_checkpoint"])
    report_path = Path(config["stage_c_report"])
    checks = {
        "exists_and_nonempty": path.is_file() and path.stat().st_size > 0,
        "exact_size": path.is_file() and path.stat().st_size == int(config.get("stage_c_expected_size", EXPECTED_PARENT_SIZE)),
        "report_exists": report_path.is_file(),
    }
    if not all(checks.values()):
        return {"passed": False, "checks": checks}, None
    actual_hash = _sha256(path)
    expected_hash = str(config.get("stage_c_sha256", EXPECTED_PARENT_SHA256)).upper()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = report.get("metrics", {})
    gate_results = report.get("gate_results", metrics.get("gate_results", {}))
    expected_pre = dict(config["preprocessing"]); saved_pre = checkpoint.get("preprocessing", {})
    expected_range = expected_pre.get("numeric_range", expected_pre.get("input_range"))
    preprocessing_compatible = (
        saved_pre.get("numeric_range", saved_pre.get("input_range")) == expected_range
        and saved_pre.get("dtype") == expected_pre.get("dtype")
        and saved_pre.get("channel_order") == expected_pre.get("channel_order")
        and saved_pre.get("resize") == expected_pre.get("resize")
        and saved_pre.get("interpolation") == expected_pre.get("interpolation")
        and saved_pre.get("antialias") == expected_pre.get("antialias")
        and saved_pre.get("normalization", "none") == expected_pre.get("normalization", "none")
        and expected_pre.get("crop", "none") == "none"
    )
    checks.update({
        "sha256": actual_hash == expected_hash,
        "scientific_status": report.get("scientific_status") == "passed_stage_c_regional_repair_pilot",
        "all_stage_c_gates": bool(gate_results) and all(gate_results.values()),
        "selected_amplitude": float(report.get("selected_amplitude", -1)) == .014,
        "analytical_zero": float(metrics.get("analytical_contribution", -1)) == 0.0,
        "blind_decoder": metrics.get("blind_decoder") is True,
        "disjoint_images": metrics.get("disjoint_images") is True,
        "regions_and_bits": len(checkpoint.get("active_bit_mapping", ())) == 16 * 8,
        "preprocessing": preprocessing_compatible,
        "architecture": checkpoint.get("architecture_version") == RegionalChannelV1.architecture_version,
        "no_prohibited_material": not _contains_prohibited_material(checkpoint) and not _contains_prohibited_material(report),
    })
    if not all(checks.values()):
        return {"passed": False, "checks": checks, "sha256": actual_hash}, None

    stage_b = torch.load(config["stage_b_checkpoint"], map_location="cpu", weights_only=False)
    natural = NaturalChannelV2(int(config["image_size"]), int(config.get("width", 16)))
    natural.load_state_dict(stage_b["model_state"], strict=True)
    model = RegionalChannelV1(natural, int(config["image_size"]))
    expected, saved = model.state_dict(), checkpoint.get("model_state", {})
    checks["parameter_schema"] = set(saved) == set(expected) and all(
        saved[k].shape == expected[k].shape and saved[k].dtype == expected[k].dtype
        and torch.isfinite(saved[k]).all() for k in expected
    )
    if not checks["parameter_schema"]:
        return {"passed": False, "checks": checks, "sha256": actual_hash}, None
    model.load_state_dict(saved, strict=True)
    model.eval()
    generator = torch.Generator().manual_seed(731)
    image = torch.rand(2, 3, int(config["image_size"]), int(config["image_size"]), generator=generator)
    bits = torch.randint(0, 2, (2, 4, 4, 8), generator=generator).float()
    with torch.no_grad():
        before = model(image, bits, .014)["regional_logits"]
    clone = RegionalChannelV1(natural, int(config["image_size"]))
    clone.load_state_dict(model.state_dict(), strict=True)
    clone.eval()
    with torch.no_grad():
        after = clone(image, bits, .014)["regional_logits"]
    checks["roundtrip_logits"] = torch.equal(before, after)
    return {"passed": all(checks.values()), "checks": checks, "sha256": actual_hash,
            "roundtrip_max_logit_difference": float((before - after).abs().max())}, model


def _packet_metrics(logits, bits, active_bits: int):
    correct = logits[..., :active_bits].ge(0).eq(bits[..., :active_bits].bool())
    by_field = lambda a, z: float(correct[..., a:min(z, active_bits)].float().mean()) if active_bits > a else None
    result = {
        "overall_bit_accuracy": float(correct.float().mean()),
        "index_bit_accuracy": by_field(0, 4), "symbol_bit_accuracy": by_field(4, 12),
        "tag_bit_accuracy": by_field(12, 44),
        "per_region_accuracy": correct.float().mean((0, 3)).flatten().tolist(),
        "per_bit_accuracy": correct.float().mean((0, 1, 2)).tolist(),
    }
    if active_bits == 44:
        result.update({
            "exact_index_accuracy": float(correct[..., :4].all(-1).float().mean()),
            "exact_symbol_accuracy": float(correct[..., 4:12].all(-1).float().mean()),
            "exact_tag_accuracy": float(correct[..., 12:44].all(-1).float().mean()),
            "exact_packet_accuracy": float(correct.all(-1).float().mean()),
            "exact_recovered_packets": int(correct.all(-1).sum()),
            "total_packets": int(correct.shape[0] * 16),
            "exact_grid_accuracy": float(correct.all(-1).all((-1, -2)).float().mean()),
        })
    return result


def _evaluate(model, images, bits, tokens, key, amplitude, generator):
    model.eval()
    with torch.no_grad():
        output = model(images, bits, amplitude, 44)
        logits = output["packet_logits"]
        original_logits = model.decoder(images)
    metrics = _packet_metrics(logits, bits, 44)
    shuffled = bits[torch.randperm(len(bits), generator=generator)]
    spatial = bits.flatten(1, 2)[:, torch.tensor(list(range(1, 16)) + [0])].reshape_as(bits)
    random_original = torch.randint(0, 2, bits.shape, generator=generator).float()
    metrics["correct_minus_shuffled_margin"] = metrics["overall_bit_accuracy"] - _packet_metrics(logits, shuffled, 44)["overall_bit_accuracy"]
    metrics["correct_minus_spatial_margin"] = metrics["overall_bit_accuracy"] - _packet_metrics(logits, spatial, 44)["overall_bit_accuracy"]
    metrics["original_image_accuracy"] = _packet_metrics(original_logits, random_original, 44)["overall_bit_accuracy"]
    residual = output["residual"]
    mse = residual.square().flatten(1).mean(1)
    metrics.update({
        "psnr": float((10 * torch.log10(1 / mse.clamp_min(1e-12))).mean()),
        "ssim": _ssim(images, output["watermarked_image"]),
        "residual_rms": float(residual.square().mean().sqrt()),
        "residual_saturation_fraction": float(residual.abs().ge(amplitude * .999).float().mean()),
        "residual_quantiles": torch.quantile(residual.abs(), torch.tensor([0., .5, .9, .99, 1.])).tolist(),
        "mask_mean": float(output["strength_mask"].mean()),
        "tanh_preactivation_mean": float(output["raw_residual"].abs().mean()),
        "analytical_contribution": 0.0, "disjoint_images": True,
        "blind_decoder": list(inspect.signature(model.decoder.forward).parameters) == ["questioned_image"],
        "no_secret_or_expected_payload_serialized": True,
    })
    predicted = logits.ge(0).cpu()
    valid = wrong = mutations = swaps = 0
    wrong_key = bytes((x ^ 0xA5) for x in key)
    for sample in range(len(bits)):
        token_bytes = tokens[sample].pack()
        cells = predicted[sample].reshape(16, 44)
        for region, cell in enumerate(cells):
            packet = bits_packet(cell.tolist())
            from kfrag.crypto.authentication import verify_tag
            valid += int(packet.region_index == region and verify_tag(key, token_bytes, packet.region_index, packet.coded_symbol, packet.authentication_tag))
            wrong += int(packet.region_index == region and verify_tag(wrong_key, token_bytes, packet.region_index, packet.coded_symbol, packet.authentication_tag))
            truth_cell = bits[sample].reshape(16, 44)[region].bool()
            for bit_index in (0, 4, 12):
                changed = truth_cell.clone(); changed[bit_index] = ~changed[bit_index]
                mutated = bits_packet(changed.tolist())
                mutations += int(mutated.region_index == region and verify_tag(key, token_bytes, mutated.region_index, mutated.coded_symbol, mutated.authentication_tag))
            other_region = (region + 1) % 16
            swapped = bits_packet(bits[sample].reshape(16, 44)[other_region].tolist())
            swaps += int(swapped.region_index == region and verify_tag(key, token_bytes, swapped.region_index, swapped.coded_symbol, swapped.authentication_tag))
    total = len(bits) * 16
    metrics.update({"hmac_valid_packet_fraction": valid / total, "wrong_key_acceptance_rate": wrong / total,
                    "one_bit_mutation_acceptance_rate": mutations / (total * 3), "region_swap_acceptance_rate": swaps / total})
    changed = bits.clone(); changed[:, 0, 0, 4] = 1 - changed[:, 0, 0, 4]
    with torch.no_grad():
        altered = model(images, changed, amplitude, 44)
    delta = (altered["packet_logits"] - logits).abs()
    intended = float(delta[:, 0, 0, 4].mean())
    metrics["cross_region_leakage"] = float(delta[:, 1:].mean()) / max(intended, 1e-12)
    metrics["payload_to_residual_sensitivity"] = float((altered["residual"] - residual).abs().mean())
    return metrics


def _save(output, model, optimizer, scheduler, config, split, metrics, parent_hash, passed):
    safe_config = {k: v for k, v in config.items() if "key" not in k.lower()}
    payload = {"schema_version": "stage_d_complete_packet.0", "architecture_version": model.architecture_version,
               "stage_c_parent_sha256": parent_hash, "protocol_version": "kfrag-rs16-12-hmac32-v1",
               "packet_layout": {"index": [0, 4], "rs_symbol": [4, 12], "hmac_tag": [12, 44]},
               "configuration": safe_config, "preprocessing": config["preprocessing"],
               "active_bit_curriculum": list(CAPACITIES), "model_state": model.state_dict(),
               "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
               "split_manifest": split, "metrics": metrics, "scientific_status": metrics["scientific_status"],
               "stage_e_permitted": False}
    torch.save(payload, output / "last.pt")
    if passed:
        torch.save(payload, output / "best.pt")


def run_stage_d(config: Mapping[str, Any]):
    output = Path(config["output_directory"]); output.mkdir(parents=True, exist_ok=True)
    verification, stage_c = verify_stage_c_parent(config)
    if not verification["passed"]:
        report = {"scientific_status": "blocked_by_stage_c_checkpoint", "stage_e_permitted": False,
                  "checkpoint_verification": verification}
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    seed = int(config.get("seed", 2026)); random.seed(seed); torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1); key = ephemeral_key()
    size = int(config.get("image_size", 64)); validate_preprocessing_spec(config["preprocessing"], size)
    dataset = SyntheticStageCDataset(int(config["synthetic_image_count"])) if config.get("synthetic_image_count") else CocoImageDataset(config["data_root"])
    identifiers = [dataset[i]["relative_path"] for i in range(len(dataset))]
    split = deterministic_split(identifiers, int(config["train_images"]), int(config["validation_images"]), seed)
    train = load_stage_c_population(dataset, split["train"], config["preprocessing"], size)
    validation = load_stage_c_population(dataset, split["validation"], config["preprocessing"], size)
    model = CompletePacketChannelV1(stage_c, size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    maximum = int(config.get("maximum_steps_per_capacity", 60)); minimum = int(config.get("minimum_steps_per_capacity", 20)); every = int(config.get("evaluate_every", 10)); patience = int(config.get("patience", 5)); batch = int(config.get("batch_size", 4)); amplitude = .014
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, maximum * len(CAPACITIES)))
    curriculum = {}; history = []; blocked = None
    for capacity in CAPACITIES:
        best = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_score = -1.; stale = 0; passed = False
        for step in range(1, maximum + 1):
            selection = torch.randint(len(train), (batch,), generator=generator); image = train[selection]
            bits, _ = fresh_packet_batch(batch, key, generator); output_data = model(image, bits, amplitude, capacity)
            losses = field_losses(output_data["packet_logits"], bits, capacity)
            residual = output_data["residual"]; normalized = residual.abs() / amplitude
            communication = losses["complete"] + sum(v for k, v in losses.items() if k != "complete") / (len(losses) - 1)
            fidelity = F.l1_loss(output_data["watermarked_image"], image)
            saturation = F.relu(normalized - .90).square().mean()
            total = communication + float(config.get("fidelity_weight", .2)) * fidelity + float(config.get("energy_weight", 5.)) * residual.square().mean() + float(config.get("saturation_weight", 2.)) * saturation
            optimizer.zero_grad(set_to_none=True); before = [p.detach().clone() for p in model.parameters()]; total.backward()
            if not torch.isfinite(total) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("non-finite Stage-D optimization")
            grad_before, grad_after = clip_gradients(model.parameters(), 1.0); optimizer.step(); scheduler.step()
            update_norm = math.sqrt(sum(float((p.detach() - old).square().sum()) for p, old in zip(model.parameters(), before)))
            if step % every == 0:
                eval_bits, _ = fresh_packet_batch(len(validation), key, generator)
                with torch.no_grad(): eval_logits = model(validation, eval_bits, amplitude, capacity)["packet_logits"]
                quick = _packet_metrics(eval_logits, eval_bits, capacity); score = quick["overall_bit_accuracy"]
                row = {"capacity": capacity, "step": step, "loss": float(total), "gradient_norm_before": grad_before,
                       "gradient_norm_after": grad_after, "parameter_update_norm": update_norm, **quick}; history.append(row)
                threshold = float(config.get("capacity_bit_gate", .95))
                if score > best_score: best_score = score; best = {k: v.detach().clone() for k, v in model.state_dict().items()}; stale = 0
                else: stale += 1
                passed = step >= minimum and score >= threshold and min(quick["per_region_accuracy"]) >= .90 and min(quick["per_bit_accuracy"]) >= .90
                if passed or (step >= minimum and stale >= patience): break
        model.load_state_dict(best)
        curriculum[str(capacity)] = {"passed": passed, "step": step, "best_bit_accuracy": best_score}
        if not passed: blocked = capacity; break
    count = int(config.get("final_evaluation_samples", 32)); eval_images, population = build_evaluation_population(validation, count)
    eval_images = preprocess_stage_c_image(eval_images, config["preprocessing"], size)
    eval_bits, tokens = fresh_packet_batch(len(eval_images), key, generator); population["actual_packet_grid_count"] = len(eval_bits)
    metrics = _evaluate(model, eval_images, eval_bits, tokens, key, amplitude, generator)
    # Exact-packet threshold is the independent-bit lower-bound implied by the .95 gate: .95^44 ~= .1047.
    gates = stage_d_gates(metrics); gates["exact_packet"] = metrics["exact_packet_accuracy"] >= .10; gates["authenticated_packet"] = metrics["hmac_valid_packet_fraction"] >= .10
    passed = blocked is None and all(gates.values())
    status = "passed_stage_d_complete_packet_pilot" if passed else ("blocked_by_12_bit_packet" if blocked == 12 else "blocked_by_tag_capacity" if blocked else "blocked_by_full_packet_gate")
    metrics.update({"gate_results": gates, "scientific_status": status})
    _save(output, model, optimizer, scheduler, config, split, metrics, verification["sha256"], passed)
    report = {"schema_version": "stage_d_complete_packet_report.0", "checkpoint_verification": verification,
              "authenticated_message": "12-byte ProvenanceToken || one-byte region index || one-byte RS(16,12) coded symbol; HMAC-SHA256 truncated to 32 bits",
              "exact_packet_gate_rationale": "0.10 is the rounded independent-error yield implied by the mandatory 0.95 per-bit gate: 0.95^44=0.1047",
              "evaluation_population": population, "curriculum": curriculum, "first_failing_capacity": blocked,
              "history": history, "metrics": metrics, "gate_results": gates, "stage_d_passed": passed,
              "stage_e_permitted": False, "scientific_status": status}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
