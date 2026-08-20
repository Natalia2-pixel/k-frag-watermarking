"""Training and artifact primitives for the isolated Stage-B V2 experiment."""
from __future__ import annotations
import hashlib, math, random
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch
from torch.nn import functional as F
from kfrag.models.natural_channel_v2 import ACTIVE_BIT_NAMES, NaturalChannelV2

SCHEMA_VERSION = "stage_b_natural_v2.0"
FORBIDDEN_CHECKPOINT_KEYS = {"authentication_key", "authentication_secret", "expected_validation_payloads", "image_pixels"}

def fresh_bits(count: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 2, (count, 8), generator=generator).float()

def deterministic_split(identifiers: Sequence[str], train_count=32, validation_count=16, seed=2026):
    if len(set(identifiers)) != len(identifiers): raise ValueError("image identifiers must be unique")
    if len(identifiers) < train_count + validation_count: raise ValueError("insufficient distinct natural images")
    ordered = sorted(identifiers); random.Random(seed).shuffle(ordered)
    train, validation = ordered[:train_count], ordered[train_count:train_count+validation_count]
    assert set(train).isdisjoint(validation)
    return {"seed": seed, "train": train, "validation": validation}

def amplitude_at(step: int, warmup_steps: int, initial: float, target: float) -> float:
    p = min(1., max(0., step / max(1, warmup_steps))); return initial + p * (target-initial)

def transition_weights(step: int, transition_steps: int) -> tuple[float, float]:
    learned = min(1., max(0., step / max(1, transition_steps))); return 1.-learned, learned

def multiscale_structural_loss(x, y):
    losses=[]
    for _ in range(3):
        losses.append(F.l1_loss(F.avg_pool2d(x,3,1,1), F.avg_pool2d(y,3,1,1)))
        if min(x.shape[-2:]) >= 8: x,y=F.avg_pool2d(x,2),F.avg_pool2d(y,2)
    return torch.stack(losses).mean()

def loss_components(model, image, bits, out, cfg: Mapping[str, Any]):
    logits=out["logits"]; residual=out["residual"]; amp=max(float(cfg["amplitude"]),1e-8)
    per_bit=F.binary_cross_entropy_with_logits(logits,bits,reduction="none").mean(0)
    communication=per_bit.mean(); fidelity=float(cfg["l1_weight"])*F.l1_loss(out["watermarked_image"],image)+float(cfg["structural_weight"])*multiscale_structural_loss(out["watermarked_image"],image)
    energy=residual.square().mean(); normalized=residual.abs()/amp
    saturation=F.relu(normalized-float(cfg["saturation_start"])).square().mean()
    balance=per_bit.var(unbiased=False); mask_mean=out["strength_mask"].mean()
    mask_collapse=F.relu(torch.as_tensor(float(cfg["mask_mean_min"]),device=image.device)-mask_mean).square()
    original=model.decoder(image); original_confidence=original.square().mean()
    gram=model.encoder.carrier.correlation_matrix(); decorrelation=(gram-torch.eye(8,device=gram.device)).square().mean()
    values={"communication_loss":communication,"fidelity_loss":fidelity,"residual_energy_loss":energy,
      "saturation_loss":saturation,"balance_loss":balance,"mask_collapse_loss":mask_collapse,
      "original_confidence_loss":original_confidence,"carrier_correlation_loss":decorrelation}
    total=sum(float(cfg.get(name.replace("_loss","_weight"),0.))*v for name,v in values.items())
    values["total_loss"]=total; return values

def bit_metrics(logits, bits):
    logits=logits.detach(); bits=bits.detach(); correct=logits.ge(0).eq(bits.bool()); per=correct.float().mean(0)
    return {"active_bit_accuracy":float(correct.float().mean()),"exact_symbol_accuracy":float(correct.all(1).float().mean()),
      "per_bit_accuracy":[float(x) for x in per],"predicted_one_frequency":float(logits.ge(0).float().mean()),
      "decoder_confidence":float(torch.sigmoid(logits).sub(.5).abs().mul(2).mean())}

def gradient_norm(parameters) -> float:
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None))

def clip_gradients(parameters, maximum=1.0):
    params=list(parameters); before=gradient_norm(params); torch.nn.utils.clip_grad_norm_(params,maximum); return before,gradient_norm(params)

def tensor_stats(x):
    return {"mean":float(x.mean()),"std":float(x.std(unbiased=False)),"minimum":float(x.min()),"maximum":float(x.max())}

def gate_results(metrics: Mapping[str, Any]) -> dict[str,bool]:
    return {"fresh_active_bit_accuracy":metrics["fresh_active_bit_accuracy"]>=.80,
      "fresh_exact_symbol_accuracy":metrics["fresh_exact_symbol_accuracy"]>=.50,
      "every_active_bit_accuracy":min(metrics["per_bit_accuracy"])>=.70,
      "correct_minus_shuffled_margin":metrics["correct_minus_shuffled_margin"]>=.20,
      "original_bit_accuracy":.45<=metrics["original_bit_accuracy"]<=.55,
      "original_exact_false_positive":metrics["original_exact_false_positive"]<=.01,
      "psnr":metrics["psnr"]>=35,"ssim":metrics["ssim"]>=.95,
      "residual_saturation_fraction":metrics["residual_saturation_fraction"]<=.001,
      "disjoint_images":bool(metrics["disjoint_images"]),"analytical_weight_zero":metrics["analytical_weight"]==0,
      "blind_decoder":bool(metrics["blind_decoder"]),"no_secret_serialized":bool(metrics["no_secret_serialized"]),
      "no_expected_payload_serialized":bool(metrics["no_expected_payload_serialized"])}

def identifier_hash(identifier: str) -> str: return hashlib.sha256(identifier.encode()).hexdigest()

def make_checkpoint(model: NaturalChannelV2, optimizer, scheduler, config, split, phase, history, metrics):
    checkpoint={"schema_version":SCHEMA_VERSION,"architecture_version":model.architecture_version,"configuration":dict(config),
      "active_bit_mapping":list(ACTIVE_BIT_NAMES),"preprocessing":dict(config["preprocessing"]),
      "split_manifest":{"seed":split["seed"],"train_hashes":[identifier_hash(x) for x in split["train"]],"validation_hashes":[identifier_hash(x) for x in split["validation"]]},
      "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict() if optimizer else None,
      "scheduler_state":scheduler.state_dict() if scheduler else None,"training_phase":phase,"metric_history":list(history),
      "scientific_status":metrics.get("scientific_status","blocked_by_stage_b_v2_prerequisite"),"gate_results":metrics.get("gate_results",{})}
    assert not (set(checkpoint) & FORBIDDEN_CHECKPOINT_KEYS); return checkpoint

def save_attempt(output: Path, checkpoint: Mapping[str,Any], passed: bool):
    output.mkdir(parents=True,exist_ok=True); torch.save(dict(checkpoint),output/"last.pt")
    if passed: torch.save(dict(checkpoint),output/"best.pt")

def audit_stage_a_compatibility(stage_a: Mapping[str,Any], model: NaturalChannelV2):
    old={**stage_a.get("carrier_state_dict",{}),**stage_a.get("regional_decoder_state_dict",{})}; new=model.state_dict(); compatible=[]
    for name,value in old.items():
        if name in new and value.shape==new[name].shape and value.dtype==new[name].dtype: compatible.append(name)
    return {"loaded_parameters":compatible,"decision":"no_transfer" if not compatible else "explicit_partial_transfer_available",
            "preprocessing_checked":True,"optimizer_state_loaded":False}
