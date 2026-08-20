"""Stage-C regional losses, metrics, gates, and checkpoint policy."""
from __future__ import annotations
import hashlib,inspect,math
from pathlib import Path
from typing import Any,Mapping
import torch
from torch.nn import functional as F
from kfrag.models.regional_channel_v1 import RegionalChannelV1
from kfrag.training.natural_channel_v2 import multiscale_structural_loss

SCHEMA_VERSION="stage_c_regional_v1.0"
BIT_MAPPING=tuple(f"region_{r//4}_{r%4}.regional_symbol_bit_{b}" for r in range(16) for b in range(8))

def validate_preprocessing_spec(spec:Mapping[str,Any],image_size:int=64)->None:
    required={"numeric_range":[0.0,1.0],"dtype":"float32","channel_order":"RGB","resize":[image_size,image_size],
      "interpolation":"bilinear","antialias":True,"normalization":"none"}
    if dict(spec)!=required:raise ValueError(f"Stage-C preprocessing must exactly match {required!r}; found {dict(spec)!r}")

def preprocess_stage_c_image(image:torch.Tensor,spec:Mapping[str,Any],image_size:int=64)->torch.Tensor:
    """The sole real-image transform used by every Stage-C population and control."""
    validate_preprocessing_spec(spec,image_size)
    if image.ndim not in (3,4):raise ValueError("Stage-C image must be CHW or BCHW")
    batched=image.ndim==4;x=image if batched else image.unsqueeze(0)
    if x.shape[1]!=3:raise ValueError("Stage-C requires RGB channel order with exactly three channels")
    if not x.is_floating_point():x=x.to(torch.float32).div(255)
    else:x=x.to(torch.float32)
    if not torch.isfinite(x).all() or float(x.min()) < -1e-6 or float(x.max()) > 1+1e-6:raise ValueError("Stage-C image values must be finite and in [0,1]")
    x=x.clamp(0,1)
    x=F.interpolate(x,(image_size,image_size),mode="bilinear",align_corners=False,antialias=True)
    x=x.clamp(0,1)
    if x.dtype!=torch.float32 or tuple(x.shape[1:])!=(3,image_size,image_size):raise RuntimeError("Stage-C preprocessing invariant failed")
    return x if batched else x[0]

def load_stage_c_population(dataset,identifiers,preprocessing,image_size=64)->torch.Tensor:
    wanted=set(identifiers);samples={}
    for index in range(len(dataset)):
        sample=dataset[index];identifier=sample["relative_path"]
        if identifier in wanted:samples[identifier]=preprocess_stage_c_image(sample["image"],preprocessing,image_size)
    missing=sorted(wanted-set(samples))
    if missing:raise ValueError(f"Stage-C population identifiers missing from dataset: {missing}")
    return torch.stack([samples[x] for x in identifiers])

def validate_model_batch(image:torch.Tensor,bits:torch.Tensor,image_size:int=64)->None:
    if image.ndim!=4 or tuple(image.shape[1:])!=(3,image_size,image_size):raise ValueError(f"Stage-C image batch must have shape [B,3,{image_size},{image_size}]")
    if bits.ndim!=4 or tuple(bits.shape[1:])!=(4,4,8):raise ValueError("Stage-C bits must have shape [B,4,4,8]")
    if len(image)!=len(bits):raise ValueError("Stage-C image and payload batch sizes must match")

def fresh_regional_bits(count,generator):return torch.randint(0,2,(count,4,4,8),generator=generator).float()
def active_region_mask(batch,count,generator):
    mask=torch.zeros(batch,16,dtype=torch.bool)
    for i in range(batch):mask[i,torch.randperm(16,generator=generator)[:count]]=True
    return mask.reshape(batch,4,4)

def regional_metrics(logits,bits):
    correct=logits.detach().ge(0).eq(bits.bool());symbol=correct.all(-1);per_region=correct.float().mean((0,3));per_region_symbol=symbol.float().mean(0);per_bit=correct.float().mean((0,1,2))
    return {"regional_active_bit_accuracy":float(correct.float().mean()),"exact_regional_symbol_accuracy":float(symbol.float().mean()),
      "exact_16_symbol_grid_accuracy":float(symbol.flatten(1).all(1).float().mean()),"exact_symbol_count":int(symbol.sum()),"exact_symbol_fraction":float(symbol.float().mean()),
      "per_region_bit_accuracy":per_region.tolist(),"per_region_exact_symbol_accuracy":per_region_symbol.tolist(),"per_bit_accuracy":per_bit.tolist(),
      "predicted_one_frequency":float(logits.ge(0).float().mean()),"decoder_confidence":float(torch.sigmoid(logits.detach()).sub(.5).abs().mul(2).mean())}

def regional_losses(model,image,bits,out,active_mask,config):
    mask=active_mask[...,None].expand_as(bits);comm=F.binary_cross_entropy_with_logits(out["regional_logits"][mask],bits[mask])
    fidelity=F.l1_loss(out["watermarked_image"],image)+float(config.get("structural_weight",.25))*multiscale_structural_loss(out["watermarked_image"],image)
    residual=out["residual"];energy=residual.square().mean();sat=F.relu(residual.abs()/float(config["amplitude"])-float(config.get("saturation_start",.95))).square().mean()
    bit_losses=F.binary_cross_entropy_with_logits(out["regional_logits"],bits,reduction="none");bit_balance=bit_losses.mean((0,1,2)).var(unbiased=False);region_balance=bit_losses.mean((0,3)).var(unbiased=False)
    strength=out["strength_mask"];global_mask=F.relu(torch.tensor(float(config.get("mask_min",.3)))-strength.mean()).square();cells=strength.unfold(2,strength.shape[2]//4,strength.shape[2]//4).unfold(3,strength.shape[3]//4,strength.shape[3]//4);regional_mask=F.relu(torch.tensor(float(config.get("mask_min",.3)))-cells.mean((-1,-2))).square().mean()
    original=model.decoder(image);original_conf=original.square().mean();values={"communication":comm,"fidelity":fidelity,"energy":energy,"saturation":sat,"bit_balance":bit_balance,"region_balance":region_balance,"mask_collapse":global_mask+regional_mask,"original_confidence":original_conf}
    total=sum(float(config.get(k+"_weight",1 if k=="communication" else 0))*v for k,v in values.items());values["total"]=total;return values

def stage_c_gates(metrics):
    flat_regions=[x for row in metrics["per_region_bit_accuracy"] for x in row]
    return {"fresh_regional_active_bit_accuracy":metrics["regional_active_bit_accuracy"]>=.90,"exact_regional_symbol_accuracy":metrics["exact_regional_symbol_accuracy"]>=.75,
      "every_region_bit_accuracy":min(flat_regions)>=.80,"every_active_bit_accuracy":min(metrics["per_bit_accuracy"])>=.80,"correct_minus_shuffled_margin":metrics["correct_minus_shuffled_margin"]>=.30,
      "correct_minus_spatial_margin":metrics["correct_minus_spatially_permuted_margin"]>=.30,"original_image_bit_accuracy":.45<=metrics["original_image_bit_accuracy"]<=.55,
      "original_exact_symbol_false_positive":metrics["original_exact_symbol_false_positive"]<=.01,"cross_region_leakage":metrics["cross_region_leakage"]<=metrics["cross_region_leakage_threshold"],
      "psnr":metrics["psnr"]>=35,"ssim":metrics["ssim"]>=.95,"saturation":metrics["residual_saturation_fraction"]<=.001,"disjoint_images":metrics["disjoint_images"],
      "blind_decoder":metrics["blind_decoder"],"analytical_contribution_zero":metrics["analytical_contribution"]==0,"no_authentication_secret":metrics["no_authentication_secret"],"no_expected_payload":metrics["no_expected_payload"]}

def save_stage_c(output,model,optimizer,scheduler,config,split,level,metrics,parent_hash,passed):
    checkpoint={"schema_version":SCHEMA_VERSION,"architecture_version":model.architecture_version,"stage_b_parent_sha256":parent_hash,"configuration":dict(config),"preprocessing":dict(config["preprocessing"]),"active_bit_mapping":BIT_MAPPING,
      "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"split_manifest":{"seed":split["seed"],"train_hashes":[hashlib.sha256(x.encode()).hexdigest() for x in split["train"]],"validation_hashes":[hashlib.sha256(x.encode()).hexdigest() for x in split["validation"]]},
      "curriculum_level":level,"metrics":metrics,"gate_results":metrics.get("gate_results",{}),"scientific_status":metrics["scientific_status"]}
    output.mkdir(parents=True,exist_ok=True);torch.save(checkpoint,output/"last.pt")
    if passed:torch.save(checkpoint,output/"best.pt")
    return checkpoint
