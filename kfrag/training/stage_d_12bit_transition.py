"""Metrics and gates for the isolated Stage-D 12-bit transition repair."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def active_mapping(level:int):return tuple(range(4,12))+tuple(range(level))

def transition_loss(logits,bits,level,parent_rs_logits,distillation_weight=1.0):
    active=active_mapping(level);communication=F.binary_cross_entropy_with_logits(logits[...,active],bits[...,active])
    distillation=F.mse_loss(logits[...,4:12],parent_rs_logits.detach())
    return {"communication":communication,"rs_logit_distillation":distillation,"total":communication+distillation_weight*distillation}

def transition_metrics(logits,bits,level):
    active=active_mapping(level);correct=logits[...,active].ge(0).eq(bits[...,active].bool());rs=logits[...,4:12].ge(0).eq(bits[...,4:12].bool())
    index=logits[...,:level].ge(0).eq(bits[...,:level].bool()) if level else None
    return {"overall_active_bit_accuracy":float(correct.float().mean()),"index_bit_accuracy":None if index is None else float(index.float().mean()),
      "rs_bit_accuracy":float(rs.float().mean()),"per_active_bit_accuracy":correct.float().mean((0,1,2)).tolist(),"per_region_accuracy":correct.float().mean((0,3)).flatten().tolist(),
      "exact_index_accuracy":None if index is None else float(index.all(-1).float().mean()),"exact_rs_symbol_accuracy":float(rs.all(-1).float().mean())}

def repair_gates(metrics):
    return {"overall":metrics["overall_active_bit_accuracy"]>=.95,"index":metrics["index_bit_accuracy"]>=.98,"rs":metrics["rs_bit_accuracy"]>=.95,
      "regions":min(metrics["per_region_accuracy"])>=.90,"bits":min(metrics["per_active_bit_accuracy"])>=.90,"shuffled_margin":metrics["correct_minus_shuffled_margin"]>=.40,
      "spatial_margin":metrics["correct_minus_spatial_margin"]>=.40,"original_randomized_fields":.45<=metrics["original_image_randomized_field_accuracy"]<=.55,
      "psnr":metrics["psnr"]>=35,"ssim":metrics["ssim"]>=.95,"saturation":metrics["residual_saturation_fraction"]<=.001,
      "analytical_zero":metrics["analytical_contribution"]==0,"blind":metrics["blind_decoder"],"disjoint":metrics["disjoint_images"]}
