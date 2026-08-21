"""Field-aware loss, metrics and gates for P0-P4 tag progression."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def active_bits(tag_bits):return tuple(range(12+tag_bits))
def tag_capacity_loss(logits,bits,tag_bits,parent_logits,weights):
    index=F.binary_cross_entropy_with_logits(logits[...,:4],bits[...,:4]);rs=F.binary_cross_entropy_with_logits(logits[...,4:12],bits[...,4:12]);tag=F.binary_cross_entropy_with_logits(logits[...,12:12+tag_bits],bits[...,12:12+tag_bits])
    distill=F.mse_loss(logits[...,:12],parent_logits[...,:12].detach());total=float(weights.get("index",1))*index+float(weights.get("rs",1))*rs+float(weights.get("tag",2))*tag+float(weights.get("distillation",1))*distill
    return {"index":index,"rs":rs,"tag":tag,"distillation":distill,"total":total}
def capacity_metrics(logits,bits,tag_bits):
    correct=logits[...,:12+tag_bits].ge(0).eq(bits[...,:12+tag_bits].bool());scalar=lambda x:float(x.detach().cpu());index=correct[...,:4];rs=correct[...,4:12];tag=correct[...,12:]
    return {"overall_bit_accuracy":scalar(correct.float().mean()),"index_bit_accuracy":scalar(index.float().mean()),"rs_bit_accuracy":scalar(rs.float().mean()),"tag_bit_accuracy":scalar(tag.float().mean()),
      "exact_index_accuracy":scalar(index.all(-1).float().mean()),"exact_rs_symbol_accuracy":scalar(rs.all(-1).float().mean()),"exact_active_tag_accuracy":scalar(tag.all(-1).float().mean()),
      "exact_packet_accuracy":scalar(correct.all(-1).float().mean()),"per_region_accuracy":correct.float().mean((0,3)).flatten().detach().cpu().tolist(),"per_bit_accuracy":correct.float().mean((0,1,2)).detach().cpu().tolist()}
def capacity_gates(m):
    return {"overall":m["overall_bit_accuracy"]>=.95,"index":m["index_bit_accuracy"]>=.98,"rs":m["rs_bit_accuracy"]>=.95,"tag":m["tag_bit_accuracy"]>=.95,
      "regions":min(m["per_region_accuracy"])>=.90,"bits":min(m["per_bit_accuracy"])>=.90,"shuffled_margin":m["correct_minus_shuffled_margin"]>=.40,"spatial_margin":m["correct_minus_spatial_margin"]>=.40,
      "original":.45<=m["original_randomized_field_accuracy"]<=.55,"psnr":m["psnr"]>=35,"ssim":m["ssim"]>=.95,"saturation":m["residual_saturation_fraction"]<=.001,
      "analytical_zero":m["analytical_contribution"]==0,"blind":m["blind_decoder"],"disjoint":m["disjoint_images"]}

def balanced_bit_loss(logits,bits,active_count,loss_ema,decay=.9,maximum_weight=3.0):
    """Balance bits using only current training BCE statistics."""
    per_element=F.binary_cross_entropy_with_logits(logits[...,:active_count],bits[...,:active_count],reduction="none");per_bit=per_element.mean((0,1,2))
    updated=float(decay)*loss_ema+(1-float(decay))*per_bit.detach();weights=(updated/updated.mean().clamp_min(1e-8)).square().clamp(.5,float(maximum_weight));weights=weights/weights.mean()
    return (per_bit*weights).mean(),updated,weights.detach()
