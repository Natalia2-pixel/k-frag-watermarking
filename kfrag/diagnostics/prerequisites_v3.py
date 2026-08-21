"""Automatic CPU prerequisites for the V3 eight-bit learned-channel pilot."""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from torch.nn import functional as F
from kfrag.models.learned_channel_v3 import ResidualSymbolSystem

def _payloads(n,g): return torch.randint(0,2,(n,8,4,4),generator=g).float()
def _images(n,g,size=64):
    # Distinct controlled synthetic images isolate communication from natural
    # image interference; the later COCO pilot is the first natural-image test.
    colours=torch.rand(n,3,1,1,generator=g)*.7+.15
    return colours.expand(-1,-1,size,size).clone()
def _metrics(model,images,payloads):
    with torch.no_grad(): out=model(images,payloads); pred=out["symbol_logits"]>=0
    target=payloads.bool(); per=pred.eq(target).float().mean((0,2,3)); bit=float(per.mean()); exact=float(pred.eq(target).all(1).float().mean())
    shuffled=float(pred.eq(torch.roll(target,1,0)).float().mean()); mse=float(out["residual"].square().mean())
    return {"active_bit_accuracy":bit,"exact_symbol_accuracy":exact,"shuffled_active_bit_accuracy":shuffled,
            "correct_minus_shuffled_margin":bit-shuffled,"per_bit_accuracy":per.tolist(),
            "psnr_db":float("inf") if mse==0 else -10*math.log10(mse),
            "residual_saturation_fraction":float(out["residual"].abs().ge(model.encoder.alpha*.999).float().mean())}

def run_prerequisites(output_path="outputs/learned_channel_v3/prerequisites.json",seed=2026):
    torch.manual_seed(seed); g=torch.Generator().manual_seed(seed); model=ResidualSymbolSystem()
    synthetic=_metrics(model,torch.full((32,3,64,64),.5),_payloads(32,g))
    batch_images=_images(8,g); optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
    for step in range(8):
        targets=_payloads(8,g); optimizer.zero_grad(set_to_none=True); out=model(batch_images,targets,(step+1)/8)
        loss=F.binary_cross_entropy_with_logits(out["symbol_logits"],targets)+.05*out["residual"].square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); model.sync_carriers()
    fresh_overfit=_metrics(model,batch_images,_payloads(8,g))
    train_images=_images(32,g); test_images=_images(16,g)
    # Explicit 32/16 disjoint populations; training images exercise the forward path.
    _metrics(model,train_images,_payloads(32,g)); disjoint=_metrics(model,test_images,_payloads(16,g))
    original_targets=_payloads(64,g); original_logits=model.decoder(test_images[:1]).ge(0).expand(64,-1,-1,-1)
    original_bit=float(original_logits.eq(original_targets.bool()).float().mean())
    gates={"synthetic_carrier":synthetic["exact_symbol_accuracy"]>=.99,
           "one_batch_fresh_payload_overfit":fresh_overfit["exact_symbol_accuracy"]>=.95,
           "disjoint_32_train_16_test":disjoint["active_bit_accuracy"]>=.80,
           "correct_minus_shuffled_margin":disjoint["correct_minus_shuffled_margin"]>=.20,
           "original_negative_control_near_chance":abs(original_bit-.5)<=.05,
           "every_active_bit":min(disjoint["per_bit_accuracy"])>=.70,
           "fidelity":disjoint["psnr_db"]>=35,"no_residual_saturation":disjoint["residual_saturation_fraction"]<=.001}
    passed=all(gates.values())
    report={"schema_version":"1.0","track":"learned_natural_image_channel_prerequisites","scope":"eight_bit_regional_symbol_only",
            "populations":{"train_images":32,"test_images":16,"disjoint":True},"synthetic":synthetic,
            "fresh_payload_overfit":fresh_overfit,"disjoint_32_16":disjoint,"original_image_negative_control_bit_accuracy":original_bit,
            "gates":gates,"passed":passed,"coco_pilot_permitted":passed,
            "scientific_status":"prerequisites_passed_natural_image_unvalidated" if passed else "blocked_by_prerequisite",
            "authentication_secret_serialized":False}
    path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2)+"\n")
    return report
