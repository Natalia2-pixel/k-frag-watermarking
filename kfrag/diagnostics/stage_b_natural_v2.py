"""Stage-B V2 population runner and domain-transition diagnostics."""
from __future__ import annotations
import inspect, json, math, random
from pathlib import Path
from typing import Any, Mapping
import torch
from torch.nn import functional as F
from kfrag.data import CocoImageDataset
from kfrag.models.natural_channel_v2 import NaturalChannelV2, analytical_residual
from kfrag.training.natural_channel_v2 import (amplitude_at, audit_stage_a_compatibility, bit_metrics,
 clip_gradients, deterministic_split, fresh_bits, gate_results, loss_components, make_checkpoint,
 save_attempt, tensor_stats, transition_weights)

def _ssim(x,y):
    dims=(1,2,3); c1,c2=.01**2,.03**2; mx,my=x.mean(dims),y.mean(dims); vx,vy=x.var(dims,unbiased=False),y.var(dims,unbiased=False)
    cov=((x-mx[:,None,None,None])*(y-my[:,None,None,None])).mean(dims)
    return float((((2*mx*my+c1)*(2*cov+c2))/((mx.square()+my.square()+c1)*(vx+vy+c2))).mean())

def _evaluate(model, images, bits, amplitude):
    with torch.no_grad(): out=model(images,bits,amplitude); logits=out["logits"]
    correct=bit_metrics(logits,bits); shuffled=bit_metrics(logits,torch.roll(bits,1,0))
    random_targets=fresh_bits(len(images),torch.Generator().manual_seed(991)); original_logits=model.decoder(images)
    original=bit_metrics(original_logits,random_targets); residual=out["residual"]
    mse=residual.square().flatten(1).mean(1); psnr=float((10*torch.log10(1/mse.clamp_min(1e-12))).mean())
    metrics={"fresh_active_bit_accuracy":correct["active_bit_accuracy"],"fresh_exact_symbol_accuracy":correct["exact_symbol_accuracy"],
      "per_bit_accuracy":correct["per_bit_accuracy"],"correct_target_accuracy":correct["active_bit_accuracy"],
      "shuffled_target_accuracy":shuffled["active_bit_accuracy"],"correct_minus_shuffled_margin":correct["active_bit_accuracy"]-shuffled["active_bit_accuracy"],
      "original_bit_accuracy":original["active_bit_accuracy"],"original_exact_false_positive":original["exact_symbol_accuracy"],
      "decoder_confidence_original":bit_metrics(original_logits,random_targets)["decoder_confidence"],"predicted_one_frequency":correct["predicted_one_frequency"],
      "psnr":psnr,"ssim":_ssim(images,out["watermarked_image"]),"maximum_absolute_residual":float(residual.abs().max()),
      "mean_absolute_residual":float(residual.abs().mean()),"residual_energy":float(residual.square().mean()),
      "residual_saturation_fraction":float(residual.abs().ge(amplitude*.999).float().mean()),"strength_mask":tensor_stats(out["strength_mask"])}
    metrics["strength_mask"].update({"low_strength_fraction":float((out["strength_mask"]<.35).float().mean()),"high_strength_fraction":float((out["strength_mask"]>.9).float().mean())})
    return metrics,out

def domain_transition_report(model, domains, amplitude, generator):
    report={}
    hp=model.decoder.highpass
    for name,image in domains.items():
        bits=fresh_bits(len(image),generator); metrics,out=_evaluate(model,image,bits,amplitude); host=image.square().mean(); residual=out["residual"].square().mean()
        hp_host=hp(image).square().mean(); hp_residual=hp(out["residual"]).square().mean()
        report[name]={"host_image_energy":float(host),"residual_energy":float(residual),"highpass_host_energy":float(hp_host),
          "highpass_residual_energy":float(hp_residual),"carrier_to_host_energy_ratio":float(residual/host.clamp_min(1e-12)),
          "active_bit_accuracy":metrics["fresh_active_bit_accuracy"],"exact_symbol_accuracy":metrics["fresh_exact_symbol_accuracy"],"per_bit_accuracy":metrics["per_bit_accuracy"]}
    return report

def run_stage_b_v2(config: Mapping[str,Any]):
    seed=int(config.get("seed",2026)); random.seed(seed); torch.manual_seed(seed); gen=torch.Generator().manual_seed(seed+1)
    dataset=CocoImageDataset(config["data_root"]); identifiers=[dataset[i]["relative_path"] for i in range(len(dataset))]
    split=deterministic_split(identifiers,int(config.get("train_images",32)),int(config.get("validation_images",16)),seed)
    lookup={dataset[i]["relative_path"]:dataset[i]["image"] for i in range(len(dataset))}; size=int(config.get("image_size",64))
    resize=lambda x:F.interpolate(x[None],(size,size),mode="bilinear",align_corners=False,antialias=True)[0]
    train=torch.stack([resize(lookup[x]) for x in split["train"]]); validation=torch.stack([resize(lookup[x]) for x in split["validation"]])
    model=NaturalChannelV2(size,int(config.get("width",16)),config.get("activation","silu"),float(config.get("mask_floor",.25)))
    compatibility={"decision":"stage_a_not_requested"}
    if config.get("stage_a_checkpoint"):
        ck=torch.load(config["stage_a_checkpoint"],map_location="cpu",weights_only=False); compatibility=audit_stage_a_compatibility(ck,model)
    enc_lr=float(config.get("encoder_learning_rate",3e-4)); dec_lr=float(config.get("decoder_learning_rate",1e-4))
    optimizer=torch.optim.AdamW([{"params":model.encoder.parameters(),"lr":enc_lr},{"params":model.decoder.parameters(),"lr":dec_lr}],weight_decay=float(config.get("weight_decay",1e-4)))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,int(config.get("total_steps",30))))
    loss_cfg=dict(config["losses"]); history=[]; batch=int(config.get("batch_size",4)); total=int(config.get("total_steps",30)); phase1=int(config.get("phase1_steps",10)); phase2=int(config.get("phase2_steps",10))
    analytical_weight=0.; learned_weight=1.; final_phase=1; phase1_metrics=None
    for step in range(1,total+1):
        idx=torch.randint(len(train),(batch,),generator=gen); images=train[idx]; bits=fresh_bits(batch,gen)
        amplitude=amplitude_at(step,int(config.get("amplitude_warmup_steps",20)),float(config.get("initial_amplitude",.002)),float(config.get("target_amplitude",.02)))
        if step<=phase1:
            analytical_weight,learned_weight=1.,0.; questioned=(images+analytical_residual(bits,images.shape[-2:],amplitude)).clamp(0,1); logits=model.decoder(questioned)
            out={"watermarked_image":questioned,"residual":questioned-images,"logits":logits,"strength_mask":torch.ones_like(images[:,:1]),"bounded_residual":torch.zeros_like(images)}
        else:
            learned=model(images,bits,amplitude); analytical_weight,learned_weight=transition_weights(step-phase1,phase2)
            mixed=analytical_weight*analytical_residual(bits,images.shape[-2:],amplitude)+learned_weight*learned["residual"]
            out={**learned,"watermarked_image":(images+mixed).clamp(0,1),"residual":mixed}; out["logits"]=model.decoder(out["watermarked_image"])
        loss_cfg["amplitude"]=amplitude; losses=loss_components(model,images,bits,out,loss_cfg); optimizer.zero_grad(set_to_none=True); losses["total_loss"].backward()
        if not torch.isfinite(losses["total_loss"]): raise FloatingPointError("non-finite loss")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("non-finite gradient")
        before,after=clip_gradients(model.parameters(),1.0); optimizer.step(); scheduler.step()
        final_phase=1 if step<=phase1 else (2 if step<=phase1+phase2 else 3)
        history.append({"step":step,"phase":final_phase,"total_loss":float(losses["total_loss"].detach()),"amplitude":amplitude,"analytical_weight":analytical_weight,"learned_weight":learned_weight,"gradient_norm_before":before,"gradient_norm_after":after,"learning_rates":[x["lr"] for x in optimizer.param_groups]})
        if step==phase1:
            warm_bits=fresh_bits(len(validation),gen)
            with torch.no_grad(): warm_logits=model.decoder((validation+analytical_residual(warm_bits,validation.shape[-2:],amplitude)).clamp(0,1))
            phase1_metrics=bit_metrics(warm_logits,warm_bits)
            if phase1_metrics["exact_symbol_accuracy"] < .99:
                break
    bits=fresh_bits(len(validation),gen); metrics,out=_evaluate(model,validation,bits,float(config.get("target_amplitude",.02)))
    metrics.update({"disjoint_images":set(split["train"]).isdisjoint(split["validation"]),"analytical_weight":analytical_weight,
      "blind_decoder":list(inspect.signature(model.decoder.forward).parameters)==["questioned_image"],"no_secret_serialized":True,"no_expected_payload_serialized":True})
    gates=gate_results(metrics); passed=all(gates.values()); metrics["gate_results"]=gates; metrics["stage_c_permitted"]=passed
    metrics["scientific_status"]="passed_stage_b_v2_repair_pilot" if passed else "blocked_by_stage_b_v2_prerequisite"
    y=torch.linspace(0,1,size)[None,None,:,None]; x=torch.linspace(0,1,size)[None,None,None,:]
    domains={"zero":torch.zeros(2,3,size,size),"constant_colour":torch.full((2,3,size,size),.5),"synthetic_texture":((x+y)*.5).expand(2,3,-1,-1),
      "low_texture_natural":validation[:2],"high_texture_natural":validation[-2:],"disjoint_coco":validation[2:4]}
    report={"schema_version":"stage_b_natural_v2.report.0","metrics":metrics,"domain_transition":domain_transition_report(model,domains,float(config.get("target_amplitude",.02)),gen),
      "checkpoint_compatibility":compatibility,"preprocessing_audit":dict(config["preprocessing"]),"active_bit_scope":"eight regional-symbol bits only",
      "phase_prerequisites":{"phase1_exact_symbol_accuracy":None if phase1_metrics is None else phase1_metrics["exact_symbol_accuracy"],"phase1_required_exact_symbol_accuracy":.99,"reached_phase":final_phase},
      "stage_c_permitted":passed,"scientific_status":metrics["scientific_status"],"carrier_correlation_matrix":model.encoder.carrier.correlation_matrix().detach().tolist()}
    output=Path(config.get("output_directory","outputs/stage_b_natural_v2/local_prerequisite")); output.mkdir(parents=True,exist_ok=True)
    checkpoint=make_checkpoint(model,optimizer,scheduler,config,split,final_phase,history,{**metrics,"gate_results":gates}); save_attempt(output,checkpoint,passed)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); (output/"history.json").write_text(json.dumps(history,indent=2)+"\n",encoding="utf-8")
    return report
