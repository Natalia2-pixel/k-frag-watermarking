"""Strict analytical-carrier sanity ladder for Stage-B V2 Phase 1 only."""
from __future__ import annotations
import json, math, random
from pathlib import Path
from typing import Any, Mapping
import torch
from torch.nn import functional as F
from kfrag.data import CocoImageDataset
from kfrag.models.natural_channel_v2 import NaturalChannelV2, analytical_carrier_bases, analytical_residual
from kfrag.training.natural_channel_v2 import bit_metrics, clip_gradients, deterministic_split, fresh_bits

def balanced_bits(batch: int, generator: torch.Generator) -> torch.Tensor:
    if batch % 2: raise ValueError("balanced Phase-1 batch size must be even")
    half=torch.randint(0,2,(batch//2,8),generator=generator).float(); order=torch.randperm(batch,generator=generator)
    return torch.cat((half,1-half))[order]

def _inputs(level, hosts, bits, amplitude):
    residual=analytical_residual(bits,hosts.shape[-2:],amplitude)
    if level=="carrier_alone": questioned=residual
    elif level=="zero_image": questioned=residual.clamp(0,1)
    else: questioned=(hosts+residual).clamp(0,1)
    return questioned,residual

def _metrics(decoder, questioned, bits):
    with torch.no_grad(): logits=decoder.forward_analytical(questioned)
    result=bit_metrics(logits,bits); shuffled=bit_metrics(logits,torch.roll(bits,1,0))
    result.update({"bce":float(F.binary_cross_entropy_with_logits(logits,bits)),"shuffled_accuracy":shuffled["active_bit_accuracy"],
      "correct_minus_shuffled_margin":result["active_bit_accuracy"]-shuffled["active_bit_accuracy"],
      "logit_mean":float(logits.mean()),"logit_std":float(logits.std(unbiased=False))})
    result["passed"]=result["active_bit_accuracy"]>=.995 and result["exact_symbol_accuracy"]>=.99 and min(result["per_bit_accuracy"])>=.99 and result["correct_minus_shuffled_margin"]>=.45
    return result

def _signal_audit(model, hosts, bits, amplitude):
    questioned,residual=_inputs("natural_images",hosts,bits,amplitude); hp=model.decoder.highpass
    effective=questioned-hosts; mse=effective.square().flatten(1).mean(1)
    bases=analytical_carrier_bases(hosts.shape[-2:]); gram=F.normalize(bases.flatten(1),dim=1)@F.normalize(bases.flatten(1),dim=1).T
    return {"basis_means":[float(x) for x in bases.mean((-2,-1))],"basis_rms":[float(x) for x in bases.square().mean((-2,-1)).sqrt()],
      "correlation_matrix":gram.tolist(),"maximum_off_diagonal_correlation":float((gram-torch.eye(8)).abs().max()),
      "analytical_residual_rms":float(residual.square().mean().sqrt()),"maximum_residual":float(residual.abs().max()),
      "highpass_residual_energy":float(hp(effective).square().mean()),"highpass_host_energy":float(hp(hosts).square().mean()),
      "carrier_to_host_ratio":float(effective.square().mean()/hosts.square().mean().clamp_min(1e-12)),
      "clamped_pixel_fraction":float((effective-residual).abs().gt(1e-8).float().mean()),"psnr":float((10*torch.log10(1/mse.clamp_min(1e-12))).mean())}

def run_phase1_repair(config: Mapping[str,Any]):
    seed=int(config.get("seed",2026)); random.seed(seed); torch.manual_seed(seed); gen=torch.Generator().manual_seed(seed+1)
    dataset=CocoImageDataset(config["data_root"]); ids=[dataset[i]["relative_path"] for i in range(len(dataset))]
    split=deterministic_split(ids,32,16,seed); lookup={dataset[i]["relative_path"]:dataset[i]["image"] for i in range(len(dataset))}
    size=int(config.get("image_size",64)); resize=lambda x:F.interpolate(x[None],(size,size),mode="bilinear",align_corners=False,antialias=True)[0]
    train=torch.stack([resize(lookup[x]) for x in split["train"]]); validation=torch.stack([resize(lookup[x]) for x in split["validation"]])
    model=NaturalChannelV2(size,int(config.get("width",16))); decoder=model.decoder
    with torch.no_grad(): decoder.output.weight[:,:-8].zero_()
    optimizer=torch.optim.AdamW(decoder.output.parameters(),lr=float(config.get("decoder_learning_rate",1e-3)),weight_decay=float(config.get("weight_decay",1e-4)))
    batch=int(config.get("batch_size",16)); target=float(config.get("target_amplitude",.02)); start=float(config.get("initial_phase1_amplitude",.06))
    maximum=int(config.get("maximum_phase1_steps",1000)); minimum=int(config.get("minimum_phase1_steps",100)); every=int(config.get("evaluate_every",25)); patience=int(config.get("patience_evaluations",10))
    eval_bits=balanced_bits(256,torch.Generator().manual_seed(seed+91)); grey=torch.full((batch,3,size,size),.5)
    texture=torch.linspace(0,1,size)[None,None,:,None].expand(batch,3,-1,size)*.5+torch.linspace(0,1,size)[None,None,None,:].expand(batch,3,size,-1)*.5
    natural_eval=validation.repeat(math.ceil(256/len(validation)),1,1,1)[:256]
    hosts={"carrier_alone":torch.zeros(batch,3,size,size),"zero_image":torch.zeros(batch,3,size,size),"constant_grey":grey,"synthetic_texture":texture,"natural_images":train[:batch]}
    eval_hosts={"carrier_alone":torch.zeros(256,3,size,size),"zero_image":torch.zeros(256,3,size,size),"constant_grey":torch.full((256,3,size,size),.5),
      "synthetic_texture":texture.repeat(math.ceil(256/batch),1,1,1)[:256],"natural_images":natural_eval}
    audit_bits=balanced_bits(len(validation),torch.Generator().manual_seed(seed+7)); audit=_signal_audit(model,validation,audit_bits,target)
    history=[]; results={}; global_step=0; blocked=None
    for level in hosts:
        successes=0; stale=0; best=-1.; level_result=None
        for local_step in range(1,maximum+1):
            global_step+=1; bits=balanced_bits(batch,gen); amplitude=max(target,start-(start-target)*min(1,local_step/minimum))
            host=hosts[level]
            if level=="natural_images": host=train[torch.randint(len(train),(batch,),generator=gen)]
            questioned,_=_inputs(level,host,bits,amplitude); before_state=[p.detach().clone() for p in decoder.parameters()]
            logits=decoder.forward_analytical(questioned); loss=F.binary_cross_entropy_with_logits(logits,bits); optimizer.zero_grad(set_to_none=True); loss.backward()
            logit_grad=torch.autograd.grad(F.binary_cross_entropy_with_logits(logits,bits),logits,retain_graph=False)[0] if False else (torch.sigmoid(logits.detach())-bits)/bits.numel()
            per_bit_gradient=[float(logit_grad[:,i].abs().sum()) for i in range(8)]
            grad_before,grad_after=clip_gradients(decoder.parameters(),1.0); optimizer.step()
            update=math.sqrt(sum(float((p.detach()-old).square().sum()) for p,old in zip(decoder.parameters(),before_state)))
            if local_step%every==0:
                eval_q,_=_inputs(level,eval_hosts[level],eval_bits,target); level_result=_metrics(decoder,eval_q,eval_bits)
                row={"level":level,"global_step":global_step,"level_step":local_step,"training_amplitude":amplitude,"evaluation_amplitude":target,
                  "gradient_norm_before":grad_before,"gradient_norm_after":grad_after,"parameter_update_norm":update,"per_bit_logit_gradient":per_bit_gradient,**level_result}
                history.append(row)
                if level_result["logit_std"]<1e-6: raise RuntimeError("Phase-1 logits are effectively constant")
                if min(per_bit_gradient)<=0: raise RuntimeError("a decoder output bit has zero gradient")
                if update<=1e-12: raise RuntimeError("decoder parameter update is effectively zero")
                if level_result["predicted_one_frequency"]<.05 or level_result["predicted_one_frequency"]>.95: raise RuntimeError("predicted-one frequency collapsed")
                successes=successes+1 if level_result["passed"] and local_step>=minimum else 0
                stale=0 if level_result["active_bit_accuracy"]>best+1e-6 else stale+1; best=max(best,level_result["active_bit_accuracy"])
                if successes>=3: level_result={**level_result,"passed_step":global_step,"level_step":local_step}; break
                if local_step>=minimum and stale>=patience: break
        results[level]=level_result or {"passed":False,"passed_step":None}
        if not results[level].get("passed") or successes<3:
            results[level]["passed"]=False; results[level]["passed_step"]=None; blocked=level; break
    # One-batch and fresh-payload controls use the strict first-level representation.
    fixed=balanced_bits(batch,torch.Generator().manual_seed(seed+111)); fixed_q,_=_inputs("carrier_alone",torch.zeros_like(grey),fixed,target)
    overfit=_metrics(decoder,fixed_q,fixed); fresh=balanced_bits(256,torch.Generator().manual_seed(seed+112)); fresh_q,_=_inputs("carrier_alone",torch.zeros(256,3,size,size),fresh,target); fresh_result=_metrics(decoder,fresh_q,fresh)
    passed=blocked is None and all(x.get("passed") for x in results.values())
    report={"schema_version":"stage_b_v2.phase1_repair.0","signal_audit":audit,"sanity_ladder":results,"history":history,
      "one_batch_overfit":overfit,"fresh_payload_control":fresh_result,"phase1_passed":passed,"blocked_level":blocked,
      "stage_b_v2_passed":False,"stage_c_permitted":False,"scientific_status":"phase1_repaired_stage_b_v2_still_blocked" if passed else "blocked_by_stage_b_v2_phase1_prerequisite"}
    output=Path(config.get("output_directory","outputs/stage_b_natural_v2/phase1_repair")); output.mkdir(parents=True,exist_ok=True)
    torch.save({"schema_version":report["schema_version"],"decoder_state":decoder.state_dict(),"configuration":dict(config),"sanity_ladder":results,
      "scientific_status":report["scientific_status"],"stage_c_permitted":False},output/"last.pt")
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); return report
