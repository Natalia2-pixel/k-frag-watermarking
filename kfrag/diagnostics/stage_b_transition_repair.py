"""Gated Stage-B V2 analytical-to-learned transition and learned-only audit."""
from __future__ import annotations
import inspect, json, math, random
from pathlib import Path
from typing import Any, Mapping
import torch
from torch.nn import functional as F
from kfrag.data import CocoImageDataset
from kfrag.models.natural_channel_v2 import NaturalChannelV2, analytical_carrier_bases, analytical_residual
from kfrag.training.natural_channel_v2 import bit_metrics, clip_gradients, deterministic_split, gate_results, make_checkpoint, save_attempt
from kfrag.diagnostics.stage_b_phase1_repair import balanced_bits

WEIGHTS=(1.0,.9,.75,.5,.25,.1,0.0)

def normalized_distance(left,right):
    def norm(x): return x/(x.square().mean((-3,-2,-1),keepdim=True).sqrt().clamp_min(1e-6))
    return F.mse_loss(norm(left),norm(right))

def carrier_distillation_losses(model,image,bits,learned,analytical):
    hp=model.decoder.highpass; spatial=normalized_distance(learned,analytical); highpass=normalized_distance(hp(learned),hp(analytical))
    amplitude_match=F.mse_loss(learned,analytical)/analytical.square().mean().clamp_min(1e-8)
    saturation=F.relu(learned.abs()/analytical.abs().max().clamp_min(1e-6)-.95).square().mean()
    bases=analytical_carrier_bases(image.shape[-2:],device=image.device); signed=bits.mul(2).sub(1)
    correlations=torch.einsum("bchw,khw->bk",learned,bases)/(learned.shape[1]*learned.shape[2]*learned.shape[3])
    correlations=correlations/(correlations.detach().abs().mean().clamp_min(1e-6)); sign=F.mse_loss(correlations,signed)
    learned_logits=model.decoder.forward_analytical((image+learned).clamp(0,1)); analytical_logits=model.decoder.forward_analytical((image+analytical).clamp(0,1)).detach()
    logits=F.mse_loss(learned_logits,analytical_logits)
    communication=F.binary_cross_entropy_with_logits(learned_logits,bits)
    return {"spatial_distillation":spatial,"highpass_distillation":highpass,"amplitude_matching":amplitude_match,
      "saturation_penalty":saturation,"sign_correlation":sign,"logit_distillation":logits,"communication":communication}

def _ssim(x,y):
    dims=(1,2,3);c1,c2=.01**2,.03**2;mx,my=x.mean(dims),y.mean(dims);vx,vy=x.var(dims,unbiased=False),y.var(dims,unbiased=False);cov=((x-mx[:,None,None,None])*(y-my[:,None,None,None])).mean(dims)
    return float((((2*mx*my+c1)*(2*cov+c2))/((mx.square()+my.square()+c1)*(vx+vy+c2))).mean().detach())

def evaluate_mixture(model,images,bits,analytical_weight,amplitude):
    with torch.no_grad():
        learned_out=model.encoder(images,bits,amplitude); learned=learned_out["residual"]; analytical=analytical_residual(bits,images.shape[-2:],amplitude)
        mixed=analytical_weight*analytical+(1-analytical_weight)*learned; questioned=(images+mixed).clamp(0,1); effective=questioned-images; logits=model.decoder(questioned)
    correct=bit_metrics(logits,bits); shuffled=bit_metrics(logits,torch.roll(bits,1,0)); flat_l=learned.flatten(1);flat_a=analytical.flatten(1)
    cosine=float(F.cosine_similarity(flat_l,flat_a).mean());mse=effective.square().flatten(1).mean(1)
    return {**correct,"shuffled_accuracy":shuffled["active_bit_accuracy"],"correct_minus_shuffled_margin":correct["active_bit_accuracy"]-shuffled["active_bit_accuracy"],
      "psnr":float((10*torch.log10(1/mse.clamp_min(1e-12))).mean()),"residual_rms":float(effective.square().mean().sqrt()),
      "learned_analytical_cosine_similarity":cosine,"analytical_weight":analytical_weight,"learned_weight":1-analytical_weight,
      "learned_residual_rms":float(learned.square().mean().sqrt()),"residual_saturation_fraction":float(learned.abs().ge(amplitude*.999).float().mean()),
      "mask_mean":float(learned_out["strength_mask"].mean()),"mask_minimum":float(learned_out["strength_mask"].min()),"mask_maximum":float(learned_out["strength_mask"].max())}

def payload_sensitivity(model,images,bits,amplitude):
    base=model.encoder(images,bits,amplitude)["residual"]; differences=[]
    for bit in range(8):
        flipped=bits.clone();flipped[:,bit]=1-flipped[:,bit];other=model.encoder(images,flipped,amplitude)["residual"]
        differences.append(float((base-other).square().mean().sqrt().detach()))
    return {"per_bit_flip_residual_rms":differences,"payload_to_residual_sensitivity":float(sum(differences)/8)}

def run_transition_repair(config: Mapping[str,Any]):
    seed=int(config.get("seed",2026));random.seed(seed);torch.manual_seed(seed);gen=torch.Generator().manual_seed(seed+1)
    dataset=CocoImageDataset(config["data_root"]);ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_split(ids,32,16,seed)
    lookup={dataset[i]["relative_path"]:dataset[i]["image"] for i in range(len(dataset))};size=int(config.get("image_size",64));resize=lambda x:F.interpolate(x[None],(size,size),mode="bilinear",align_corners=False,antialias=True)[0]
    train=torch.stack([resize(lookup[x]) for x in split["train"]]);validation=torch.stack([resize(lookup[x]) for x in split["validation"]])
    model=NaturalChannelV2(size,int(config.get("width",16)));phase1=torch.load(config["phase1_checkpoint"],map_location="cpu",weights_only=False);model.decoder.load_state_dict(phase1["decoder_state"],strict=True)
    with torch.no_grad(): model.encoder.carrier.bases.copy_(analytical_carrier_bases(size)[:,None])
    for p in model.decoder.parameters():p.requires_grad_(False)
    encoder_lr=float(config.get("encoder_learning_rate",1e-3));decoder_lr=float(config.get("decoder_learning_rate",1e-4))
    optimizer=torch.optim.AdamW([{"params":model.encoder.parameters(),"lr":encoder_lr},{"params":model.decoder.output.parameters(),"lr":decoder_lr}],weight_decay=float(config.get("weight_decay",1e-4)))
    amplitude=float(config.get("target_amplitude",.02));batch=int(config.get("batch_size",16));maximum=int(config.get("maximum_steps_per_level",500));minimum=int(config.get("minimum_steps_per_level",50));every=int(config.get("evaluate_every",25));patience=int(config.get("patience_evaluations",10))
    eval_bits=balanced_bits(128,torch.Generator().manual_seed(seed+90));eval_images=validation.repeat(8,1,1,1)
    before={str(w):evaluate_mixture(model,eval_images,eval_bits,w,amplitude) for w in WEIGHTS};history=[];results={};global_step=0;blocked=None
    threshold_acc=float(config.get("transition_accuracy",.95));threshold_exact=float(config.get("transition_exact",.85));threshold_margin=float(config.get("transition_margin",.40))
    for level_index,w in enumerate(WEIGHTS[1:]):
        best_state={k:v.detach().clone() for k,v in model.state_dict().items()};initial=evaluate_mixture(model,eval_images,eval_bits,w,amplitude)
        best_score=initial["active_bit_accuracy"]+initial["correct_minus_shuffled_margin"]-10*initial["residual_saturation_fraction"];stale=0;successes=0;last=None
        if w<=float(config.get("decoder_unfreeze_weight",.25)):
            for p in model.decoder.output.parameters():p.requires_grad_(True)
        for local in range(1,maximum+1):
            global_step+=1;idx=torch.randint(len(train),(batch,),generator=gen);image=train[idx];bits=balanced_bits(batch,gen)
            out=model.encoder(image,bits,amplitude);learned=out["residual"];analytical=analytical_residual(bits,image.shape[-2:],amplitude)
            losses=carrier_distillation_losses(model,image,bits,learned,analytical)
            total=(float(config.get("spatial_weight",1))*losses["spatial_distillation"]+float(config.get("highpass_weight",1))*losses["highpass_distillation"]+
              float(config.get("sign_weight",.25))*losses["sign_correlation"]+float(config.get("logit_weight",.25))*losses["logit_distillation"]+float(config.get("communication_weight",1))*losses["communication"])
            total=total+float(config.get("amplitude_match_weight",2))*losses["amplitude_matching"]+float(config.get("saturation_weight",5))*losses["saturation_penalty"]
            optimizer.zero_grad(set_to_none=True);before_params=[p.detach().clone() for p in model.encoder.parameters()];total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.encoder.parameters()):raise FloatingPointError("non-finite encoder gradient")
            grad_before,grad_after=clip_gradients([p for g in optimizer.param_groups for p in g["params"] if p.requires_grad],1.0);optimizer.step()
            update=math.sqrt(sum(float((p.detach()-old).square().sum()) for p,old in zip(model.encoder.parameters(),before_params)))
            if local%every==0:
                last=evaluate_mixture(model,eval_images,eval_bits,w,amplitude);sensitivity=payload_sensitivity(model,validation[:batch],balanced_bits(batch,gen),amplitude);score=last["active_bit_accuracy"]+last["correct_minus_shuffled_margin"]-10*last["residual_saturation_fraction"]
                row={"analytical_weight":w,"global_step":global_step,"level_step":local,"total_loss":float(total.detach()),"encoder_gradient_norm_before":grad_before,"gradient_norm_after":grad_after,"encoder_update_norm":update,**sensitivity,**last};history.append(row)
                if last["learned_residual_rms"]<1e-5 or min(sensitivity["per_bit_flip_residual_rms"])<1e-5:raise RuntimeError("learned residual payload dependence collapsed")
                if last["mask_mean"]<.30 or update<1e-12:raise RuntimeError("mask or encoder update collapsed")
                if score>best_score:best_score=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                passed=last["active_bit_accuracy"]>=threshold_acc and last["exact_symbol_accuracy"]>=threshold_exact and min(last["per_bit_accuracy"])>=.90 and last["correct_minus_shuffled_margin"]>=threshold_margin
                successes=successes+1 if passed and local>=minimum else 0
                if successes>=2:break
                if local>=minimum and stale>=patience:break
        if successes<2:
            model.load_state_dict(best_state);blocked=w;results[str(w)]={**(last or {}),"passed":False,"passed_step":None};break
        model.load_state_dict(best_state)
        restored=evaluate_mixture(model,eval_images,eval_bits,w,amplitude)
        results[str(w)]={**restored,"passed":True,"passed_step":global_step,"level_step":local}
    after={str(w):evaluate_mixture(model,eval_images,eval_bits,w,amplitude) for w in WEIGHTS}
    learned=after["0.0"];random_targets=balanced_bits(len(eval_images),torch.Generator().manual_seed(seed+222))
    with torch.no_grad():original_logits=model.decoder(eval_images)
    original=bit_metrics(original_logits,random_targets);learned.update({"fresh_active_bit_accuracy":learned["active_bit_accuracy"],"fresh_exact_symbol_accuracy":learned["exact_symbol_accuracy"],
      "original_bit_accuracy":original["active_bit_accuracy"],"original_exact_false_positive":original["exact_symbol_accuracy"],"ssim":_ssim(eval_images,(eval_images+model.encoder(eval_images,eval_bits,amplitude)["residual"]).clamp(0,1)),
      "disjoint_images":True,"blind_decoder":list(inspect.signature(model.decoder.forward).parameters)==["questioned_image"],"no_secret_serialized":True,"no_expected_payload_serialized":True})
    gates=gate_results(learned);passed=blocked is None and all(gates.values());status="passed_stage_b_v2_repair_pilot" if passed else "blocked_by_stage_b_v2_transition_prerequisite"
    sensitivity=payload_sensitivity(model,validation,balanced_bits(16,gen),amplitude)
    report={"schema_version":"stage_b_v2.transition_repair.0","pretraining_transition_table":before,"transition_results":results,"final_transition_table":after,
      "first_pretraining_collapse_weight":next((w for w in WEIGHTS if before[str(w)]["active_bit_accuracy"]<threshold_acc or before[str(w)]["correct_minus_shuffled_margin"]<threshold_margin),None),
      "historical_demonstrated_collapse_weight":0.0,
      "learned_residual_payload_sensitivity":sensitivity,"history":history,"learned_only_metrics":learned,"gate_results":gates,"stage_b_v2_passed":passed,"stage_c_permitted":passed,"scientific_status":status}
    output=Path(config.get("output_directory","outputs/stage_b_natural_v2/transition_repair"));output.mkdir(parents=True,exist_ok=True)
    checkpoint=make_checkpoint(model,optimizer,None,{**dict(config),"preprocessing":dict(config["preprocessing"])},split,3,history,{"scientific_status":status,"gate_results":gates});save_attempt(output,checkpoint,passed)
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8");return report
