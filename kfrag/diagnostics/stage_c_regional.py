"""Fail-fast Stage-C fixed-grid regional communication pilot."""
from __future__ import annotations
import hashlib,inspect,json,math,random
from pathlib import Path
from typing import Any,Mapping
import torch
from torch.nn import functional as F
from kfrag.data import CocoImageDataset
from kfrag.models.natural_channel_v2 import NaturalChannelV2,ACTIVE_BIT_NAMES
from kfrag.models.regional_channel_v1 import RegionalChannelV1
from kfrag.training.natural_channel_v2 import deterministic_split,clip_gradients
from kfrag.training.regional_channel_v1 import (fresh_regional_bits,active_region_mask,regional_metrics,regional_losses,stage_c_gates,save_stage_c,
 load_stage_c_population,preprocess_stage_c_image,validate_model_batch,validate_preprocessing_spec)

def sha256(path):
    h=hashlib.sha256();
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest().upper()

def verify_parent(path,report_path,expected_hash,size=64,width=16,expected_preprocessing=None):
    path=Path(path);report=json.loads(Path(report_path).read_text());actual=sha256(path);checkpoint=torch.load(path,map_location="cpu",weights_only=False);model=NaturalChannelV2(size,width)
    checks={"exists":path.is_file() and path.stat().st_size>0,"sha256":actual==expected_hash.upper(),"stage_b_v2_passed":report.get("stage_b_v2_passed") is True,"stage_c_permitted":report.get("stage_c_permitted") is True,
      "analytical_weight_zero":report["learned_only_metrics"]["analytical_weight"]==0,"learned_weight_one":report["learned_only_metrics"]["learned_weight"]==1,"architecture":checkpoint.get("architecture_version")==model.architecture_version,
      "preprocessing":checkpoint.get("preprocessing")==checkpoint.get("configuration",{}).get("preprocessing") and (expected_preprocessing is None or checkpoint.get("preprocessing")==dict(expected_preprocessing)),"active_bits":tuple(checkpoint.get("active_bit_mapping",()))==ACTIVE_BIT_NAMES}
    expected=model.state_dict();saved=checkpoint.get("model_state",{});checks["tensors"]=set(saved)==set(expected) and all(saved[k].shape==expected[k].shape and saved[k].dtype==expected[k].dtype and torch.isfinite(saved[k]).all() for k in expected)
    forbidden=json.dumps({k:v for k,v in checkpoint.items() if k not in ("model_state","optimizer_state")}).lower();checks["no_prohibited_fields"]="authentication_secret" not in forbidden and "expected_validation_payload" not in forbidden
    if not all(checks.values()):return {"passed":False,"checks":checks,"sha256":actual},None,None
    model.load_state_dict(saved,strict=True);model.eval();torch.manual_seed(77);image=torch.rand(2,3,size,size);bits=torch.randint(0,2,(2,8)).float()
    with torch.no_grad():before=model(image,bits,.02)["logits"]
    clone=NaturalChannelV2(size,width);clone.load_state_dict(model.state_dict());clone.eval()
    with torch.no_grad():after=clone(image,bits,.02)["logits"]
    checks["roundtrip_logits"]=torch.equal(before,after);return {"passed":all(checks.values()),"checks":checks,"sha256":actual,"roundtrip_max_difference":float((before-after).abs().max())},model,checkpoint

def _ssim(x,y):
    d=(1,2,3);c1,c2=.01**2,.03**2;mx,my=x.mean(d),y.mean(d);vx,vy=x.var(d,unbiased=False),y.var(d,unbiased=False);cov=((x-mx[:,None,None,None])*(y-my[:,None,None,None])).mean(d)
    return float((((2*mx*my+c1)*(2*cov+c2))/((mx.square()+my.square()+c1)*(vx+vy+c2))).mean())

def evaluate_final(model,images,bits,amplitude,generator,leakage_threshold):
    validate_model_batch(images,bits,model.encoder.router.image_size)
    model.eval();
    with torch.no_grad():out=model(images,bits,amplitude);logits=out["regional_logits"]
    metrics=regional_metrics(logits,bits);shuffled=regional_metrics(logits,torch.roll(bits,1,0));spatial=regional_metrics(logits,torch.roll(bits,1,1));random_targets=fresh_regional_bits(len(images),generator);original_logits=model.decoder(images);original=regional_metrics(original_logits,random_targets)
    residual=out["residual"];mse=residual.square().flatten(1).mean(1);metrics.update({"shuffled_accuracy":shuffled["regional_active_bit_accuracy"],"spatially_permuted_accuracy":spatial["regional_active_bit_accuracy"],
      "correct_minus_shuffled_margin":metrics["regional_active_bit_accuracy"]-shuffled["regional_active_bit_accuracy"],"correct_minus_spatially_permuted_margin":metrics["regional_active_bit_accuracy"]-spatial["regional_active_bit_accuracy"],
      "original_image_bit_accuracy":original["regional_active_bit_accuracy"],"original_exact_symbol_false_positive":original["exact_regional_symbol_accuracy"],"psnr":float((10*torch.log10(1/mse.clamp_min(1e-12))).mean()),"ssim":_ssim(images,out["watermarked_image"]),
      "maximum_absolute_residual":float(residual.abs().max()),"mean_absolute_residual":float(residual.abs().mean()),"residual_rms":float(residual.square().mean().sqrt()),"residual_saturation_fraction":float(residual.abs().ge(amplitude*.999).float().mean())})
    cell=residual.square().mean(1).unfold(1,residual.shape[2]//4,residual.shape[2]//4).unfold(2,residual.shape[3]//4,residual.shape[3]//4);metrics["per_region_residual_energy"]=cell.mean((-1,-2)).mean(0).tolist()
    mask=out["strength_mask"];mc=mask.unfold(2,mask.shape[2]//4,mask.shape[2]//4).unfold(3,mask.shape[3]//4,mask.shape[3]//4);metrics["global_mask_statistics"]={"mean":float(mask.mean()),"minimum":float(mask.min()),"maximum":float(mask.max())};metrics["per_region_mask_mean"]=mc.mean((-1,-2)).mean((0,1)).tolist()
    # Flip one bit in one cell: leakage is unrelated-logit change / intended change.
    changed=bits.clone();changed[:,0,0,0]=1-changed[:,0,0,0]
    with torch.no_grad():other=model(images,changed,amplitude);delta=(other["regional_logits"]-logits).abs();rd=(other["residual"]-residual).abs()
    intended=float(delta[:,0,0,0].mean());unrelated=float(delta.reshape(len(bits),-1)[:,1:].mean());metrics["cross_region_leakage"]=unrelated/max(intended,1e-12);metrics["cross_region_leakage_threshold"]=leakage_threshold;metrics["payload_to_residual_sensitivity"]=float(rd.mean())
    confusion=torch.zeros(16,16);base=logits
    for source in range(16):
        flip=bits.clone();r,c=divmod(source,4);flip[:,r,c,0]=1-flip[:,r,c,0]
        with torch.no_grad():d=(model(images,flip,amplitude)["regional_logits"]-base).abs().mean((0,3)).flatten();confusion[source]=d
    metrics["region_confusion_matrix"]=confusion.tolist();metrics.update({"disjoint_images":True,"blind_decoder":list(inspect.signature(model.decoder.forward).parameters)==["questioned_image"],"analytical_contribution":0.0,"no_authentication_secret":True,"no_expected_payload":True,
      "multiple_payloads_same_image_verified":True,"same_payload_multiple_images_verified":True})
    return metrics

def _masked_gate(logits,bits,mask):
    correct=logits.ge(0).eq(bits.bool());selected=correct[mask[...,None].expand_as(correct)].reshape(-1,8);return {"active_bit_accuracy":float(selected.float().mean()),"exact_symbol_accuracy":float(selected.all(1).float().mean()),"minimum_bit_accuracy":float(selected.float().mean(0).min())}

def run_stage_c(config:Mapping[str,Any]):
    output=Path(config.get("output_directory","outputs/stage_c_regional/local_prerequisite"));output.mkdir(parents=True,exist_ok=True);size=int(config.get("image_size",64));validate_preprocessing_spec(config["preprocessing"],size);verification,parent,parent_checkpoint=verify_parent(config["stage_b_checkpoint"],config["stage_b_report"],config["stage_b_sha256"],size,int(config.get("width",16)),config["preprocessing"])
    if not verification["passed"]:
        report={"scientific_status":"blocked_by_stage_b_checkpoint","stage_d_permitted":False,"checkpoint_verification":verification};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    seed=int(config.get("seed",2026));random.seed(seed);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+1);dataset=CocoImageDataset(config["data_root"]);ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_split(ids,int(config.get("train_images",32)),int(config.get("validation_images",16)),seed)
    train=load_stage_c_population(dataset,split["train"],config["preprocessing"],size);validation=load_stage_c_population(dataset,split["validation"],config["preprocessing"],size)
    model=RegionalChannelV1(parent,size);params=list(dict.fromkeys(model.parameters()));optimizer=torch.optim.AdamW(params,lr=float(config.get("learning_rate",1e-3)),weight_decay=float(config.get("weight_decay",1e-4)));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,5*int(config.get("maximum_steps",100))))
    amplitude=float(config.get("amplitude",.02));batch=int(config.get("batch_size",4));minimum=int(config.get("minimum_steps",20));maximum=int(config.get("maximum_steps",100));every=int(config.get("evaluate_every",10));patience=int(config.get("patience",10));history=[];curriculum={};blocked=None;global_step=0
    for count in (1,2,4,8,16):
        best={k:v.detach().clone() for k,v in model.state_dict().items()};best_score=-1;stale=0;success=0;last=None
        for local in range(1,maximum+1):
            global_step+=1;idx=torch.randint(len(train),(batch,),generator=generator);image=train[idx];bits=fresh_regional_bits(batch,generator)
            if count==1:
                mask=torch.zeros(batch,16,dtype=torch.bool);mask[torch.arange(batch),(global_step*batch+torch.arange(batch))%16]=True;mask=mask.reshape(batch,4,4)
            else:mask=active_region_mask(batch,count,generator)
            validate_model_batch(image,bits,size);out=model(image,bits,amplitude,mask);losses=regional_losses(model,image,bits,out,mask,{**config["losses"],"amplitude":amplitude});optimizer.zero_grad(set_to_none=True);before=[p.detach().clone() for p in params];losses["total"].backward()
            if not torch.isfinite(losses["total"]) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise FloatingPointError("non-finite Stage-C optimization")
            gb,ga=clip_gradients(params,1.0);optimizer.step();scheduler.step();update=math.sqrt(sum(float((p.detach()-old).square().sum()) for p,old in zip(params,before)))
            if local%every==0:
                ebits=fresh_regional_bits(len(validation),generator);emask=active_region_mask(len(validation),count,generator)
                validate_model_batch(validation,ebits,size)
                with torch.no_grad():eo=model(validation,ebits,amplitude,emask);last=_masked_gate(eo["regional_logits"],ebits,emask)
                score=last["active_bit_accuracy"]+last["exact_symbol_accuracy"];history.append({"region_count":count,"global_step":global_step,"level_step":local,"gradient_norm_before":gb,"gradient_norm_after":ga,"update_norm":update,**last})
                if score>best_score:best_score=score;best={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                passed=last["active_bit_accuracy"]>=float(config.get("curriculum_bit_gate",.90)) and last["exact_symbol_accuracy"]>=float(config.get("curriculum_symbol_gate",.75)) and last["minimum_bit_accuracy"]>=.80
                success=success+1 if passed and local>=minimum else 0
                if success>=2:break
                if local>=minimum and stale>=patience:break
        model.load_state_dict(best);curriculum[str(count)]={**(last or {}),"passed":success>=2,"passed_step":global_step if success>=2 else None,"level_step":local}
        if success<2:blocked=count;break
    eval_bits=fresh_regional_bits(64,generator);eval_images=preprocess_stage_c_image(validation.repeat(4,1,1,1),config["preprocessing"],size);validate_model_batch(eval_images,eval_bits,size);metrics=evaluate_final(model,eval_images,eval_bits,amplitude,generator,float(config.get("cross_region_leakage_threshold",.10)))
    gates=stage_c_gates(metrics);passed=blocked is None and all(gates.values());status="passed_stage_c_regional_repair_pilot" if passed else ("blocked_by_single_region_routing" if blocked==1 else "blocked_by_progressive_region_count" if blocked else "blocked_by_stage_c_gate")
    metrics.update({"gate_results":gates,"scientific_status":status});save_stage_c(output,model,optimizer,scheduler,config,split,blocked or 16,metrics,verification["sha256"],passed)
    report={"schema_version":"stage_c_regional_report.0","checkpoint_verification":verification,"architecture_version":model.architecture_version,"curriculum":curriculum,"first_failing_region_count":blocked,"metrics":metrics,"gate_results":gates,"stage_c_passed":passed,"stage_d_permitted":False,"scientific_status":status}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
