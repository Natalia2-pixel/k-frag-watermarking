"""Gated Stage-C residual-energy and saturation repair."""
from __future__ import annotations
import json,math,random
from pathlib import Path
from typing import Any,Mapping
import torch
from torch.nn import functional as F
from kfrag.data import CocoImageDataset
from kfrag.models.regional_channel_v1 import RegionalChannelV1
from kfrag.training.natural_channel_v2 import deterministic_split,clip_gradients
from kfrag.training.regional_channel_v1 import (fresh_regional_bits,regional_losses,stage_c_gates,save_stage_c,load_stage_c_population,
 build_evaluation_population,preprocess_stage_c_image,validate_model_batch)
from kfrag.diagnostics.stage_c_regional import verify_parent,evaluate_final,SyntheticStageCDataset

def _quick_metrics(model,images,bits,amplitude):
    with torch.no_grad():out=model(images,bits,amplitude);correct=out["regional_logits"].ge(0).eq(bits.bool());residual=out["residual"];mse=residual.square().flatten(1).mean(1)
    return {"regional_active_bit_accuracy":float(correct.float().mean()),"exact_regional_symbol_accuracy":float(correct.all(-1).float().mean()),
      "psnr":float((10*torch.log10(1/mse.clamp_min(1e-12))).mean()),"residual_saturation_fraction":float(residual.abs().ge(amplitude*.999).float().mean()),
      "residual_rms":float(residual.square().mean().sqrt()),"mask_mean":float(out["strength_mask"].mean()),"raw_absolute_mean":float(out["raw_residual"].abs().mean())}

def run_fidelity_repair(config:Mapping[str,Any]):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);size=int(config.get("image_size",64));verification,parent,_=verify_parent(config["stage_b_checkpoint"],config["stage_b_report"],config["stage_b_sha256"],size,int(config.get("width",16)),config["preprocessing"])
    if not verification["passed"]:
        report={"scientific_status":"blocked_by_stage_b_checkpoint","stage_d_permitted":False,"checkpoint_verification":verification};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    model=RegionalChannelV1(parent,size);source=torch.load(config["stage_c_checkpoint"],map_location="cpu",weights_only=False);model.load_state_dict(source["model_state"],strict=True)
    seed=int(config.get("seed",2026));random.seed(seed);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+1);dataset=SyntheticStageCDataset(int(config["synthetic_image_count"])) if config.get("synthetic_image_count") else CocoImageDataset(config["data_root"]);ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_split(ids,int(config["train_images"]),int(config["validation_images"]),seed)
    train=load_stage_c_population(dataset,split["train"],config["preprocessing"],size);validation=load_stage_c_population(dataset,split["validation"],config["preprocessing"],size);eval_images,population=build_evaluation_population(validation,int(config["final_evaluation_samples"]));eval_images=preprocess_stage_c_image(eval_images,config["preprocessing"],size);eval_bits=fresh_regional_bits(len(eval_images),generator);population["actual_payload_grid_count"]=len(eval_bits)
    encoder_params=list(dict.fromkeys(model.encoder.parameters()));decoder_params=[p for p in model.decoder.parameters() if id(p) not in {id(x) for x in encoder_params}]
    optimizer=torch.optim.AdamW([{"params":encoder_params,"lr":float(config.get("encoder_learning_rate",5e-4))},{"params":decoder_params,"lr":float(config.get("decoder_learning_rate",5e-5))}],weight_decay=float(config.get("weight_decay",1e-4)))
    amplitudes=[float(x) for x in config.get("repair_amplitudes",[.02,.018,.016,.014])];maximum=int(config.get("maximum_steps_per_level",60));minimum=int(config.get("minimum_steps_per_level",20));every=int(config.get("evaluate_every",10));patience=int(config.get("patience",6));batch=int(config.get("batch_size",4));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,len(amplitudes)*maximum));history=[];levels=[];best_passing=None;best_passing_score=-1e9;global_step=0
    initial_curve=[]
    for amplitude in amplitudes:initial_curve.append({"amplitude":amplitude,**_quick_metrics(model,eval_images,eval_bits,amplitude)})
    for amplitude in amplitudes:
        best={k:v.detach().clone() for k,v in model.state_dict().items()};best_score=-1e9;stale=0
        for local in range(1,maximum+1):
            global_step+=1;idx=torch.randint(len(train),(batch,),generator=generator);image=train[idx];bits=fresh_regional_bits(batch,generator);validate_model_batch(image,bits,size);mask=torch.ones(batch,4,4,dtype=torch.bool);out=model(image,bits,amplitude,mask)
            losses=regional_losses(model,image,bits,out,mask,{**config["losses"],"amplitude":amplitude});optimizer.zero_grad(set_to_none=True);before=[p.detach().clone() for p in model.parameters()];losses["total"].backward()
            if not torch.isfinite(losses["total"]) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError("non-finite fidelity repair")
            gb,ga=clip_gradients(model.parameters(),1.0);optimizer.step();scheduler.step();update=math.sqrt(sum(float((p.detach()-old).square().sum()) for p,old in zip(model.parameters(),before)))
            if local%every==0:
                quick=_quick_metrics(model,validation,fresh_regional_bits(len(validation),generator),amplitude);communication_ok=quick["regional_active_bit_accuracy"]>=.90 and quick["exact_regional_symbol_accuracy"]>=.75
                score=quick["psnr"]-100*quick["residual_saturation_fraction"]+quick["regional_active_bit_accuracy"]
                history.append({"amplitude":amplitude,"global_step":global_step,"level_step":local,"gradient_norm_before":gb,"gradient_norm_after":ga,"update_norm":update,**quick})
                if communication_ok and score>best_score:best_score=score;best={k:v.detach().clone() for k,v in model.state_dict().items()};stale=0
                else:stale+=1
                if local>=minimum and quick["psnr"]>=35 and quick["residual_saturation_fraction"]<=.001 and stale>=2:break
                if local>=minimum and stale>=patience:break
        model.load_state_dict(best);metrics=evaluate_final(model,eval_images,eval_bits,amplitude,generator,float(config.get("cross_region_leakage_threshold",.10)));gates=stage_c_gates(metrics);passed=all(gates.values());levels.append({"amplitude":amplitude,"metrics":metrics,"gate_results":gates,"passed":passed,"completed_step":global_step})
        score=metrics["psnr"]-metrics["residual_saturation_fraction"]*100
        if passed and score>best_passing_score:best_passing_score=score;best_passing={k:v.detach().clone() for k,v in model.state_dict().items()}
    passed=best_passing is not None
    if passed:model.load_state_dict(best_passing)
    selected=next((x for x in reversed(levels) if x["passed"] and x["metrics"]["psnr"]-100*x["metrics"]["residual_saturation_fraction"]==best_passing_score),None)
    final_metrics=selected["metrics"] if selected else levels[-1]["metrics"];final_gates=stage_c_gates(final_metrics);status="passed_stage_c_regional_repair_pilot" if passed else "blocked_by_stage_c_gate";final_metrics.update({"gate_results":final_gates,"scientific_status":status})
    save_stage_c(output,model,optimizer,scheduler,config,split,"fidelity_repair",final_metrics,verification["sha256"],passed)
    report={"schema_version":"stage_c_fidelity_repair.0","checkpoint_verification":verification,"source_stage_c_checkpoint":str(config["stage_c_checkpoint"]),"evaluation_population":population,
      "initial_communication_psnr_amplitude_curve":initial_curve,"repair_levels":levels,"selected_amplitude":None if selected is None else selected["amplitude"],"metrics":final_metrics,"gate_results":final_gates,"stage_c_passed":passed,"stage_d_permitted":False,"scientific_status":status}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
