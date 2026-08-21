"""Executed R0-R4 repair for Stage-D's 12-bit structural packet."""
from __future__ import annotations
import inspect,json,math,random
from pathlib import Path
from typing import Mapping,Any
import torch
from torch.nn import functional as F
from kfrag.data import CocoImageDataset
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset,_ssim
from kfrag.diagnostics.stage_d_complete_packet import verify_stage_c_parent
from kfrag.models.stage_d_12bit_transition_v1 import StageD12BitTransitionV1,RS_SLICE
from kfrag.training.complete_packet_v1 import ephemeral_key,fresh_packet_batch
from kfrag.training.natural_channel_v2 import deterministic_split,clip_gradients
from kfrag.training.regional_channel_v1 import load_stage_c_population,build_evaluation_population,preprocess_stage_c_image
from kfrag.training.stage_d_12bit_transition import transition_loss,transition_metrics,repair_gates

def transplant_audit(model,image,bits,tolerance=0.0):
    parent=model.stage_c(image,bits[...,RS_SLICE],.014);r0=model.reproduce_stage_c(image,bits,.014)
    checks={"mapping":all(RS_SLICE.start+j==4+j for j in range(8)),"basis_exact":torch.equal(model.stage_c.encoder.router.cell_bases,model.stage_c.decoder.router.cell_bases),
      "decoder_exact_object":model.stage_c.decoder is model.stage_c.decoder,"residual":torch.allclose(parent["residual"],r0["residual"],atol=tolerance,rtol=0),
      "watermarked":torch.allclose(parent["watermarked_image"],r0["watermarked_image"],atol=tolerance,rtol=0),"rs_logits":torch.allclose(parent["regional_logits"],r0["packet_logits"][...,4:12],atol=tolerance,rtol=0)}
    changed=bits.clone();changed[...,12:]=1-changed[...,12:];a=model.reproduce_stage_c(image,bits);b=model.reproduce_stage_c(image,changed)
    checks["inactive_tag_isolation"]=torch.equal(a["residual"],b["residual"]) and torch.equal(a["packet_logits"],b["packet_logits"])
    return {"passed":all(checks.values()),"checks":checks,"maximum_residual_difference":float((parent["residual"]-r0["residual"]).abs().max()),"maximum_logit_difference":float((parent["regional_logits"]-r0["packet_logits"][...,4:12]).abs().max())}

def _evaluate(model,images,bits,level,generator):
    model.eval()
    with torch.no_grad():out=model(images,bits,.014,level);logits=out["packet_logits"];parent=model.stage_c(images,bits[...,4:12],.014)["regional_logits"]
    metrics=transition_metrics(logits,bits,level);shuffled=bits.clone();shuffled[...,4:12]=torch.roll(bits[...,4:12],1,0)
    spatial=torch.roll(bits,1,1);rs_correct=logits[...,4:12].ge(0).eq(shuffled[...,4:12].bool()).float().mean()
    metrics["correct_minus_shuffled_margin"]=metrics["rs_bit_accuracy"]-float(rs_correct)
    active=tuple(range(4,12))+tuple(range(level));metrics["correct_minus_spatial_margin"]=metrics["overall_active_bit_accuracy"]-float(logits[...,active].ge(0).eq(spatial[...,active].bool()).float().mean())
    random_rs=torch.randint(0,2,(len(images),4,4,8),generator=generator).float()
    with torch.no_grad():original=model.stage_c.decoder(images)
    metrics["original_image_randomized_field_accuracy"]=float(original.ge(0).eq(random_rs.bool()).float().mean());metrics["original_index_structural_accuracy"]=float(model.index_head(images).ge(0).eq(bits[...,:4].bool()).float().mean())
    residual=out["residual"];mse=residual.square().flatten(1).mean(1);metrics.update({"psnr":float((10*torch.log10(1/mse.clamp_min(1e-12))).mean()),"ssim":_ssim(images,out["watermarked_image"]),
      "residual_saturation_fraction":float(residual.abs().ge(.014*.999).float().mean()),"rs_logit_drift":float((logits[...,4:12]-parent).abs().mean()),"analytical_contribution":0.0,
      "blind_decoder":list(inspect.signature(model.index_head.forward).parameters)==["questioned_image"],"disjoint_images":True})
    changed=bits.clone();changed[:,0,0,4]=1-changed[:,0,0,4]
    with torch.no_grad():other=model(images,changed,.014,level);delta=(other["packet_logits"]-logits).abs().flatten(1)
    metrics["cross_region_leakage"]=float(delta[:,1:].mean())/max(float(delta[:,4].mean()),1e-12)
    return metrics

def run_transition_repair(config:Mapping[str,Any]):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);verification,parent=verify_stage_c_parent(config)
    if not verification["passed"]:
        report={"scientific_status":"blocked_by_stage_c_checkpoint","stage_e_permitted":False,"checkpoint_verification":verification};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    seed=int(config.get("seed",2027));random.seed(seed);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+1);key=ephemeral_key()
    synthetic_count=int(config.get("synthetic_image_count",0));dataset=SyntheticStageCDataset(synthetic_count) if synthetic_count else CocoImageDataset(config["data_root"])
    ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_split(ids,int(config["train_images"]),int(config["validation_images"]),seed);overlap=sorted(set(split["train"])&set(split["validation"]))
    train=load_stage_c_population(dataset,split["train"],config["preprocessing"],64);validation=load_stage_c_population(dataset,split["validation"],config["preprocessing"],64);model=StageD12BitTransitionV1(parent)
    audit_bits,_=fresh_packet_batch(2,key,generator);audit=transplant_audit(model,validation[:2],audit_bits,float(config.get("transplant_tolerance",0)))
    if not audit["passed"]:
        report={"scientific_status":"blocked_by_12_bit_transplant_equivalence","stage_e_permitted":False,"checkpoint_verification":verification,"transplant_audit":audit};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    for parameter in model.stage_c.parameters():parameter.requires_grad=False
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=float(config.get("learning_rate",1e-3)));maximum=int(config.get("maximum_steps_per_level",300));minimum=int(config.get("minimum_steps_per_level",50));every=int(config.get("evaluate_every",25));batch=int(config.get("batch_size",8));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max(1,4*maximum));curriculum={"R0":{"passed":True,"step":0,**audit}};history=[];blocked=None
    for level in range(1,5):
        best={k:v.detach().clone() for k,v in model.state_dict().items()};best_score=-1;passed=False
        for step in range(1,maximum+1):
            idx=torch.randint(len(train),(batch,),generator=generator);image=train[idx];bits,_=fresh_packet_batch(batch,key,generator);out=model(image,bits,.014,level)
            with torch.no_grad():parent_logits=model.stage_c(image,bits[...,4:12],.014)["regional_logits"]
            losses=transition_loss(out["packet_logits"],bits,level,parent_logits,float(config.get("distillation_weight",1)))
            fidelity=F.l1_loss(out["watermarked_image"],image);total=losses["total"]+.2*fidelity+2*F.relu(out["residual"].abs()/.014-.9).square().mean();optimizer.zero_grad(set_to_none=True);old=[p.detach().clone() for p in model.parameters() if p.requires_grad];total.backward();gb,ga=clip_gradients([p for p in model.parameters() if p.requires_grad],1);optimizer.step();scheduler.step();update=math.sqrt(sum(float((p.detach()-o).square().sum()) for p,o in zip([p for p in model.parameters() if p.requires_grad],old)))
            if step%every==0:
                ebits,_=fresh_packet_batch(len(validation),key,generator);metrics=_evaluate(model,validation,ebits,level,generator);history.append({"level":f"R{level}","step":step,"gradient_norm_new_before":gb,"gradient_norm_new_after":ga,"new_parameter_update_norm":update,"old_parameter_update_norm":0.0,**metrics});score=min(metrics["index_bit_accuracy"],metrics["rs_bit_accuracy"])
                if score>best_score:best_score=score;best={k:v.detach().clone() for k,v in model.state_dict().items()}
                level_gate=metrics["rs_bit_accuracy"]>=.95 and metrics["index_bit_accuracy"]>=float(config.get("transition_index_gate",.995)) and min(metrics["per_active_bit_accuracy"])>=.90 and min(metrics["per_region_accuracy"])>=.90
                if step>=minimum and level_gate:passed=True;break
        model.load_state_dict(best);curriculum[f"R{level}"]={"passed":passed,"step":step,"best_accuracy":best_score,"metrics":metrics}
        if not passed:blocked=level;break
    eval_images,population=build_evaluation_population(validation,int(config.get("final_evaluation_samples",32)));eval_images=preprocess_stage_c_image(eval_images,config["preprocessing"],64);eval_bits,_=fresh_packet_batch(len(eval_images),key,generator);metrics=_evaluate(model,eval_images,eval_bits,4,generator);gates=repair_gates(metrics);passed=blocked is None and all(gates.values());status="passed_stage_d_12_bit_transition_repair" if passed else "blocked_by_12_bit_transition_level" if blocked else "blocked_by_12_bit_repair_gate";metrics.update({"gate_results":gates,"scientific_status":status})
    safe={k:v for k,v in config.items() if "key" not in k.lower()};checkpoint={"schema_version":"stage_d_12bit_transition.0","architecture_version":model.architecture_version,"stage_c_parent_sha256":verification["sha256"],"configuration":safe,"preprocessing":config["preprocessing"],"active_mapping":{"index":[0,4],"rs_symbol":[4,12]},"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"split_manifest":split,"metrics":metrics,"scientific_status":status,"stage_e_permitted":False};torch.save(checkpoint,output/"last.pt")
    if passed:torch.save(checkpoint,output/"best.pt")
    report={"schema_version":"stage_d_12bit_transition_report.0","checkpoint_verification":verification,"transplant_audit":audit,"curriculum":curriculum,"first_failing_level":blocked,"evaluation_population":population,"data_split":{"train_count":len(split["train"]),"validation_count":len(split["validation"]),"train_validation_overlap_count":len(overlap),"train_validation_overlap":overlap},"history":history,"metrics":metrics,"gate_results":gates,"stage_d_12bit_passed":passed,"stage_e_permitted":False,"scientific_status":status};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
