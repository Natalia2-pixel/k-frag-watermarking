"""Separate P2 repair with four-bit bridging and training-only bit balancing."""
from __future__ import annotations
import copy,hashlib,json,math,random
from pathlib import Path
from typing import Mapping,Any
import torch
from torch.nn import functional as F
from kfrag.data import CocoImageDataset
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent,transplant_audit,_evaluate
from kfrag.models.stage_d_tag_capacity_v1 import StageDTagCapacityV1
from kfrag.training.complete_packet_v1 import ephemeral_key,fresh_packet_batch
from kfrag.training.natural_channel_v2 import deterministic_split,clip_gradients
from kfrag.training.regional_channel_v1 import load_stage_c_population,build_evaluation_population,preprocess_stage_c_image
from kfrag.training.stage_d_tag_capacity import balanced_bit_loss,capacity_gates

def weak_bit_diagnosis(report):
    result={}
    for level in ("P1","P2"):
        rows=[x for x in report.get("history",[]) if x.get("level")==level]
        if not rows:continue
        result[level]=[{"packet_bit":i,"field":"index" if i<4 else "rs" if i<12 else "tag","minimum":min(x["per_bit_accuracy"][i] for x in rows),"mean":sum(x["per_bit_accuracy"][i] for x in rows)/len(rows),"below_gate_count":sum(x["per_bit_accuracy"][i]<.90 for x in rows)} for i in range(len(rows[0]["per_bit_accuracy"])) if any(x["per_bit_accuracy"][i]<.90 for x in rows)]
    return result

def deterministic_ephemeral_key(seed):return hashlib.sha256(f"stage-d-p2-repair:{int(seed)}".encode()).digest()

def _safe_checkpoint(model,optimizer,scheduler,config,verification,metrics,level):
    safe={k:v for k,v in config.items() if "key" not in k.lower()};return {"schema_version":"stage_d_p2_tag_repair.0","architecture_version":model.architecture_version,"stage_d_12bit_parent_sha256":verification["sha256"],"configuration":safe,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"selected_level":level,"metrics":metrics,"scientific_status":metrics.get("scientific_status","p2_repair_in_progress"),"stage_e_permitted":False}

def run_p2_repair(config:Mapping[str,Any]):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);verification,parent=verify_12bit_parent(config)
    if not verification["passed"]:
        report={"scientific_status":"blocked_by_12_bit_parent","stage_e_permitted":False,"parent_verification":verification};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    baseline_path=Path(config["blocked_baseline_report"]);diagnosis=weak_bit_diagnosis(json.loads(baseline_path.read_text()))
    seed=int(config.get("seed",2029));random.seed(seed);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+1);key=deterministic_ephemeral_key(seed);count=int(config.get("synthetic_image_count",0));dataset=SyntheticStageCDataset(count) if count else CocoImageDataset(config["data_root"]);ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_split(ids,int(config["train_images"]),int(config["validation_images"]),seed);train=load_stage_c_population(dataset,split["train"],config["preprocessing"],64);validation=load_stage_c_population(dataset,split["validation"],config["preprocessing"],64);model=StageDTagCapacityV1(parent);audit_bits,_=fresh_packet_batch(2,key,generator);audit=transplant_audit(model,validation[:2],audit_bits)
    for parameter in model.parent.parameters():parameter.requires_grad=False
    schedule=(("P1",8,True),("P1_to_P2_10tag",10,False),("P1_to_P2_12tag",12,False),("P1_to_P2_14tag",14,False),("P2",16,True));maximum=int(config.get("maximum_steps_per_level",500));minimum=int(config.get("minimum_steps_per_level",50));every=int(config.get("evaluate_every",25));batch=int(config.get("batch_size",8));history=[];levels={"P0":{"passed":audit["passed"],"audit":audit}};blocked=None;p1_checkpoint=None;optimizer=scheduler=None;previous_active=12
    for name,tag_bits,official in schedule:
        teacher=copy.deepcopy(model).eval();
        for parameter in teacher.parameters():parameter.requires_grad=False
        trainable=[p for p in model.parameters() if p.requires_grad];level_lr=float(config.get("learning_rate",3e-4) if name=="P1" else config.get("bridge_learning_rate",1e-4));optimizer=torch.optim.AdamW(trainable,lr=level_lr,weight_decay=float(config.get("weight_decay",1e-4)));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,maximum);ema=torch.ones(12+tag_bits);best=None;best_metrics=None;best_score=-float("inf");terminal=None;streak=0
        selected_history=[]
        for step in range(1,maximum+1):
            image=train[torch.randint(len(train),(batch,),generator=generator)];bits,_=fresh_packet_batch(batch,key,generator);out=model(image,bits,.014,tag_bits)
            loss,ema,weights=balanced_bit_loss(out["packet_logits"],bits,12+tag_bits,ema,float(config.get("balance_decay",.9)),float(config.get("maximum_bit_weight",3)))
            with torch.no_grad():teacher_logits=teacher(image,bits,.014,0 if name=="P1" else previous_active-12)["packet_logits"]
            distill=F.mse_loss(out["packet_logits"][...,:previous_active],teacher_logits[...,:previous_active]);residual=out["residual"];distillation_weight=float(config.get("distillation_weight",1) if name=="P1" else config.get("bridge_distillation_weight",10));total=loss+distillation_weight*distill+.2*F.l1_loss(out["watermarked_image"],image)+2*F.relu(residual.abs()/.014-.9).square().mean();optimizer.zero_grad(set_to_none=True);total.backward();gb,ga=clip_gradients(trainable,1);optimizer.step();scheduler.step()
            if step%every==0:
                ebits,_=fresh_packet_batch(len(validation),key,generator);metrics=_evaluate(model,validation,ebits,tag_bits,generator);gates=capacity_gates(metrics);passed=all(gates.values());score=min(metrics["per_bit_accuracy"]);row={"level":name,"official_milestone":official,"tag_bits":tag_bits,"step":step,"learning_rate":level_lr,"distillation_weight":distillation_weight,"training_bit_loss_ema":ema.tolist(),"training_bit_weights":weights.tolist(),"gradient_norm_before":gb,"gradient_norm_after":ga,"gate_results":gates,**metrics};history.append(row);selected_history.append(row);terminal=metrics
                if passed and score>best_score:best_score=score;best={k:v.detach().clone() for k,v in model.state_dict().items()};best_metrics=dict(metrics)
                streak=streak+1 if passed and step>=minimum else 0
                if streak>=2:break
        level_passed=best is not None
        if level_passed:model.load_state_dict(best)
        levels[name]={"official_milestone":official,"tag_bits":tag_bits,"passed":level_passed,"step":step,"selected_metrics":best_metrics,"terminal_metrics":terminal,"selected_per_bit_history":[x["per_bit_accuracy"] for x in selected_history if all(x["gate_results"].values())],"terminal_per_bit_accuracy":terminal["per_bit_accuracy"]}
        if name=="P1" and level_passed:
            p1_dir=output/"P1";p1_dir.mkdir(exist_ok=True);p1_metrics={**best_metrics,"scientific_status":"passed_p1_20bit_prerequisite"};p1_checkpoint=_safe_checkpoint(model,optimizer,scheduler,config,verification,p1_metrics,"P1_20bit");torch.save(p1_checkpoint,p1_dir/"best.pt")
        if not level_passed:blocked=name;break
        previous_active=12+tag_bits
    final_tag_bits=levels[list(levels)[-1]].get("tag_bits",8);eval_images,pop=build_evaluation_population(validation,int(config.get("final_evaluation_samples",32)));eval_images=preprocess_stage_c_image(eval_images,config["preprocessing"],64);eval_bits,_=fresh_packet_batch(len(eval_images),key,generator);metrics=_evaluate(model,eval_images,eval_bits,final_tag_bits,generator);gates=capacity_gates(metrics);p2_passed=blocked is None and final_tag_bits==16 and all(gates.values());status="passed_stage_d_p2_tag_capacity_repair" if p2_passed else "blocked_by_p2_tag_capacity_repair";metrics.update({"gate_results":gates,"scientific_status":status,"evaluated_tag_bits":final_tag_bits});torch.save(_safe_checkpoint(model,optimizer,scheduler,config,verification,metrics,blocked or "P2_28bit"),output/"last.pt")
    report={"schema_version":"stage_d_p2_tag_repair_report.0","parent_verification":verification,"p0_audit":audit,"baseline_weak_bit_diagnosis":diagnosis,"rejected_hypothesis":"protecting carrier channel 4 caused P1 to fail; channel identity alone is not causal","curriculum":levels,"first_failing_level":blocked,"evaluation_population":pop,"history":history,"metrics":metrics,"gate_results":gates,"p2_repair_passed":p2_passed,"p1_checkpoint_saved":p1_checkpoint is not None,"p4_best_withheld":True,"stage_e_permitted":False,"scientific_status":status};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
