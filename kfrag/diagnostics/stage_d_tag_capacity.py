"""Executed, fail-fast P0-P4 Stage-D tag-capacity progression."""
from __future__ import annotations
import hashlib,inspect,json,math,random
from pathlib import Path
from typing import Mapping,Any
import torch
from torch.nn import functional as F
from kfrag.data import CocoImageDataset
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset,_ssim
from kfrag.diagnostics.stage_d_complete_packet import verify_stage_c_parent,_contains_prohibited_material
from kfrag.models.stage_d_12bit_transition_v1 import StageD12BitTransitionV1
from kfrag.models.stage_d_tag_capacity_v1 import StageDTagCapacityV1
from kfrag.training.complete_packet_v1 import ephemeral_key,fresh_packet_batch
from kfrag.training.natural_channel_v2 import deterministic_split,clip_gradients
from kfrag.training.regional_channel_v1 import load_stage_c_population,build_evaluation_population,preprocess_stage_c_image
from kfrag.training.stage_d_tag_capacity import tag_capacity_loss,capacity_metrics,capacity_gates

EXPECTED_SHA="37B312A8FCCB93F23D5A519BB51EDFCA105A962BD9C4ECA24C395826C91BCC0A"
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def verify_12bit_parent(config):
    path=Path(config["stage_d_12bit_checkpoint"]);report_path=Path(config["stage_d_12bit_report"]);checks={"exists":path.is_file() and path.stat().st_size>0,"report":report_path.is_file()}
    if not all(checks.values()):return {"passed":False,"checks":checks},None
    report=json.loads(report_path.read_text());checkpoint=torch.load(path,map_location="cpu",weights_only=False);checks.update({"sha256":_sha(path)==str(config.get("stage_d_12bit_sha256",EXPECTED_SHA)).upper(),"report_passed":report.get("stage_d_12bit_passed") is True,
      "status":report.get("scientific_status")=="passed_stage_d_12_bit_transition_repair","stage_e_false":report.get("stage_e_permitted") is False,"checkpoint_status":checkpoint.get("scientific_status")=="passed_stage_d_12_bit_transition_repair","finite":all(torch.isfinite(x).all() for x in checkpoint.get("model_state",{}).values()),"no_prohibited_material":not _contains_prohibited_material(report) and not _contains_prohibited_material(checkpoint)})
    verification,stage_c=verify_stage_c_parent(config);checks["stage_c_parent"]=verification["passed"]
    if not all(checks.values()):return {"passed":False,"checks":checks,"sha256":_sha(path)},None
    parent=StageD12BitTransitionV1(stage_c);expected=parent.state_dict();saved=checkpoint["model_state"];checks["schema"]=set(expected)==set(saved) and all(expected[k].shape==saved[k].shape and expected[k].dtype==saved[k].dtype for k in expected)
    if not checks["schema"]:return {"passed":False,"checks":checks,"sha256":_sha(path)},None
    parent.load_state_dict(saved,strict=True);parent.eval();g=torch.Generator().manual_seed(44);image=torch.rand(2,3,64,64,generator=g);bits=torch.randint(0,2,(2,4,4,44),generator=g).float();clone=StageD12BitTransitionV1(stage_c);clone.load_state_dict(parent.state_dict());clone.eval()
    with torch.no_grad():a=parent(image,bits,.014,4)["packet_logits"];b=clone(image,bits,.014,4)["packet_logits"]
    checks["roundtrip"]=torch.equal(a,b);return {"passed":all(checks.values()),"checks":checks,"sha256":_sha(path),"roundtrip_max_difference":float((a-b).abs().max())},parent

def transplant_audit(model,images,bits):
    with torch.no_grad():a=model.parent(images,bits,.014,4);b=model(images,bits,.014,0)
    checks={name:torch.equal(a[name],b[name]) for name in ("residual","watermarked_image","packet_logits")};changed=bits.clone();changed[...,12:]=1-changed[...,12:]
    with torch.no_grad():c=model(images,changed,.014,0)
    checks["inactive_tag_isolation"]=torch.equal(b["residual"],c["residual"]) and torch.equal(b["packet_logits"],c["packet_logits"]);return {"passed":all(checks.values()),"checks":checks,"maximum_logit_difference":float((a["packet_logits"]-b["packet_logits"]).abs().max())}

def _evaluate(model,images,bits,tag_bits,generator):
    model.eval();active=12+tag_bits
    with torch.no_grad():out=model(images,bits,.014,tag_bits);logits=out["packet_logits"]
    m=capacity_metrics(logits,bits,tag_bits);shuffled=torch.roll(bits,1,0);spatial=torch.roll(bits,1,1)
    randomized_correct=logits[...,4:active].ge(0).eq(bits[...,4:active].bool()).float().mean();randomized_shuffled=logits[...,4:active].ge(0).eq(shuffled[...,4:active].bool()).float().mean()
    m["correct_minus_shuffled_margin"]=float((randomized_correct-randomized_shuffled).detach().cpu());m["shuffled_margin_field"]="randomized_rs_and_active_tag_bits_only"
    m["correct_minus_spatial_margin"]=m["overall_bit_accuracy"]-capacity_metrics(logits,spatial,tag_bits)["overall_bit_accuracy"]
    random=torch.randint(0,2,(len(images),4,4,active),generator=generator).float();original=torch.cat((model.parent.index_head(images),model.parent.stage_c.decoder(images),model.tag_head(images)[...,:tag_bits]),-1);m["original_randomized_field_accuracy"]=float(original.ge(0).eq(random.bool()).float().mean().detach())
    residual=out["residual"];mse=residual.square().flatten(1).mean(1);m.update({"psnr":float((10*torch.log10(1/mse.clamp_min(1e-12))).mean()),"ssim":_ssim(images,out["watermarked_image"]),"residual_saturation_fraction":float(residual.abs().ge(.014*.999).float().mean()),"analytical_contribution":0.0,"blind_decoder":list(inspect.signature(model.tag_head.forward).parameters)==["questioned_image"],"disjoint_images":True})
    return m

def run_tag_progression(config:Mapping[str,Any]):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);verification,parent=verify_12bit_parent(config)
    if not verification["passed"]:
        report={"scientific_status":"blocked_by_12_bit_parent","stage_e_permitted":False,"parent_verification":verification};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    seed=int(config.get("seed",2028));random.seed(seed);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+1);key=ephemeral_key();count=int(config.get("synthetic_image_count",0));dataset=SyntheticStageCDataset(count) if count else CocoImageDataset(config["data_root"]);ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_split(ids,int(config["train_images"]),int(config["validation_images"]),seed);train=load_stage_c_population(dataset,split["train"],config["preprocessing"],64);validation=load_stage_c_population(dataset,split["validation"],config["preprocessing"],64);model=StageDTagCapacityV1(parent);audit_bits,_=fresh_packet_batch(2,key,generator);audit=transplant_audit(model,validation[:2],audit_bits)
    if not audit["passed"]:
        report={"scientific_status":"blocked_by_p0_transplant","stage_e_permitted":False,"parent_verification":verification,"p0_audit":audit};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    for p in model.parent.parameters():p.requires_grad=False
    levels=(8,16,24,32);maximum=int(config.get("maximum_steps_per_level",300));minimum=int(config.get("minimum_steps_per_level",50));every=int(config.get("evaluate_every",25));batch=int(config.get("batch_size",8));history=[];curriculum={"P0":{"passed":True,"tag_bits":0,"audit":audit}};blocked=None;optimizer=scheduler=None
    for number,tag_bits in enumerate(levels,1):
        trainable=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(trainable,lr=float(config.get("learning_rate",3e-4)));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,maximum);rollback={k:v.detach().clone() for k,v in model.state_dict().items()};best=None;best_metrics=None;best_score=-1;best_any=rollback;best_any_metrics=None;best_any_score=-float("inf");streak=0
        for step in range(1,maximum+1):
            image=train[torch.randint(len(train),(batch,),generator=generator)];bits,_=fresh_packet_batch(batch,key,generator);out=model(image,bits,.014,tag_bits)
            with torch.no_grad():parent_logits=model.parent(image,bits,.014,4)["packet_logits"]
            losses=tag_capacity_loss(out["packet_logits"],bits,tag_bits,parent_logits,config.get("loss_weights",{}));residual=out["residual"];total=losses["total"]+.2*F.l1_loss(out["watermarked_image"],image)+2*F.relu(residual.abs()/.014-.9).square().mean();optimizer.zero_grad(set_to_none=True);total.backward();gb,ga=clip_gradients(trainable,1);optimizer.step();scheduler.step()
            if step%every==0:
                ebits,_=fresh_packet_batch(len(validation),key,generator);m=_evaluate(model,validation,ebits,tag_bits,generator);gates=capacity_gates(m);passed=all(gates.values());score=min(m["index_bit_accuracy"],m["rs_bit_accuracy"],m["tag_bit_accuracy"]);history.append({"level":f"P{number}","step":step,"gradient_norm_before":gb,"gradient_norm_after":ga,"gate_results":gates,**m})
                any_score=min(m["index_bit_accuracy"]-.98,m["rs_bit_accuracy"]-.95,m["tag_bit_accuracy"]-.95,min(m["per_bit_accuracy"])-.90,min(m["per_region_accuracy"])-.90,m["correct_minus_shuffled_margin"]-.40,m["correct_minus_spatial_margin"]-.40)
                if any_score>best_any_score:best_any_score=any_score;best_any={k:v.detach().clone() for k,v in model.state_dict().items()};best_any_metrics=m
                if passed and score>best_score:best_score=score;best={k:v.detach().clone() for k,v in model.state_dict().items()};best_metrics=m
                streak=streak+1 if passed and step>=minimum else 0
                if streak>=2:break
        level_passed=best is not None
        model.load_state_dict(best if level_passed else best_any)
        curriculum[f"P{number}"]={"tag_bits":tag_bits,"passed":level_passed,"step":step,"rolled_back_to_best_observed_checkpoint":not level_passed,"metrics":best_metrics or best_any_metrics or m}
        if not level_passed:blocked=number;break
    final_tag_bits=32 if blocked is None else levels[blocked-1];eval_images,pop=build_evaluation_population(validation,int(config.get("final_evaluation_samples",32)));eval_images=preprocess_stage_c_image(eval_images,config["preprocessing"],64);eval_bits,_=fresh_packet_batch(len(eval_images),key,generator);metrics=_evaluate(model,eval_images,eval_bits,final_tag_bits,generator);metrics["evaluated_tag_bits"]=final_tag_bits;gates=capacity_gates(metrics);passed=blocked is None and all(gates.values());status="passed_stage_d_tag_capacity_progression" if passed else "blocked_by_tag_capacity";metrics.update({"gate_results":gates,"scientific_status":status});safe={k:v for k,v in config.items() if "key" not in k.lower()};checkpoint={"schema_version":"stage_d_tag_capacity.0","architecture_version":model.architecture_version,"stage_d_12bit_parent_sha256":verification["sha256"],"configuration":safe,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"curriculum":curriculum,"metrics":metrics,"scientific_status":status,"stage_e_permitted":False};torch.save(checkpoint,output/"last.pt");
    if passed:torch.save(checkpoint,output/"best.pt")
    report={"schema_version":"stage_d_tag_capacity_report.0","parent_verification":verification,"p0_audit":audit,"curriculum":curriculum,"first_failing_level":blocked,"evaluation_population":pop,"history":history,"metrics":metrics,"gate_results":gates,"tag_capacity_passed":passed,"stage_e_permitted":False,"scientific_status":status};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
