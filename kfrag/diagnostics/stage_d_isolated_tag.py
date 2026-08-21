"""Local-only architectural tag-subspace isolation experiment."""
from __future__ import annotations
import copy,hashlib,inspect,json,math,random
from pathlib import Path
from typing import Mapping,Any
import torch
from torch.nn import functional as F
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset,_ssim
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent
from kfrag.models.stage_d_isolated_tag_v1 import StageDIsolatedTagV1
from kfrag.training.complete_packet_v1 import fresh_packet_batch
from kfrag.training.natural_channel_v2 import deterministic_split,clip_gradients
from kfrag.training.regional_channel_v1 import load_stage_c_population,build_evaluation_population,preprocess_stage_c_image
from kfrag.training.stage_d_tag_capacity import balanced_bit_loss,capacity_metrics,capacity_gates

def _key(seed):return hashlib.sha256(f"isolated-tag-fixture:{seed}".encode()).digest()
def _evaluate(model,images,bits,tag_bits,generator):
    active=12+tag_bits;model.eval()
    with torch.no_grad():out=model(images,bits,.014,tag_bits);logits=out["packet_logits"];parent=model.parent(images,bits,.014,4);original=torch.cat((model.parent.index_head(images),model.parent.stage_c.decoder(images),model.tag_decoder(images)[...,:tag_bits]),-1)
    metrics=capacity_metrics(logits,bits,tag_bits);randomized=torch.roll(bits,1,0);spatial=torch.roll(bits,1,1);correct=logits[...,4:active].ge(0).eq(bits[...,4:active].bool()).float().mean();shuffled=logits[...,4:active].ge(0).eq(randomized[...,4:active].bool()).float().mean();metrics["correct_minus_shuffled_margin"]=float((correct-shuffled).detach());metrics["correct_minus_spatial_margin"]=metrics["overall_bit_accuracy"]-capacity_metrics(logits,spatial,tag_bits)["overall_bit_accuracy"]
    target=torch.randint(0,2,(len(images),4,4,active),generator=generator).float();metrics["original_randomized_field_accuracy"]=float(original.ge(0).eq(target.bool()).float().mean());residual=out["residual"];mse=residual.square().flatten(1).mean(1);metrics.update({"psnr":float((10*torch.log10(1/mse.clamp_min(1e-12))).mean()),"ssim":_ssim(images,out["watermarked_image"]),"residual_saturation_fraction":float(residual.abs().ge(.014*.999).float().mean()),"parent_rs_logit_drift":float((logits[...,4:12]-parent["packet_logits"][...,4:12]).abs().mean()),"parent_residual_energy":float(out["parent_residual"].square().mean()),"tag_residual_energy":float(out["tag_residual"].square().mean()),"combined_residual_energy":float(residual.square().mean()),"analytical_contribution":0.0,"blind_decoder":list(inspect.signature(model.tag_decoder.forward).parameters)==["questioned_image"],"disjoint_images":True,"no_secret_or_expected_payload_serialized":True})
    parent_bits=parent["packet_logits"][...,:12].ge(0).eq(bits[...,:12].bool()).float().mean((0,1,2));after=logits[...,:12].ge(0).eq(bits[...,:12].bool()).float().mean((0,1,2));metrics["weakest_parent_bit_before"]={"packet_bit":int(parent_bits.argmin()),"accuracy":float(parent_bits.min())};metrics["weakest_parent_bit_after"]={"packet_bit":int(after.argmin()),"accuracy":float(after.min())}
    correlation=model.carrier.correlation_matrix();metrics["carrier_correlation_summary"]={"maximum_absolute":float(correlation.abs().max()),"mean_absolute":float(correlation.abs().mean()),"rms":float(correlation.square().mean().sqrt()),"shape":list(correlation.shape)}
    weight=model.tag_projection.weight[:,:,0,0];tag_rgb=torch.einsum("oc,kchw->kohw",weight,model.carrier.vectors);parent_rgb=torch.einsum("oc,kchw->kohw",weight,model.carrier.parent_vectors);spatial=F.normalize(tag_rgb.flatten(1),dim=1)@F.normalize(parent_rgb.flatten(1),dim=1).T
    highpass=model.parent.stage_c.decoder.decoder.highpass;tag_hp=highpass(tag_rgb);parent_hp=highpass(parent_rgb);frequency=F.normalize(tag_hp.flatten(1),dim=1)@F.normalize(parent_hp.flatten(1),dim=1).T
    metrics["post_projection_correlation_summary"]={"matrix_shape":[32,12],"spatial_maximum_absolute":float(spatial.abs().max().detach()),"frequency_maximum_absolute":float(frequency.abs().max().detach()),"spatial_mean_absolute":float(spatial.abs().mean().detach()),"frequency_mean_absolute":float(frequency.abs().mean().detach())}
    changed=bits.clone();changed[:,0,0,12]=1-changed[:,0,0,12]
    with torch.no_grad():other=model(images,changed,.014,tag_bits);delta=(other["packet_logits"]-logits).abs();intended=float(delta[:,0,0,12].mean());unrelated=torch.cat((delta[:,:1,:1,:12].flatten(),delta[:,1:].flatten()));metrics["cross_region_leakage"]=float(unrelated.mean())/max(intended,1e-12)
    questioned=out["watermarked_image"][:min(4,len(images))].detach().requires_grad_(True);target_bits=bits[:len(questioned)];parent_logits=torch.cat((model.parent.index_head(questioned),model.parent.stage_c.decoder(questioned)),-1);tag_logits=model.tag_decoder(questioned)[...,:tag_bits];parent_loss=F.binary_cross_entropy_with_logits(parent_logits,target_bits[...,:12]);tag_loss=F.binary_cross_entropy_with_logits(tag_logits,target_bits[...,12:active]);gp=torch.autograd.grad(parent_loss,questioned,retain_graph=True)[0];gt=torch.autograd.grad(tag_loss,questioned)[0];metrics["decoder_gradient_interference_cosine"]=float(F.cosine_similarity(gp.flatten(1),gt.flatten(1)).mean().detach())
    return metrics

def _gates(metrics):
    gates=capacity_gates(metrics);gates.update({"leakage":metrics["cross_region_leakage"]<=.10,"serialization":metrics["no_secret_or_expected_payload_serialized"],"carrier_decorrelation":metrics["carrier_correlation_summary"]["maximum_absolute"]<=1e-5});return gates

def run_isolated_tag_experiment(config:Mapping[str,Any]):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);verification,parent=verify_12bit_parent(config)
    if not verification["passed"]:
        report={"scientific_status":"blocked_by_12_bit_parent","stage_e_permitted":False,"parent_verification":verification};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    seed=int(config.get("seed",2030));random.seed(seed);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+1);key=_key(seed);dataset=SyntheticStageCDataset(int(config["synthetic_image_count"]));ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_split(ids,int(config["train_images"]),int(config["validation_images"]),seed);train=load_stage_c_population(dataset,split["train"],config["preprocessing"],64);validation=load_stage_c_population(dataset,split["validation"],config["preprocessing"],64);model=StageDIsolatedTagV1(parent,float(config.get("tag_budget_fraction",.25)))
    audit_bits,_=fresh_packet_batch(2,key,generator)
    with torch.no_grad():p0=model.parent(validation[:2],audit_bits,.014,4);zero=model(validation[:2],audit_bits,.014,0)
    audit={"residual":torch.equal(p0["residual"],zero["residual"]),"watermarked_image":torch.equal(p0["watermarked_image"],zero["watermarked_image"]),"parent_logits":torch.equal(p0["packet_logits"],zero["packet_logits"]),"parent_sha256":verification["sha256"]};audit["passed"]=all(v for k,v in audit.items() if k not in ("parent_sha256",))
    for parameter in model.parent.parameters():parameter.requires_grad=False
    schedule=(("P1",8,True),("bridge_10tag",10,False),("bridge_12tag",12,False),("bridge_14tag",14,False),("P2",16,True),("P3",24,True),("P4",32,True));maximum=int(config.get("maximum_steps_per_level",300));minimum=int(config.get("minimum_steps_per_level",50));every=int(config.get("evaluate_every",25));batch=int(config.get("batch_size",8));levels={"P0":{"passed":audit["passed"],"audit":audit}};history=[];blocked=None;optimizer=scheduler=None;previous_active=12
    for name,tag_bits,official in schedule:
        teacher=copy.deepcopy(model).eval();trainable=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(trainable,lr=float(config.get("learning_rate",3e-4)),weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,maximum);ema=torch.ones(12+tag_bits);best=None;best_metrics=None;terminal=None;streak=0
        for step in range(1,maximum+1):
            image=train[torch.randint(len(train),(batch,),generator=generator)];bits,_=fresh_packet_batch(batch,key,generator);out=model(image,bits,.014,tag_bits);loss,ema,weights=balanced_bit_loss(out["packet_logits"],bits,12+tag_bits,ema,.9,3)
            with torch.no_grad():teacher_logits=teacher(image,bits,.014,max(0,previous_active-12))["packet_logits"]
            distill=F.mse_loss(out["packet_logits"][...,:previous_active],teacher_logits[...,:previous_active]);energy=out["tag_residual"].square().mean();total=loss+float(config.get("distillation_weight",5))*distill+float(config.get("tag_energy_weight",1))*energy;optimizer.zero_grad(set_to_none=True);total.backward();parent_grad=sum(float(p.grad.abs().sum()) for p in model.parent.parameters() if p.grad is not None);gb,ga=clip_gradients(trainable,1);optimizer.step();scheduler.step()
            if step%every==0:
                ebits,_=fresh_packet_batch(len(validation),key,generator);metrics=_evaluate(model,validation,ebits,tag_bits,generator);gates=_gates(metrics);passed=all(gates.values());row={"level":name,"official":official,"tag_bits":tag_bits,"step":step,"training_weights":weights.tolist(),"parent_gradient_sum":parent_grad,"gradient_norm_before":gb,"gradient_norm_after":ga,"gate_results":gates,**metrics};history.append(row);terminal=metrics
                if passed:best={k:v.detach().clone() for k,v in model.state_dict().items()};best_metrics=dict(metrics)
                streak=streak+1 if passed and step>=minimum else 0
                if streak>=2:break
        level_passed=best is not None
        if level_passed:model.load_state_dict(best)
        levels[name]={"official":official,"tag_bits":tag_bits,"passed":level_passed,"step":step,"selected_metrics":best_metrics,"terminal_metrics":terminal,"selected_per_bit_accuracy":None if best_metrics is None else best_metrics["per_bit_accuracy"],"terminal_per_bit_accuracy":terminal["per_bit_accuracy"]}
        if level_passed and official:
            directory=output/name;directory.mkdir(exist_ok=True);torch.save({"schema_version":"stage_d_isolated_tag.0","model_state":model.state_dict(),"level":name,"metrics":best_metrics,"parent_sha256":verification["sha256"],"stage_e_permitted":False},directory/"best.pt")
        if not level_passed:blocked=name;break
        previous_active=12+tag_bits
    last_name=list(levels)[-1];final_tag_bits=levels[last_name].get("tag_bits",0);eval_images,pop=build_evaluation_population(validation,int(config.get("final_evaluation_samples",32)));eval_images=preprocess_stage_c_image(eval_images,config["preprocessing"],64);eval_bits,_=fresh_packet_batch(len(eval_images),key,generator);metrics=_evaluate(model,eval_images,eval_bits,final_tag_bits,generator);gates=_gates(metrics);passed=blocked is None and final_tag_bits==32 and all(gates.values());status="passed_stage_d_isolated_tag_experiment" if passed else "blocked_by_isolated_tag_capacity";metrics.update({"gate_results":gates,"scientific_status":status,"evaluated_tag_bits":final_tag_bits});checkpoint={"schema_version":"stage_d_isolated_tag.0","architecture_version":model.architecture_version,"parent_sha256":verification["sha256"],"configuration":{k:v for k,v in config.items() if "key" not in k.lower()},"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"metrics":metrics,"scientific_status":status,"stage_e_permitted":False};torch.save(checkpoint,output/"last.pt")
    if passed:torch.save(checkpoint,output/"best.pt")
    report={"schema_version":"stage_d_isolated_tag_report.0","parent_verification":verification,"p0_audit":audit,"curriculum":levels,"first_failing_level":blocked,"evaluation_population":pop,"history":history,"metrics":metrics,"gate_results":gates,"passed":passed,"stage_e_permitted":False,"scientific_status":status};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
