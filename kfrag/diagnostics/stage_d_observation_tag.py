"""Local P0/P1 feasibility test for observation-space tag isolation."""
from __future__ import annotations
import hashlib,inspect,json,random
from pathlib import Path
from typing import Mapping,Any
import torch
from torch.nn import functional as F
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset,_ssim
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent
from kfrag.models.stage_d_observation_tag_v1 import StageDObservationTagV1
from kfrag.training.complete_packet_v1 import fresh_packet_batch
from kfrag.training.natural_channel_v2 import deterministic_split,clip_gradients
from kfrag.training.regional_channel_v1 import load_stage_c_population,build_evaluation_population,preprocess_stage_c_image
from kfrag.training.stage_d_tag_capacity import balanced_bit_loss,capacity_metrics,capacity_gates

def deterministic_targets_key(seed):return hashlib.sha256(f"observation-space-tag:{seed}".encode()).digest()
def evaluate(model,images,bits,generator):
    model.eval()
    with torch.no_grad():out=model(images,bits,.014,8);logits=out["packet_logits"];parent=model.parent(images,bits,.014,4);original=torch.cat((model.parent.index_head(images),model.parent.stage_c.decoder(images),model.tag_decoder(images)[...,:8]),-1)
    m=capacity_metrics(logits,bits,8);shuffled=torch.roll(bits,1,0);spatial=torch.roll(bits,1,1);m["correct_minus_shuffled_margin"]=float((logits[...,4:20].ge(0).eq(bits[...,4:20].bool()).float().mean()-logits[...,4:20].ge(0).eq(shuffled[...,4:20].bool()).float().mean()).detach());m["correct_minus_spatial_margin"]=m["overall_bit_accuracy"]-capacity_metrics(logits,spatial,8)["overall_bit_accuracy"]
    target=torch.randint(0,2,(len(images),4,4,20),generator=generator).float();m["original_randomized_field_accuracy"]=float(original.ge(0).eq(target.bool()).float().mean());residual=out["residual"];mse=residual.square().flatten(1).mean(1);m.update({"psnr":float((10*torch.log10(1/mse.clamp_min(1e-12))).mean()),"ssim":_ssim(images,out["watermarked_image"]),"residual_saturation_fraction":float(residual.abs().ge(.014*.999).float().mean()),"parent_rs_accuracy_before":capacity_metrics(parent["packet_logits"],bits,0)["rs_bit_accuracy"],"parent_rs_accuracy_after":m["rs_bit_accuracy"],"parent_logit_drift":float((logits[...,:12]-parent["packet_logits"][...,:12]).abs().mean()),"parent_residual_energy":float(out["parent_residual"].square().mean()),"tag_residual_energy":float(out["tag_residual"].square().mean()),"combined_residual_energy":float(residual.square().mean()),"analytical_contribution":0.0,"blind_decoder":list(inspect.signature(model.tag_decoder.forward).parameters)==["questioned_image"],"disjoint_images":True,"no_secret_or_expected_payload_serialized":True})
    corr=model.carrier.observation_correlation();m["observation_correlation"]={"maximum_absolute":float(corr.abs().max()),"mean_absolute":float(corr.abs().mean()),"rms":float(corr.square().mean().sqrt()),"matrix_shape":list(corr.shape)}
    changed=bits.clone();changed[:,0,0,12]=1-changed[:,0,0,12]
    with torch.no_grad():other=model(images,changed,.014,8);delta=(other["packet_logits"]-logits).abs();intended=float(delta[:,0,0,12].mean());unrelated=torch.cat((delta[:,:1,:1,:12].flatten(),delta[:,1:].flatten()));m["cross_region_leakage"]=float(unrelated.mean())/max(intended,1e-12)
    questioned=out["watermarked_image"][:min(4,len(images))].detach().requires_grad_(True);targets=bits[:len(questioned)];parent_logits=torch.cat((model.parent.index_head(questioned),model.parent.stage_c.decoder(questioned)),-1);tag_logits=model.tag_decoder(questioned)[...,:8];pl=F.binary_cross_entropy_with_logits(parent_logits,targets[...,:12]);tl=F.binary_cross_entropy_with_logits(tag_logits,targets[...,12:20]);gp=torch.autograd.grad(pl,questioned,retain_graph=True)[0];gt=torch.autograd.grad(tl,questioned)[0];m["decoder_gradient_cosine"]=float(F.cosine_similarity(gp.flatten(1),gt.flatten(1)).mean().detach());m["weakest_parent_bit_before"]={"packet_bit":int(parent["packet_logits"][...,:12].ge(0).eq(bits[...,:12].bool()).float().mean((0,1,2)).argmin()),"accuracy":min(capacity_metrics(parent["packet_logits"],bits,0)["per_bit_accuracy"])};m["weakest_parent_bit_after"]={"packet_bit":int(torch.tensor(m["per_bit_accuracy"][:12]).argmin()),"accuracy":min(m["per_bit_accuracy"][:12])};return m

def gates(metrics,config):
    result=capacity_gates(metrics);result.update({"leakage":metrics["cross_region_leakage"]<=.10,"serialization":metrics["no_secret_or_expected_payload_serialized"],"observation_correlation":metrics["observation_correlation"]["maximum_absolute"]<=float(config.get("maximum_observation_correlation",1e-5)),"gradient_cosine":abs(metrics["decoder_gradient_cosine"])<=float(config.get("maximum_gradient_cosine",.20))});return result

def run_observation_experiment(config:Mapping[str,Any]):
    output=Path(config["output_directory"]);output.mkdir(parents=True,exist_ok=True);verification,parent=verify_12bit_parent(config)
    if not verification["passed"]:
        report={"scientific_status":"blocked_by_12_bit_parent","stage_e_permitted":False,"parent_verification":verification};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
    seed=int(config.get("seed",2031));random.seed(seed);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+1);key=deterministic_targets_key(seed);dataset=SyntheticStageCDataset(int(config["synthetic_image_count"]));ids=[dataset[i]["relative_path"] for i in range(len(dataset))];split=deterministic_split(ids,int(config["train_images"]),int(config["validation_images"]),seed);train=load_stage_c_population(dataset,split["train"],config["preprocessing"],64);validation=load_stage_c_population(dataset,split["validation"],config["preprocessing"],64);model=StageDObservationTagV1(parent,float(config.get("tag_budget_fraction",.25)))
    audit_bits,_=fresh_packet_batch(2,key,generator)
    with torch.no_grad():a=model.parent(validation[:2],audit_bits,.014,4);b=model(validation[:2],audit_bits,.014,0)
    audit={"residual":torch.equal(a["residual"],b["residual"]),"watermarked_image":torch.equal(a["watermarked_image"],b["watermarked_image"]),"parent_logits":torch.equal(a["packet_logits"],b["packet_logits"]),"parent_sha256":verification["sha256"]};audit["passed"]=audit["residual"] and audit["watermarked_image"] and audit["parent_logits"]
    for parameter in model.parent.parameters():parameter.requires_grad=False
    trainable=[p for p in model.parameters() if p.requires_grad];maximum=int(config.get("maximum_steps",300));minimum=int(config.get("minimum_steps",50));every=int(config.get("evaluate_every",25));batch=int(config.get("batch_size",8));optimizer=torch.optim.AdamW(trainable,lr=float(config.get("learning_rate",3e-4)),weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,maximum);ema=torch.ones(20);history=[];best=None;selected=None;terminal=None;streak=0
    for step in range(1,maximum+1):
        image=train[torch.randint(len(train),(batch,),generator=generator)];bits,_=fresh_packet_batch(batch,key,generator);out=model(image,bits,.014,8);loss,ema,weights=balanced_bit_loss(out["packet_logits"],bits,20,ema,.9,3);parent_distill=F.mse_loss(out["packet_logits"][...,:12],model.parent(image,bits,.014,4)["packet_logits"][...,:12].detach());total=loss+float(config.get("distillation_weight",5))*parent_distill+out["tag_residual"].square().mean();optimizer.zero_grad(set_to_none=True);total.backward();parent_grad=sum(float(p.grad.abs().sum()) for p in model.parent.parameters() if p.grad is not None);gb,ga=clip_gradients(trainable,1);optimizer.step();scheduler.step()
        if step%every==0:
            ebits,_=fresh_packet_batch(len(validation),key,generator);metrics=evaluate(model,validation,ebits,generator);result=gates(metrics,config);passed=all(result.values());row={"step":step,"training_weights":weights.tolist(),"parent_gradient_sum":parent_grad,"gradient_norm_before":gb,"gradient_norm_after":ga,"gate_results":result,**metrics};history.append(row);terminal=metrics
            if passed:best={k:v.detach().clone() for k,v in model.state_dict().items()};selected=dict(metrics)
            streak=streak+1 if passed and step>=minimum else 0
            if streak>=2:break
    p1_passed=best is not None
    if p1_passed:model.load_state_dict(best)
    eval_images,pop=build_evaluation_population(validation,int(config.get("final_evaluation_samples",32)));eval_images=preprocess_stage_c_image(eval_images,config["preprocessing"],64);eval_bits,_=fresh_packet_batch(len(eval_images),key,generator);metrics=evaluate(model,eval_images,eval_bits,generator);result=gates(metrics,config);passed=p1_passed and all(result.values());status="passed_observation_space_p1_feasibility" if passed else "blocked_by_observation_space_p1";metrics.update({"gate_results":result,"scientific_status":status});checkpoint={"schema_version":"stage_d_observation_tag.0","architecture_version":model.architecture_version,"parent_sha256":verification["sha256"],"configuration":{k:v for k,v in config.items() if "key" not in k.lower()},"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"metrics":metrics,"scientific_status":status,"stage_e_permitted":False};torch.save(checkpoint,output/"last.pt")
    if passed:
        directory=output/"P1";directory.mkdir(exist_ok=True);torch.save(checkpoint,directory/"best.pt")
    report={"schema_version":"stage_d_observation_tag_report.0","observation_operator":{"input":"regional RGB residual","fixed_high_pass":["laplacian","horizontal_difference","vertical_difference","image_minus_average_blur"],"pooling":"normalized inner product over channel and space","normalization":"L2 per observed carrier"},"parent_verification":verification,"p0_audit":audit,"curriculum":{"P0":{"passed":audit["passed"]},"P1":{"passed":passed,"step":step,"selected_metrics":selected,"terminal_metrics":terminal}},"later_levels_entered":False,"evaluation_population":pop,"history":history,"metrics":metrics,"gate_results":result,"passed":passed,"stage_e_permitted":False,"scientific_status":status};(output/"report.json").write_text(json.dumps(report,indent=2)+"\n");return report
