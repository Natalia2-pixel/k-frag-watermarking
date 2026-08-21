"""Executed local-only P0->P1 neural feasibility pilot for distributed auth v2."""
from __future__ import annotations

import hashlib, inspect, json, random
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from kfrag.data import CocoImageDataset
from kfrag.diagnostics.stage_c_regional import SyntheticStageCDataset, _ssim
from kfrag.diagnostics.stage_d_tag_capacity import verify_12bit_parent, transplant_audit
from kfrag.models.stage_d_tag_capacity_v1 import StageDTagCapacityV1
from kfrag.training.distributed_auth_neural_v2 import (
    deterministic_scientific_key, evaluate_protocol_controls,
    fresh_distributed_packet_batch,
)
from kfrag.training.natural_channel_v2 import clip_gradients
from kfrag.training.regional_channel_v1 import load_stage_c_population, build_evaluation_population
from kfrag.training.stage_d_tag_capacity import capacity_metrics, capacity_gates, balanced_bit_loss


def _three_way_split(ids, train_count, selection_count, final_count, seed):
    if train_count + selection_count + final_count > len(ids): raise ValueError("three disjoint populations require more source images")
    order = list(ids); random.Random(seed).shuffle(order)
    return {"train": order[:train_count], "selection": order[train_count:train_count+selection_count], "final_test": order[train_count+selection_count:train_count+selection_count+final_count]}


def _parent_digest(model):
    digest = hashlib.sha256()
    for name, value in model.parent.state_dict().items(): digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _evaluate(model, images, bits, generator):
    model.eval()
    with torch.no_grad():
        out = model(images, torch.cat((bits, bits.new_zeros((*bits.shape[:-1], 24))), -1), .014, 8)
        logits = out["packet_logits"]
    m = capacity_metrics(logits, torch.cat((bits, bits.new_zeros((*bits.shape[:-1], 24))), -1), 8)
    hard = logits[..., :20].ge(0); shuffled = torch.roll(bits, 1, 0); spatial = torch.roll(bits, 1, 1)
    randomized = hard[..., 4:20].eq(bits[..., 4:20].bool()).float().mean()
    m["correct_minus_shuffled_margin"] = float(randomized - hard[..., 4:20].eq(shuffled[..., 4:20].bool()).float().mean())
    m["correct_minus_spatial_margin"] = m["overall_bit_accuracy"] - float(hard.eq(spatial.bool()).float().mean())
    original_logits = torch.cat((model.parent.index_head(images), model.parent.stage_c.decoder(images), model.tag_head(images)[..., :8]), -1)
    random_targets = torch.randint(0, 2, original_logits.shape, generator=generator).bool()
    m["original_randomized_field_accuracy"] = float(original_logits[..., 4:].ge(0).eq(random_targets[..., 4:]).float().mean())
    residual = out["residual"]; mse = residual.square().flatten(1).mean(1)
    # Region-local sensitivity: changed-region energy divided by all-region change energy.
    flipped = bits.clone(); flipped[:, 0, 0, 12] = 1 - flipped[:, 0, 0, 12]
    full_flipped = torch.cat((flipped, flipped.new_zeros((*flipped.shape[:-1], 24))), -1)
    with torch.no_grad(): changed = model(images, full_flipped, .014, 8)["residual"] - residual
    cells = changed.reshape(len(images), 3, 4, 16, 4, 16).square().mean((1,3,5)); target = cells[:,0,0]
    m.update({
        "cross_region_leakage": float(((cells.sum((1,2))-target)/cells.sum((1,2)).clamp_min(1e-12)).mean()),
        "parent_rs_logit_drift": float((logits[...,4:12]-model.parent(images, torch.cat((bits, bits.new_zeros((*bits.shape[:-1],24))),-1), .014, 4)["packet_logits"][...,4:12]).abs().mean()),
        "psnr": float((10*torch.log10(1/mse.clamp_min(1e-12))).mean()), "ssim": _ssim(images,out["watermarked_image"]),
        "residual_saturation_fraction": float(residual.abs().ge(.014*.999).float().mean()), "analytical_contribution": 0.0,
        "blind_decoder": list(inspect.signature(model.tag_head.forward).parameters)==["questioned_image"], "disjoint_images": True,
    })
    return m, logits


def _all_gates(metrics):
    gates = capacity_gates(metrics); gates["leakage"] = metrics["cross_region_leakage"] <= .10
    return gates


def _safe_checkpoint(model, optimizer, scheduler, config, verification, metrics, status):
    safe = {k:v for k,v in config.items() if "key" not in k.lower() and "payload" not in k.lower()}
    return {"schema_version":"stage_d_v2_20bit_neural.0", "architecture_version":model.architecture_version,
        "stage_d_12bit_parent_sha256":verification["sha256"], "configuration":safe, "model_state":model.state_dict(),
        "optimizer_state":optimizer.state_dict(), "scheduler_state":scheduler.state_dict(), "metrics":metrics,
        "scientific_status":status, "stage_e_permitted":False}


def run_stage_d_v2_20bit(config: Mapping[str, Any]):
    output = Path(config["output_directory"]); output.mkdir(parents=True, exist_ok=True)
    verification, parent = verify_12bit_parent(config)
    if not verification["passed"]:
        report={"scientific_status":"blocked_by_12_bit_parent","stage_e_permitted":False,"parent_verification":verification}; (output/"report.json").write_text(json.dumps(report,indent=2)+"\n"); return report
    seed=int(config.get("seed",2040)); random.seed(seed); torch.manual_seed(seed); generator=torch.Generator().manual_seed(seed+1); key=deterministic_scientific_key(seed)
    count=int(config.get("synthetic_image_count",0)); dataset=SyntheticStageCDataset(count) if count else CocoImageDataset(config["data_root"])
    ids=[dataset[i]["relative_path"] for i in range(len(dataset))]; split=_three_way_split(ids,int(config["train_images"]),int(config["selection_images"]),int(config["final_test_images"]),seed)
    prep=config["preprocessing"]; train=load_stage_c_population(dataset,split["train"],prep,64); selection=load_stage_c_population(dataset,split["selection"],prep,64); final=load_stage_c_population(dataset,split["final_test"],prep,64)
    model=StageDTagCapacityV1(parent); audit_bits,_=fresh_distributed_packet_batch(2,key,generator); audit44=torch.cat((audit_bits,audit_bits.new_zeros((2,4,4,24))),-1); audit=transplant_audit(model,selection[:2],audit44)
    for parameter in model.parent.parameters(): parameter.requires_grad=False
    parent_before=_parent_digest(model); parent_snapshot={k:v.detach().clone() for k,v in model.parent.state_dict().items()}
    trainable=[p for p in model.parameters() if p.requires_grad]; maximum=int(config.get("maximum_steps",500)); minimum=int(config.get("minimum_steps",50)); every=int(config.get("evaluate_every",25)); batch=int(config.get("batch_size",8))
    optimizer=torch.optim.AdamW(trainable,lr=float(config.get("learning_rate",3e-4)),weight_decay=float(config.get("weight_decay",1e-4))); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,maximum); ema=torch.ones(20); history=[]; best=None; selected=None; terminal=None; streak=0
    for step in range(1,maximum+1):
        image=train[torch.randint(len(train),(batch,),generator=generator)]; bits,_=fresh_distributed_packet_batch(batch,key,generator); packet44=torch.cat((bits,bits.new_zeros((*bits.shape[:-1],24))),-1); out=model(image,packet44,.014,8)
        balanced,ema,weights=balanced_bit_loss(out["packet_logits"],packet44,20,ema,float(config.get("balance_decay",.9)),float(config.get("maximum_bit_weight",3)))
        with torch.no_grad(): teacher=model.parent(image,packet44,.014,4)["packet_logits"]
        distill=F.mse_loss(out["packet_logits"][...,:12],teacher[...,:12]); residual=out["residual"]
        loss=balanced+float(config.get("distillation_weight",2))*distill+.2*F.l1_loss(out["watermarked_image"],image)+2*F.relu(residual.abs()/.014-.9).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); gb,ga=clip_gradients(trainable,1); optimizer.step(); scheduler.step()
        if step%every==0:
            eval_bits,_=fresh_distributed_packet_batch(len(selection),key,generator); metrics,_=_evaluate(model,selection,eval_bits,generator); gates=_all_gates(metrics); terminal=dict(metrics); history.append({"step":step,"loss":float(loss.detach()),"gradient_norm_before":gb,"gradient_norm_after":ga,"training_bit_weights":weights.tolist(),"gate_results":gates,**metrics})
            passed=all(gates.values()) and step>=minimum
            if passed and (selected is None or min(metrics["per_bit_accuracy"])>min(selected["per_bit_accuracy"])): best={k:v.detach().clone() for k,v in model.state_dict().items()}; selected=dict(metrics); selected["step"]=step
            streak=streak+1 if passed else 0
            if streak>=2: break
    if best is not None: model.load_state_dict(best)
    actual=int(config.get("final_evaluation_samples",32)); eval_images,pop=build_evaluation_population(final,actual); eval_bits,metadata=fresh_distributed_packet_batch(len(eval_images),key,generator); final_metrics,logits=_evaluate(model,eval_images,eval_bits,generator)
    second_bits,second_meta=fresh_distributed_packet_batch(len(eval_images),key,generator)
    with torch.no_grad(): second_logits=model(eval_images,torch.cat((second_bits,second_bits.new_zeros((*second_bits.shape[:-1],24))),-1),.014,8)["packet_logits"]
    protocol=evaluate_protocol_controls(logits,metadata,key,second_logits,second_meta); gates=_all_gates(final_metrics)
    frozen_equal=parent_before==_parent_digest(model) and all(torch.equal(v,model.parent.state_dict()[k]) for k,v in parent_snapshot.items()); parent_gradients_zero=all(p.grad is None or int(p.grad.count_nonzero())==0 for p in model.parent.parameters()); gates.update({"p0_exact":audit["passed"],"parent_frozen":frozen_equal,"parent_gradients_zero":parent_gradients_zero,"protocol_from_decoded":protocol["protocol_input"]=="thresholded_blind_decoder_logits"})
    passed=best is not None and all(gates.values()); status="passed_stage_d_v2_20bit_neural_feasibility" if passed else "blocked_by_stage_d_v2_20bit_neural_gate"; final_metrics.update({"gate_results":gates,"scientific_status":status})
    checkpoint=_safe_checkpoint(model,optimizer,scheduler,config,verification,final_metrics,status); torch.save(checkpoint,output/"last.pt");
    if passed: torch.save(checkpoint,output/"best.pt")
    report={"schema_version":"stage_d_v2_20bit_neural_report.0","parent_verification":verification,"p0_audit":audit,"packet_layout":{"index":[0,4],"rs":[4,12],"distributed_auth_share":[12,20]},"actual_packet_generator":"JointFragmentCode(RS(16,8)-encoded 64-bit global HMAC)","split":{"identifiers":split,"overlap":False},"selection_metrics":selected,"terminal_metrics":terminal,"history":history,"final_population":pop,"neural_metrics":final_metrics,"protocol_metrics":protocol,"parent_gradients_zero":parent_gradients_zero,"parent_updates_zero":frozen_equal,"gate_results":gates,"stage_d_v2_20bit_passed":passed,"best_checkpoint_saved":passed,"last_checkpoint_saved":True,"no_secret_or_expected_payload_serialized":True,"stage_e_permitted":False,"scientific_status":status}; (output/"report.json").write_text(json.dumps(report,indent=2)+"\n"); return report
