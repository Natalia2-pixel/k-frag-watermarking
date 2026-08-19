"""Streaming fresh-payload training with bounded batch residency."""
from __future__ import annotations
import csv, random, time
from pathlib import Path
import torch
from torch.nn import functional as F
from kfrag.models.kfrag_system import KFragSystem
from kfrag.evaluation.metrics import communication_metrics, fidelity_metrics

def fresh_payloads(batch,device="cpu",generator=None):
    # Generate on CPU from a dedicated generator, then transfer only this batch.
    return torch.randint(0,2,(batch,44,4,4),generator=generator).float().to(device)

def create_run_directory(root,experiment,run_id):
    path=Path(root)/experiment/run_id
    if path.exists(): raise FileExistsError(f"immutable run already exists: {path}")
    path.mkdir(parents=True); return path

def _images(batch,device):
    return batch["image"].to(device,non_blocking=device.type=="cuda")

def _average(rows):
    result={}
    for key in rows[0]:
        values=[row[key] for row in rows]
        if values[0] is None: result[key]=None
        elif isinstance(values[0],list): result[key]=[sum(items)/len(items) for items in zip(*values)]
        else: result[key]=sum(values)/len(values)
    return result

def validate_streaming(model,loader,device,seed,max_batches=None):
    """Evaluate four controls without retaining images or logits between batches."""
    model.eval(); fixed_gen=torch.Generator().manual_seed(seed+10_000); fresh_gen=torch.Generator().manual_seed(seed+20_000)
    fixed_rows=[]; fresh_rows=[]; shuffled_rows=[]; original_rows=[]; fidelity_rows=[]; sensitivity=[]
    with torch.no_grad():
        for batch_index,batch in enumerate(loader):
            if max_batches is not None and batch_index>=max_batches: break
            image=_images(batch,device); fixed=fresh_payloads(len(image),device,fixed_gen)
            out=model(image,fixed); fixed_rows.append(communication_metrics(out["packet_logits"],fixed))
            shuffled_rows.append(communication_metrics(out["packet_logits"],torch.roll(fixed,1,0)))
            fidelity_rows.append(fidelity_metrics(image,out["watermarked_image"],out["residual"]))
            fresh=fresh_payloads(len(image),device,fresh_gen); fresh_out=model(image,fresh)
            fresh_rows.append(communication_metrics(fresh_out["packet_logits"],fresh))
            original_rows.append(communication_metrics(model.decode(image)["packet_logits"],fixed))
            sensitivity.append(float((out["residual"]-model.encode(image,1-fixed)["residual"]).abs().mean().detach()))
            del image,fixed,out,fresh,fresh_out
    if not fixed_rows: raise RuntimeError("validation loader produced no images")
    return {"fixed_held_out_payloads":_average(fixed_rows),"fresh_post_training_payloads":_average(fresh_rows),"shuffled_targets":_average(shuffled_rows),"original_unwatermarked_images":_average(original_rows),"payload_sensitivity":sum(sensitivity)/len(sensitivity)},_average(fidelity_rows)

def train(config,train_loader,validation_loader,manifest_hash="synthetic",resume_state=None):
    seed=int(config.get("seed",2026)); random.seed(seed); torch.manual_seed(seed); device=torch.device(config.get("device","cpu")); payload_gen=torch.Generator().manual_seed(seed+1)
    model=KFragSystem(width=int(config.get("width",16)),residual_alpha=float(config.get("residual_alpha",.05)),preprocessing=config.get("preprocessing","rgb_plus_high_pass")).to(device)
    if resume_state is not None: model.load_state_dict(resume_state["model_states"]["kfrag"],strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config.get("learning_rate",1e-3))); history=[]; start=time.perf_counter(); iterator=iter(train_loader)
    validation_interval=int(config.get("validation_interval",max(1,int(config.get("steps",2)))))
    validation=None; fidelity=None
    for step in range(1,int(config.get("steps",2))+1):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(train_loader); batch=next(iterator)
        image=_images(batch,device); target=fresh_payloads(len(image),device,payload_gen); model.train(); optimizer.zero_grad(set_to_none=True); out=model(image,target)
        active=config.get("active_channels")
        logits=out["packet_logits"] if active is None else out["packet_logits"][:,[int(x) for x in active]]
        expected=target if active is None else target[:,[int(x) for x in active]]
        packet=F.binary_cross_entropy_with_logits(logits,expected); fidelity_loss=(out["watermarked_image"]-image).square().mean(); loss=packet+float(config.get("fidelity_weight",.1))*fidelity_loss
        loss.backward(); grad=sum(float(p.grad.detach().square().sum()) for p in model.parameters() if p.grad is not None)**.5; optimizer.step()
        history.append({"step":step,"loss":float(loss.detach()),"packet_loss":float(packet.detach()),"gradient_norm":grad})
        del batch,image,target,out,logits,expected,loss,packet,fidelity_loss
        if step%validation_interval==0 or step==int(config.get("steps",2)):
            validation,fidelity=validate_streaming(model,validation_loader,device,seed)
    metrics=dict(validation["fixed_held_out_payloads"]); shuffled=validation["shuffled_targets"]
    metrics["shuffled_margin"]=metrics["bit_accuracy"]-shuffled["bit_accuracy"]
    # This remains an explicit gate while avoiding a retained alternate-logit tensor.
    metrics["payload_sensitivity"]=validation["payload_sensitivity"]
    thresholds=config.get("gates",{})
    gates={"active_bit_accuracy":metrics["bit_accuracy"]>=float(thresholds.get("active_bit_accuracy",.995)),"exact_symbol_accuracy":metrics["exact_symbol_accuracy"]>=float(thresholds.get("exact_symbol_accuracy",.96)),"shuffled_margin":metrics["shuffled_margin"]>=float(thresholds.get("shuffled_margin",.45)),"payload_sensitivity":metrics["payload_sensitivity"]>0}
    gates["passed"]=all(gates.values())
    return model,{"metrics":metrics,"validation_controls":validation,"fidelity":fidelity,"gates":gates,"history":history,"elapsed_seconds":time.perf_counter()-start,"scientific_status":"passed" if gates["passed"] else "blocked_by_prerequisite"}

def write_history(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
