"""Controlled fresh-payload training with immutable run artifacts."""
from __future__ import annotations
import csv,json,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
from kfrag.models.kfrag_system import KFragSystem
from kfrag.evaluation.metrics import communication_metrics,fidelity_metrics

def fresh_payloads(batch,device="cpu",generator=None): return torch.randint(0,2,(batch,44,4,4),generator=generator,device=device).float()
def create_run_directory(root,experiment,run_id):
    path=Path(root)/experiment/run_id
    if path.exists(): raise FileExistsError(f"immutable run already exists: {path}")
    path.mkdir(parents=True); return path
def train(config,images,manifest_hash="synthetic",resume_state=None):
    seed=int(config.get("seed",2026)); random.seed(seed); torch.manual_seed(seed); device=torch.device(config.get("device","cpu")); gen=torch.Generator().manual_seed(seed+1)
    model=KFragSystem(width=int(config.get("width",16)),residual_alpha=float(config.get("residual_alpha",.05)),preprocessing=config.get("preprocessing","rgb_plus_high_pass")).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model_states"]["kfrag"], strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config.get("learning_rate",1e-3))); history=[]; start=time.perf_counter(); images=images.to(device)
    for step in range(1,int(config.get("steps",2))+1):
        indices=torch.randint(len(images),(int(config.get("batch_size",2)),),generator=gen); image=images[indices].to(device); target=fresh_payloads(len(indices),device,gen)
        optimizer.zero_grad(set_to_none=True); out=model(image,target); packet=F.binary_cross_entropy_with_logits(out["packet_logits"],target); fidelity=(out["watermarked_image"]-image).square().mean(); loss=packet+float(config.get("fidelity_weight",.1))*fidelity; loss.backward(); grad=sum(float(p.grad.detach().square().sum()) for p in model.parameters() if p.grad is not None)**.5; optimizer.step()
        # Scalar copies are for history only. Detaching here leaves the tensors used by
        # backward and the optimizer untouched.
        history.append({"step":step,"loss":float(loss.detach()),"packet_loss":float(packet.detach()),"gradient_norm":grad})
    model.eval(); target=fresh_payloads(min(4,len(images)),device,gen); image=images[:len(target)].to(device)
    with torch.no_grad(): out=model(image,target); shuffled=torch.roll(target,1,0); metrics=communication_metrics(out["packet_logits"],target); control=communication_metrics(out["packet_logits"],shuffled); fidelity=fidelity_metrics(image,out["watermarked_image"],out["residual"]); alt=model.encode(image,1-target)
    metrics["shuffled_margin"]=metrics["bit_accuracy"]-control["bit_accuracy"]; metrics["payload_sensitivity"]=float((out["residual"]-alt["residual"]).abs().mean().detach())
    gates={"active_bit_accuracy":metrics["bit_accuracy"]>=.995,"exact_symbol_accuracy":metrics["exact_symbol_accuracy"]>=.96,"shuffled_margin":metrics["shuffled_margin"]>=.45,"payload_sensitivity":metrics["payload_sensitivity"]>0}; gates["passed"]=all(gates.values())
    return model,{"metrics":metrics,"fidelity":fidelity,"gates":gates,"history":history,"elapsed_seconds":time.perf_counter()-start,"scientific_status":"passed" if gates["passed"] else "blocked_by_prerequisite"}
def write_history(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
