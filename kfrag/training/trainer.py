"""Streaming, active-symbol-only natural-image pilot training."""
from __future__ import annotations
import csv, random, time
from pathlib import Path
import torch
from torch.nn import functional as F
from kfrag.models.kfrag_system import KFragSystem
from kfrag.models.regional_carrier import RegionalCarrierBank, ACTIVE_START, ACTIVE_BITS
from kfrag.evaluation.metrics import communication_metrics, fidelity_metrics

SYMBOL_CHANNELS = tuple(range(ACTIVE_START, ACTIVE_START + ACTIVE_BITS))

def active_symbol_mask(packet_bits=44, grid_size=4):
    mask=torch.zeros(1,packet_bits,grid_size,grid_size,dtype=torch.bool); mask[:,SYMBOL_CHANNELS]=True; return mask

def configured_active_channels(config):
    active=tuple(int(x) for x in config.get("active_channels",SYMBOL_CHANNELS))
    if active != SYMBOL_CHANNELS: raise ValueError(f"natural_image_communication requires active_channels={list(SYMBOL_CHANNELS)}, got {list(active)}")
    return active

def select_active(tensor,active=SYMBOL_CHANNELS): return tensor[:,list(active)]

def fresh_payloads(batch,device="cpu",generator=None):
    payload=torch.zeros(batch,44,4,4,device=device)
    payload[:,list(SYMBOL_CHANNELS)]=torch.randint(0,2,(batch,ACTIVE_BITS,4,4),generator=generator).float().to(device)
    return payload

def create_run_directory(root,experiment,run_id):
    path=Path(root)/experiment/run_id
    if path.exists(): raise FileExistsError(f"immutable run already exists: {path}")
    path.mkdir(parents=True); return path

def _images(batch,device): return batch["image"].to(device,non_blocking=device.type=="cuda")
def _average(rows):
    result={}
    for key in rows[0]:
        values=[row[key] for row in rows]
        if values[0] is None: result[key]=None
        elif isinstance(values[0],list): result[key]=[sum(items)/len(items) for items in zip(*values)]
        else: result[key]=sum(values)/len(values)
    return result

def _active_metrics(logits,targets): return communication_metrics(select_active(logits),select_active(targets),symbol_slice=slice(0,ACTIVE_BITS))

def validate_streaming(model,loader,device,seed,max_batches=None,residual_alpha=.05):
    model.eval(); fixed_gen=torch.Generator().manual_seed(seed+10_000); fresh_gen=torch.Generator().manual_seed(seed+20_000)
    fixed_rows=[]; fresh_rows=[]; shuffled_rows=[]; original_rows=[]; fidelity_rows=[]; sensitivity=[]
    with torch.no_grad():
        for batch_index,batch in enumerate(loader):
            if max_batches is not None and batch_index>=max_batches: break
            image=_images(batch,device); fixed=fresh_payloads(len(image),device,fixed_gen); out=model(image,fixed)
            fixed_rows.append(_active_metrics(out["packet_logits"],fixed)); shuffled_rows.append(_active_metrics(out["packet_logits"],torch.roll(fixed,1,0)))
            fidelity_rows.append(fidelity_metrics(image,out["watermarked_image"],out["residual"],residual_alpha))
            fresh=fresh_payloads(len(image),device,fresh_gen); fresh_out=model(image,fresh); fresh_rows.append(_active_metrics(fresh_out["packet_logits"],fresh))
            original_rows.append(_active_metrics(model.decode(image)["packet_logits"],fixed)); sensitivity.append(float((out["residual"]-model.encode(image,1-fixed)["residual"]).abs().mean().detach()))
    if not fixed_rows: raise RuntimeError("validation loader produced no images")
    return {"fixed_held_out_payloads":_average(fixed_rows),"fresh_post_training_payloads":_average(fresh_rows),"shuffled_targets":_average(shuffled_rows),"original_unwatermarked_images":_average(original_rows),"payload_sensitivity":sum(sensitivity)/len(sensitivity)},_average(fidelity_rows)

def _set_trainable(module,value):
    for parameter in module.parameters(): parameter.requires_grad_(value)

def train(config,train_loader,validation_loader,manifest_hash="synthetic",resume_state=None):
    active=configured_active_channels(config); seed=int(config.get("seed",2026)); random.seed(seed); torch.manual_seed(seed); device=torch.device(config.get("device","cpu")); payload_gen=torch.Generator().manual_seed(seed+1); alpha=float(config.get("residual_alpha",.05))
    model=KFragSystem(width=int(config.get("width",16)),residual_alpha=alpha,preprocessing=config.get("preprocessing","rgb_plus_high_pass")).to(device)
    if resume_state is not None: model.load_state_dict(resume_state["model_states"]["kfrag"],strict=True)
    phases=config.get("phases",{}); decoder_steps=int(phases.get("decoder_warmup_steps",0)); encoder_steps=int(phases.get("encoder_warmup_steps",0)); total=int(config.get("steps",2))
    if decoder_steps+encoder_steps>=total: raise ValueError("warm-up phases must leave at least one joint-training step")
    carrier=RegionalCarrierBank(image_size=int(config.get("image_size",256)),alpha=alpha,mode="fixed").to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config.get("learning_rate",1e-3))); history=[]; start=time.perf_counter(); iterator=iter(train_loader)
    interval=int(config.get("validation_interval",max(1,total))); patience=int(config.get("early_stopping_patience",0)); stale=0; best_pair=(-1.,-1.); validation=fidelity=None; stopped_early=False
    for step in range(1,total+1):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(train_loader); batch=next(iterator)
        image=_images(batch,device); target=fresh_payloads(len(image),device,payload_gen); model.train(); optimizer.zero_grad(set_to_none=True)
        if step<=decoder_steps:
            phase="decoder_warmup"; _set_trainable(model.encoder,False); _set_trainable(model.decoder,True); _set_trainable(model.synchronizer,True); scale=step/max(1,decoder_steps)
            residual=carrier(target)*scale; marked=(image+residual).clamp(0,1); out={**model.decode(marked),"residual":residual,"watermarked_image":marked}
        else:
            phase="encoder_warmup" if step<=decoder_steps+encoder_steps else "joint_training"; _set_trainable(model.encoder,True); _set_trainable(model.decoder,phase=="joint_training"); _set_trainable(model.synchronizer,phase=="joint_training")
            scale=min(1.,step/max(1,int(config.get("residual_warmup_steps",decoder_steps+encoder_steps)))); encoded=model.encode(image,target); residual=encoded["residual"]*scale; marked=(image+residual).clamp(0,1); out={**encoded,**model.decode(marked),"residual":residual,"watermarked_image":marked}
        packet=F.binary_cross_entropy_with_logits(select_active(out["packet_logits"],active),select_active(target,active)); energy=out["residual"].square().mean(); saturation=F.relu(out["residual"].abs()/alpha-float(config.get("saturation_threshold",.9))).square().mean(); fidelity_loss=(out["watermarked_image"]-image).square().mean()
        loss=packet+float(config.get("fidelity_weight",.1))*fidelity_loss+float(config.get("residual_energy_weight",.1))*energy+float(config.get("saturation_weight",.1))*saturation
        loss.backward(); unclipped=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float(config.get("gradient_clip_norm",1.0)))); optimizer.step()
        history.append({"step":step,"phase":phase,"residual_scale":scale,"loss":float(loss.detach()),"packet_loss":float(packet.detach()),"residual_energy":float(energy.detach()),"saturation_penalty":float(saturation.detach()),"gradient_norm_before_clip":unclipped})
        if step%interval==0 or step==total:
            validation,fidelity=validate_streaming(model,validation_loader,device,seed,residual_alpha=alpha); fresh=validation["fresh_post_training_payloads"]; margin=fresh["bit_accuracy"]-validation["shuffled_targets"]["bit_accuracy"]; pair=(fresh["bit_accuracy"],margin)
            improved=pair[0]>best_pair[0]+1e-6 or pair[1]>best_pair[1]+1e-6
            if improved: best_pair=(max(best_pair[0],pair[0]),max(best_pair[1],pair[1])); stale=0
            else: stale+=1
            if patience and stale>=patience: stopped_early=True; break
    metrics=dict(validation["fresh_post_training_payloads"]); metrics["active_bit_accuracy"]=metrics.pop("bit_accuracy"); metrics["per_active_bit_accuracy"]=metrics.pop("per_bit_accuracy"); metrics["shuffled_margin"]=metrics["active_bit_accuracy"]-validation["shuffled_targets"]["bit_accuracy"]; metrics["payload_sensitivity"]=validation["payload_sensitivity"]; metrics["saturation_fraction"]=fidelity["saturation_fraction"]
    thresholds=config.get("gates",{}); gates={"fresh_active_bit_accuracy":metrics["active_bit_accuracy"]>=float(thresholds.get("active_bit_accuracy",.995)),"fresh_exact_symbol_accuracy":metrics["exact_symbol_accuracy"]>=float(thresholds.get("exact_symbol_accuracy",.96)),"shuffled_margin":metrics["shuffled_margin"]>=float(thresholds.get("shuffled_margin",.45)),"payload_sensitivity":metrics["payload_sensitivity"]>0}; gates["passed"]=all(gates.values())
    return model,{"metrics":metrics,"validation_controls":validation,"fidelity":fidelity,"gates":gates,"history":history,"stopped_early":stopped_early,"completed_steps":len(history),"elapsed_seconds":time.perf_counter()-start,"scientific_status":"passed" if gates["passed"] else "blocked_by_prerequisite"}

def write_history(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
