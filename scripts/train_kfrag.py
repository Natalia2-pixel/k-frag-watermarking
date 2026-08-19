"""Train K-FRAG from bounded, deterministic image mini-batches."""
from __future__ import annotations
import argparse, json, platform, random, sys, time
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader, Dataset
from kfrag.checkpoints.provenance import load_checkpoint, save_checkpoint, sha256_file
from kfrag.data.adapters import ImageFolderAdapter
from kfrag.data.manifests import assert_disjoint, create_manifests, load_identifiers, manifest_hash, report_manifest, stable_identifiers
from kfrag.training.trainer import train, write_history

def build_parser():
    p=argparse.ArgumentParser(description="Train and validate the K-FRAG system")
    p.add_argument("--config",required=True,type=Path); p.add_argument("--run-id")
    p.add_argument("--smoke",action="store_true",help="short deterministic integration check"); p.add_argument("--dry-run",action="store_true",help="validate and print the plan without writing or training")
    p.add_argument("--device",help="PyTorch device, such as cpu or cuda:0"); p.add_argument("--seed",type=int); p.add_argument("--output-directory",type=Path); p.add_argument("--resume",type=Path)
    p.add_argument("--train-data-root",type=Path); p.add_argument("--val-data-root",type=Path)
    p.add_argument("--train-manifest",type=Path); p.add_argument("--val-manifest",type=Path)
    p.add_argument("--batch-size",type=int); p.add_argument("--num-workers",type=int)
    p.add_argument("--max-train-images",type=int); p.add_argument("--max-val-images",type=int)
    return p

def load_config(path):
    if not path.is_file(): raise FileNotFoundError(f"configuration does not exist: {path}")
    cfg=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg,dict): raise ValueError("configuration root must be a YAML mapping")
    for key in ("schema_version","experiment_name"):
        if not cfg.get(key): raise ValueError(f"configuration is missing required field: {key}")
    for key in ("steps","batch_size","image_size","max_images","max_train_images","max_val_images","width"):
        if key in cfg and (isinstance(cfg[key],bool) or int(cfg[key])<=0): raise ValueError(f"configuration field {key} must be a positive integer")
    return cfg

def resolve_device(value):
    try: device=torch.device(value)
    except (RuntimeError,ValueError) as exc: raise ValueError(f"invalid device: {value}") from exc
    if device.type not in {"cpu","cuda","mps"}: raise ValueError(f"invalid device: {value}")
    if device.type=="cuda" and (not torch.cuda.is_available() or (device.index is not None and device.index>=torch.cuda.device_count())): raise ValueError(f"requested device is unavailable: {value}")
    if device.type=="mps" and not (hasattr(torch.backends,"mps") and torch.backends.mps.is_available()): raise ValueError(f"requested device is unavailable: {value}")
    return device

class SyntheticImages(Dataset):
    def __init__(self,count,size,seed): self.count=count; self.size=size; self.seed=seed
    def __len__(self): return self.count
    def __getitem__(self,index): return {"image":torch.rand(3,self.size,self.size,generator=torch.Generator().manual_seed(self.seed+index)),"relative_id":f"synthetic/{index:04d}.png"}

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32; random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError: pass

def make_loader(dataset,batch_size,workers,shuffle,device,seed):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=workers,pin_memory=device.type=="cuda",drop_last=False,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed),persistent_workers=workers>0)

def _limited(items,maximum): return list(items[:maximum] if maximum is not None else items)

def resolve_data(cfg,args,seed):
    if args.smoke:
        ids=[f"synthetic/{i:04d}.png" for i in range(2)]
        return SyntheticImages(2,int(cfg["image_size"]),seed),SyntheticImages(2,int(cfg["image_size"]),seed+100),ids,ids,"SyntheticImages"
    legacy=cfg.get("data_root")
    train_root=args.train_data_root or (Path(legacy) if legacy else None); val_root=args.val_data_root or train_root
    if train_root is None: raise ValueError("--train-data-root is required for real-dataset training")
    if val_root is None: raise ValueError("--val-data-root is required for real-dataset training")
    for name,root in (("training",train_root),("validation",val_root)):
        if not root.is_dir(): raise FileNotFoundError(f"{name} dataset root does not exist: {root}")
    if args.train_manifest: train_ids=load_identifiers(args.train_manifest,"train")
    else: train_ids=None
    if args.val_manifest: val_ids=load_identifiers(args.val_manifest,"validation")
    else: val_ids=None
    if train_ids is None and val_ids is None and train_root.resolve()==val_root.resolve():
        split=create_manifests(train_root,seed=seed,ratios=(.8,.2,0)); train_ids=split["train"]; val_ids=split["validation"]
    else:
        train_ids=train_ids if train_ids is not None else stable_identifiers(train_root)
        val_ids=val_ids if val_ids is not None else stable_identifiers(val_root)
    train_ids=_limited(train_ids,args.max_train_images if args.max_train_images is not None else cfg.get("max_train_images",cfg.get("max_images")))
    val_ids=_limited(val_ids,args.max_val_images if args.max_val_images is not None else cfg.get("max_val_images",cfg.get("max_images")))
    assert_disjoint({"train":train_ids,"validation":val_ids,"test":[]})
    size=int(cfg.get("image_size",256))
    return ImageFolderAdapter(train_root,train_ids,size),ImageFolderAdapter(val_root,val_ids,size),train_ids,val_ids,"ImageFolderAdapter"

def portable_config(cfg):
    blocked={"data_root","train_data_root","val_data_root","train_manifest","val_manifest"}
    clean={key:value for key,value in cfg.items() if key not in blocked}
    def scrub(value):
        if isinstance(value,dict): return {k:scrub(v) for k,v in value.items() if k not in blocked}
        if isinstance(value,list): return [scrub(v) for v in value]
        if isinstance(value,Path): return value.as_posix() if not value.is_absolute() else "<runtime-path>"
        if isinstance(value,str) and Path(value).is_absolute(): return "<runtime-path>"
        return value
    return scrub(clean)

def write_artifacts(output,cfg,manifests,model,summary,stage,seed,args,plan):
    safe_cfg=portable_config(cfg); (output/"configuration.yaml").write_text(yaml.safe_dump(safe_cfg,sort_keys=True),encoding="utf-8")
    (output/"run_manifest.json").write_text(json.dumps(manifests,indent=2)+"\n",encoding="utf-8")
    (output/"environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"torch":torch.__version__},indent=2)+"\n",encoding="utf-8")
    write_history(output/"history.csv",summary.pop("history")); combined_hash=manifest_hash({"splits":manifests})
    meta=dict(config=safe_cfg,manifest_hash=combined_hash,architecture="KFragSystem-v1",protocol_version=1,phase=stage,step=int(cfg["steps"]),seeds={"global":seed},residual_alpha=cfg.get("residual_alpha",.05),models={"kfrag":model},gates=summary["gates"],status=summary["scientific_status"])
    save_checkpoint(output/"best.pt",**meta); save_checkpoint(output/"last.pt",**meta); summary["checkpoint_sha256"]={name:sha256_file(output/name) for name in ("best.pt","last.pt")}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    values=summary["metrics"]["per_bit_accuracy"]; (output/"per_bit_metrics.csv").write_text("bit,accuracy\n"+"".join(f"{i},{v}\n" for i,v in enumerate(values)),encoding="utf-8")
    for name,header in (("per_region_metrics.csv","region,accuracy\n"),("attack_metrics.csv","attack,severity,reconstruction_success\n"),("reconstruction_by_survivors.csv","survivors,recovery_rate\n")): (output/name).write_text(header,encoding="utf-8")
    (output/"failure_cases.json").write_text("[]\n",encoding="utf-8"); (output/"evidence_maps").mkdir()
    note="Smoke execution succeeded; this is an integration check, not scientific gate success." if args.smoke else "Training execution completed; inspect scientific_status and gates."
    (output/"README.txt").write_text(note+"\nA failed gate blocks scientific progression.\n",encoding="utf-8"); print(json.dumps({**plan,"smoke":args.smoke,"message":note,**summary["gates"]},indent=2))

def main(argv=None):
    args=build_parser().parse_args(argv); cfg=load_config(args.config)
    if args.device is not None: cfg["device"]=args.device
    if args.seed is not None: cfg["seed"]=args.seed
    if args.batch_size is not None: cfg["batch_size"]=args.batch_size
    if args.num_workers is not None: cfg["num_workers"]=args.num_workers
    if args.smoke: cfg.update({"steps":1,"batch_size":1,"image_size":min(int(cfg.get("image_size",64)),64),"num_workers":0})
    for name in ("batch_size","num_workers"):
        if int(cfg.get(name,0 if name=="num_workers" else 1))<0 or (name=="batch_size" and int(cfg[name])==0): raise ValueError(f"{name} must be {'positive' if name=='batch_size' else 'non-negative'}")
    device=resolve_device(str(cfg.get("device","cpu"))); seed=int(cfg.get("seed",2026)); train_ds,val_ds,train_ids,val_ids,adapter=resolve_data(cfg,args,seed)
    if args.resume and not args.resume.is_file(): raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
    resume=load_checkpoint(args.resume,architecture="KFragSystem-v1") if args.resume else None
    run_id=args.run_id or time.strftime("%Y%m%dT%H%M%S"); output=args.output_directory or Path(cfg.get("output_root","outputs/kfrag"))/cfg["experiment_name"]/run_id
    if output.exists(): raise FileExistsError(f"immutable run already exists: {output}")
    stage=cfg.get("phase",cfg.get("stage","natural_image_communication")); plan={"output_directory":output.as_posix(),"scientific_stage":stage,"device":str(device),"seed":seed,"train_images":len(train_ds),"validation_images":len(val_ds)}
    if args.dry_run: print(json.dumps({"dry_run":True,**plan},indent=2)); return 0
    preprocessing={"color_mode":"RGB","resize":{"height":int(cfg["image_size"]),"width":int(cfg["image_size"]),"method":"bicubic"},"tensor_scale":"0..1"}
    manifests=[report_manifest(adapter,"train",train_ids,preprocessing),report_manifest(adapter,"validation",val_ids,preprocessing)]
    random.seed(seed); torch.manual_seed(seed); batch=int(cfg.get("batch_size",2)); workers=int(cfg.get("num_workers",0))
    train_loader=make_loader(train_ds,batch,workers,True,device,seed); val_loader=make_loader(val_ds,batch,workers,False,device,seed+1)
    output.mkdir(parents=True); model,summary=train(cfg,train_loader,val_loader,manifest_hash({"splits":manifests}),resume_state=resume); write_artifacts(output,cfg,manifests,model,summary,stage,seed,args,plan); return 0

if __name__=="__main__": raise SystemExit(main())
