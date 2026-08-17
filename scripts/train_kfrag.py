"""Train K-FRAG without overwriting an existing experiment run."""
from __future__ import annotations
import argparse, json, platform, random, sys, time
from pathlib import Path
import torch, yaml
from kfrag.checkpoints.provenance import load_checkpoint, save_checkpoint, sha256_file
from kfrag.data.adapters import ImageFolderAdapter
from kfrag.data.manifests import create_manifests
from kfrag.training.trainer import train, write_history

def build_parser():
    p=argparse.ArgumentParser(description="Train and validate the K-FRAG system")
    p.add_argument("--config",required=True,type=Path); p.add_argument("--run-id")
    p.add_argument("--smoke",action="store_true",help="short deterministic integration check")
    p.add_argument("--dry-run",action="store_true",help="validate and print the plan without writing or training")
    p.add_argument("--device",help="PyTorch device, such as cpu or cuda:0"); p.add_argument("--seed",type=int)
    p.add_argument("--output-directory",type=Path,help="exact directory for this immutable run")
    p.add_argument("--resume",type=Path,help="checkpoint used to initialize this run")
    return p

def load_config(path):
    if not path.is_file(): raise FileNotFoundError(f"configuration does not exist: {path}")
    cfg=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg,dict): raise ValueError("configuration root must be a YAML mapping")
    for key in ("schema_version","experiment_name"):
        if not cfg.get(key): raise ValueError(f"configuration is missing required field: {key}")
    for key in ("steps","batch_size","image_size","max_images","width"):
        if key in cfg and (isinstance(cfg[key],bool) or int(cfg[key])<=0): raise ValueError(f"configuration field {key} must be a positive integer")
    for key in ("learning_rate","residual_alpha","fidelity_weight"):
        if key in cfg and float(cfg[key])<0: raise ValueError(f"configuration field {key} must be non-negative")
    return cfg

def resolve_device(value):
    try: device=torch.device(value)
    except (RuntimeError,ValueError) as exc: raise ValueError(f"invalid device: {value}") from exc
    if device.type not in {"cpu","cuda","mps"}: raise ValueError(f"invalid device: {value}")
    if device.type=="cuda" and (not torch.cuda.is_available() or (device.index is not None and device.index>=torch.cuda.device_count())): raise ValueError(f"requested device is unavailable: {value}")
    if device.type=="mps" and not (hasattr(torch.backends,"mps") and torch.backends.mps.is_available()): raise ValueError(f"requested device is unavailable: {value}")
    return device

def validate(cfg,args):
    device=resolve_device(args.device or str(cfg.get("device","cpu")))
    root=Path(cfg.get("data_root","data/raw/coco_val2017_100"))
    if not root.is_dir() and not args.smoke: raise FileNotFoundError(f"referenced dataset directory does not exist: {root}")
    manifest=create_manifests(root) if root.is_dir() else {"train":[],"validation":[],"test":[],"sha256":"synthetic"}
    paths=manifest["train"] or manifest["test"]
    if not paths and not args.smoke: raise RuntimeError(f"no images found under {root}")
    resume=None
    initialization=cfg.get("initialization",{})
    if isinstance(initialization,dict) and initialization.get("require_checkpoint"):
        required=Path(str(initialization.get("checkpoint","")))
        if not required.is_file(): raise FileNotFoundError(f"required checkpoint does not exist: {required}")
    if args.resume:
        if not args.resume.is_file(): raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
        resume=load_checkpoint(args.resume,architecture="KFragSystem-v1")
    run_id=args.run_id or time.strftime("%Y%m%dT%H%M%S")
    output=args.output_directory or Path(cfg.get("output_root","outputs/kfrag"))/cfg["experiment_name"]/run_id
    if output.exists(): raise FileExistsError(f"immutable run already exists: {output}")
    return device,root,manifest,paths,resume,output

def write_artifacts(output,cfg,manifest,model,summary,stage,seed,args,plan):
    (output/"configuration.yaml").write_text(yaml.safe_dump(cfg,sort_keys=True),encoding="utf-8")
    (output/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (output/"environment.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"torch":torch.__version__},indent=2)+"\n",encoding="utf-8")
    write_history(output/"history.csv",summary.pop("history"))
    meta=dict(config=cfg,manifest_hash=manifest["sha256"],architecture="KFragSystem-v1",protocol_version=1,phase=stage,step=int(cfg["steps"]),seeds={"global":seed},residual_alpha=cfg.get("residual_alpha",.05),models={"kfrag":model},gates=summary["gates"],status=summary["scientific_status"])
    save_checkpoint(output/"best.pt",**meta); save_checkpoint(output/"last.pt",**meta)
    summary["checkpoint_sha256"]={name:sha256_file(output/name) for name in ("best.pt","last.pt")}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    values=summary["metrics"]["per_bit_accuracy"]; (output/"per_bit_metrics.csv").write_text("bit,accuracy\n"+"".join(f"{i},{v}\n" for i,v in enumerate(values)),encoding="utf-8")
    for name,header in (("per_region_metrics.csv","region,accuracy\n"),("attack_metrics.csv","attack,severity,reconstruction_success\n"),("reconstruction_by_survivors.csv","survivors,recovery_rate\n")): (output/name).write_text(header,encoding="utf-8")
    (output/"failure_cases.json").write_text("[]\n",encoding="utf-8"); (output/"evidence_maps").mkdir()
    note="Smoke execution succeeded; this is an integration check, not scientific gate success." if args.smoke else "Training execution completed; inspect scientific_status and gates."
    (output/"README.txt").write_text(note+"\nA failed gate blocks scientific progression.\n",encoding="utf-8")
    print(json.dumps({**plan,"smoke":args.smoke,"message":note,**summary["gates"]},indent=2))

def main(argv=None):
    args=build_parser().parse_args(argv); cfg=load_config(args.config)
    if args.device is not None: cfg["device"]=args.device
    if args.seed is not None: cfg["seed"]=args.seed
    if args.smoke: cfg.update({"steps":1,"batch_size":1,"max_images":2,"image_size":min(int(cfg.get("image_size",64)),64)})
    device,root,manifest,paths,resume,output=validate(cfg,args)
    stage=cfg.get("phase",cfg.get("stage","natural_image_communication")); seed=int(cfg.get("seed",2026))
    plan={"output_directory":output.as_posix(),"scientific_stage":stage,"device":str(device),"seed":seed}
    if args.dry_run: print(json.dumps({"dry_run":True,**plan},indent=2)); return 0
    random.seed(seed); torch.manual_seed(seed)
    selected=paths[:int(cfg.get("max_images",8))]
    if selected:
        dataset=ImageFolderAdapter(root,selected,int(cfg.get("image_size",256))); images=torch.stack([dataset[i]["image"] for i in range(len(dataset))])
    else: images=torch.rand(2,3,int(cfg["image_size"]),int(cfg["image_size"]))
    output.mkdir(parents=True)
    model,summary=train(cfg,images,manifest["sha256"],resume_state=resume)
    write_artifacts(output,cfg,manifest,model,summary,stage,seed,args,plan); return 0

if __name__=="__main__": raise SystemExit(main())
