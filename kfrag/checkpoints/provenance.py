from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
import torch
SCHEMA_VERSION="1.0"
def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
def config_hash(config): return hashlib.sha256(json.dumps(config,sort_keys=True,default=str,separators=(",",":")).encode()).hexdigest()
def git_commit():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def save_checkpoint(path,*,config,manifest_hash,architecture,protocol_version,phase,step,seeds,residual_alpha,models,optimizers=None,schedulers=None,gates=None,status="implemented_unvalidated"):
    destination=Path(path)
    if destination.exists(): raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    state={"schema_version":SCHEMA_VERSION,"git_commit":git_commit(),"configuration":config,"configuration_hash":config_hash(config),"dataset_manifest_hash":manifest_hash,"architecture":architecture,"protocol_version":protocol_version,"completed_phase":phase,"completed_step":step,"random_seeds":seeds,"residual_alpha":residual_alpha,"model_states":{k:v.state_dict() for k,v in models.items()},"optimizer_states":{k:v.state_dict() for k,v in (optimizers or {}).items()},"scheduler_states":{k:v.state_dict() for k,v in (schedulers or {}).items()},"gate_results":gates or {},"scientific_status":status}
    destination.parent.mkdir(parents=True,exist_ok=True); torch.save(state,destination); return sha256_file(destination)
def load_checkpoint(path,*,configuration=None,manifest_hash=None,architecture=None):
    state=torch.load(path,map_location="cpu",weights_only=False)
    if state.get("schema_version")!=SCHEMA_VERSION: raise ValueError("incompatible checkpoint schema")
    if configuration is not None and state.get("configuration_hash")!=config_hash(configuration): raise ValueError("checkpoint configuration mismatch")
    if manifest_hash is not None and state.get("dataset_manifest_hash")!=manifest_hash: raise ValueError("checkpoint dataset mismatch")
    if architecture is not None and state.get("architecture")!=architecture: raise ValueError("checkpoint architecture mismatch")
    return state
