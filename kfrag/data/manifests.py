"""Deterministic, portable image split manifests."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
def stable_identifiers(root):
    root=Path(root); return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
def manifest_hash(manifest):
    portable={key:manifest[key] for key in sorted(manifest) if key!="sha256"}
    return hashlib.sha256(json.dumps(portable,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def create_manifests(root,seed=2026,ratios=(.8,.1,.1)):
    ids=stable_identifiers(root); random.Random(seed).shuffle(ids); a=int(len(ids)*ratios[0]); b=a+int(len(ids)*ratios[1])
    result={"train":sorted(ids[:a]),"validation":sorted(ids[a:b]),"test":sorted(ids[b:])}
    result["sha256"]=manifest_hash(result); assert_disjoint(result); return result
def assert_disjoint(manifest):
    sets=[set(manifest.get(k,[])) for k in ("train","validation","test")]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise ValueError("dataset manifest splits overlap")

def load_identifiers(path,split):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f"{split} manifest does not exist: {path}")
    text=path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml",".yml"}:
        import yaml
        value=yaml.safe_load(text)
    else: value=json.loads(text)
    if isinstance(value,dict): value=value.get(split,value.get("images"))
    if not isinstance(value,list) or not all(isinstance(item,str) for item in value):
        raise ValueError(f"{split} manifest must contain a list of image identifiers")
    identifiers=[Path(item).as_posix() for item in value]
    for identifier in identifiers:
        candidate=Path(identifier)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{split} manifest identifiers must be relative: {identifier}")
    if len(identifiers)!=len(set(identifiers)): raise ValueError(f"{split} manifest contains duplicate identifiers")
    return identifiers

def report_manifest(adapter,split,identifiers,preprocessing):
    body={"dataset_adapter":adapter,"relative_image_identifiers":list(identifiers),"image_count":len(identifiers),"split_name":split,"deterministic_preprocessing":dict(preprocessing)}
    body["manifest_hash"]=manifest_hash(body)
    return body
