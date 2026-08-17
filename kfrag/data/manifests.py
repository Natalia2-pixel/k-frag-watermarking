from __future__ import annotations
import hashlib, json
from pathlib import Path
IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
def stable_identifiers(root):
    root=Path(root); return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def create_manifests(root,seed=2026,ratios=(.8,.1,.1)):
    import random
    ids=stable_identifiers(root); random.Random(seed).shuffle(ids); a=int(len(ids)*ratios[0]); b=a+int(len(ids)*ratios[1])
    result={"schema_version":"1.0","preprocessing":"RGB deterministic resize","train":sorted(ids[:a]),"validation":sorted(ids[a:b]),"test":sorted(ids[b:])}
    result["sha256"]=manifest_hash(result); assert_disjoint(result); return result
def assert_disjoint(manifest):
    sets=[set(manifest.get(k,[])) for k in ("train","validation","test")]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): raise ValueError("dataset manifest splits overlap")
