from __future__ import annotations
from pathlib import Path
from torch.utils.data import Dataset
from .manifests import stable_identifiers
class ImageFolderAdapter(Dataset):
    def __init__(self,root,identifiers=None,image_size=256):
        self.root=Path(root); self.identifiers=list(identifiers if identifiers is not None else stable_identifiers(root)); self.image_size=image_size
        if not self.root.is_dir(): raise FileNotFoundError(f"dataset root does not exist: {self.root}")
        if not self.identifiers: raise RuntimeError(f"no images selected under dataset root: {self.root}")
        missing=[item for item in self.identifiers if not (self.root/item).is_file()]
        if missing: raise FileNotFoundError(f"manifest image does not exist under dataset root: {missing[0]}")
    def __len__(self): return len(self.identifiers)
    def __getitem__(self,index):
        from PIL import Image
        import numpy as np, torch
        identifier=self.identifiers[index]; image=Image.open(self.root/identifier).convert("RGB").resize((self.image_size,self.image_size),Image.Resampling.BICUBIC)
        return {"image":torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255,"relative_id":identifier}
class CocoAdapter(ImageFolderAdapter): pass
class ImageNetAdapter(ImageFolderAdapter): pass
class OpenImagesAdapter(ImageFolderAdapter): pass
class DIV2KAdapter(ImageFolderAdapter): pass
class FaceHQAdapter(ImageFolderAdapter): pass
