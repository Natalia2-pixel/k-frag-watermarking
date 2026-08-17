from __future__ import annotations
from pathlib import Path
from torch.utils.data import Dataset
from .manifests import stable_identifiers
class ImageFolderAdapter(Dataset):
    def __init__(self,root,identifiers=None,image_size=256): self.root=Path(root); self.identifiers=list(identifiers or stable_identifiers(root)); self.image_size=image_size
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
