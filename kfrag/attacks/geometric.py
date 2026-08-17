from __future__ import annotations
import torch
from torch.nn import functional as F
def _warp(x,theta): return F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),align_corners=False)
def rotate(x,degrees=5.):
    a=torch.tensor(degrees*torch.pi/180,device=x.device,dtype=x.dtype); c,s=torch.cos(a),torch.sin(a); t=torch.tensor([[c,-s,0],[s,c,0]],device=x.device,dtype=x.dtype).expand(x.shape[0],-1,-1); return _warp(x,t)
def translate(x,dx=.05,dy=.05):
    t=torch.tensor([[1,0,dx],[0,1,dy]],device=x.device,dtype=x.dtype).expand(x.shape[0],-1,-1); return _warp(x,t)
