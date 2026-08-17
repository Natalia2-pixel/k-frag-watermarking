from __future__ import annotations
import torch
from torch.nn import functional as F
def jpeg(x,quality=75):
    levels=max(8,int(256*quality/100)); return ((x*levels).round()/levels).clamp(0,1)
def resize(x,scale=.75):
    size=x.shape[-2:]; small=F.interpolate(x,scale_factor=scale,mode="bilinear",align_corners=False); return F.interpolate(small,size,mode="bilinear",align_corners=False)
def blur(x,sigma=1.):
    k=max(3,int(2*round(2*sigma)+1)); return F.avg_pool2d(x,k,1,k//2)
def noise(x,std=.01,generator=None): return (x+torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)*std).clamp(0,1)
def color(x,brightness=0.,contrast=1.): return ((x-.5)*contrast+.5+brightness).clamp(0,1)
