"""Questioned-image-only interference suppression."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class BlindNormalizer(nn.Module):
    MODES=("raw_rgb","high_pass","local_mean_removed","region_standardized","rgb_plus_high_pass")
    def __init__(self, mode="rgb_plus_high_pass", region_size=64, eps=1e-5):
        super().__init__()
        if mode not in self.MODES: raise ValueError(f"mode must be one of {self.MODES}")
        self.mode,self.region_size,self.eps=mode,region_size,eps
    @property
    def output_channels(self): return 6 if self.mode=="rgb_plus_high_pass" else 3
    def forward(self, questioned_image):
        if not isinstance(questioned_image,torch.Tensor) or questioned_image.ndim!=4 or questioned_image.shape[1]!=3:
            raise ValueError("questioned_image must have shape [B,3,H,W]")
        x=questioned_image
        mean=F.avg_pool2d(x,5,1,2); hp=x-mean
        if self.mode=="raw_rgb": return x
        if self.mode in ("high_pass","local_mean_removed"): return hp
        if self.mode=="rgb_plus_high_pass": return torch.cat((x,hp),1)
        h,w=x.shape[-2:]; pooled=F.adaptive_avg_pool2d(x,(4,4)); sq=F.adaptive_avg_pool2d(x*x,(4,4))
        mu=F.interpolate(pooled,(h,w),mode="nearest"); std=F.interpolate((sq-pooled*pooled).clamp_min(0).sqrt(),(h,w),mode="nearest")
        return (x-mu)/(std+self.eps)
