"""Blind global affine prediction plus deterministic affine search grid."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class GlobalSynchronizationHead(nn.Module):
    def __init__(self,width=16,max_rotation_degrees=15):
        super().__init__(); self.max_rotation_degrees=max_rotation_degrees
        self.net=nn.Sequential(nn.Conv2d(3,width,5,2,2),nn.SiLU(),nn.Conv2d(width,width*2,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1)); self.out=nn.Linear(width*2,5)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self,questioned_image):
        p=self.out(self.net(questioned_image).flatten(1)); return {"log_scale":p[:,0],"translation":torch.tanh(p[:,1:3]),"rotation_degrees":torch.tanh(p[:,3])*self.max_rotation_degrees,"confidence":torch.sigmoid(p[:,4])}
    def align(self,image,prediction=None):
        p=self(image) if prediction is None else prediction; angle=p["rotation_degrees"]*torch.pi/180; scale=p["log_scale"].exp(); c,s=torch.cos(angle)/scale,torch.sin(angle)/scale
        theta=torch.stack((c,-s,p["translation"][:,0],s,c,p["translation"][:,1]),1).reshape(-1,2,3)
        return F.grid_sample(image,F.affine_grid(theta,image.shape,align_corners=False),align_corners=False),p
