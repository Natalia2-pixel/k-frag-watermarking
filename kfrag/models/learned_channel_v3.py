"""V3 residual regional-symbol channel, isolated from failed V1/V2 baselines."""
from __future__ import annotations
import math
import torch
from torch import nn
from .regional_carrier import _orthogonal_carriers

class SpatialSymbolProjector(nn.Module):
    def __init__(self,image_size: int=64) -> None:
        super().__init__(); self.image_size=image_size; self.region_size=image_size//4
        self.carriers=nn.Parameter(_orthogonal_carriers(self.region_size)*self.region_size)
    def normalized(self):
        flat=self.carriers.flatten(1); flat=flat-flat.mean(1,keepdim=True)
        q,_=torch.linalg.qr(flat.T,mode="reduced")
        return (q.T*math.sqrt(flat.shape[1])).reshape_as(self.carriers)
    def forward(self,symbol_bits):
        if symbol_bits.ndim!=4 or tuple(symbol_bits.shape[1:])!=(8,4,4): raise ValueError("symbol_bits must have shape [B,8,4,4]")
        patches=torch.einsum("bkrc,khw->brchw",symbol_bits.float()*2-1,self.normalized())/math.sqrt(8)
        return patches.permute(0,1,3,2,4).reshape(len(symbol_bits),1,self.image_size,self.image_size)

class ImageConditionedResidualEncoder(nn.Module):
    def __init__(self,image_size=64,alpha=.018,width=16):
        super().__init__(); self.alpha=float(alpha); self.projector=SpatialSymbolProjector(image_size)
        self.refine=nn.Sequential(nn.Conv2d(4,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,3,1))
        nn.init.zeros_(self.refine[-1].weight); nn.init.zeros_(self.refine[-1].bias)
    def scheduled_alpha(self,progress=1.0): return self.alpha*(.25+.75*min(1.,max(0.,float(progress))))
    def forward(self,image,symbol_bits,progress=1.0):
        carrier=self.projector(symbol_bits); refinement=self.refine(torch.cat((image,carrier),1))
        residual=self.scheduled_alpha(progress)*torch.tanh(carrier.expand(-1,3,-1,-1)+.1*refinement)
        return image+residual,residual

class BlindSymbolDecoder(nn.Module):
    def __init__(self,carriers,image_size=64):
        super().__init__(); self.image_size=image_size; self.region_size=image_size//4
        self.register_buffer("carriers",carriers.detach().clone()); self.log_gain=nn.Parameter(torch.tensor(0.)); self.bias=nn.Parameter(torch.zeros(8))
    def forward(self,questioned_image):
        if questioned_image.ndim!=4 or tuple(questioned_image.shape[1:])!=(3,self.image_size,self.image_size): raise ValueError("questioned_image has invalid shape")
        grey=questioned_image.mean(1)
        patches=grey.reshape(len(grey),4,self.region_size,4,self.region_size).permute(0,1,3,2,4)
        patches=patches-patches.mean((-2,-1),keepdim=True)
        logits=torch.einsum("brchw,khw->bkrc",patches,self.carriers)/self.region_size
        return logits*self.log_gain.exp()+self.bias[None,:,None,None]

class ResidualSymbolSystem(nn.Module):
    def __init__(self,image_size=64,alpha=.018):
        super().__init__(); self.encoder=ImageConditionedResidualEncoder(image_size,alpha); self.decoder=BlindSymbolDecoder(self.encoder.projector.normalized(),image_size)
    def sync_carriers(self): self.decoder.carriers.copy_(self.encoder.projector.normalized().detach())
    def forward(self,image,symbol_bits,progress=1.0):
        questioned,residual=self.encoder(image,symbol_bits,progress)
        return {"watermarked_image":questioned,"residual":residual,"symbol_logits":self.decoder(questioned)}
