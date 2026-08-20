"""Stage-C fixed-grid regionalization of the validated Stage-B V2 channel."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .natural_channel_v2 import NaturalChannelV2,analytical_carrier_bases

GRID_SIZE=4
REGION_COUNT=16
BITS_PER_REGION=8

def _native_cell_bases(size:int)->torch.Tensor:
    y=(torch.arange(size,dtype=torch.float32)+.5)[:,None];x=(torch.arange(size,dtype=torch.float32)+.5)[None,:]
    pairs=((15,15),(15,14),(14,15),(14,14),(13,15),(15,13),(13,14),(14,13))
    base=torch.stack([torch.cos(math.pi*k*x/size)*torch.cos(math.pi*l*y/size) for k,l in pairs]);base-=base.mean((-2,-1),keepdim=True)
    return base/base.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)

class RegionalCarrierRouter(nn.Module):
    def __init__(self, stage_b: NaturalChannelV2, image_size: int=64, window_floor: float=.7):
        super().__init__();self.image_size=image_size;self.cell_size=image_size//4
        self.carrier=stage_b.encoder.carrier
        base=_native_cell_bases(self.cell_size)[:,None]
        base=base-base.mean((-2,-1),keepdim=True);base=base/base.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        self.cell_bases=nn.Parameter(base[:,0].detach());w=torch.hann_window(self.cell_size,periodic=False)
        window=window_floor+(1-window_floor)*(w[:,None]*w[None,:]);self.register_buffer("window",window)

    def contributions(self,bits:torch.Tensor,active_mask:torch.Tensor|None=None)->torch.Tensor:
        if bits.ndim!=4 or tuple(bits.shape[1:])!=(4,4,8):raise ValueError("regional bits must have shape [B,4,4,8]")
        signed=bits.float().mul(2).sub(1)
        if active_mask is not None:signed=signed*active_mask.to(signed.dtype)[...,None]
        return signed[:,:,:, :,None,None]*self.cell_bases[None,None,None]*self.window[None,None,None,None]

    def forward(self,bits:torch.Tensor,active_mask:torch.Tensor|None=None)->torch.Tensor:
        patches=self.contributions(bits,active_mask) # B,R,C,K,H,W
        return patches.permute(0,3,1,4,2,5).reshape(len(bits),8,self.image_size,self.image_size)

class RegionalLearnedEncoder(nn.Module):
    def __init__(self,stage_b:NaturalChannelV2,image_size=64):
        super().__init__();self.router=RegionalCarrierRouter(stage_b,image_size);self.encoder=stage_b.encoder
    def forward(self,image,bits,amplitude,active_mask=None):
        carrier=self.router(bits,active_mask);out=self.encoder.forward_features(image,carrier,amplitude);out["regional_carrier_features"]=carrier;return out

class BlindRegionalDecoder(nn.Module):
    def __init__(self,stage_b:NaturalChannelV2,router:RegionalCarrierRouter,image_size=64):
        super().__init__();self.decoder=stage_b.decoder;self.image_size=image_size;self.cell_size=image_size//4
        self.router=router
        self.regional_features=nn.Sequential(nn.Conv2d(7,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU())
        self.regional_output=nn.Linear(32,8);nn.init.zeros_(self.regional_output.weight);nn.init.zeros_(self.regional_output.bias)
    def forward(self,questioned_image):
        if questioned_image.ndim!=4 or questioned_image.shape[1]!=3:raise ValueError("decoder accepts only questioned RGB image")
        b=len(questioned_image);x=questioned_image.reshape(b,3,4,self.cell_size,4,self.cell_size).permute(0,2,4,1,3,5).reshape(b*16,3,self.cell_size,self.cell_size)
        hp=self.decoder.highpass(x);base=self.router.cell_bases*self.router.window;base=base-base.mean((-2,-1),keepdim=True);base=base/base.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        hp_bases=self.decoder.highpass(base[:,None].expand(-1,3,-1,-1));hp_bases=hp_bases/hp_bases.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        matched=25*torch.einsum("bchw,kchw->bk",hp,hp_bases)/(hp.shape[1]*hp.shape[2]*hp.shape[3])
        zeros=x.new_zeros((len(x),self.decoder.output.in_features-8));matched_logits=self.decoder.output(torch.cat((zeros,matched),1))
        local=self.regional_output(self.regional_features(torch.cat((x,hp),1)).mean((-2,-1)))
        return (matched_logits+local).reshape(b,4,4,8)

class RegionalChannelV1(nn.Module):
    architecture_version="regional_channel_v1.0"
    def __init__(self,stage_b:NaturalChannelV2,image_size=64):
        super().__init__();self.encoder=RegionalLearnedEncoder(stage_b,image_size);self.decoder=BlindRegionalDecoder(stage_b,self.encoder.router,image_size)
    def forward(self,image,bits,amplitude,active_mask=None):
        out=self.encoder(image,bits,amplitude,active_mask);out["regional_logits"]=self.decoder(out["watermarked_image"]);return out
