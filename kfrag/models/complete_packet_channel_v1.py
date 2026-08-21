"""Stage-D shared regional channel for complete 44-bit protocol packets."""
from __future__ import annotations
import math,torch
from torch import nn
from torch.nn import functional as F
from .regional_channel_v1 import RegionalChannelV1

PACKET_BITS=44

def packet_cell_bases(size:int=16)->torch.Tensor:
    y=(torch.arange(size,dtype=torch.float32)+.5)[:,None];x=(torch.arange(size,dtype=torch.float32)+.5)[None,:]
    pairs=sorted(((k,l) for k in range(4,size) for l in range(4,size)),key=lambda p:(-(p[0]+p[1]),-min(p),-max(p)))[:PACKET_BITS]
    bases=torch.stack([torch.cos(math.pi*k*x/size)*torch.cos(math.pi*l*y/size) for k,l in pairs]);bases-=bases.mean((-2,-1),keepdim=True)
    return bases/bases.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)

class CompletePacketRouter(nn.Module):
    def __init__(self,stage_c:RegionalChannelV1,image_size=64):
        super().__init__();self.image_size=image_size;self.cell_size=image_size//4;self.encoder=stage_c.encoder.encoder
        self.bases=nn.Parameter(packet_cell_bases(self.cell_size));self.register_buffer("channel_assignment",F.one_hot(torch.arange(44)%8,8).float())
        w=torch.hann_window(self.cell_size,periodic=False);self.register_buffer("window",.7+.3*w[:,None]*w[None,:])
    def contributions(self,bits,active_bits=44):
        if bits.ndim!=4 or tuple(bits.shape[1:])!=(4,4,44):raise ValueError("packet bits must have shape [B,4,4,44]")
        signed=bits.float().mul(2).sub(1);signed[...,active_bits:]=0
        return signed[...,None,None,None]*self.bases[None,None,None,:,None]*self.window[None,None,None,None,None]*self.channel_assignment[None,None,None,:,:,None,None]
    def forward(self,bits,active_bits=44):
        routed=self.contributions(bits,active_bits).sum(3)*math.sqrt(8/max(1,active_bits));return routed.permute(0,3,1,4,2,5).reshape(len(bits),8,self.image_size,self.image_size)

class CompletePacketEncoder(nn.Module):
    def __init__(self,stage_c,image_size=64):super().__init__();self.router=CompletePacketRouter(stage_c,image_size);self.encoder=stage_c.encoder.encoder
    def forward(self,image,bits,amplitude,active_bits=44):return self.encoder.forward_features(image,self.router(bits,active_bits),amplitude)

class BlindCompletePacketDecoder(nn.Module):
    def __init__(self,stage_c:RegionalChannelV1,router:CompletePacketRouter,image_size=64):
        super().__init__();self.image_size=image_size;self.cell_size=image_size//4;self.highpass=stage_c.decoder.decoder.highpass;self.router=router;self.carrier_skip=stage_c.encoder.encoder.carrier_skip
        self.matched_output=nn.Linear(44,44);nn.init.eye_(self.matched_output.weight);nn.init.zeros_(self.matched_output.bias)
        self.features=nn.Sequential(nn.Conv2d(7,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.SiLU())
        self.output=nn.Linear(64,44);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,questioned_image):
        if questioned_image.ndim!=4 or tuple(questioned_image.shape[1:])!=(3,self.image_size,self.image_size):raise ValueError("decoder accepts only questioned RGB image")
        b=len(questioned_image);x=questioned_image.reshape(b,3,4,self.cell_size,4,self.cell_size).permute(0,2,4,1,3,5).reshape(b*16,3,self.cell_size,self.cell_size);hp=self.highpass(x)
        weights=self.carrier_skip.weight[:,:,0,0];rgb=weights[:,torch.arange(44)%8].T[:,:,None,None]*self.router.bases[:,None]*self.router.window[None,None]
        hp_bases=self.highpass(rgb);hp_bases=hp_bases/hp_bases.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        matched=12*torch.einsum("bchw,kchw->bk",hp,hp_bases)/(hp.shape[1]*hp.shape[2]*hp.shape[3])
        logits=self.matched_output(matched)+self.output(self.features(torch.cat((x,hp),1)).mean((-2,-1)))
        return logits.reshape(b,4,4,44)

class CompletePacketChannelV1(nn.Module):
    architecture_version="complete_packet_channel_v1.0"
    def __init__(self,stage_c:RegionalChannelV1,image_size=64):
        super().__init__();self.encoder=CompletePacketEncoder(stage_c,image_size);self.decoder=BlindCompletePacketDecoder(stage_c,self.encoder.router,image_size)
    def forward(self,image,bits,amplitude,active_bits=44):
        if image.ndim!=4 or tuple(image.shape[1:])!=(3,self.encoder.router.image_size,self.encoder.router.image_size) or len(image)!=len(bits):raise ValueError("image must be [B,3,64,64] with matching packet batch")
        out=self.encoder(image,bits,amplitude,active_bits);out["packet_logits"]=self.decoder(out["watermarked_image"]);return out
