"""Stage-D D1 repair with an exact Stage-C RS pathway transplant."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .regional_channel_v1 import RegionalChannelV1

INDEX_BITS=4
RS_SLICE=slice(4,12)

def _index_bases(size:int)->torch.Tensor:
    y=(torch.arange(size,dtype=torch.float32)+.5)[:,None];x=(torch.arange(size,dtype=torch.float32)+.5)[None,:]
    pairs=((12,15),(15,12),(12,14),(14,12))
    bases=torch.stack([torch.cos(math.pi*k*x/size)*torch.cos(math.pi*l*y/size) for k,l in pairs]);bases-=bases.mean((-2,-1),keepdim=True)
    return bases/bases.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)

class IndexCarrier(nn.Module):
    def __init__(self,stage_c:RegionalChannelV1):
        super().__init__();router=stage_c.encoder.router;self.image_size=router.image_size;self.cell_size=router.cell_size
        self.bases=nn.Parameter(_index_bases(self.cell_size));self.register_buffer("window",router.window.detach().clone())
        assignment=F.one_hot(torch.arange(4)+4,8).float();self.register_buffer("channel_assignment",assignment)
    def forward(self,index_bits,active_index_bits):
        signed=index_bits.float().mul(2).sub(1);signed[...,active_index_bits:]=0
        patches=signed[...,None,None,None]*self.bases[None,None,None,:,None]*self.window[None,None,None,None,None]*self.channel_assignment[None,None,None,:,:,None,None]
        patches=patches.sum(3).permute(0,3,1,4,2,5).reshape(len(index_bits),8,self.image_size,self.image_size)
        return patches*math.sqrt(8/max(1,8+active_index_bits))

class BlindIndexHead(nn.Module):
    def __init__(self,stage_c:RegionalChannelV1,index_carrier:IndexCarrier):
        super().__init__();self.stage_c=stage_c;self.carrier=index_carrier;self.image_size=index_carrier.image_size;self.cell_size=index_carrier.cell_size
        self.mix=nn.Linear(4,4);nn.init.eye_(self.mix.weight);nn.init.zeros_(self.mix.bias)
    def forward(self,questioned_image):
        if questioned_image.ndim!=4 or tuple(questioned_image.shape[1:])!=(3,self.image_size,self.image_size):raise ValueError("blind decoder requires [B,3,64,64]")
        b=len(questioned_image);cells=questioned_image.reshape(b,3,4,self.cell_size,4,self.cell_size).permute(0,2,4,1,3,5).reshape(b*16,3,self.cell_size,self.cell_size)
        hp=self.stage_c.decoder.decoder.highpass(cells);bases=self.carrier.bases*self.carrier.window
        hp_bases=self.stage_c.decoder.decoder.highpass(bases[:,None].expand(-1,3,-1,-1));hp_bases=hp_bases/hp_bases.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        matched=25*torch.einsum("bchw,kchw->bk",hp,hp_bases)/(hp.shape[1]*hp.shape[2]*hp.shape[3])
        return self.mix(matched).reshape(b,4,4,4)

class StageD12BitTransitionV1(nn.Module):
    architecture_version="stage_d_12bit_transition_v1.0"
    def __init__(self,stage_c:RegionalChannelV1):
        super().__init__();self.stage_c=stage_c;self.index_carrier=IndexCarrier(stage_c);self.index_head=BlindIndexHead(stage_c,self.index_carrier)
    def forward(self,image,packet_bits,amplitude=.014,active_index_bits=4):
        if packet_bits.ndim!=4 or tuple(packet_bits.shape[1:])!=(4,4,44) or len(image)!=len(packet_bits):raise ValueError("packet bits must be [B,4,4,44] with matching image batch")
        rs=packet_bits[...,RS_SLICE];rs_carrier=self.stage_c.encoder.router(rs);index_carrier=self.index_carrier(packet_bits[...,:4],active_index_bits)
        carrier=rs_carrier+index_carrier
        out=self.stage_c.encoder.encoder.forward_features(image,carrier,amplitude);watermarked=out["watermarked_image"]
        index_logits=self.index_head(watermarked);rs_logits=self.stage_c.decoder(watermarked)
        logits=watermarked.new_zeros((len(image),4,4,44));logits[...,:4]=index_logits;logits[...,4:12]=rs_logits
        out.update({"packet_logits":logits,"rs_carrier":rs_carrier,"index_carrier":index_carrier});return out
    def reproduce_stage_c(self,image,packet_bits,amplitude=.014):
        """R0 uses the exact parent call, guaranteeing an auditable identity path."""
        parent=self.stage_c(image,packet_bits[...,RS_SLICE],amplitude)
        logits=image.new_zeros((len(image),4,4,44));logits[...,RS_SLICE]=parent["regional_logits"]
        return {**parent,"packet_logits":logits}
