"""P0-P4 tag-capacity expansion of the validated Stage-D 12-bit channel."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .stage_d_12bit_transition_v1 import StageD12BitTransitionV1

TAG_BITS=32
CAPACITIES=(12,20,28,36,44)

def _tag_bases(size:int)->torch.Tensor:
    excluded={(15,15),(15,14),(14,15),(14,14),(13,15),(15,13),(13,14),(14,13),(12,15),(15,12),(12,14),(14,12)}
    pairs=[p for p in sorted(((a,b) for a in range(4,size) for b in range(4,size)),key=lambda p:(-(p[0]+p[1]),-min(p),-max(p))) if p not in excluded][:TAG_BITS]
    y=(torch.arange(size,dtype=torch.float32)+.5)[:,None];x=(torch.arange(size,dtype=torch.float32)+.5)[None,:]
    bases=torch.stack([torch.cos(math.pi*a*x/size)*torch.cos(math.pi*b*y/size) for a,b in pairs]);bases-=bases.mean((-2,-1),keepdim=True)
    return bases/bases.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)

class TagCarrier(nn.Module):
    def __init__(self,parent:StageD12BitTransitionV1):
        super().__init__();router=parent.stage_c.encoder.router;self.image_size=router.image_size;self.cell_size=router.cell_size
        self.bases=nn.Parameter(_tag_bases(self.cell_size));self.register_buffer("window",router.window.detach().clone());self.register_buffer("assignment",F.one_hot(torch.arange(TAG_BITS)%8,8).float())
    def forward(self,tags,active_tag_bits):
        signed=tags.float().mul(2).sub(1);signed[...,active_tag_bits:]=0
        patches=signed[...,None,None,None]*self.bases[None,None,None,:,None]*self.window[None,None,None,None,None]*self.assignment[None,None,None,:,:,None,None]
        return patches.sum(3).permute(0,3,1,4,2,5).reshape(len(tags),8,self.image_size,self.image_size)*math.sqrt(12/max(12,12+active_tag_bits))

class BlindTagHead(nn.Module):
    def __init__(self,parent,carrier):
        super().__init__();self.parent=parent;self.carrier=carrier;self.image_size=carrier.image_size;self.cell_size=carrier.cell_size;self.mix=nn.Linear(32,32);nn.init.eye_(self.mix.weight);nn.init.zeros_(self.mix.bias)
    def forward(self,questioned_image):
        if questioned_image.ndim!=4 or tuple(questioned_image.shape[1:])!=(3,self.image_size,self.image_size):raise ValueError("blind tag decoder requires [B,3,64,64]")
        b=len(questioned_image);cells=questioned_image.reshape(b,3,4,self.cell_size,4,self.cell_size).permute(0,2,4,1,3,5).reshape(b*16,3,self.cell_size,self.cell_size);hp=self.parent.stage_c.decoder.decoder.highpass(cells)
        bases=self.carrier.bases*self.carrier.window;responses=self.parent.stage_c.decoder.decoder.highpass(bases[:,None].expand(-1,3,-1,-1));responses=responses/responses.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
        matched=25*torch.einsum("bchw,kchw->bk",hp,responses)/(hp.shape[1]*hp.shape[2]*hp.shape[3]);return self.mix(matched).reshape(b,4,4,32)

class StageDTagCapacityV1(nn.Module):
    architecture_version="stage_d_tag_capacity_v1.0"
    def __init__(self,parent:StageD12BitTransitionV1):super().__init__();self.parent=parent;self.tag_carrier=TagCarrier(parent);self.tag_head=BlindTagHead(parent,self.tag_carrier)
    def forward(self,image,bits,amplitude=.014,active_tag_bits=32):
        if bits.ndim!=4 or tuple(bits.shape[1:])!=(4,4,44) or len(image)!=len(bits):raise ValueError("complete packets must be [B,4,4,44] with matching images")
        if active_tag_bits==0:return self.reproduce_parent(image,bits,amplitude)
        base=self.parent.stage_c.encoder.router(bits[...,4:12])+self.parent.index_carrier(bits[...,:4],4);tag=self.tag_carrier(bits[...,12:44],active_tag_bits)
        out=self.parent.stage_c.encoder.encoder.forward_features(image,base+tag,amplitude);watermarked=out["watermarked_image"];logits=watermarked.new_zeros((len(image),4,4,44));logits[...,:4]=self.parent.index_head(watermarked);logits[...,4:12]=self.parent.stage_c.decoder(watermarked);logits[...,12:44]=self.tag_head(watermarked);out.update({"packet_logits":logits,"tag_carrier":tag});return out
    def reproduce_parent(self,image,bits,amplitude=.014):return self.parent(image,bits,amplitude,4)
