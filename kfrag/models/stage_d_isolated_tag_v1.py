"""Experimental Stage-D tag channel isolated from the frozen 12-bit parent."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .stage_d_12bit_transition_v1 import StageD12BitTransitionV1

def parent_carrier_vectors(parent:StageD12BitTransitionV1):
    router=parent.stage_c.encoder.router;h=router.cell_size;vectors=[]
    for bit in range(8):
        value=torch.zeros(8,h,h);value[bit]=router.cell_bases[bit].detach()*router.window;vectors.append(value)
    for bit in range(4):
        value=torch.zeros(8,h,h);channel=int(parent.index_carrier.channel_assignment[bit].argmax());value[channel]=parent.index_carrier.bases[bit].detach()*parent.index_carrier.window;vectors.append(value)
    return F.normalize(torch.stack(vectors).flatten(1),dim=1).reshape(12,8,h,h)

def isolated_tag_vectors(parent,count=32):
    frozen=parent_carrier_vectors(parent).flatten(1);q=torch.linalg.qr(frozen.T,mode="reduced").Q.T;h=parent.stage_c.encoder.router.cell_size;y=(torch.arange(h,dtype=torch.float32)+.5)[:,None];x=(torch.arange(h,dtype=torch.float32)+.5)[None,:];candidates=[]
    for a in range(h-1,3,-1):
        for b in range(h-1,3,-1):
            spatial=torch.cos(math.pi*a*x/h)*torch.cos(math.pi*b*y/h)
            for channel in range(8):
                value=torch.zeros(8,h,h);value[channel]=spatial;candidates.append(value.flatten())
    selected=[]
    for candidate in candidates:
        value=candidate-q.T@(q@candidate)
        if selected:
            basis=torch.stack(selected);value=value-basis.T@(basis@value)
        norm=value.norm()
        if norm>1e-5:selected.append(value/norm)
        if len(selected)==count:break
    if len(selected)!=count:raise RuntimeError("insufficient orthogonal tag-carrier dimension")
    return torch.stack(selected).reshape(count,8,h,h)*math.sqrt(8*h*h)

class IsolatedTagCarrier(nn.Module):
    def __init__(self,parent):
        super().__init__();router=parent.stage_c.encoder.router;self.image_size=router.image_size;self.cell_size=router.cell_size;self.register_buffer("parent_vectors",parent_carrier_vectors(parent));self.register_buffer("vectors",isolated_tag_vectors(parent))
    def contributions(self,tags,active):
        signed=tags.float().mul(2).sub(1);signed[...,active:]=0;return signed[...,None,None,None]*self.vectors[None,None,None]
    def forward(self,tags,active):
        patches=self.contributions(tags,active).sum(3)/math.sqrt(max(1,active));return patches.permute(0,3,1,4,2,5).reshape(len(tags),8,self.image_size,self.image_size)
    def correlation_matrix(self):return F.normalize(self.vectors.flatten(1),dim=1)@F.normalize(self.parent_vectors.flatten(1),dim=1).T

class BlindIsolatedTagDecoder(nn.Module):
    def __init__(self,parent,carrier,projection):
        super().__init__();self.parent=parent;self.carrier=carrier;self.projection=projection;self.image_size=carrier.image_size;self.cell_size=carrier.cell_size;self.mix=nn.Linear(32,32);nn.init.eye_(self.mix.weight);nn.init.zeros_(self.mix.bias)
    def forward(self,questioned_image):
        if questioned_image.ndim!=4 or tuple(questioned_image.shape[1:])!=(3,self.image_size,self.image_size):raise ValueError("blind isolated tag decoder accepts only [B,3,64,64]")
        b=len(questioned_image);cells=questioned_image.reshape(b,3,4,self.cell_size,4,self.cell_size).permute(0,2,4,1,3,5).reshape(b*16,3,self.cell_size,self.cell_size);hp=self.parent.stage_c.decoder.decoder.highpass(cells)
        rgb=torch.einsum("oc,kchw->kohw",self.projection.weight[:,:,0,0],self.carrier.vectors);responses=self.parent.stage_c.decoder.decoder.highpass(rgb);responses=F.normalize(responses.flatten(1),dim=1).reshape_as(responses)
        matched=25*torch.einsum("bchw,kchw->bk",hp,responses)/(hp.shape[1]*hp.shape[2]*hp.shape[3]);return self.mix(matched).reshape(b,4,4,32)

class StageDIsolatedTagV1(nn.Module):
    architecture_version="stage_d_isolated_tag_v1.0"
    def __init__(self,parent:StageD12BitTransitionV1,tag_budget_fraction=.25):
        super().__init__();self.parent=parent;self.carrier=IsolatedTagCarrier(parent);self.tag_projection=nn.Conv2d(8,3,1,bias=False);self.tag_projection.weight.data.copy_(parent.stage_c.encoder.encoder.carrier_skip.weight.detach());self.tag_decoder=BlindIsolatedTagDecoder(parent,self.carrier,self.tag_projection);self.tag_budget_fraction=float(tag_budget_fraction)
    def forward(self,image,bits,amplitude=.014,active_tag_bits=32):
        if active_tag_bits==0:return self.parent(image,bits,amplitude,4)
        with torch.set_grad_enabled(torch.is_grad_enabled()):base=self.parent(image,bits,amplitude,4)
        feature=self.carrier(bits[...,12:44],active_tag_bits);bounded=torch.tanh(self.tag_projection(feature))*self.tag_budget_fraction;parent_residual=base["residual"];positive_room=amplitude-parent_residual;negative_room=amplitude+parent_residual;tag_residual=torch.where(bounded>=0,bounded*positive_room,bounded*negative_room);residual=parent_residual+tag_residual;watermarked=(image+residual).clamp(0,1)
        logits=watermarked.new_zeros((len(image),4,4,44));logits[...,:4]=self.parent.index_head(watermarked);logits[...,4:12]=self.parent.stage_c.decoder(watermarked);logits[...,12:44]=self.tag_decoder(watermarked);return {**base,"parent_residual":parent_residual,"tag_residual":tag_residual,"residual":residual,"watermarked_image":watermarked,"packet_logits":logits,"tag_carrier_features":feature}
