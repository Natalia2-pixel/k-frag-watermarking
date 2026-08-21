"""Tag carriers isolated under the blind decoder's observation operator."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .stage_d_12bit_transition_v1 import StageD12BitTransitionV1
from .stage_d_isolated_tag_v1 import parent_carrier_vectors

class DecoderObservationOperator(nn.Module):
    """RGB -> fixed HP bank -> flatten -> L2 normalize; pooling is its inner product."""
    def __init__(self,parent):super().__init__();self.highpass=parent.stage_c.decoder.decoder.highpass
    def raw(self,rgb):return self.highpass(rgb).flatten(1)
    def forward(self,rgb):return F.normalize(self.raw(rgb),dim=1)
    def pooled_similarity(self,a,b):return self(a)@self(b).T

def parent_rgb_patterns(parent):
    feature=parent_carrier_vectors(parent).detach();weight=parent.stage_c.encoder.encoder.carrier_skip.weight[:,:,0,0].detach()
    return torch.einsum("oc,kchw->kohw",weight,feature).detach()

def observation_isolated_tag_patterns(parent,count=32):
    operator=DecoderObservationOperator(parent);parents=parent_rgb_patterns(parent);parent_obs=operator.raw(parents);u,s,vh=torch.linalg.svd(parent_obs,full_matrices=False);rank=int((s>s.max()*1e-5).sum());parent_basis_obs=vh[:rank];parent_basis_rgb=((u[:,:rank].T/s[:rank,None])@parents.flatten(1)).reshape(rank,3,*parents.shape[-2:]);h=parents.shape[-1];y=(torch.arange(h,dtype=torch.float32)+.5)[:,None];x=(torch.arange(h,dtype=torch.float32)+.5)[None,:];selected_rgb=[];selected_obs=[]
    for a in range(h-1,1,-1):
        for b in range(h-1,1,-1):
            spatial=torch.cos(math.pi*a*x/h)*torch.cos(math.pi*b*y/h)
            for channel in range(3):
                candidate=torch.zeros(3,h,h);candidate[channel]=spatial;observed=operator.raw(candidate[None])[0];coeff=parent_basis_obs@observed;value=candidate-torch.einsum("k,kchw->chw",coeff,parent_basis_rgb);obs=operator.raw(value[None])[0]
                if selected_obs:
                    basis=torch.stack(selected_obs);rgb_basis=torch.stack(selected_rgb);projection=basis@obs;value=value-torch.einsum("k,kchw->chw",projection,rgb_basis);obs=operator.raw(value[None])[0]
                # Re-project after tag/tag subtraction to control float32 drift.
                coeff=parent_basis_obs@obs;value=value-torch.einsum("k,kchw->chw",coeff,parent_basis_rgb);obs=operator.raw(value[None])[0]
                norm=obs.norm()
                if norm>5e-2:selected_rgb.append(value/norm);selected_obs.append(obs/norm)
                if len(selected_rgb)==count:return (torch.stack(selected_rgb)*math.sqrt(obs.numel())).detach()
    raise RuntimeError("insufficient observation-orthogonal tag patterns")

class ObservationTagCarrier(nn.Module):
    def __init__(self,parent):
        super().__init__();self.image_size=parent.stage_c.encoder.router.image_size;self.cell_size=parent.stage_c.encoder.router.cell_size;self.operator=DecoderObservationOperator(parent);self.register_buffer("parent_patterns",parent_rgb_patterns(parent));self.register_buffer("patterns",observation_isolated_tag_patterns(parent))
    def contributions(self,tags,active):
        signed=tags.float().mul(2).sub(1);signed[...,active:]=0;return signed[...,None,None,None]*self.patterns[None,None,None]
    def forward(self,tags,active):
        patches=self.contributions(tags,active).sum(3)/math.sqrt(max(1,active));return patches.permute(0,3,1,4,2,5).reshape(len(tags),3,self.image_size,self.image_size)
    def observation_correlation(self):return self.operator(self.patterns)@self.operator(self.parent_patterns).T

class BlindObservationTagDecoder(nn.Module):
    def __init__(self,parent,carrier):super().__init__();self.parent=parent;self.carrier=carrier;self.image_size=carrier.image_size;self.cell_size=carrier.cell_size;self.mix=nn.Linear(32,32);nn.init.eye_(self.mix.weight);nn.init.zeros_(self.mix.bias)
    def forward(self,questioned_image):
        if questioned_image.ndim!=4 or tuple(questioned_image.shape[1:])!=(3,self.image_size,self.image_size):raise ValueError("blind observation tag decoder accepts only [B,3,64,64]")
        b=len(questioned_image);cells=questioned_image.reshape(b,3,4,self.cell_size,4,self.cell_size).permute(0,2,4,1,3,5).reshape(b*16,3,self.cell_size,self.cell_size);observed=self.carrier.operator(cells);patterns=self.carrier.operator(self.carrier.patterns);return self.mix(25*(observed@patterns.T)).reshape(b,4,4,32)

class StageDObservationTagV1(nn.Module):
    architecture_version="stage_d_observation_tag_v1.0"
    def __init__(self,parent:StageD12BitTransitionV1,tag_budget_fraction=.25):super().__init__();self.parent=parent;self.carrier=ObservationTagCarrier(parent);self.tag_decoder=BlindObservationTagDecoder(parent,self.carrier);self.tag_budget_fraction=float(tag_budget_fraction)
    def forward(self,image,bits,amplitude=.014,active_tag_bits=8):
        if active_tag_bits==0:return self.parent(image,bits,amplitude,4)
        base=self.parent(image,bits,amplitude,4);carrier=self.carrier(bits[...,12:44],active_tag_bits);bounded=torch.tanh(carrier)*self.tag_budget_fraction;parent_residual=base["residual"];tag_residual=torch.where(bounded>=0,bounded*(amplitude-parent_residual),bounded*(amplitude+parent_residual));residual=parent_residual+tag_residual;watermarked=(image+residual).clamp(0,1);logits=watermarked.new_zeros((len(image),4,4,44));logits[...,:4]=self.parent.index_head(watermarked);logits[...,4:12]=self.parent.stage_c.decoder(watermarked);logits[...,12:44]=self.tag_decoder(watermarked);return {**base,"parent_residual":parent_residual,"tag_residual":tag_residual,"residual":residual,"watermarked_image":watermarked,"packet_logits":logits,"tag_carrier_rgb":carrier}
