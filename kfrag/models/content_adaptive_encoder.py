"""Compact FiLM-conditioned bounded residual encoder with carrier bias."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class ContentAdaptiveEncoder(nn.Module):
    def __init__(self, packet_bits=44, width=32, residual_alpha=.05, learned_carriers=True):
        super().__init__(); self.packet_bits=packet_bits; self.residual_alpha=float(residual_alpha)
        self.image=nn.Sequential(nn.Conv2d(3,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU())
        self.condition=nn.Conv2d(packet_bits,width*2,1)
        self.output=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,3,1))
        bank=torch.randn(packet_bits,3,16,16)*.05
        self.carrier_bank=nn.Parameter(bank,requires_grad=learned_carriers)
    def forward(self,image,regional_packets):
        if image.ndim!=4 or image.shape[1]!=3: raise ValueError("image must be [B,3,H,W]")
        if regional_packets.ndim!=4 or regional_packets.shape[1:]!=(self.packet_bits,4,4) or image.shape[0]!=regional_packets.shape[0]: raise ValueError("regional_packets must be [B,packet_bits,4,4]")
        cond=F.interpolate(regional_packets*2-1,image.shape[-2:],mode="nearest")
        gamma,beta=self.condition(cond).chunk(2,1); feature=self.image(image)*(1+.1*gamma)+.1*beta
        learned=self.output(feature)
        carrier=torch.einsum("bkhw,kcxy->bchwxy",regional_packets*2-1,self.carrier_bank)
        carrier=carrier.permute(0,1,2,4,3,5).reshape(image.shape[0],3,64,64)
        carrier=F.interpolate(carrier,image.shape[-2:],mode="bilinear",align_corners=False)
        residual=self.residual_alpha*torch.tanh(learned+carrier)
        return {"watermarked_image":(image+residual).clamp(0,1),"residual":residual}
