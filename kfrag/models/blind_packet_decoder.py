"""Blind multiscale packet and discovery decoder."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from kfrag.preprocessing import BlindNormalizer

class BlindPacketDecoder(nn.Module):
    def __init__(self,packet_bits=44,width=32,preprocessing="rgb_plus_high_pass"):
        super().__init__(); self.normalizer=BlindNormalizer(preprocessing); c=self.normalizer.output_channels
        self.features=nn.Sequential(nn.Conv2d(c,width,3,padding=1),nn.SiLU(),nn.Conv2d(width,width,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(width,width*2,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(width*2,width*2,3,padding=1),nn.SiLU())
        self.packet=nn.Conv2d(width*2,packet_bits,1); self.presence=nn.Conv2d(width*2,1,1); self.uncertainty=nn.Conv2d(width*2,1,1); self.position=nn.Conv2d(width*2,2,1)
    def forward(self,questioned_image):
        f=self.features(self.normalizer(questioned_image)); size=(4,4)
        pool=lambda x:F.adaptive_avg_pool2d(x,size)
        return {"packet_logits":pool(self.packet(f)),"index_logits":pool(self.packet(f))[:,:4],"presence_logits":pool(self.presence(f)),"uncertainty":torch.sigmoid(pool(self.uncertainty(f))),"position_offsets":torch.tanh(pool(self.position(f)))}
