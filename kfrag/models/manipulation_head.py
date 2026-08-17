from __future__ import annotations
from torch import nn
class ManipulationHead(nn.Module):
    def __init__(self,input_channels=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(input_channels,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,4,1))
    def forward(self,features): return self.net(features)
