from __future__ import annotations
from torch import nn
from .content_adaptive_encoder import ContentAdaptiveEncoder
from .blind_packet_decoder import BlindPacketDecoder
from .synchronization import GlobalSynchronizationHead
class KFragSystem(nn.Module):
    def __init__(self,packet_bits=44,width=32,residual_alpha=.05,preprocessing="rgb_plus_high_pass"):
        super().__init__(); self.encoder=ContentAdaptiveEncoder(packet_bits,width,residual_alpha); self.synchronizer=GlobalSynchronizationHead(max(8,width//2)); self.decoder=BlindPacketDecoder(packet_bits,width,preprocessing)
    def encode(self,image,regional_packets): return self.encoder(image,regional_packets)
    def decode(self,questioned_image):
        aligned,sync=self.synchronizer.align(questioned_image); result=self.decoder(aligned); result["synchronization"]=sync; return result
    def forward(self,image,regional_packets):
        encoded=self.encode(image,regional_packets); return {**encoded,**self.decode(encoded["watermarked_image"])}
