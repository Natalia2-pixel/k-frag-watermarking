from __future__ import annotations
from dataclasses import dataclass
from kfrag.protocol.packet import FragmentPacket
@dataclass(frozen=True)
class Candidate:
    packet: FragmentPacket|None
    location: tuple[float,float]
    confidence: float
    observed: bool=True
    decodable: bool=True
