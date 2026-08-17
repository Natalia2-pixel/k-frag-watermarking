"""Serialization of independently authenticated logical fragments."""
from __future__ import annotations
from dataclasses import dataclass
from .authentication import PROTOCOL_VERSION, tag
from .identity import RegisteredIdentity
from kfrag.crypto.reed_solomon import encode

@dataclass(frozen=True)
class FragmentPacket:
    index: int
    symbol: int
    authentication_tag: bytes
    version: int = PROTOCOL_VERSION
    def __post_init__(self) -> None:
        if not 0 <= self.index < 16 or not 0 <= self.symbol < 256: raise ValueError("invalid fragment fields")
        if not isinstance(self.authentication_tag, bytes) or not self.authentication_tag: raise ValueError("authentication_tag must be bytes")
    def to_bytes(self) -> bytes:
        """Packed representation with a zero padding nibble; wire payload is 44 bits."""
        return bytes((self.index, self.symbol)) + self.authentication_tag
    def to_bits(self) -> tuple[int,...]:
        return tuple((self.index >> shift)&1 for shift in range(3,-1,-1)) + tuple((byte >> shift)&1 for byte in bytes((self.symbol,))+self.authentication_tag for shift in range(7,-1,-1))
    @classmethod
    def from_bytes(cls, raw: bytes, tag_bits: int = 32) -> "FragmentPacket":
        if not isinstance(raw, bytes) or len(raw) != 2 + tag_bits//8: raise ValueError("invalid packet length")
        if raw[0] > 15: raise ValueError("non-zero packet padding/version nibble")
        return cls(raw[0], raw[1], raw[2:])
    @classmethod
    def from_bits(cls, bits, tag_bits: int = 32) -> "FragmentPacket":
        values=list(bits)
        if len(values) != 12+tag_bits or any(x not in (0,1,False,True) for x in values): raise ValueError("invalid packet bits")
        index=sum(int(values[j]) << (3-j) for j in range(4)); tail=values[4:]
        raw=bytes(sum(int(tail[i+j]) << (7-j) for j in range(8)) for i in range(0,len(tail),8))
        return cls(index,raw[0],raw[1:])

def create_fragments(identity: RegisteredIdentity, key: bytes, namespace: bytes, tag_bits: int = 32, version: int = PROTOCOL_VERSION):
    return tuple(FragmentPacket(i,s,tag(key,namespace,i,s,tag_bits,version),version) for i,s in enumerate(encode(identity.value)))
