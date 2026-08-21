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
        """Pack the 44-bit packet into six bytes (the final low nibble is padding)."""
        value = self.index
        value = (value << 8) | self.symbol
        value = (value << (8 * len(self.authentication_tag))) | int.from_bytes(self.authentication_tag, "big")
        return (value << 4).to_bytes(6, "big")
    def to_bits(self) -> tuple[int,...]:
        return tuple((self.index >> shift)&1 for shift in range(3,-1,-1)) + tuple((byte >> shift)&1 for byte in bytes((self.symbol,))+self.authentication_tag for shift in range(7,-1,-1))
    @classmethod
    def from_bytes(cls, raw: bytes, tag_bits: int = 32) -> "FragmentPacket":
        if tag_bits != 32: raise ValueError("packed wire format currently requires a 32-bit tag")
        if not isinstance(raw, bytes) or len(raw) != 6: raise ValueError("invalid packet length")
        value=int.from_bytes(raw,"big")
        if value & 0xF: raise ValueError("non-zero packet padding nibble")
        value >>= 4
        authentication_tag=(value & 0xFFFFFFFF).to_bytes(4,"big"); value >>= 32
        symbol=value & 0xFF; index=value >> 8
        return cls(index,symbol,authentication_tag)
    @classmethod
    def from_bits(cls, bits, tag_bits: int = 32) -> "FragmentPacket":
        values=list(bits)
        if len(values) != 12+tag_bits or any(x not in (0,1,False,True) for x in values): raise ValueError("invalid packet bits")
        index=sum(int(values[j]) << (3-j) for j in range(4)); tail=values[4:]
        raw=bytes(sum(int(tail[i+j]) << (7-j) for j in range(8)) for i in range(0,len(tail),8))
        return cls(index,raw[0],raw[1:])

def create_fragments(identity: RegisteredIdentity, key: bytes, namespace: bytes, tag_bits: int = 32, version: int = PROTOCOL_VERSION):
    return tuple(FragmentPacket(i,s,tag(key,namespace,i,s,tag_bits,version),version) for i,s in enumerate(encode(identity.value)))
