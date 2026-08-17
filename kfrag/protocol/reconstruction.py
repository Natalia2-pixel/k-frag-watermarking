"""Authenticated erasure-only threshold reconstruction."""
from __future__ import annotations
from collections.abc import Iterable
from .authentication import verify
from .identity import RegisteredIdentity
from .packet import FragmentPacket
from kfrag.crypto.reed_solomon import reconstruct

def authenticated_fragments(packets: Iterable[FragmentPacket], key: bytes, namespace: bytes, tag_bits: int = 32):
    accepted={}
    for packet in packets:
        if not isinstance(packet, FragmentPacket): raise ValueError("invalid packet format")
        if not verify(key,namespace,packet.index,packet.symbol,packet.authentication_tag,tag_bits,packet.version): continue
        previous=accepted.get(packet.index)
        if previous is not None:
            if previous != packet: raise ValueError("duplicate or conflicting authenticated fragments")
            raise ValueError("duplicate indices are not allowed")
        accepted[packet.index]=packet
    return tuple(accepted[i] for i in sorted(accepted))

def reconstruct_identity(packets: Iterable[FragmentPacket], key: bytes, namespace: bytes, tag_bits: int = 32) -> RegisteredIdentity:
    valid=authenticated_fragments(packets,key,namespace,tag_bits)
    if len(valid) < 12: raise ValueError("fewer than 12 authenticated distinct fragments")
    return RegisteredIdentity(reconstruct((p.index,p.symbol) for p in valid))
