"""Cryptography is the sole admission criterion for reconstruction."""
from __future__ import annotations
from collections import defaultdict
from .candidate import Candidate
from kfrag.protocol.authentication import verify

def verify_candidates(candidates,key,namespace,tag_bits=32):
    grouped=defaultdict(list); rejected=[]
    for c in candidates:
        if c.packet is None or not c.decodable: rejected.append((c,"undecodable")); continue
        p=c.packet
        if verify(key,namespace,p.index,p.symbol,p.authentication_tag,tag_bits,p.version): grouped[p.index].append(c)
        else: rejected.append((c,"invalid_authentication"))
    accepted={}; conflicts=set()
    for index,items in grouped.items():
        distinct={(x.packet.symbol,x.packet.authentication_tag) for x in items}
        if len(items)>1: conflicts.add(index); rejected.extend((x,"duplicate_or_conflicting") for x in items)
        else: accepted[index]=max(items,key=lambda x:x.confidence)
    return accepted,rejected,conflicts
