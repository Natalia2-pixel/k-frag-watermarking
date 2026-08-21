"""Bounded blind soft-decision decoder for 20-bit distributed-auth fragments."""
from __future__ import annotations

import math, time
from dataclasses import dataclass
from itertools import combinations

import torch

from kfrag.crypto.reed_solomon import reconstruct as reconstruct_token
from kfrag.crypto.token import ProvenanceToken
from kfrag.protocols.distributed_auth_v2 import AuthFragment, JointFragmentCode


@dataclass(frozen=True)
class ValueCandidate:
    value: int
    log_likelihood: float


@dataclass
class RegionObservation:
    ordinal: int
    packet_log_likelihood: float
    index_confidence: float
    rs_confidence: float
    auth_confidence: float
    bit_probabilities: list[float]
    indices: list[ValueCandidate]
    symbols: list[ValueCandidate]
    shares: list[ValueCandidate]


def top_k_field_candidates(logits: torch.Tensor, k: int) -> list[ValueCandidate]:
    """Return candidates from logits alone using bounded bitwise beam search."""
    if logits.ndim != 1: raise ValueError("field logits must be one-dimensional")
    beam=[(0,0.0)]
    for logit in logits.detach().float().cpu():
        lp1=float(torch.nn.functional.logsigmoid(logit)); lp0=float(torch.nn.functional.logsigmoid(-logit))
        beam=sorted(((value<<1|bit,score+(lp1 if bit else lp0)) for value,score in beam for bit in (0,1)),key=lambda x:(-x[1],x[0]))[:k]
    return [ValueCandidate(value,score) for value,score in beam]


def _confidence(candidates):
    if len(candidates)<2:return 1.0
    return float(1-math.exp(min(0.0,candidates[1].log_likelihood-candidates[0].log_likelihood)))


def observations_from_logits(regional_logits: torch.Tensor, field_top_k: int=2) -> list[RegionObservation]:
    """Candidate generation deliberately has no payload, token, or key argument."""
    if regional_logits.ndim!=2 or regional_logits.shape[1]!=20:raise ValueError("regional logits must be [N,20]")
    result=[]
    for ordinal,row in enumerate(regional_logits):
        index=top_k_field_candidates(row[:4],min(16,field_top_k)); symbol=top_k_field_candidates(row[4:12],field_top_k); share=top_k_field_candidates(row[12:20],field_top_k)
        probabilities=torch.sigmoid(row.detach()).cpu().tolist()
        result.append(RegionObservation(ordinal,index[0].log_likelihood+symbol[0].log_likelihood+share[0].log_likelihood,_confidence(index),_confidence(symbol),_confidence(share),probabilities,index,symbol,share))
    return result


def likelihood_assignment(observations: list[RegionObservation]):
    """Resolve unordered/duplicate indices using deterministic likelihood edges."""
    edges=sorted(((candidate.log_likelihood,obs.ordinal,candidate.value,obs) for obs in observations for candidate in obs.indices),key=lambda x:(-x[0],x[1],x[2]))
    assigned={};used=set()
    for _,ordinal,index,obs in edges:
        if ordinal not in used and index not in assigned:assigned[index]=obs;used.add(ordinal)
    return assigned,[obs for obs in observations if obs.ordinal not in used]


def _field_beam(assigned,field,beam_width):
    beam=[({},0.0)]
    for index,obs in sorted(assigned.items()):
        candidates=getattr(obs,field)
        beam=sorted(((dict(values,**{str(index):candidate.value}),score+candidate.log_likelihood) for values,score in beam for candidate in candidates),key=lambda x:-x[1])[:beam_width]
    return [({int(k):v for k,v in values.items()},score) for values,score in beam]


class SoftAuthenticatedFragmentDecoder:
    def __init__(self,field_top_k=2,beam_width=64,search_budget=512,uncertain_confidence=.25):
        self.field_top_k=int(field_top_k);self.beam_width=int(beam_width);self.search_budget=int(search_budget);self.uncertain_confidence=float(uncertain_confidence)
        if min(self.field_top_k,self.beam_width,self.search_budget)<1:raise ValueError("search bounds must be positive")

    def decode(self,regional_logits:torch.Tensor,key:bytes,candidate_sources:list[bytes]):
        """Decode without expected token/payload; HMAC appears only in final verification."""
        started=time.perf_counter();observations=observations_from_logits(regional_logits,self.field_top_k);assigned,unassigned=likelihood_assignment(observations);attempts=0;exhausted=False;tokens={}
        symbol_beam=_field_beam(assigned,"symbols",self.beam_width)
        for beam_index,(values,score) in enumerate(symbol_beam):
            if len(values)<12:continue
            variants=[values]
            if beam_index==0:
                ranked=sorted(values,key=lambda i:assigned[i].rs_confidence)
                for erased in range(1,min(4,len(values)-11)):
                    variants.extend({i:v for i,v in values.items() if i not in removed} for removed in combinations(ranked[:5],erased))
            for variant in variants:
                attempts+=1
                if attempts>self.search_budget:exhausted=True;break
                try:token=ProvenanceToken.unpack(reconstruct_token(sorted(variant.items())))
                except ValueError:continue
                tokens[token]=max(score,tokens.get(token,-float("inf")))
            if exhausted:break
        auth_values=[]
        if not exhausted:
            protocol=JointFragmentCode()
            for values,score in _field_beam(assigned,"shares",self.beam_width):
                attempts+=1
                if attempts>self.search_budget:exhausted=True;break
                fragments=[AuthFragment(i,assigned[i].symbols[0].value,share) for i,share in values.items()]
                try:auth_values.append((protocol.reconstruct_authenticator(fragments),score))
                except ValueError:pass
        matches=[];auth_scores={}
        if not exhausted:
            protocol=JointFragmentCode()
            for value,score in auth_values:auth_scores[value]=max(score,auth_scores.get(value,-float("inf")))
            for token,token_score in tokens.items():
                for source in candidate_sources:
                    expected=protocol._shares(token,source,key)[:8]  # exact HMAC verification, never candidate generation
                    attempts+=1
                    if attempts>self.search_budget:exhausted=True;break
                    if expected in auth_scores:matches.append((token,source,token_score+auth_scores[expected]))
                if exhausted:break
        unique={(token,source):(token,source,score) for token,source,score in matches}
        status="authenticated" if not exhausted and len(unique)==1 and len(assigned)>=12 and not unassigned else "search_budget_exceeded" if exhausted else "ambiguous" if len(unique)>1 else "insufficient" if len(assigned)<12 else "rejected"
        states={i:"missing" for i in range(16)}
        if status=="authenticated":
            token,source,_=next(iter(unique.values()));protocol=JointFragmentCode();expected_symbols=protocol.issue(token,source,key);expected={x.index:x for x in expected_symbols}
            for index,obs in assigned.items():
                hard_symbol=obs.symbols[0].value;hard_share=obs.shares[0].value
                if hard_symbol==expected[index].symbol and hard_share==expected[index].share:states[index]="valid"
                elif min(obs.rs_confidence,obs.auth_confidence)<self.uncertain_confidence:states[index]="uncertain"
                else:states[index]="manipulated"
        else:
            for index in assigned:states[index]="uncertain"
        return {"status":status,"token":next(iter(unique))[0] if status=="authenticated" else None,"source_id":next(iter(unique))[1] if status=="authenticated" else None,"states":states,"candidate_count":len(tokens)*max(1,len(auth_values))*max(1,len(candidate_sources)),"token_candidates":len(tokens),"authenticator_candidates":len(auth_values),"_authenticator_values":tuple(auth_scores),"search_attempts":attempts,"search_budget":self.search_budget,"search_budget_exhausted":exhausted,"unassigned_observations":len(unassigned),"runtime_ms":(time.perf_counter()-started)*1000}
