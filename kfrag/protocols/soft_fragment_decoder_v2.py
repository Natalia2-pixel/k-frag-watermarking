"""Calibrated, bounded soft decoder with likelihood assignment and RS lists."""
from __future__ import annotations
import math,time
from dataclasses import dataclass
from itertools import combinations
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from kfrag.crypto.reed_solomon import reconstruct as reconstruct_token
from kfrag.crypto.token import ProvenanceToken
from kfrag.protocols.distributed_auth_v2 import AuthFragment,JointFragmentCode
from kfrag.protocols.soft_fragment_decoder_v1 import ValueCandidate,top_k_field_candidates

@dataclass
class ObservationV2:
    ordinal:int;probabilities:list[float];index_log_likelihoods:list[float];indices:list[ValueCandidate];symbols:list[ValueCandidate];shares:list[ValueCandidate];index_confidence:float;rs_confidence:float;auth_confidence:float

def _confidence(candidates):return 1-math.exp(min(0.,candidates[1].log_likelihood-candidates[0].log_likelihood)) if len(candidates)>1 else 1.
def _value_log_likelihood(logits,value,width):
    return sum(float(torch.nn.functional.logsigmoid(logits[i] if value&(1<<(width-1-i)) else -logits[i])) for i in range(width))

def calibrated_observations(regional_logits,field_top_k=4,temperatures=(1.,1.,1.)):
    """Uses logits only; no key, expected token, packet, or oracle input is accepted."""
    if regional_logits.ndim!=2 or regional_logits.shape[1]!=20:raise ValueError("regional logits must be [N,20]")
    result=[]
    for ordinal,row in enumerate(regional_logits.detach().float().cpu()):
        index_logits=row[:4]/temperatures[0];rs_logits=row[4:12]/temperatures[1];auth_logits=row[12:20]/temperatures[2]
        indices=top_k_field_candidates(index_logits,min(16,field_top_k));symbols=top_k_field_candidates(rs_logits,field_top_k);shares=top_k_field_candidates(auth_logits,field_top_k)
        result.append(ObservationV2(ordinal,torch.sigmoid(row).tolist(),[_value_log_likelihood(index_logits,v,4) for v in range(16)],indices,symbols,shares,_confidence(indices),_confidence(symbols),_confidence(shares)))
    return result

def maximum_likelihood_assignment(observations):
    if len(observations)>16:raise ValueError("at most sixteen observations are supported")
    costs=np.asarray([[-score for score in obs.index_log_likelihoods] for obs in observations]);rows,cols=linear_sum_assignment(costs)
    return {int(index):observations[int(row)] for row,index in zip(rows,cols)}

def _beam(assigned,field,width,indices=None):
    indices=sorted(assigned) if indices is None else [i for i in indices if i in assigned];beam=[({},0.)]
    for index in indices:
        beam=sorted(((dict(values,**{str(index):candidate.value}),score+candidate.log_likelihood) for values,score in beam for candidate in getattr(assigned[index],field)),key=lambda x:(-x[1],tuple(sorted(x[0].items()))))[:width]
    return [({int(k):v for k,v in values.items()},score) for values,score in beam]

class SoftAuthenticatedFragmentDecoderV2:
    def __init__(self,field_top_k=4,beam_width=128,search_budget=2048,temperatures=(1.,1.,1.),uncertain_confidence=.25):
        self.field_top_k=int(field_top_k);self.beam_width=int(beam_width);self.search_budget=int(search_budget);self.temperatures=tuple(float(x) for x in temperatures);self.uncertain_confidence=float(uncertain_confidence)
        if min(self.field_top_k,self.beam_width,self.search_budget)<1 or len(self.temperatures)!=3:raise ValueError("invalid bounded decoder configuration")
    def decode(self,regional_logits,key,candidate_sources):
        started=time.perf_counter();observations=calibrated_observations(regional_logits,self.field_top_k,self.temperatures);assigned=maximum_likelihood_assignment(observations);exact_duplicate=len({regional_logits[i].detach().cpu().numpy().tobytes() for i in range(len(regional_logits))})<len(regional_logits);attempts=0;exhausted=False;tokens={}
        # Systematic positions directly propose token bytes; parity likelihood is checked by HMAC/codeword verification later.
        if all(i in assigned for i in range(12)):
            for values,score in _beam(assigned,"symbols",self.beam_width,range(12)):
                attempts+=1
                if attempts>self.search_budget:exhausted=True;break
                try:token=ProvenanceToken.unpack(bytes(values[i] for i in range(12)))
                except ValueError:continue
                tokens[token]=max(score,tokens.get(token,-float("inf")))
        # Full-codeword and adaptive low-confidence erasure candidates.
        if not exhausted and len(assigned)>=12:
            ranked=sorted(assigned,key=lambda i:assigned[i].rs_confidence)
            for beam_index,(values,score) in enumerate(_beam(assigned,"symbols",self.beam_width)):
                variants=[values]
                if beam_index<4:
                    max_erase=min(4,len(values)-12)
                    for erased in range(1,max_erase+1):variants.extend({i:v for i,v in values.items() if i not in removed} for removed in combinations(ranked[:min(6,len(ranked))],erased))
                for variant in variants:
                    attempts+=1
                    if attempts>self.search_budget:exhausted=True;break
                    try:token=ProvenanceToken.unpack(reconstruct_token(sorted(variant.items())))
                    except ValueError:continue
                    tokens[token]=max(score,tokens.get(token,-float("inf")))
                if exhausted:break
        auth_scores={}
        if not exhausted and len(assigned)>=8:
            for values,score in _beam(assigned,"shares",self.beam_width):
                attempts+=1
                if attempts>self.search_budget:exhausted=True;break
                fragments=[AuthFragment(i,assigned[i].symbols[0].value,share) for i,share in values.items()]
                try:value=JointFragmentCode().reconstruct_authenticator(fragments);auth_scores[value]=max(score,auth_scores.get(value,-float("inf")))
                except ValueError:pass
        matches=[];correct_hmac_checks=0
        if not exhausted:
            protocol=JointFragmentCode()
            for token,token_score in tokens.items():
                for source in candidate_sources:
                    attempts+=1
                    if attempts>self.search_budget:exhausted=True;break
                    expected=protocol._shares(token,source,key)[:8];correct_hmac_checks+=1
                    if expected in auth_scores:matches.append((token,source,token_score+auth_scores[expected]))
                if exhausted:break
        unique={(token,source):(token,source,score) for token,source,score in matches};status="authenticated" if not exhausted and len(unique)==1 and len(assigned)>=12 and not exact_duplicate else "search_budget_exceeded" if exhausted else "ambiguous" if len(unique)>1 else "insufficient" if len(assigned)<12 else "rejected";states={i:"missing" for i in range(16)}
        authenticated=next(iter(unique.values())) if status=="authenticated" else None
        if authenticated:
            token,source,_=authenticated;expected={f.index:f for f in JointFragmentCode().issue(token,source,key)}
            for index,obs in assigned.items():
                symbol=obs.symbols[0].value;share=obs.shares[0].value
                if symbol==expected[index].symbol and share==expected[index].share:states[index]="valid"
                elif min(obs.rs_confidence,obs.auth_confidence)>=self.uncertain_confidence:states[index]="manipulated"
                else:states[index]="uncertain"
        else:
            for index in assigned:states[index]="uncertain"
        return {"status":status,"token":authenticated[0] if authenticated else None,"source_id":authenticated[1] if authenticated else None,"states":states,"candidate_count":len(tokens)*max(1,len(auth_scores))*max(1,len(candidate_sources)),"token_candidates":len(tokens),"authenticator_candidates":len(auth_scores),"_token_values":tuple(tokens),"_authenticator_values":tuple(auth_scores),"search_attempts":attempts,"search_budget":self.search_budget,"search_budget_exhausted":exhausted,"hmac_candidates_checked":correct_hmac_checks,"exact_duplicate_observed":exact_duplicate,"runtime_ms":(time.perf_counter()-started)*1000}

