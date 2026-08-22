"""Deterministic bounded blind crop hypothesis search over frozen neural logits."""
from __future__ import annotations
from dataclasses import dataclass
import math,time
import torch
from torch.nn import functional as F
from kfrag.protocols.soft_fragment_decoder_v2 import calibrated_observations,maximum_likelihood_assignment

@dataclass(frozen=True)
class CropSearchLimits:
    scales:tuple[float,...]=(1.,.9,.75,.6,.5,.4,.25)
    aspect_ratios:tuple[float,...]=(.75,1.,4/3)
    offsets:tuple[tuple[float,float],...]=((0.,0.),(1.,0.),(0.,1.),(1.,1.),(.5,.5))
    regional_top_k:int=4
    token_beam_limit:int=256
    hmac_attempt_limit:int=4096
    total_search_budget:int=8192
    hypothesis_decode_limit:int=1
    runtime_limit_ms:float=1000.

def blind_regional_logits(model,questioned_images):
    """The frozen decoder receives only questioned RGB images."""
    if questioned_images.ndim!=4 or questioned_images.shape[1:]!=(3,64,64):raise ValueError("questioned images must be [B,3,64,64]")
    with torch.no_grad():
        return torch.cat((model.parent.index_head(questioned_images),model.parent.stage_c.decoder(questioned_images),model.tag_head(questioned_images)[...,:8]),-1)

def _content(questioned):
    mask=(questioned-.5).abs().amax(0)>1e-5
    if not bool(mask.any()):return questioned
    ys,xs=torch.where(mask);return questioned[:,int(ys.min()):int(ys.max())+1,int(xs.min()):int(xs.max())+1]

def geometric_hypotheses(questioned,limits):
    """Uses questioned pixels and predeclared geometry only; accepts no crop metadata."""
    if questioned.shape!=(3,256,256):raise ValueError("questioned crop representation must be [3,256,256]")
    content=_content(questioned);result=[F.interpolate(questioned[None],(64,64),mode="bilinear",align_corners=False,antialias=True)[0]]
    for area in limits.scales:
        for aspect in limits.aspect_ratios:
            h=min(256,max(1,round(256*math.sqrt(area/aspect))));w=min(256,max(1,round(256*math.sqrt(area*aspect))))
            resized=F.interpolate(content[None],(h,w),mode="bilinear",align_corners=False,antialias=True)[0]
            for oy,ox in limits.offsets:
                y=round((256-h)*oy);x=round((256-w)*ox);canvas=questioned.new_full((3,256,256),.5);canvas[:,y:y+h,x:x+w]=resized
                result.append(F.interpolate(canvas[None],(64,64),mode="bilinear",align_corners=False,antialias=True)[0])
    return torch.stack(result)

def _rank_score(logits,limits,temperatures):
    obs=calibrated_observations(logits.reshape(16,20),limits.regional_top_k,temperatures);assigned=maximum_likelihood_assignment(obs)
    return sum(o.indices[0].log_likelihood+o.symbols[0].log_likelihood+o.shares[0].log_likelihood for o in assigned.values())

def blind_crop_search(questioned,model,decoder,key,candidate_sources,limits:CropSearchLimits):
    """No original image, expected packet/token, coordinates, grid labels, or oracle survivor set is accepted."""
    started=time.perf_counter();hypotheses=geometric_hypotheses(questioned,limits)
    if len(hypotheses)>limits.total_search_budget:return {"status":"search_budget_exceeded","search_budget_exhausted":True,"states":{i:"uncertain" for i in range(16)},"runtime_ms":(time.perf_counter()-started)*1000}
    logits=[] 
    for batch in hypotheses.split(32):logits.extend(blind_regional_logits(model,batch))
    ranked=sorted(range(len(logits)),key=lambda i:(-_rank_score(logits[i],limits,decoder.temperatures),i))
    decoded=[];attempts=len(hypotheses)
    for index in ranked[:limits.hypothesis_decode_limit]:
        result=decoder.decode(logits[index].reshape(16,20),key,candidate_sources);attempts+=result["search_attempts"]
        if result["search_attempts"]>limits.hmac_attempt_limit or attempts>limits.total_search_budget:
            return {"status":"search_budget_exceeded","search_budget_exhausted":True,"states":{i:"uncertain" for i in range(16)},"runtime_ms":(time.perf_counter()-started)*1000,"hypothesis_count":len(hypotheses),"total_attempts":attempts}
        decoded.append((index,result))
    authenticated={(x["token"],x["source_id"]) for _,x in decoded if x["status"]=="authenticated"}
    chosen=next(((i,x) for i,x in decoded if x["status"]=="authenticated"),decoded[0])
    index,result=chosen;status="authenticated" if len(authenticated)==1 else "ambiguous" if len(authenticated)>1 else result["status"]
    states={i:("uncertain" if value=="manipulated" else value) for i,value in result["states"].items()}
    elapsed=(time.perf_counter()-started)*1000;exhausted=elapsed>limits.runtime_limit_ms or result["search_budget_exhausted"]
    if exhausted:status="search_budget_exceeded"
    return {"status":status,"token":result.get("token") if not exhausted else None,"source_id":result.get("source_id") if not exhausted else None,"states":states,"candidate_count":result["candidate_count"],"hypothesis_count":len(hypotheses),"selected_hypothesis":index,"total_attempts":attempts,"search_budget_exhausted":exhausted,"runtime_ms":elapsed,"_selected_logits":logits[index],"oracle_inputs_used":False}
