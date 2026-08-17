from __future__ import annotations
import time, torch
from kfrag.evidence import Candidate,verify_candidates,protocol_evidence_map
from kfrag.payload.regional_tensor import bits_to_packet
from kfrag.protocol.packet import FragmentPacket
from kfrag.protocol.reconstruction import reconstruct_identity
def candidates_from_output(output,threshold=.5):
    bits=output["packet_logits"].ge(0).cpu(); presence=torch.sigmoid(output["presence_logits"]).cpu(); offsets=output["position_offsets"].cpu(); result=[]
    for b in range(bits.shape[0]):
        for r in range(4):
            for c in range(4):
                legacy=bits_to_packet(bits[b,:,r,c]); packet=FragmentPacket(legacy.region_index,legacy.coded_symbol,legacy.authentication_tag)
                result.append(Candidate(packet,(float(((c+.5+offsets[b,0,r,c])/4).detach()),float(((r+.5+offsets[b,1,r,c])/4).detach())),float(presence[b,0,r,c].detach()),bool(presence[b,0,r,c]>=threshold),True))
    return result
def evaluate_questioned(model,image,key,namespace):
    start=time.perf_counter()
    with torch.no_grad(): output=model.decode(image)
    decode_ms=(time.perf_counter()-start)*1000; candidates=candidates_from_output(output); accepted,rejected,conflicts=verify_candidates(candidates,key,namespace); evidence=protocol_evidence_map(candidates,accepted,rejected,conflicts); identity=None; failure=None
    try: identity=reconstruct_identity([x.packet for x in accepted.values()],key,namespace).hex()
    except ValueError as exc: failure=str(exc)
    return {"discovered_candidates":len(candidates),"authenticated_fragments":len(accepted),"rejected_fragments":len(rejected),"reconstruction_success":identity is not None,"identity":identity,"failure_reason":failure,"protocol_evidence_map":evidence,"synchronization_success":float(output["synchronization"]["confidence"].mean().detach())>=.5,"decoding_latency_ms":decode_ms}
