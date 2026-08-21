import inspect
import torch
from kfrag.protocols.soft_fragment_decoder_v1 import SoftAuthenticatedFragmentDecoder,observations_from_logits,top_k_field_candidates
from kfrag.training.distributed_auth_neural_v2 import deterministic_scientific_key,fresh_distributed_packet_batch,evaluate_protocol_controls
from kfrag.protocols.distributed_auth_v2 import JointFragmentCode

def perfect_fixture(seed=10):
    g=torch.Generator().manual_seed(seed);key=deterministic_scientific_key(seed);bits,metadata=fresh_distributed_packet_batch(1,key,g);return (bits[0].reshape(16,20)*2-1)*12,metadata[0],key

def test_candidate_generation_has_no_truth_key_or_payload_interface():
    assert list(inspect.signature(observations_from_logits).parameters)==["regional_logits","field_top_k"]
    assert list(inspect.signature(top_k_field_candidates).parameters)==["logits","k"]
    assert list(inspect.signature(SoftAuthenticatedFragmentDecoder.decode).parameters)==["self","regional_logits","key","candidate_sources"]

def test_hmac_is_not_used_during_candidate_generation(monkeypatch):
    logits,_,_=perfect_fixture();calls=[]
    monkeypatch.setattr(JointFragmentCode,"_shares",lambda *args,**kwargs:calls.append(1) or bytes(16))
    observations_from_logits(logits);assert calls==[]

def test_perfect_logits_authenticate_and_wrong_key_rejects():
    logits,meta,key=perfect_fixture();decoder=SoftAuthenticatedFragmentDecoder(search_budget=512)
    assert decoder.decode(logits,key,[meta.source_id])["status"]=="authenticated"
    assert decoder.decode(logits,bytes(32),[meta.source_id])["status"]!="authenticated"

def test_random_and_uninformative_logits_fail_closed():
    _,meta,key=perfect_fixture();decoder=SoftAuthenticatedFragmentDecoder(search_budget=128)
    assert decoder.decode(torch.zeros(16,20),key,[meta.source_id])["status"]!="authenticated"
    assert decoder.decode(torch.randn(16,20,generator=torch.Generator().manual_seed(9)),key,[meta.source_id])["status"]!="authenticated"

def test_search_is_deterministic_and_bounded():
    logits,meta,key=perfect_fixture();decoder=SoftAuthenticatedFragmentDecoder(field_top_k=4,beam_width=64,search_budget=10)
    a=decoder.decode(logits,key,[meta.source_id]);b=decoder.decode(logits,key,[meta.source_id]);assert a["status"]==b["status"]=="search_budget_exceeded" and a["search_attempts"]==b["search_attempts"]==11

def test_unordered_perfect_fragments_remain_authenticatable():
    logits,meta,key=perfect_fixture();order=torch.randperm(16,generator=torch.Generator().manual_seed(2));result=SoftAuthenticatedFragmentDecoder(search_budget=512).decode(logits[order],key,[meta.source_id]);assert result["status"]=="authenticated"

def test_no_authentication_credit_when_share_reconstruction_fails():
    logits,meta,key=perfect_fixture();logits[:,12:20]=0
    result=SoftAuthenticatedFragmentDecoder(search_budget=256).decode(logits,key,[meta.source_id]);assert result["status"]!="authenticated" and result["token"] is None

def test_hard_baseline_is_reproducible():
    logits,meta,key=perfect_fixture();grid=logits.reshape(1,4,4,20)
    assert evaluate_protocol_controls(grid,[meta],key)==evaluate_protocol_controls(grid,[meta],key)
