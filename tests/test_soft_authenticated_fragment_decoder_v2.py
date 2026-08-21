import inspect,torch
from kfrag.protocols.soft_fragment_decoder_v2 import SoftAuthenticatedFragmentDecoderV2,calibrated_observations
from kfrag.training.distributed_auth_neural_v2 import deterministic_scientific_key,fresh_distributed_packet_batch
from kfrag.diagnostics.soft_authenticated_fragment_decoder_v2 import _split,run_soft_decoder_v2
def fixture(seed=31):
    g=torch.Generator().manual_seed(seed);key=deterministic_scientific_key(seed);bits,meta=fresh_distributed_packet_batch(1,key,g);return (bits[0].reshape(16,20)*2-1)*12,meta[0],key
def test_oracle_truth_cannot_enter_candidate_generation_or_decode():
    assert list(inspect.signature(calibrated_observations).parameters)==["regional_logits","field_top_k","temperatures"]
    assert list(inspect.signature(SoftAuthenticatedFragmentDecoderV2.decode).parameters)==["self","regional_logits","key","candidate_sources"]
def test_perfect_and_shuffled_authenticate():
    logits,meta,key=fixture();decoder=SoftAuthenticatedFragmentDecoderV2();assert decoder.decode(logits,key,[meta.source_id])["status"]=="authenticated";assert decoder.decode(logits.flip(0),key,[meta.source_id])["status"]=="authenticated"
def test_duplicate_fails_closed():
    logits,meta,key=fixture();duplicate=torch.cat((logits[:15],logits[:1]));assert SoftAuthenticatedFragmentDecoderV2().decode(duplicate,key,[meta.source_id])["status"]!="authenticated"
def test_corruption_is_not_automatically_labelled_valid():
    logits,meta,key=fixture();logits[0,4:20]*=-1;result=SoftAuthenticatedFragmentDecoderV2(uncertain_confidence=.1).decode(logits,key,[meta.source_id]);assert result["status"]=="authenticated" and result["states"][0] in ("manipulated","uncertain") and result["states"][0]!="valid"
def test_uncertain_distinct_from_manipulated():
    logits,meta,key=fixture();logits[0,4:20]=0;result=SoftAuthenticatedFragmentDecoderV2(uncertain_confidence=.25).decode(logits,key,[meta.source_id]);assert "uncertain" in result["states"].values() and "uncertain"!="manipulated"
def test_deterministic_budget_enforced():
    logits,meta,key=fixture();decoder=SoftAuthenticatedFragmentDecoderV2(field_top_k=4,beam_width=256,search_budget=5);a=decoder.decode(logits,key,[meta.source_id]);b=decoder.decode(logits,key,[meta.source_id]);assert a["status"]==b["status"]=="search_budget_exceeded" and a["search_attempts"]==b["search_attempts"]==6
def test_population_split_is_disjoint_and_final_is_created_after_selection():
    split=_split([str(i) for i in range(12)],(4,4,4),7);assert not set(split["development"])&set(split["selection_validation"]) and not set(split["development"])&set(split["locked_final_test"]) and not set(split["selection_validation"])&set(split["locked_final_test"])
    source=inspect.getsource(run_soft_decoder_v2);assert source.index("chosen=max")<source.index("final_images=")
