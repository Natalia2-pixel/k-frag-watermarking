import inspect,json
import numpy as np
from kfrag.protocols.content_bound_fragment_state_v1 import *
def fixture(strategy=RobustPerceptualDigest(),seed=1):
    rng=np.random.default_rng(seed);features=rng.normal(size=(16,32));key=b"k"*32;record,reference=strategy.enroll(features,b"source",key);thresholds=strategy.calibrate(rng.normal(size=(64,16,32)),seed);return strategy,features,key,record,reference,thresholds
def test_four_outputs_are_separate_and_missing_means_no_observation():
    strategy,features,key,record,reference,thresholds=fixture();present=np.ones(16,bool);present[3]=False;out=strategy.verify(record,reference,features,present,key,True,thresholds);assert out.authenticated_source_identity and 3 in out.missing_fragment_evidence and 3 not in out.valid_fragment_evidence
def test_manipulated_requires_authenticated_identity_and_strong_contradiction():
    strategy,features,key,record,reference,thresholds=fixture();changed=-features;present=np.ones(16,bool);authenticated=strategy.verify(record,reference,changed,present,key,True,thresholds);unauthenticated=strategy.verify(record,reference,changed,present,key,False,thresholds);assert authenticated.manipulated_local_evidence and not unauthenticated.manipulated_local_evidence and unauthenticated.uncertain_local_evidence
def test_uncertain_is_not_manipulated():assert "uncertain" in STATES and "manipulated" in STATES and "uncertain"!="manipulated"
def test_wrong_key_prevents_manipulation_claim():
    strategy,features,key,record,reference,thresholds=fixture();out=strategy.verify(record,reference,-features,np.ones(16,bool),b"x"*32,True,thresholds);assert not out.authenticated_source_identity and len(out.uncertain_local_evidence)==16
def test_comparison_does_not_call_eight_bits_strong_authentication():
    rows=strategy_comparison();assert len(rows)>=3 and all(row["authentication_share_bits"]==8 for row in rows) and all("independent" not in row["security_assumptions"].lower() for row in rows)
def test_simulation_is_deterministic_and_keys_are_not_reported():
    a=simulate_content_binding(32,32,seed=9);b=simulate_content_binding(32,32,seed=9);assert a==b;assert "key" not in json.dumps(a).lower()
def test_strategy_api_has_no_original_image_requirement():
    assert "original_image" not in inspect.signature(ContentBindingStrategy.verify).parameters
def test_mixed_source_fails_to_claim_manipulation_without_identity():
    report=simulate_content_binding(16,16,seed=7)
    for strategy in report["strategies"].values():assert strategy["scenarios"]["mixed_source"]["identity_authenticated"]==0 and strategy["scenarios"]["mixed_source"]["missed_manipulation_rate"]==1

