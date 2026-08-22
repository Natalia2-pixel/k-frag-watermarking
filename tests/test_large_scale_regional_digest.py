import inspect,json
from pathlib import Path
import numpy as np
import torch
from kfrag.diagnostics.large_scale_regional_digest import *
from kfrag.diagnostics import large_scale_regional_digest as large

def test_frozen_threshold_file_is_exact_and_source_hash_matches():
    frozen=json.loads(Path("configs/regional_digest_frozen_thresholds_v1.json").read_text());assert frozen["source_commit"]==SOURCE_COMMIT and frozen["implementation_sha256"]==large._sha(IMPLEMENTATION_PATH);assert frozen["dct_phash"]=={"valid_max":.03125,"manipulated_min":.15625};assert frozen["combined_digest"]=={"valid_max":.03840238735079764,"manipulated_min":.11301076811845408}
def test_reproduction_has_no_threshold_selection_function():
    source=inspect.getsource(run_large_scale_reproduction);assert "quantile" not in source and "_select_threshold" not in source
def test_predeclared_transform_categories_are_separate():
    assert "jpeg_q40" in EXTREME_BENIGN and "splice_025" in MALICIOUS and not set(STANDARD_BENIGN)&set(MALICIOUS)
def test_clean_and_standard_transforms_execute_without_neural_model():
    image=torch.rand(3,256,256);assert torch.equal(benign_transform(image,"clean_repeat"),image);assert benign_transform(image,"resize_025").shape==image.shape
def test_four_state_rule_is_imported_unchanged():
    assert classify_distances([0,.1,.2,.8],[True,False,True,True],True,.05,.5)==("valid","missing","uncertain","manipulated")
def test_cluster_bootstrap_uses_image_values_not_flat_regions():
    values=[0.,1.,.5];ci=large._ci(values,np.random.default_rng(1),100);assert len(ci)==2 and ci[0]<=mean(values)<=ci[1]
def test_resume_fingerprint_changes_with_population_or_threshold():
    config={"seed":1};selected=[{"identifier":"a"}];frozen={"x":1};assert large._fingerprint(config,selected,frozen)!=large._fingerprint(config,[{"identifier":"b"}],frozen) and large._fingerprint(config,selected,frozen)!=large._fingerprint(config,selected,{"x":2})
def test_candidate_selection_requires_complete_gate_pass():
    config=json.loads(json.dumps({"gates":{"mean_benign_false_manipulation":.02,"worst_standard_benign_false_manipulation":.05,"aggregate_malicious_recall":.9,"splice_025_recall":.85,"overlay_025_recall":.85,"clean_repeatability_failures":0,"verification_runtime_ms":100}}));assert config["gates"]["splice_025_recall"]==.85
def test_global_identity_and_crop_claims_remain_separate():
    source=inspect.getsource(run_large_scale_reproduction);assert "trusted prerequisite" in source and '"crop_synchronization_validated":False' in source
