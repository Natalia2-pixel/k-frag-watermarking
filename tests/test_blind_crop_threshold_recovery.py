import inspect,json
from pathlib import Path
import pytest,torch,yaml
from kfrag.protocols import blind_crop_search_v1 as search
from kfrag.diagnostics import blind_crop_threshold_recovery as diag
from kfrag.protocols.soft_fragment_decoder_v2 import SoftAuthenticatedFragmentDecoderV2

def config():
    return yaml.safe_load(Path("configs/blind_crop_threshold_recovery_v1.yaml").read_text())

def test_blind_search_api_excludes_all_oracle_and_expected_inputs():
    names=list(inspect.signature(search.blind_crop_search).parameters)
    for forbidden in ("original","expected","payload","token","crop_coordinates","grid_coordinates","surviving_regions","oracle"):
        assert all(forbidden not in name for name in names)

def test_geometric_search_is_deterministic_and_bounded():
    limits=search.CropSearchLimits(scales=(1.,.5),aspect_ratios=(1.,),offsets=((0.,0.),(.5,.5)),total_search_budget=20)
    image=torch.rand(3,256,256);a=search.geometric_hypotheses(image,limits);b=search.geometric_hypotheses(image,limits)
    assert a.shape==(5,3,64,64) and torch.equal(a,b)

def test_predeclared_limits_and_gates_are_frozen():
    cfg=config();assert cfg["search"]["token_beam_limit"]==256 and cfg["search"]["regional_top_k"]==4
    assert cfg["gates"]=={"clean_authenticated_acceptance":.90,"grid_aware_12_of_16_acceptance":.75,"eligible_blind_crop_acceptance":.75,"insufficient_evidence_rejection":.99,"missing_region_f1":.90,"search_budget_exhaustion":.01,"p95_runtime_ms":1000}

def test_less_than_eight_and_less_than_twelve_fail_closed():
    decoder=SoftAuthenticatedFragmentDecoderV2(field_top_k=1,beam_width=1,search_budget=16)
    assert decoder.decode(torch.randn(7,20),bytes(32),[bytes(8)])["status"]=="insufficient"
    assert decoder.decode(torch.randn(11,20),bytes(32),[bytes(8)])["status"]=="insufficient"

def test_crop_survival_is_oracle_evaluation_only():
    assert diag._survival((0,0,256,256))==[1.]*16
    fractions=diag._survival((0,0,128,128));assert sum(x==1 for x in fractions)==4 and sum(x==0 for x in fractions)==12
    assert "_survival" not in inspect.getsource(search.blind_crop_search)

def test_uncertain_is_not_merged_into_manipulated():
    source=inspect.getsource(search.blind_crop_search)
    assert '"uncertain" if value=="manipulated"' in source

def test_locked_final_is_evaluated_once_after_selection_freezes():
    source=inspect.getsource(diag.run_experiment)
    assert source.index("frozen_selection.json") < source.index("locked_final_started.json")
    assert "locked-final population already evaluated" in source and '"locked_final_evaluations":1' in source

def test_report_boundaries_are_hard_coded():
    source=inspect.getsource(diag.run_experiment)
    assert '"neural_stage_passed":False' in source and '"stage_e_permitted":False' in source
    assert '"larger_beam_used":False' in source and '"novelty_claimed":False' in source

def test_shards_do_not_serialize_secrets_expected_payloads_or_coordinates():
    source=inspect.getsource(diag._evaluate_crops)
    assert '"contains_expected_payload":False' in source and '"contains_secret":False' in source
    assert "crop_coordinates" not in source
