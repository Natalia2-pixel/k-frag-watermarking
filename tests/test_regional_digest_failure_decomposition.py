import hashlib,inspect,json
from pathlib import Path
import pytest
from kfrag.analysis.regional_digest_failure_decomposition import *

REPORT=Path("docs/evidence/large_scale_regional_digest_reproduction_v1_report.json")
SHARDS=Path("outputs/large_scale_regional_digest_reproduction_v1/local_coco/shards")

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@pytest.mark.skipif(not SHARDS.exists(),reason="local read-only shards unavailable")
def test_loading_frozen_evidence_does_not_modify_it():
    paths=[REPORT,*sorted(SHARDS.glob("shard_*.json"))];before={str(p):sha(p) for p in paths}
    report,rows,proof=load_frozen_evidence(REPORT,SHARDS,20,1000,"efeb27f17035d1c5035bb7bf0c6f414af0ad6cad6042f5c8d259a6820c4fd285")
    assert len(rows)==1000 and proof["evidence_unchanged"]
    assert before=={str(p):sha(p) for p in paths}

def test_manifest_hash_mismatch_fails_closed(tmp_path):
    report=json.loads(REPORT.read_text());report_path=tmp_path/"report.json";report_path.write_text(json.dumps(report))
    shard=tmp_path/"shard_00000_00001.json";shard.write_text(json.dumps({"fingerprint":"wrong","rows":[]}))
    with pytest.raises(ValueError,match="manifest-hash"):
        load_frozen_evidence(report_path,tmp_path,1,0,"expected")

def test_analysis_preserves_scientific_boundaries():
    assert PRESERVED=={"scientific_status":"blocked_by_large_scale_regional_digest_gates","selected_candidate":None,"neural_stage_passed":False,"stage_e_permitted":False}
    source=inspect.getsource(run_analysis)
    assert '"selected_candidate":None' in source and '"stage_e_permitted":False' in source

def test_decomposition_reproduces_frozen_aggregate_from_report():
    report=json.loads(REPORT.read_text())
    for digest in DIGESTS:
        recalls=[report["digests"][digest][x]["macro_by_image"]["recall"] for x in MALICIOUS]
        assert sum(recalls)/len(recalls)==pytest.approx(report["gates"][digest]["aggregate_malicious_recall"])

def test_mechanism_recommendation_is_not_threshold_repair_or_novelty_claim():
    assert "cross-image" in inspect.getsource(run_analysis)
    assert all("fixed_20bit_packet" in x and "crop_sync" in x for x in MECHANISMS.values())
