import json
from pathlib import Path
from scripts.validate_run import REQUIRED, validate_run

def test_failed_but_complete_run_is_structurally_valid(tmp_path: Path):
    for name in REQUIRED:
        path=tmp_path/name
        if "." not in name: path.mkdir()
        elif name=="summary.json": path.write_text(json.dumps({"scientific_status":"blocked_by_prerequisite"}))
        else: path.write_bytes(b"placeholder")
    result=validate_run(tmp_path)
    assert result["valid"] and result["structurally_complete"]
    assert not result["scientifically_passed"] and result["scientific_status"]=="blocked_by_prerequisite"
