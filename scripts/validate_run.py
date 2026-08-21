"""Validate run structure independently from scientific gate success."""
from __future__ import annotations
import argparse,json
from pathlib import Path
REQUIRED=("summary.json","configuration.yaml","run_manifest.json","environment.json","history.csv","per_bit_metrics.csv","per_region_metrics.csv","attack_metrics.csv","reconstruction_by_survivors.csv","failure_cases.json","best.pt","last.pt","README.txt","evidence_maps")
STATUSES={"passed","blocked_by_prerequisite","implemented_unvalidated","prerequisites_passed_natural_image_unvalidated"}

def validate_run(root: Path) -> dict:
    missing=[name for name in REQUIRED if not (root/name).exists()]; errors=[]; summary={}
    if not missing:
        try: summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc: errors.append(f"invalid summary.json: {exc}")
    status=summary.get("scientific_status")
    if not missing and status not in STATUSES: errors.append("missing or unsupported scientific_status")
    return {"valid":not missing and not errors,"structurally_complete":not missing,"missing":missing,"errors":errors,
            "scientific_status":status,"scientifically_passed":status=="passed"}

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("run"); args=parser.parse_args(); result=validate_run(Path(args.run))
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result["valid"] else 1)
if __name__=="__main__": main()
