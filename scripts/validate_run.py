import argparse,json
from pathlib import Path
REQUIRED=("summary.json","configuration.yaml","run_manifest.json","environment.json","history.csv","per_bit_metrics.csv","per_region_metrics.csv","attack_metrics.csv","reconstruction_by_survivors.csv","failure_cases.json","best.pt","last.pt","README.txt","evidence_maps")
p=argparse.ArgumentParser(); p.add_argument("run"); a=p.parse_args(); root=Path(a.run); missing=[x for x in REQUIRED if not (root/x).exists()]; summary=json.loads((root/"summary.json").read_text()) if not missing else {}; print(json.dumps({"valid":not missing,"missing":missing,"scientific_status":summary.get("scientific_status")},indent=2)); raise SystemExit(bool(missing))
