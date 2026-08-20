import argparse,json
from kfrag.diagnostics.prerequisites_v3 import run_prerequisites
p=argparse.ArgumentParser(); p.add_argument("--output",default="outputs/learned_channel_v3/prerequisites.json")
a=p.parse_args(); result=run_prerequisites(a.output); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["passed"] else 2)
