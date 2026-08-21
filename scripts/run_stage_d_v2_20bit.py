import argparse, yaml
from kfrag.diagnostics.stage_d_v2_20bit import run_stage_d_v2_20bit

parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
with open(args.config,encoding="utf-8") as handle: config=yaml.safe_load(handle)
report=run_stage_d_v2_20bit(config); print(report["scientific_status"])

