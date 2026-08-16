"""Run Stage B on one fixed COCO validation image."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kfrag.diagnostics.stage_b_natural import run_stage_b

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
args = parser.parse_args()
with args.config.open(encoding="utf-8") as handle:
    report = run_stage_b(yaml.safe_load(handle))
na = lambda x: "N/A" if x is None else x
fresh = report["evaluations"]["fresh_on_the_fly_payloads"]
print(f"Stage B: {'PASS' if report['passed'] else 'FAIL'}")
print(f"fresh active accuracy={fresh['active_bit_accuracy']:.6f} exact symbol={fresh['exact_8bit_regional_symbol_accuracy']:.6f}")
for key, value in report["inactive_metrics"].items(): print(f"{key}: {na(value)}")
print("Stage-C progression is manual.")
