"""Run the isolated Stage-B V2 local prerequisite."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kfrag.diagnostics.stage_b_natural_v2 import run_stage_b_v2
parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); args=parser.parse_args()
report=run_stage_b_v2(yaml.safe_load(args.config.read_text(encoding="utf-8")))
m=report["metrics"]; print(f"Stage-B V2: {'PASS' if report['stage_c_permitted'] else 'BLOCKED'}")
print(f"fresh bit={m['fresh_active_bit_accuracy']:.6f} exact={m['fresh_exact_symbol_accuracy']:.6f} margin={m['correct_minus_shuffled_margin']:.6f}")
print(f"stage_c_permitted={str(report['stage_c_permitted']).lower()}")
