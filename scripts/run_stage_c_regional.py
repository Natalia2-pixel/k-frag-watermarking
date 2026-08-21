"""Run the fail-fast Stage-C regional-symbol prerequisite."""
import argparse,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kfrag.diagnostics.stage_c_regional import run_stage_c
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);a=p.parse_args();r=run_stage_c(yaml.safe_load(a.config.read_text(encoding="utf-8")))
print(f"Stage C: {'PASS' if r.get('stage_c_passed') else 'BLOCKED'}; status={r['scientific_status']}")
if "metrics" in r:print(f"bit={r['metrics']['regional_active_bit_accuracy']:.6f} symbol={r['metrics']['exact_regional_symbol_accuracy']:.6f} grid={r['metrics']['exact_16_symbol_grid_accuracy']:.6f}")
print("stage_d_permitted=false")
