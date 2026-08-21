"""Run only the Stage-B V2 analytical-carrier Phase-1 repair."""
import argparse,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kfrag.diagnostics.stage_b_phase1_repair import run_phase1_repair
p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",type=Path,required=True); a=p.parse_args()
r=run_phase1_repair(yaml.safe_load(a.config.read_text(encoding="utf-8")))
print(f"Phase-1 repair: {'PASS' if r['phase1_passed'] else 'BLOCKED'}; blocked_level={r['blocked_level']}")
for name,value in r["sanity_ladder"].items(): print(f"{name}: pass={value.get('passed')} step={value.get('passed_step')} bit={value.get('active_bit_accuracy')} exact={value.get('exact_symbol_accuracy')} margin={value.get('correct_minus_shuffled_margin')}")
print("stage_c_permitted=false")
