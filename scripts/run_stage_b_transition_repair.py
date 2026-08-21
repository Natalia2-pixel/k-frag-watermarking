"""Run the gated Stage-B V2 Phase-2/3 transition repair."""
import argparse,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kfrag.diagnostics.stage_b_transition_repair import run_transition_repair
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);a=p.parse_args();r=run_transition_repair(yaml.safe_load(a.config.read_text(encoding="utf-8")))
print(f"Transition repair: {'PASS' if r['stage_b_v2_passed'] else 'BLOCKED'}; first_pretraining_collapse={r['first_pretraining_collapse_weight']}")
m=r["learned_only_metrics"];print(f"learned-only bit={m['fresh_active_bit_accuracy']:.6f} exact={m['fresh_exact_symbol_accuracy']:.6f} margin={m['correct_minus_shuffled_margin']:.6f}");print(f"stage_c_permitted={str(r['stage_c_permitted']).lower()}")
