"""Run the separate Stage-C fidelity/saturation repair."""
import argparse,sys
from pathlib import Path
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kfrag.diagnostics.stage_c_fidelity_repair import run_fidelity_repair
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);a=p.parse_args();r=run_fidelity_repair(yaml.safe_load(a.config.read_text()));m=r.get("metrics",{})
print(f"Stage-C fidelity repair: {'PASS' if r.get('stage_c_passed') else 'BLOCKED'}; selected_amplitude={r.get('selected_amplitude')}")
if m:print(f"bit={m['regional_active_bit_accuracy']:.6f} symbol={m['exact_regional_symbol_accuracy']:.6f} PSNR={m['psnr']:.4f} saturation={m['residual_saturation_fraction']:.8f}")
print("stage_d_permitted=false")
