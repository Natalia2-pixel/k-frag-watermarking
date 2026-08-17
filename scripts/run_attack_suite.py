"""Attack evaluation entry point; refuses absent/incompatible checkpoints."""
import argparse,json
from pathlib import Path
from kfrag.checkpoints.provenance import load_checkpoint
p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--output",required=True); a=p.parse_args(); output=Path(a.output)
if output.exists(): raise FileExistsError(output)
state=load_checkpoint(a.checkpoint)
if not state.get("gate_results",{}).get("passed"): raise RuntimeError("prerequisite communication gate failed; attack suite blocked")
output.mkdir(parents=True); (output/"summary.json").write_text(json.dumps({"scientific_status":"implemented_unvalidated","note":"configure images and runtime key for full attack evaluation"},indent=2)+"\n")
