from __future__ import annotations
import argparse,json,os
from pathlib import Path
import torch
from kfrag.checkpoints.provenance import load_checkpoint
from kfrag.models import KFragSystem
from kfrag.protocol.authentication import runtime_key
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--image",required=True); p.add_argument("--namespace",required=True); a=p.parse_args(); state=load_checkpoint(a.checkpoint); model=KFragSystem(width=int(state["configuration"].get("width",16)),residual_alpha=state["residual_alpha"],preprocessing=state["configuration"].get("preprocessing","rgb_plus_high_pass")); model.load_state_dict(state["model_states"]["kfrag"]); from PIL import Image; import numpy as np; im=Image.open(a.image).convert("RGB").resize((256,256)); x=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float().unsqueeze(0)/255; from kfrag.evaluation.evaluator import evaluate_questioned; print(json.dumps(evaluate_questioned(model,x,runtime_key(),a.namespace.encode()),indent=2))
if __name__=="__main__": main()
