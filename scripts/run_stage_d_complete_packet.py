"""Run the Stage-D complete regional-packet pilot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from kfrag.diagnostics.stage_d_complete_packet import run_stage_d

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/stage_d_complete_packet_local.yaml");args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"));print(json.dumps(run_stage_d(config),indent=2))
if __name__=="__main__":main()
