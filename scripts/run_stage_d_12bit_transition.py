import argparse,json
from pathlib import Path
import yaml
from kfrag.diagnostics.stage_d_12bit_transition import run_transition_repair
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/stage_d_12bit_transition_local.yaml");a=p.parse_args();print(json.dumps(run_transition_repair(yaml.safe_load(Path(a.config).read_text())),indent=2))
