import argparse,json
from pathlib import Path
import yaml
from kfrag.diagnostics.stage_d_tag_capacity import run_tag_progression
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/stage_d_tag_capacity_local.yaml");a=p.parse_args();print(json.dumps(run_tag_progression(yaml.safe_load(Path(a.config).read_text())),indent=2))
