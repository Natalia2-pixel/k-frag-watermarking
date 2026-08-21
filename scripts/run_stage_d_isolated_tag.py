import argparse,json
from pathlib import Path
import yaml
from kfrag.diagnostics.stage_d_isolated_tag import run_isolated_tag_experiment
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/stage_d_isolated_tag_local.yaml");a=p.parse_args();print(json.dumps(run_isolated_tag_experiment(yaml.safe_load(Path(a.config).read_text())),indent=2))
