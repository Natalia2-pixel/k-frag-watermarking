import argparse,json
from pathlib import Path
import yaml
from kfrag.diagnostics.stage_d_12bit_transition import run_transition_repair
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/stage_d_12bit_transition_local.yaml")
    p.add_argument("--stage-c-checkpoint");p.add_argument("--stage-c-report");p.add_argument("--stage-b-checkpoint");p.add_argument("--data-root");p.add_argument("--output-directory");a=p.parse_args();config=yaml.safe_load(Path(a.config).read_text())
    for name in ("stage_c_checkpoint","stage_c_report","stage_b_checkpoint","data_root","output_directory"):
        value=getattr(a,name);config[name]=value if value is not None else config.get(name)
    print(json.dumps(run_transition_repair(config),indent=2))
