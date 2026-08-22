import argparse,yaml
from kfrag.diagnostics.blind_crop_threshold_recovery import run_experiment
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    with open(args.config,encoding="utf-8") as handle:config=yaml.safe_load(handle)
    print(run_experiment(config)["scientific_status"])
if __name__=="__main__":main()
