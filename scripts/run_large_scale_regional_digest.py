import argparse,yaml
from kfrag.diagnostics.large_scale_regional_digest import run_large_scale_reproduction
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    with open(args.config,encoding="utf-8") as handle:config=yaml.safe_load(handle)
    print(run_large_scale_reproduction(config)["scientific_status"])
if __name__=="__main__":main()
