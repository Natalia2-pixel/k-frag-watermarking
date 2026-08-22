import argparse,yaml
from kfrag.diagnostics.real_image_regional_digest import run_real_image_digest
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    with open(args.config,encoding="utf-8") as handle:config=yaml.safe_load(handle)
    print(run_real_image_digest(config)["scientific_status"])
if __name__=="__main__":main()

