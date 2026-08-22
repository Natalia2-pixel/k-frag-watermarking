import argparse,yaml
from kfrag.analysis.regional_digest_failure_decomposition import run_analysis
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    with open(args.config,encoding="utf-8") as handle:config=yaml.safe_load(handle)
    report=run_analysis(config);print(report["scientific_status"])
if __name__=="__main__":main()
