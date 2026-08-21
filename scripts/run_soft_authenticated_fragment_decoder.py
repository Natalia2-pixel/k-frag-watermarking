import argparse,yaml
from kfrag.diagnostics.soft_authenticated_fragment_decoder import run_soft_decoder_experiment
parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
with open(args.config,encoding="utf-8") as handle:config=yaml.safe_load(handle)
print(run_soft_decoder_experiment(config)["scientific_status"])

