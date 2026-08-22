import argparse,json
from pathlib import Path
from kfrag.protocols.content_bound_fragment_state_v1 import simulate_content_binding
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--trials",type=int,default=4000);parser.add_argument("--calibration-trials",type=int,default=512);parser.add_argument("--seed",type=int,default=404);parser.add_argument("--output",default="outputs/protocol/content_bound_fragment_state_v1/report.json");args=parser.parse_args();report=simulate_content_binding(args.trials,args.calibration_trials,seed=args.seed);path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n");print(path)
if __name__=="__main__":main()

