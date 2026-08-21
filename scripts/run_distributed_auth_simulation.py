import argparse,json
from pathlib import Path
from kfrag.protocols.distributed_auth_v2 import IndependentMAC,DistributedGlobalMAC,JointFragmentCode,construction_comparison,monte_carlo
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--trials",type=int,default=10000);p.add_argument("--output",default="outputs/protocol/distributed_auth_v2/report.json");a=p.parse_args();constructions=[IndependentMAC(8),IndependentMAC(12),IndependentMAC(16),DistributedGlobalMAC(64),DistributedGlobalMAC(128),JointFragmentCode()];report={"comparison":construction_comparison(),"monte_carlo_trials":a.trials,"simulations":{f"{type(c).__name__}_{getattr(c,'bits',getattr(c,'tag_bits','joint'))}":monte_carlo(c,a.trials,101+i) for i,c in enumerate(constructions)}};path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
