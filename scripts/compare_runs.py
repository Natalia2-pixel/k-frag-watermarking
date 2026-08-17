import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("runs",nargs="+"); a=p.parse_args(); rows=[]
for run in a.runs:
 s=json.loads((Path(run)/"summary.json").read_text()); rows.append({"run":Path(run).as_posix(),"status":s.get("scientific_status"),"metrics":s.get("metrics"),"gates":s.get("gates")})
print(json.dumps(rows,indent=2))
