import argparse,json
from pathlib import Path
from kfrag.data.manifests import create_manifests
p=argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("output"); a=p.parse_args(); out=Path(a.output)
if out.exists(): raise FileExistsError(out)
out.write_text(json.dumps(create_manifests(a.root),indent=2)+"\n")
