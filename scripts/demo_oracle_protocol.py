"""Deterministic Track-1 protocol demonstration; no learned image claims."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from kfrag.protocol import IdentityRegistry, RegisteredIdentity, embed, verify_image

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",default="outputs/oracle_protocol_demo.json")
    args=parser.parse_args(); registry=IdentityRegistry()
    registration=registry.register(b"demo-asset-0001",RegisteredIdentity(bytes(range(12))))
    questioned=embed({"synthetic_image":"checkerboard"},registration,b"demo-runtime-key")
    report={"registration":registration.public_metadata(),"verification":verify_image(questioned,registry,b"demo-runtime-key").to_dict(),
            "disclaimer":"Oracle-channel protocol validation; not learned-image experimental success."}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    colours={"valid":"#3a9d5d","missing":"#c9c9c9","invalid/manipulated":"#d64b4b","unavailable/uncertain":"#e3a52b"}
    cells=[]
    for row,values in enumerate(report["verification"]["evidence_map"]):
        for column,label in enumerate(values):
            cells.append(f'<rect x="{column*100}" y="{row*100}" width="98" height="98" fill="{colours[label]}"/><title>{label}</title>')
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400" viewBox="0 0 400 400">'+''.join(cells)+'</svg>\n'
    path.with_suffix(".svg").write_text(svg,encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
