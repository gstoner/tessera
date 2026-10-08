"""Reproducible native independent-prefix reverse package census."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from tests.unit.test_native_independent_scale_batch import source
from tessera.compiler.native_scaled_program import package_native_scaled_vjp

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for mask in range(1,16):
        for ta,tb in ((False,False),(False,True),(True,False),(True,True)):
            prefixes=tuple((2,3) if mask & (1<<i) else () for i in range(4))
            rows.append(dict(mask=mask,prefixes=prefixes,output_prefix=(2,3),
                transposeA=ta,transposeB=tb,
                package=package_native_scaled_vjp(source(prefixes,ta,tb)).to_manifest()))
    for ta,tb in ((False,False),(False,True),(True,False),(True,True)):
        for prefixes,output in [(((2,1),(3,),(),(1,3)),(2,3)),(((),(),(),()),())]:
            rows.append(dict(mask="singleton" if output else "unbatched",
                prefixes=prefixes,output_prefix=output,transposeA=ta,transposeB=tb,
                package=package_native_scaled_vjp(source(prefixes,ta,tb,output)).to_manifest()))
    compiler=Path(os.environ["TESSERA_OPT"])
    payload={"compiler_sha256":hashlib.sha256(compiler.read_bytes()).hexdigest(),
             "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "rows":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload)+"\n")
    print("Exported",len(rows),"native two-member reverse packages")
if __name__=="__main__":main()
