#!/usr/bin/env python3
"""Native LLVM execution proof for the bounded Python source-CFG producer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from tessera.compiler.trace import trace  # noqa: E402
from tessera.compiler.source_control_flow import to_native_source_ir  # noqa: E402
from tessera import _jit_boundary as jit  # noqa: E402


def branch(x,y):
    if x < y:
        if x < x-y:
            return x*x
        x = x+y
    else:
        x = x-y
    return x*y


def loop(x,step,limit):
    while x < limit:
        x = x+step
    return x*x



def mixed_edges(x, step, limit):
    total = x - x
    while x < limit:
        x = x + step
        if x < step:
            continue
        total = total + x
        if total > limit:
            break
    assert total >= x - x, "negative accumulated value"
    return total * total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for fn,values in [(branch,(-3.,2.)),(branch,(-3.,-2.)),(branch,(-1.,-2.)),(branch,(3.,2.)),
                      (loop,(0.,1.,3.)),(loop,(3.,1.,3.)),(loop,(-2.,.5,1.)),(mixed_edges,(-2.,1.,4.)),(mixed_edges,(0.,1.,3.)),(mixed_edges,(5.,1.,3.))]:
        arrays=[np.array([v],np.float32) for v in values]
        native=to_native_source_ir(trace(fn,*arrays,source_control_flow=True,max_steps=8))
        handle=jit.compile_module(native)
        try:
            result=np.empty((1,),np.float32)
            jit.invoke(handle,'source_program',arrays,result)
            np.testing.assert_allclose(result,fn(*arrays))
            rows.append(dict(family=fn.__name__,inputs=values,result=result.tolist(),
                             native_ir_sha256=hashlib.sha256(native.encode()).hexdigest()))
        finally:jit.destroy(handle)
    names=['python/tessera/compiler/trace.py','python/tessera/compiler/source_control_flow.py',
           'benchmarks/record_source_cfg_native.py','python/tessera/compiler/graph_ir.py']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    library=Path(os.environ['TESSERA_JIT_LIB'])
    args.output.write_text(json.dumps(dict(execution_kind='native_cpu',rows=rows,
        jit_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        promotion_eligible=False),indent=2)+'\n')


if __name__=='__main__': main()
