#!/usr/bin/env python3
"""Owning-device native GPU arbiter admission, fusion and budget exclusion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_native_ann_execution import source  # noqa: E402
from tessera.compiler.native_ann import prepare_native_ann, _affine, _exact_output  # noqa: E402
from tessera.compiler.native_ann_gpu import materialize_native_ann_gpu, NativeANNDeviceRegistration, ANN_GPU  # noqa: E402
from tessera.compiler.emit.candidate import arbitrate, ArbiterError, candidates_for  # noqa: E402


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('rocm','nvidia'),required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parallel-rows',action='store_true')
    args=p.parse_args();Device(args.backend)
    os.environ['TESSERA_OPT']=str(args.compiler.resolve())
    chip='sm_120' if args.backend=='nvidia' else 'gfx1151'
    rows=[]
    for activation in ('relu','abs','sum'):
        text=source()
        if activation=='abs':text=text.replace('%r = "tessera.relu"(%o) : (tensor<3x2xf32>) -> tensor<3x2xf32>','%r = math.absf %o : tensor<3x2xf32>')
        if activation=='sum':
            text=text.replace(') -> tensor<3x2xf32> {', ') -> tensor<3xf32> {', 1).replace(
                '%r = "tessera.relu"(%o) : (tensor<3x2xf32>) -> tensor<3x2xf32>',
                '%r = "tessera.reduce"(%o) {axis = 1 : i64, kind = "sum"} : (tensor<3x2xf32>) -> tensor<3xf32>').replace('return %r : tensor<3x2xf32>', 'return %r : tensor<3xf32>')
        logical=prepare_native_ann(text,allow_reassociation=True)
        physical=materialize_native_ann_gpu(logical,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),
                                            backend=args.backend,chip=chip,fuse_elementwise=True,parallel_rows=args.parallel_rows)
        samples=[np.ones((3,2),np.float32),-np.ones((3,2),np.float32),np.array([[-1,1],[.5,-.5],[1,-1]],np.float32)]
        for budget in (0.0,.001):
            with NativeANNDeviceRegistration(physical,samples,input_bound=1.0,absolute_budget=budget) as registered:
                region=registered.region
                original=arbitrate(region,ANN_GPU,args.backend)
                assert original is registered.candidates[0]
                for value in samples:
                    actual,tag=original.run(region,value)
                    assert tag=='native_gpu'
                    np.testing.assert_allclose(actual,np.asarray(_exact_output(_affine(logical.original),value),np.float32),rtol=1e-6,atol=1e-5)
                if budget:
                    rewrite=arbitrate(region,ANN_GPU,args.backend,force=registered.candidates[1].name)
                    assert rewrite is registered.candidates[1]
                else:
                    try:arbitrate(region,ANN_GPU,args.backend,force=registered.candidates[1].name)
                    except ArbiterError:pass
                    else:raise AssertionError('over-budget rewrite was admitted')
                rows.append(dict(activation=activation,budget=budget,region=region.digest,
                                 original=physical.original.binding_digest,transformed=physical.transformed.binding_digest,
                                 incumbent_retained=True,rewrite_admitted=bool(budget)))
            assert not candidates_for(args.backend,ANN_GPU)
    names=['python/tessera/compiler/native_ann.py','python/tessera/compiler/native_ann_gpu.py',
           'src/transforms/lib/AutodiffPairedPass.cpp','src/transforms/lib/NativeTapeToGPUPass.cpp']
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,parallel_rows=args.parallel_rows,rows=rows,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),promotion_eligible=False),indent=2)+'\n')
    print('Six native GPU arbiter cases passed')


if __name__=='__main__':main()
