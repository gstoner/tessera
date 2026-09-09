#!/usr/bin/env python3
"""One independent native ANN package timing run, including its host bridge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_ann import prepare_native_ann  # noqa: E402
from tessera.compiler.native_ann_gpu import materialize_native_ann_gpu  # noqa: E402


def _relu_source(rows=3,width=2):
    if type(rows) is not int or type(width) is not int or not 2 <= rows <= 64 or not 2 <= width <= 8:
        raise ValueError('ANN workload requires 2..64 rows and 2..8 features')
    if (rows,width)!=(3,2):
        shape=f'{rows}x{width}'
        matrix=str([[1.0 if i==j else .125 for j in range(width)] for i in range(width)])
        return f'''module {{ func.func @ann(%x: tensor<{shape}xf32>) -> tensor<{shape}xf32> {{
          %w = arith.constant dense<{matrix}> : tensor<{width}x{width}xf32>
          %b = arith.constant dense<0.125> : tensor<{shape}xf32>
          %m = "tessera.matmul"(%x,%w) : (tensor<{shape}xf32>,tensor<{width}x{width}xf32>) -> tensor<{shape}xf32>
          %a = "tessera.add"(%m,%b) : (tensor<{shape}xf32>,tensor<{shape}xf32>) -> tensor<{shape}xf32>
          %n = "tessera.matmul"(%a,%w) : (tensor<{shape}xf32>,tensor<{width}x{width}xf32>) -> tensor<{shape}xf32>
          %o = "tessera.add"(%n,%b) : (tensor<{shape}xf32>,tensor<{shape}xf32>) -> tensor<{shape}xf32>
          %r = "tessera.relu"(%o) : (tensor<{shape}xf32>) -> tensor<{shape}xf32>
          return %r : tensor<{shape}xf32>
        }} }}'''
    return '''module {
      func.func @ann(%x: tensor<3x2xf32>) -> tensor<3x2xf32> {
        %w1 = arith.constant dense<[[2.0, 1.0], [-1.0, 2.0]]> : tensor<2x2xf32>
        %w2 = arith.constant dense<[[3.0, -1.0], [2.0, 1.0]]> : tensor<2x2xf32>
        %b1 = arith.constant dense<1.0> : tensor<3x2xf32>
        %b2 = arith.constant dense<4.0> : tensor<3x2xf32>
        %m = "tessera.matmul"(%x,%w1) : (tensor<3x2xf32>,tensor<2x2xf32>) -> tensor<3x2xf32>
        %a = "tessera.add"(%m,%b1) : (tensor<3x2xf32>,tensor<3x2xf32>) -> tensor<3x2xf32>
        %n = "tessera.matmul"(%a,%w2) : (tensor<3x2xf32>,tensor<2x2xf32>) -> tensor<3x2xf32>
        %o = "tessera.add"(%n,%b2) : (tensor<3x2xf32>,tensor<3x2xf32>) -> tensor<3x2xf32>
        %r = "tessera.relu"(%o) : (tensor<3x2xf32>) -> tensor<3x2xf32>
        return %r : tensor<3x2xf32>
      }
    }'''


def source(rows=3,width=2,activation='relu'):
    if activation not in ('relu','abs','square'):
        raise ValueError('ANN measured activation requires relu, abs or square')
    text=_relu_source(rows,width)
    if activation in ('abs', 'square'):
        shape=f'{rows}x{width}'
        text=text.replace(f'"tessera.relu"(%o) : (tensor<{shape}xf32>) -> tensor<{shape}xf32>',
                          (f'math.absf %o : tensor<{shape}xf32>' if activation == 'abs' else
                           f'"tessera.mul"(%o,%o) : (tensor<{shape}xf32>,tensor<{shape}xf32>) -> tensor<{shape}xf32>'))
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--fuse-elementwise',action='store_true')
    p.add_argument('--parallel-rows',action='store_true')
    p.add_argument('--rows',type=int,default=3)
    p.add_argument('--width',type=int,default=2)
    p.add_argument('--tune-transformed',action='store_true')
    p.add_argument('--activation',choices=('relu','abs','square'),default='relu')
    args=p.parse_args()
    Device(args.backend)
    chip='sm_120' if args.backend=='nvidia' else 'gfx1151'
    os.environ['TESSERA_OPT']=str(args.compiler.resolve())
    pair=prepare_native_ann(source(args.rows,args.width,args.activation),allow_reassociation=True)
    physical=materialize_native_ann_gpu(pair,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,fuse_elementwise=args.fuse_elementwise,parallel_rows=args.parallel_rows,tune_transformed=args.tune_transformed)
    shape=(args.rows,args.width)
    values=[np.zeros(shape,np.float32),np.ones(shape,np.float32),-np.ones(shape,np.float32),
            np.linspace(-1,1,args.rows*args.width,dtype=np.float32).reshape(shape)]
    medians=[]
    with physical.bind(input_bound=1.0,absolute_budget=.001) as runner:
        assert runner.verify(values)
        for _ in range(5):
            runner.run(values[3]);runner.run(values[3],transformed=True)
        samples=[[],[]]
        order=np.random.default_rng(args.seed).integers(0,2,size=31)
        for first in order:
            for variant in (int(first),1-int(first)):
                start=time.perf_counter_ns()
                runner.run(values[3],transformed=bool(variant))
                samples[variant].append((time.perf_counter_ns()-start)/1e6)
        medians=[statistics.median(v) for v in samples]
        assert runner.verify(values)
        bounds=[str(v) for v in runner.bounds]
    names=['python/tessera/compiler/native_ann.py','python/tessera/compiler/native_ann_gpu.py',
           'python/tessera/compiler/native_gpu_storage.py','src/transforms/lib/NativeTapeToGPUPass.cpp']
    args.output.write_text(json.dumps(dict(schema=1,rows=args.rows,width=args.width,activation=args.activation,tune_transformed=args.tune_transformed,fuse_elementwise=args.fuse_elementwise,parallel_rows=args.parallel_rows,backend=args.backend,chip=chip,pid=os.getpid(),seed=args.seed,
        pair=pair.digest,original=physical.original.binding_digest,transformed=physical.transformed.binding_digest,
        input_bound=1.0,absolute_budget=.001,bounds=bounds,numerical_verified=True,
        timing_domain='warm_package_h2d_dispatch_d2h_host_wall',samples_ms=samples,medians_ms=medians,
        speedup=medians[0]/medians[1],promotion_eligible=False,
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(args.backend,'ANN native oracle passed; package speedup',medians[0]/medians[1])


if __name__=='__main__':main()
