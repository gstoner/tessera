#!/usr/bin/env python3
"""Exact-device split nested SAVE products with retained residual allocations."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402


def source(inner=3,outer=2,width=4):
    return f'''module {{
  func.func @nested(%x: tensor<{width}xf32>, %w: tensor<{width}xf32>) -> tensor<{width}xf32>
    attributes {{tessera.autodiff = "reverse"}} {{
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %outer = arith.constant {outer} : index
    %inner = arith.constant {inner} : index
    %out = scf.for %i = %zero to %outer step %one iter_args(%state = %x) -> tensor<{width}xf32> {{
      %in = scf.for %j = %zero to %inner step %one iter_args(%carry = %state) -> tensor<{width}xf32> {{
        %next = "tessera.mul"(%carry,%w) : (tensor<{width}xf32>,tensor<{width}xf32>) -> tensor<{width}xf32>
        scf.yield %next : tensor<{width}xf32>
      }} {{tessera.autodiff.checkpoint_policy = "save", tessera.autodiff.checkpoint_indices = array<i64: 1, 2>}}
      scf.yield %in : tensor<{width}xf32>
    }} {{tessera.autodiff.checkpoint_policy = "save", tessera.autodiff.checkpoint_indices = array<i64: 1>}}
    return %out : tensor<{width}xf32>
  }}
}}'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=Device(args.backend)
    rows=[]
    for width in (4,8,16):
        pair=materialize_persistent_tape(source(width=width),compiler=args.compiler,
            llvm_bin='/usr/lib/llvm-23/bin',backend=args.backend,chip='sm_120' if device.cuda else 'gfx1151')
        pointers=[]
        def write(pointer,value):
            device.check(device.htod(pointer,value.ctypes.data,value.nbytes) if device.cuda else device.copy(pointer,value.ctypes.data,value.nbytes,1))
        def upload(value):
            p=ct.c_void_p(); device.check(device.alloc(ct.byref(p),value.nbytes)); pointers.append(p)
            write(p,value)
            return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,typestr=value.dtype.str,data=(p.value,False)))
        def download(value):
            spec=value.__cuda_array_interface__
            result=np.empty(spec['shape'],np.float32); p=ct.c_void_p(spec['data'][0])
            device.check(device.dtoh(result.ctypes.data,p,result.nbytes) if device.cuda else device.copy(result.ctypes.data,p,result.nbytes,2))
            return result
        frame=None
        try:
            x=np.linspace(.1,.7,width,dtype=np.float32); w=np.linspace(.8,1.1,width,dtype=np.float32)
            frame=pair.capture(upload(x),upload(w))
            np.testing.assert_allclose(download(frame.primals[0]),x*w**6,atol=3e-6,rtol=3e-6)
            residual=download(frame.residuals[0])
            np.testing.assert_allclose(residual[0],x*w**3,atol=3e-6,rtol=3e-6)
            for pointer,value in zip(pointers,(x,w),strict=True): write(pointer,np.full_like(value,19))
            for factor in (1,2):
                dx,dw=frame.backward(upload(np.full_like(x,factor)))
                np.testing.assert_allclose(download(dx),factor*w**6,atol=3e-6,rtol=3e-6)
                np.testing.assert_allclose(download(dw),factor*6*x*w**5,atol=3e-6,rtol=3e-6)
                np.testing.assert_array_equal(download(frame.residuals[0]),residual)
            # A controlled device-side mutation proves backward reads the
            # retained residual rather than silently rerunning the outer forward.
            write(ct.c_void_p(frame.residuals[0].__cuda_array_interface__['data'][0]),np.zeros_like(residual))
            dx,dw=frame.backward(upload(np.ones_like(x)))
            np.testing.assert_allclose(download(dw),3*x*w**5,atol=3e-6,rtol=3e-6)
            frame.close()
            try: dx.__cuda_array_interface__
            except ValueError: pass
            else: raise AssertionError('closed persistent tape exposed a freed buffer')
            rows.append(dict(width=width,lineage=pair.lineage_digest,forward=pair.forward.binding_digest,
                             backward=pair.backward.binding_digest,repeated_backward=True,residual_mutation_control=True))
        finally:
            if frame is not None: frame.close()
            for pointer in pointers: device.check(device.free(pointer))
    args.output.write_text(json.dumps(dict(backend=args.backend,rows=rows,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'src/transforms/lib/NativeTapeToGPUPass.cpp', 'src/transforms/lib/AutodiffPairedPass.cpp',
            'python/tessera/compiler/native_persistent_tape.py', 'python/tessera/compiler/native_gpu_storage.py',
            'python/tessera/compiler/native_gpu_tensor.py', 'python/tessera/compiler/native_storage_contract.py')},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(len(rows),'persistent split tape cases passed')

if __name__=='__main__': main()
