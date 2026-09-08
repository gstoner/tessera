#!/usr/bin/env python3
"""Owning-device proof of SSA-bounded dynamic logical buffers and copies."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.scheduled_matmul import run_tessera_opt  # noqa: E402
from tessera.compiler.native_gpu_storage import build_native_gpu_storage  # noqa: E402


def dynamic_source():
    return '''module attributes {tessera.autodiff.product_abi = "test", tessera.autodiff.product_pair = "test"} {
      func.func @dynamic(%out: memref<3x3xf32>) {
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        %three = arith.constant 3 : index
        %value = arith.constant 7.0 : f32
        scf.for %i = %z to %three step %one {
          %n = arith.addi %i, %one : index
          %a = memref.alloc(%n) : memref<?xf32>
          %b = memref.alloc(%n) : memref<?xf32>
          scf.for %j = %z to %n step %one {
            memref.store %value, %a[%j] : memref<?xf32>
          }
          memref.copy %a, %b : memref<?xf32> to memref<?xf32>
          scf.for %j = %z to %n step %one {
            %v = memref.load %b[%j] : memref<?xf32>
            memref.store %v, %out[%i,%j] : memref<3x3xf32>
          }
        }
        return
      }
    }'''


def switch_source():
    return '''module {
      func.func @choose(%x: tensor<4xf32>) -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
        %z = arith.constant 0 : index
        %v = tensor.extract %x[%z] : tensor<4xf32>
        %one = arith.constant 1.0 : f32
        %two = arith.constant 2.0 : f32
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c2 = arith.constant 2 : i32
        %p = arith.cmpf oge, %v, %one : f32
        %q = arith.cmpf oge, %v, %two : f32
        %low = arith.select %p, %c1, %c0 : i32
        %flag = arith.select %q, %c2, %low : i32
        %y = scf.execute_region -> tensor<4xf32> {
          cf.switch %flag : i32, [default: ^cube(%x : tensor<4xf32>),
            0: ^square(%x : tensor<4xf32>), 1: ^twice(%x : tensor<4xf32>)]
        ^square(%a: tensor<4xf32>):
          %aa = "tessera.mul"(%a,%a) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          scf.yield %aa : tensor<4xf32>
        ^twice(%b: tensor<4xf32>):
          %bb = "tessera.add"(%b,%b) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          scf.yield %bb : tensor<4xf32>
        ^cube(%c: tensor<4xf32>):
          %cc = "tessera.mul"(%c,%c) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          %ccc = "tessera.mul"(%cc,%c) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          scf.yield %ccc : tensor<4xf32>
        } {tessera.structured_cfg.max_steps = 2 : i64,
           tessera.structured_cfg.digest = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
        return %y : tensor<4xf32>
      }
    }'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); d=Device(args.backend)
    chip='sm_120' if d.cuda else 'gfx1151'
    rows=[]
    for zero_start in (False,True):
        source=dynamic_source()
        if zero_start:source=source.replace("%n = arith.addi %i, %one", "%n = arith.addi %i, %z")
        gpu=run_tessera_opt(args.compiler,source,'--tessera-native-tape-to-gpu=backend='+args.backend)
        physical=build_native_gpu_storage(gpu,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip)
        array=np.zeros((3,3),np.float32); pointer=ct.c_void_p()
        d.check(d.alloc(ct.byref(pointer),array.nbytes))
        bound=None
        try:
            d.check(d.htod(pointer,array.ctypes.data,array.nbytes) if d.cuda else d.copy(pointer,array.ctypes.data,array.nbytes,1))
            bound=physical.bind()
            bound.launch((pointer.value,1),grid=(1,1,1),block=(1,1,1))
            d.check(d.dtoh(array.ctypes.data,pointer,array.nbytes) if d.cuda else d.copy(array.ctypes.data,pointer,array.nbytes,2))
            np.testing.assert_array_equal(array,7*np.tri(3,k=-1 if zero_start else 0,dtype=np.float32))
        finally:
            if bound is not None:bound.close()
            d.check(d.free(pointer))
        rows.append(dict(result=array.tolist(),logical_widths=[0,1,2] if zero_start else [1,2,3],
                         slot_capacity=2 if zero_start else 3,binding_digest=physical.binding_digest))
    names=['src/transforms/lib/NativeTapeToGPUPass.cpp','python/tessera/compiler/native_gpu_storage.py']
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,
        rows=rows,isolated_iteration_slots=3,compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),promotion_eligible=False),indent=2)+'\n')
    print('Native dynamic widths 0/1/2 and 1/2/3 copied through distinct capacity-bounded slots')


if __name__=='__main__':main()
