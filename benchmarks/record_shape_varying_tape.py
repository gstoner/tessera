#!/usr/bin/env python3
"""Native x86 execution of shrinking checkpoint payloads and logical shape tapes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from tessera.compiler.scheduled_matmul import run_tessera_opt  # noqa: E402
from tessera.compiler.native_persistent_tape import _attribute  # noqa: E402
from tessera import _jit_boundary as jit  # noqa: E402


def source():
    return '''module {
      func.func @shrink(%x: tensor<?xf32>) -> tensor<?xf32> attributes {tessera.autodiff = "reverse"} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %three = arith.constant 3 : index
        %y = scf.for %i = %zero to %three step %one iter_args(%state = %x) -> tensor<?xf32> {
          %n = tensor.dim %state, %zero : tensor<?xf32>
          %m = arith.subi %n, %one : index
          %slice = tensor.extract_slice %state[0][%m][1] : tensor<?xf32> to tensor<?xf32>
          %next = "tessera.mul"(%slice,%slice) : (tensor<?xf32>,tensor<?xf32>) -> tensor<?xf32>
          scf.yield %next : tensor<?xf32>
        } {tessera.autodiff.checkpoint_policy = "save", tessera.autodiff.checkpoint_indices = array<i64: 1, 2>,
           tessera.autodiff.saved_slot_shape_envelope_indices = array<i64: 0>,
           tessera.autodiff.saved_slot_shape_envelope_ranks = array<i64: 1>,
           tessera.autodiff.saved_slot_shape_envelope_bounds = array<i64: 16>}
        return %y : tensor<?xf32>
      }
    }'''


def record(compiler):
    products=[];handles=[];rows=[]
    try:
        for role in ('forward','backward'):
            text=run_tessera_opt(compiler,source(),'--tessera-autodiff-paired=box-product-scalars=true export-product='+role)
            products.append(text)
            handles.append(jit.compile_module(text))
        f,b=[json.loads(_attribute(text,'tessera.autodiff.product_abi')) for text in products]
        assert f['results']==['tensor<?xf32>','tensor<2x16xf32>','tensor<2x1xi64>']
        assert b['inputs']==f['inputs']+f['results']
        for width in (4,8,16):
            x=np.linspace(.2,.8,width,dtype=np.float32)
            primal=np.empty(width-3,np.float32)
            payload=np.full((2,16),np.nan,np.float32)
            shapes=np.full((2,1),-1,np.int64)
            jit.invoke(handles[0],f['entry'],[x],[primal,payload,shapes])
            np.testing.assert_allclose(primal,x[:-3]**8,rtol=1e-5,atol=1e-7)
            np.testing.assert_array_equal(shapes[:,0],[width-1,width-2])
            np.testing.assert_allclose(payload[0,:width-1],x[:-1]**2,rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(payload[1,:width-2],x[:-2]**4,rtol=1e-6,atol=1e-7)
            old_payload=payload.copy();old_shapes=shapes.copy()
            for factor in (1,2):
                dx=np.empty_like(x)
                jit.invoke(handles[1],b['entry'],[x,np.full_like(primal,factor),payload,shapes],dx)
                ref=np.zeros_like(x);ref[:-3]=8*factor*x[:-3]**7
                np.testing.assert_allclose(dx,ref,rtol=1e-5,atol=1e-7)
                np.testing.assert_array_equal(shapes,old_shapes)
                np.testing.assert_array_equal(payload,old_payload)
            rows.append(dict(input_width=width,output_width=width-3,saved_widths=shapes[:,0].tolist(),repeated_backward=True))
    finally:
        for handle in handles:jit.destroy(handle)
    return dict(execution_kind='native_cpu',rows=rows,
        source_sha256=hashlib.sha256(source().encode()).hexdigest(),
        compiler_sha256=hashlib.sha256(compiler.read_bytes()).hexdigest(),
        jit_sha256=hashlib.sha256(Path(jit._load()._name).read_bytes()).hexdigest(),
        product_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in products],
        sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'tools/tessera-jit/tessera_jit.cpp','python/tessera/_jit_boundary.py',
            'src/transforms/lib/TesseraToLinalgPass.cpp','src/transforms/lib/AutodiffPairedPass.cpp')},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),promotion_eligible=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(record(args.compiler),indent=2)+'\n')
    print('Three shrinking native x86 tape cases passed')
