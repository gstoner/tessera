#!/usr/bin/env python3
"""Data-dependent native checkpoint exits and asynchronous derivative generations."""
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
from benchmarks.record_tape_checkpoint_execution import counted_while_source  # noqa: E402
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402


def data_while_source():
    return counted_while_source().replace(
        '%continue = arith.cmpi slt, %i, %three : index',
        '%bounded = arith.cmpi slt, %i, %three : index\n'
        '%v = tensor.extract %state[%zero] : tensor<4xf32>\n'
        '%limit = arith.constant 0.5 : f32\n'
        '%active = arith.cmpf olt, %v, %limit : f32\n'
        '%continue = arith.andi %bounded, %active : i1')


def predicate_source():
    return '''module {
      func.func @branch(%x: tensor<4xf32>) -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
        %z = arith.constant 0 : index
        %v = tensor.extract %x[%z] : tensor<4xf32>
        %zero = arith.constant 0.0 : f32
        %p = arith.cmpf ogt, %v, %zero : f32
        %y = scf.if %p -> tensor<4xf32> {
          %a = "tessera.mul"(%x,%x) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          scf.yield %a : tensor<4xf32>
        } else { scf.yield %x : tensor<4xf32> }
        return %y : tensor<4xf32>
      }
    }'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--strided-counter',action='store_true')
    args=p.parse_args()
    d=Device(args.backend)
    chip='sm_120' if d.cuda else 'gfx1151'
    loop_source=data_while_source()
    if args.strided_counter:
        loop_source=loop_source.replace('%one = arith.constant 1 : index',
            '%one = arith.constant 2 : index\n%start = arith.constant 2 : index').replace(
            '"scf.while"(%zero, %x)', '"scf.while"(%start, %x)').replace(
            '%three = arith.constant 3 : index', '%three = arith.constant 8 : index')
    pair=materialize_persistent_tape(loop_source,compiler=args.compiler,
        llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip)
    branch_pair=materialize_persistent_tape(predicate_source(),compiler=args.compiler,
        llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip)
    create=getattr(d.lib,'cuStreamCreate' if d.cuda else 'hipStreamCreateWithFlags')
    destroy=getattr(d.lib,'cuStreamDestroy_v2' if d.cuda else 'hipStreamDestroy')
    create.argtypes,create.restype=[ct.POINTER(ct.c_void_p),ct.c_uint],ct.c_int
    destroy.argtypes,destroy.restype=[ct.c_void_p],ct.c_int
    streams=[ct.c_void_p(),ct.c_void_p()]
    for stream in streams:d.check(create(ct.byref(stream),1))
    rows=[]
    try:
        for x0,steps in ((.6,0),(.3,1),(.15,2),(.05,3),(.6,None),(-.6,None)):
            pointers=[]
            def upload(value):
                pointer=ct.c_void_p()
                d.check(d.alloc(ct.byref(pointer),value.nbytes)); pointers.append(pointer)
                d.check(d.htod(pointer,value.ctypes.data,value.nbytes) if d.cuda else d.copy(pointer,value.ctypes.data,value.nbytes,1))
                return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,typestr=value.dtype.str,data=(pointer.value,False)))
            def download(value):
                v=value.__cuda_array_interface__; out=np.empty(v['shape'],np.dtype(v['typestr']))
                d.check(d.dtoh(out.ctypes.data,v['data'][0],out.nbytes) if d.cuda else d.copy(out.ctypes.data,v['data'][0],out.nbytes,2))
                return out
            frame=None
            try:
                x=np.full(4,x0,np.float32); w=np.full(4,2,np.float32)
                frame=branch_pair.capture(upload(x)) if steps is None else pair.capture(upload(x),upload(w))
                expected_primal=(x*x if x0>0 else x) if steps is None else x*w**steps
                np.testing.assert_allclose(download(frame.primals[0]),expected_primal,rtol=1e-6,atol=1e-7)
                saved=[download(v) for v in frame.residuals]
                # Upload all cotangents first, so host copies do not serialize
                # the two subsequent submissions. This is not an overlap claim.
                seeds=[upload(np.full(4,factor,np.float32)) for factor in (1,2)]
                tickets=[frame.backward_async(stream.value,seed) for stream,seed in zip(streams,seeds,strict=True)]
                assert len(frame._submissions)==2
                for factor,ticket in zip((1,2),tickets,strict=True):
                    outputs=ticket.wait()
                    expected=((2*factor*x if x0>0 else np.full_like(x,factor)),) if steps is None else (factor*w**steps,np.zeros_like(w) if steps==0 else factor*steps*x*w**(steps-1))
                    for v,ref in zip(outputs,expected,strict=True):np.testing.assert_allclose(download(v),ref,rtol=1e-6,atol=1e-7)
                assert frame.poll()
                assert not frame._submissions
                for v,old in zip(frame.residuals,saved,strict=True):np.testing.assert_array_equal(download(v),old)
                retained_before_release=sum(v.nbytes for v in frame.buffers)
                for ticket in tickets:ticket.release()
                released_bytes=retained_before_release-sum(v.nbytes for v in frame.buffers)
                assert released_bytes==(32 if steps is None else 64)
                rows.append(dict(released_derivative_bytes=released_bytes,exit_steps=steps,predicate_path=x0>0 if steps is None else None,residual_dtypes=[v.dtype.str for v in saved],
                    residual_bytes=sum(v.nbytes for v in saved),async_generations=2,completion_owners_retired=True))
                frame.close()
                with_closed=False
                try:tickets[0].outputs[0].__cuda_array_interface__
                except ValueError:with_closed=True
                assert with_closed
            finally:
                if frame is not None:frame.close()
                for pointer in pointers:d.check(d.free(pointer))
    finally:
        for stream in streams:d.check(destroy(stream))
    names=['src/transforms/lib/AutodiffPairedPass.cpp','src/transforms/lib/NativeTapeToGPUPass.cpp',
           'python/tessera/compiler/native_persistent_tape.py','python/tessera/compiler/native_device_tape.py',
           'python/tessera/compiler/native_gpu_storage.py']
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,strided_counter=args.strided_counter,rows=rows,
        forward=pair.forward.binding_digest,backward=pair.backward.binding_digest,
        predicate_forward=branch_pair.forward.binding_digest,predicate_backward=branch_pair.backward.binding_digest,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        promotion_eligible=False,overlap_proven=False),indent=2)+'\n')
    print('4 data-dependent exits, 2 predicate paths and 12 asynchronous generations passed')


if __name__=='__main__':main()
