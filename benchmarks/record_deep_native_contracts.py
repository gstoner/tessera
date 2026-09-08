#!/usr/bin/env python3
"""Exact-device public logical lengths, nested guards and device-gated readers."""
import argparse
import ctypes as ct
import hashlib
import json
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.native_public_result import materialize_public_results  # noqa: E402
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402


def public_source():
    return '''module attributes {tessera.native_result_program = true} {
      func.func @positive(%x: memref<8xf32>, %out: memref<8xf32> {tessera.result_shape = 2 : i64}, %shape: memref<1xi64>) {
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        %eight = arith.constant 8 : index
        %zero = arith.constant 0.0 : f32
        %length = scf.for %i = %z to %eight step %one iter_args(%count = %z) -> index {
          %v = memref.load %x[%i] : memref<8xf32>
          %positive = arith.cmpf ogt, %v, %zero : f32
          %next = scf.if %positive -> index {
            %safe = arith.cmpi ult, %count, %eight : index
            cf.assert %safe, "output capacity exceeded"
            memref.store %v, %out[%count] : memref<8xf32>
            %increment = arith.addi %count, %one : index
            scf.yield %increment : index
          } else { scf.yield %count : index }
          scf.yield %next : index
        }
        %n = arith.index_cast %length : index to i64
        memref.store %n, %shape[%z] : memref<1xi64>
        return
      }
    }'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); device=Device(args.backend); chip='sm_120' if device.cuda else 'gfx1151'
    pointers=[]; streams=[]; rows=[]
    def copy_in(pointer, a):
        device.check(device.htod(pointer,a.ctypes.data,a.nbytes) if device.cuda else device.copy(pointer,a.ctypes.data,a.nbytes,1))
    def upload(a):
        p=ct.c_void_p();device.check(device.alloc(ct.byref(p),a.nbytes));pointers.append(p);copy_in(p,a)
        return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=a.shape,typestr=a.dtype.str,data=(p.value,False)))
    def download(view):
        interface=view.__cuda_array_interface__;a=np.empty(interface['shape'],dtype=interface['typestr'])
        if a.nbytes:device.check(device.dtoh(a.ctypes.data,interface['data'][0],a.nbytes) if device.cuda else device.copy(a.ctypes.data,interface['data'][0],a.nbytes,2))
        return a
    create=getattr(device.lib,'cuStreamCreate' if device.cuda else 'hipStreamCreateWithFlags')
    create.argtypes,create.restype=[ct.POINTER(ct.c_void_p),ct.c_uint],ct.c_int
    destroy=getattr(device.lib,'cuStreamDestroy_v2' if device.cuda else 'hipStreamDestroy')
    destroy.argtypes,destroy.restype=[ct.c_void_p],ct.c_int
    try:
        public=materialize_public_results(public_source(),compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip)
        for x in (np.zeros(8,np.float32),np.ones(8,np.float32),np.arange(-4,4,dtype=np.float32)):
            with public.run(upload(x)) as frame:
                np.testing.assert_array_equal(download(frame.results[0]),x[x>0])
                rows.append(dict(case='device_computed_length',logical_length=int((x>0).sum()),capacity=8))
        for source,case in [(public_source().replace('%safe = arith.cmpi ult, %count, %eight : index','%safe = arith.constant false'),'nested_guard'),
                            (public_source().replace('%n = arith.index_cast %length : index to i64','%n = arith.constant 9 : i64'),'invalid_returned_length'),
                            (public_source().replace('memref.store %n, %shape[%z] : memref<1xi64>', ''),'missing_returned_length')]:
            failed=materialize_public_results(source,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip)
            try:failed.run(upload(np.ones(8,np.float32)))
            except RuntimeError as error:assert 'guard failed' in str(error)
            else:raise AssertionError('failed product exposed a view')
            rows.append(dict(case=case,output_refused=True))
            if case=='nested_guard':
                from tessera.compiler.native_storage_contract import generate_tensor_binding
                signature=inspect.Signature([inspect.Parameter(n,inspect.Parameter.POSITIONAL_ONLY)
                    for n in ('arg0','arg1','arg2','status','scratch')])
                binding=generate_tensor_binding(failed.package,signature)
                output=upload(np.full(8,42,np.float32))
                shape=upload(np.zeros(1,np.int64));status=upload(np.zeros(1,np.int64))
                try:
                    binding(upload(np.ones(8,np.float32)),output,shape,status,1)
                    np.testing.assert_array_equal(download(output),np.full(8,42,np.float32))
                    np.testing.assert_array_equal(download(shape),[-1])
                    np.testing.assert_array_equal(download(status),[1])
                finally:binding.close()
                rows[-1]['failed_body_and_suffix_writes_suppressed']=True
        square='''module { func.func @square(%x: tensor<4xf32>) -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
          %y = "tessera.mul"(%x,%x) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          return %y : tensor<4xf32> } }'''
        pair=materialize_persistent_tape(square,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,checked_status=True,gated_input=True)
        for _ in range(2):
            stream=ct.c_void_p();device.check(create(ct.byref(stream),1));streams.append(stream)
        with pair.capture(upload(np.full(4,2,np.float32))) as first, pair.capture(upload(np.full(4,3,np.float32))) as second:
            seed=upload(np.ones(4,np.float32))
            for inject in (False,True):
                parent=first.backward_async(streams[0].value,seed)
                if inject:
                    parent.submission.wait()
                    copy_in(parent._status_buffer.pointer,np.ones(1,np.int64))
                # No parent.wait()/poll() status readback before enqueue.
                child=parent.backward_into(second,streams[1].value)
                if inject:
                    try:child.wait()
                    except RuntimeError as error:assert 'guard failed' in str(error)
                    else:raise AssertionError('upstream failure did not gate reader')
                else:np.testing.assert_array_equal(download(child.wait()[0]),np.full(4,24,np.float32))
                parent.release();child.release()
                rows.append(dict(case='device_gated_reader',injected_upstream_failure=inject))
    finally:
        for p in pointers:device.check(device.free(p))
        for stream in streams:device.check(destroy(stream))
    names=['src/transforms/lib/NativeTapeToGPUPass.cpp','python/tessera/compiler/native_public_result.py','python/tessera/compiler/native_persistent_tape.py']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,execution_kind="native_gpu",rows=rows,promotion_eligible=False,overlap_measured=False,
        sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names},
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('Public lengths, nested refusal and device-gated reader contracts passed')


if __name__=='__main__':main()
