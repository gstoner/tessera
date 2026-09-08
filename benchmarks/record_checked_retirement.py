#!/usr/bin/env python3
"""Device AD result projection and checked pool generations with tracked readers."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from benchmarks.record_automatic_ad_results import source  # noqa: E402
from tessera.compiler.native_public_result import materialize_ad_public_results  # noqa: E402
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();d=Device(args.backend);chip='sm_120' if d.cuda else 'gfx1151'
    pointers=[];streams=[];rows=[];P=ct.c_void_p
    def bind(cu,hip,types):
        fn=getattr(d.lib,cu if d.cuda else hip);fn.argtypes,fn.restype=types,ct.c_int;return fn
    create=bind('cuStreamCreate','hipStreamCreateWithFlags',[ct.POINTER(P),ct.c_uint])
    destroy=bind('cuStreamDestroy_v2','hipStreamDestroy',[P])
    copy=bind('cuMemcpyDtoDAsync_v2','hipMemcpyDtoDAsync',[P,P,ct.c_size_t,P])
    def upload(value):
        pointer=P();d.check(d.alloc(ct.byref(pointer),value.nbytes));pointers.append(pointer)
        d.check(d.htod(pointer,value.ctypes.data,value.nbytes) if d.cuda else d.copy(pointer,value.ctypes.data,value.nbytes,1))
        return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,typestr=value.dtype.str,data=(pointer.value,False)))
    def download(view):
        data=view.__cuda_array_interface__;value=np.empty(data['shape'],np.dtype(data['typestr']))
        if value.nbytes:d.check(d.dtoh(value.ctypes.data,data['data'][0],value.nbytes) if d.cuda else d.copy(value.ctypes.data,data['data'][0],value.nbytes,2))
        return value
    def poll(owner):
        deadline=time.monotonic()+20
        while not owner.poll():
            if time.monotonic()>deadline:raise TimeoutError('asynchronous retirement failed')
            time.sleep(.001)
    try:
        program=materialize_ad_public_results(source(),compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,capacity=4)
        for first,length in ((1.,2),(-1.,4)):
            x=np.array([first,2,3,4],np.float32)
            with program.run(upload(x)) as frame:np.testing.assert_array_equal(download(frame.results[0]),x[:length]**2)
            rows.append(dict(case='automatic_forward_AD_shape',length=length))
        empty=materialize_ad_public_results(source().replace('%two = arith.constant 2 : index','%two = arith.constant 0 : index'),
            compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,capacity=4)
        with empty.run(upload(np.ones(4,np.float32))) as frame:
            assert download(frame.results[0]).shape==(0,)
        rows.append(dict(case='automatic_forward_AD_shape',length=0))
        limited=materialize_ad_public_results(source(),compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,capacity=2)
        try:limited.run(upload(np.array([-1,2,3,4],np.float32)))
        except RuntimeError:rows.append(dict(case='undersized_capacity',refused=True))
        else:raise AssertionError('capacity overflow exposed a result')
        square='''module { func.func @square(%x: tensor<4xf32>) -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
          %y = "tessera.mul"(%x,%x) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          return %y : tensor<4xf32> } }'''
        pair=materialize_persistent_tape(square,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,checked_status=True,gated_input=True)
        for _ in range(3):
            stream=P();d.check(create(ct.byref(stream),1));streams.append(stream)
        seed=upload(np.ones(4,np.float32));dest=upload(np.zeros(4,np.float32))
        with pair.capture(upload(np.full(4,2,np.float32))) as first, pair.capture(upload(np.full(4,3,np.float32))) as second:
            counts=[len(f.buffers) for f in (first,second)]
            saved=[f.sync for f in (first,second)]
            def forbidden():raise AssertionError('healthy generation used a context barrier')
            for f in (first,second):f.sync=forbidden
            try:
                for iteration in range(9):
                    parent=first.backward_async(streams[0].value,seed,tracked=True)
                    injected=iteration==8
                    if injected:
                        parent._submission.ticket.wait()
                        failure=np.ones(1,np.int64)
                        pointer=parent._status_buffer.pointer
                        d.check(d.htod(pointer,failure.ctypes.data,8) if d.cuda else d.copy(pointer,failure.ctypes.data,8,1))
                    child=parent.backward_into(second,streams[1].value)
                    parent.retire(streams[2].value);poll(parent)
                    # Explicit checked read for the numerical oracle; reclamation
                    # itself requires neither success readback nor a context wait.
                    if injected:
                        try:child.wait_success()
                        except RuntimeError:pass
                        else:raise AssertionError('failed checked generation became readable')
                        try:child.read(streams[0].value)
                        except ValueError:pass
                        else:raise AssertionError('failed generation exposed a reader')
                    else:
                        child.wait_success()
                        with child.read(streams[0].value) as output:
                            d.check(copy(P(dest.__cuda_array_interface__['data'][0]),P(output[0].__cuda_array_interface__['data'][0]),16,streams[0]))
                    child.retire(streams[2].value);poll(child)
                    if not injected:np.testing.assert_array_equal(download(dest),np.full(4,24,np.float32))
                    assert [len(f.buffers) for f in (first,second)]==counts
                rows.append(dict(case='checked_pool_generation_chain',successful_iterations=8,injected_failure_iterations=1,context_waits=0,retained_generation_buffers=0))
            finally:
                for f,sync in zip((first,second),saved,strict=True):f.sync=sync
    finally:
        for pointer in pointers:d.check(d.free(pointer))
        for stream in streams:d.check(destroy(stream))
    names=['src/transforms/lib/NativeTapeToGPUPass.cpp','python/tessera/compiler/native_public_result.py',
           'python/tessera/compiler/native_persistent_tape.py','python/tessera/compiler/native_reader_retirement.py',
           'python/tessera/compiler/native_gpu_storage.py','python/tessera/compiler/native_gpu_tensor.py',
           'benchmarks/record_automatic_ad_results.py']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,execution_kind='native_gpu',rows=rows,
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),promotion_eligible=False),indent=2)+'\n')
    print('Automatic dynamic AD forward results and checked generation retirement passed')


if __name__=='__main__':main()
