#!/usr/bin/env python3
"""Runtime-sized AD products and scoped whole-frame retirement device proof."""
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
        dynamic='''module { func.func @dynamic(%x: tensor<?xf32>) -> tensor<?xf32> attributes {tessera.autodiff = "reverse"} {
          %y = "tessera.mul"(%x,%x) : (tensor<?xf32>,tensor<?xf32>) -> tensor<?xf32>
          return %y : tensor<?xf32> } }'''
        program=materialize_ad_public_results(dynamic,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,capacity=4,input_capacity=4,role='backward')
        stream=P();d.check(create(ct.byref(stream),1));streams.append(stream)
        for length in (0,2,4):
            x=np.arange(1,5,dtype=np.float32);seed=np.ones(4,np.float32)
            shape=upload(np.array([length],np.int64))
            with program.submit(stream.value,upload(x),upload(seed),shape,upload(np.array([length],np.int64))) as frame:
                assert not hasattr(frame,'results')
                poll(frame)
                np.testing.assert_array_equal(download(frame.results[0]),2*x[:length])
            rows.append(dict(case='dynamic_backward_input_and_result',length=length))
        for left,right in ((-1,-1),(5,5),(2,3)):
            with program.submit(stream.value,upload(np.ones(4,np.float32)),upload(np.ones(4,np.float32)),
                                upload(np.array([left],np.int64)),upload(np.array([right],np.int64))) as frame:
                try:poll(frame)
                except RuntimeError:assert not hasattr(frame,'results')
                else:raise AssertionError('malformed input shape was exposed')
            rows.append(dict(case='invalid_backward_shape',shapes=[left,right],refused=True))
        matrix_backward=materialize_ad_public_results(dynamic.replace('?xf32','?x?xf32'),compiler=args.compiler,
            llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,capacity=4,input_capacity=4,role='backward')
        for dims in ((2,2),(1,4),(0,4)):
            x=np.arange(1,5,dtype=np.float32)
            with matrix_backward.run(upload(x),upload(np.ones(4,np.float32)),
                                     upload(np.array(dims,np.int64)),upload(np.array(dims,np.int64))) as frame:
                np.testing.assert_array_equal(download(frame.results[0]),(2*x[:int(np.prod(dims))]).reshape(dims))
            rows.append(dict(case='dynamic_matrix_backward',shape=dims))
        matrix='''module { func.func @matrix(%x: tensor<2x2xf32>) -> (tensor<2x2xf32>,tensor<2x2xf32>) attributes {tessera.autodiff = "reverse"} {
          %y = "tessera.mul"(%x,%x) : (tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
          %z = "tessera.add"(%y,%x) : (tensor<2x2xf32>,tensor<2x2xf32>) -> tensor<2x2xf32>
          return %y,%z : tensor<2x2xf32>,tensor<2x2xf32> } }'''
        multi=materialize_ad_public_results(matrix,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,capacity=4)
        x=np.arange(1,5,dtype=np.float32).reshape(2,2)
        with multi.run(upload(x)) as frame:
            np.testing.assert_array_equal(download(frame.results[0]),x*x)
            np.testing.assert_array_equal(download(frame.results[1]),x*x+x)
        rows.append(dict(case='multiple_matrix_results',shapes=[[2,2],[2,2]]))
        square='''module { func.func @square(%x: tensor<4xf32>) -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
          %y = "tessera.mul"(%x,%x) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
          return %y : tensor<4xf32> } }'''
        pair=materialize_persistent_tape(square,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,checked_status=True,gated_input=True)
        for _ in range(3):
            stream=P();d.check(create(ct.byref(stream),1));streams.append(stream)
        seed=upload(np.ones(4,np.float32));dest=upload(np.zeros(4,np.float32))
        for iteration in range(8):
            frame=pair.capture(upload(np.full(4,2,np.float32)),scoped=True,stream=streams[0].value)
            assert not hasattr(frame,'primals') and not hasattr(frame,'residuals')
            def forbidden():raise AssertionError('scoped retirement used a context barrier')
            frame.sync=forbidden
            for binding in frame._bindings:
                if binding._bound is not None:binding._bound._sync=forbidden
            generation=frame.backward_async(streams[1].value,seed,tracked=True)
            for binding in frame._bindings:
                if binding._bound is not None:binding._bound._sync=forbidden
            generation.wait_success()
            with generation.read(streams[0].value) as output:
                d.check(copy(P(dest.__cuda_array_interface__['data'][0]),P(output[0].__cuda_array_interface__['data'][0]),16,streams[0]))
            frame.retire(streams[2].value)
            deadline=time.monotonic()+20
            while not frame.poll_retired():
                if time.monotonic()>deadline:raise TimeoutError('whole-frame retirement failed')
                time.sleep(.001)
            assert frame.closed and not frame.buffers
            np.testing.assert_array_equal(download(dest),np.full(4,4,np.float32))
        rows.append(dict(case='scoped_whole_frame_retirement',iterations=8,explicit_context_waits=0,retained_buffers=0))
    finally:
        for pointer in pointers:d.check(d.free(pointer))
        for stream in streams:d.check(destroy(stream))
    names=['src/transforms/lib/NativeTapeToGPUPass.cpp','python/tessera/compiler/native_public_result.py',
           'python/tessera/compiler/native_persistent_tape.py','python/tessera/compiler/native_reader_retirement.py',
           'python/tessera/compiler/native_gpu_storage.py','python/tessera/compiler/native_gpu_tensor.py',
           'python/tessera/compiler/native_storage_contract.py']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,execution_kind='native_gpu',rows=rows,
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),promotion_eligible=False),indent=2)+'\n')
    print('Dynamic backward, matrix results and scoped frame retirement passed')


if __name__=='__main__':main()
