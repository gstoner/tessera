#!/usr/bin/env python3
"""Exact-device mapped source views and checked exception completion; no promotion."""
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
from tessera.compiler.trace import trace  # noqa: E402
from tessera.compiler.source_control_flow import to_native_source_ir  # noqa: E402
from tessera.compiler.native_source_state import materialize_source_state  # noqa: E402
from tessera.compiler.native_public_result import materialize_ad_public_results  # noqa: E402


def mapped(x):
    part=x[::-1,::2]
    part[:]=part*part
    return x+x


def failure(x):
    x[:]=x+x
    if x>x+x:
        try:raise ValueError('device inner')
        except ValueError as inner:raise RuntimeError('device outer') from inner
    return x*x


def dynamic_failure(x):
    x[:]=x+x
    if x>x+x:raise ValueError(x*x)
    return x*x


def context_failure(x):
    try:raise ValueError(x+x)
    except ValueError as inner:
        x[:]=x+x
        raise RuntimeError(x*x) from inner


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=Device(args.backend);chip='sm_120' if device.cuda else 'gfx1151'
    rows=[]
    for fn in (mapped,failure,dynamic_failure,context_failure):
        shape=(2,4) if fn is mapped else (1,)
        specs=() if fn is mapped else (((1,),'f32'),)
        source=to_native_source_ir(trace(fn,np.ones(shape,np.float32),source_control_flow=True,
            source_state_groups=((0,),),source_error_specs=specs))
        program=materialize_source_state(source,compiler=args.compiler,llvm_bin='/usr/lib/llvm-23/bin',
            backend=args.backend,chip=chip,capacity=int(np.prod(shape)))
        for asynchronous in (False,True):
            for negative in ((True,) if fn is context_failure else (False,True)) if fn is not mapped else (False,):
                x=np.arange(1,np.prod(shape)+1,dtype=np.float32).reshape(shape)*(-1 if negative else 1)
                pointer=ct.c_void_p();stream=ct.c_void_p();frame=None
                device.check(device.alloc(ct.byref(pointer),x.nbytes))
                try:
                    device.check(device.htod(pointer,x.ctypes.data,x.nbytes) if device.cuda else device.copy(pointer,x.ctypes.data,x.nbytes,1))
                    view=SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=shape,typestr='<f4',data=(pointer.value,False)))
                    error=None
                    try:
                        if asynchronous:
                            create=getattr(device.lib,'cuStreamCreate' if device.cuda else 'hipStreamCreateWithFlags')
                            create.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint];create.restype=ct.c_int
                            device.check(create(ct.byref(stream),1))
                            frame=program.submit(stream.value,view)
                            deadline=time.monotonic()+20
                            while not frame.poll():
                                if time.monotonic()>deadline:raise TimeoutError('source result pending')
                                time.sleep(.001)
                        else:frame=program.run(view)
                    except (ValueError,RuntimeError) as caught:error=caught
                    if negative:
                        assert error is not None
                        if fn is context_failure:
                            assert type(error) is RuntimeError
                            np.testing.assert_array_equal(error.args[0],x*x*4)
                            np.testing.assert_array_equal(error.__context__.args[0],x*2)
                            assert error.__cause__ is error.__context__
                        elif fn is failure:
                            assert type(error) is RuntimeError and error.args==('device outer',)
                            assert error.__cause__ is error.__context__
                            assert error.__cause__.args==('device inner',)
                        else:
                            assert type(error) is ValueError
                            np.testing.assert_array_equal(error.args[0],x*x*4)
                        if asynchronous:
                            assert frame is not None and not hasattr(frame,'results')
                            try:frame.poll()
                            except (ValueError,RuntimeError) as again:assert again is error
                            else:raise AssertionError('failed frame became readable')
                        rows.append(dict(family=fn.__name__,asynchronous=asynchronous,exception=type(error).__name__,result_exposed=False))
                    else:
                        if error is not None:raise error
                        assert frame is not None
                        expected=x.copy();oracle=fn(expected)
                        for output,wanted in zip(frame.results[:2],(oracle,expected),strict=True):
                            actual=np.empty_like(wanted)
                            device.check(device.dtoh(actual.ctypes.data,output.pointer,actual.nbytes) if device.cuda else device.copy(actual.ctypes.data,output.pointer,actual.nbytes,2))
                            np.testing.assert_array_equal(actual,wanted)
                        rows.append(dict(family=fn.__name__,asynchronous=asynchronous,matched=True))
                    original=np.empty_like(x)
                    device.check(device.dtoh(original.ctypes.data,pointer,x.nbytes) if device.cuda else device.copy(original.ctypes.data,pointer,x.nbytes,2))
                    np.testing.assert_array_equal(original,x)
                finally:
                    if frame is not None:frame.close()
                    if stream.value:
                        destroy=getattr(device.lib,'cuStreamDestroy_v2' if device.cuda else 'hipStreamDestroy')
                        destroy.argtypes=[ct.c_void_p];destroy.restype=ct.c_int;device.check(destroy(stream))
                    device.check(device.free(pointer))
    source=to_native_source_ir(trace(mapped,np.ones((2,4),np.float32),source_control_flow=True,
        source_state_groups=((0,),)),autodiff='reverse')
    backward=materialize_ad_public_results(source,compiler=args.compiler,llvm_bin='/usr/lib/llvm-23/bin',
        backend=args.backend,chip=chip,capacity=8,role='backward')
    x=np.arange(1,9,dtype=np.float32).reshape(2,4)
    pointers=[];inputs=[]
    try:
        for array in (x,np.ones_like(x),np.zeros_like(x)):
            pointer=ct.c_void_p();device.check(device.alloc(ct.byref(pointer),array.nbytes));pointers.append(pointer)
            device.check(device.htod(pointer,array.ctypes.data,array.nbytes) if device.cuda else device.copy(pointer,array.ctypes.data,array.nbytes,1))
            inputs.append(SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=array.shape,typestr=array.dtype.str,data=(pointer.value,False))))
        with backward.run(*inputs) as frame:
            actual=np.empty_like(x);output=frame.results[0]
            device.check(device.dtoh(actual.ctypes.data,output.pointer,actual.nbytes) if device.cuda else device.copy(actual.ctypes.data,output.pointer,actual.nbytes,2))
            expected=np.full_like(x,2);expected[:,::2]=4*x[:,::2]
            np.testing.assert_array_equal(actual,expected)
            rows.append(dict(family='mapped_adjoint',matched=True))
    finally:
        for pointer in reversed(pointers):device.check(device.free(pointer))
    dynamic=(ROOT/'tests/fixtures/source_maps/runtime_even_columns.mlir').read_text()
    for role in ('forward','backward'):
        program=materialize_ad_public_results(dynamic,compiler=args.compiler,llvm_bin='/usr/lib/llvm-23/bin',
            backend=args.backend,chip=chip,capacity=16,input_capacity=16,role=role)
        pointers=[]
        def upload(array):
            pointer=ct.c_void_p();device.check(device.alloc(ct.byref(pointer),array.nbytes));pointers.append(pointer)
            device.check(device.htod(pointer,array.ctypes.data,array.nbytes) if device.cuda else device.copy(pointer,array.ctypes.data,array.nbytes,1))
            return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=array.shape,typestr=array.dtype.str,data=(pointer.value,False)))
        try:
            for cols in (2,5,6):
                data=np.arange(1,17,dtype=np.float32)
                arguments=[upload(data)]
                if role=='backward':arguments.append(upload(np.ones(16,np.float32)))
                arguments.append(upload(np.array([2,cols],np.int64)))
                if role=='backward':arguments.append(upload(np.array([2,(cols+1)//2],np.int64)))
                with program.run(*arguments) as frame:
                    result=frame.results[0];interface=result.__cuda_array_interface__
                    actual=np.empty(interface['shape'],np.float32)
                    device.check(device.dtoh(actual.ctypes.data,result.pointer,actual.nbytes) if device.cuda else device.copy(actual.ctypes.data,result.pointer,actual.nbytes,2))
                    expected=data[:2*cols].reshape(2,cols)[:,::2] if role=='forward' else np.zeros((2,cols),np.float32)
                    if role=='backward':expected[:,::2]=1
                    np.testing.assert_array_equal(actual,expected)
                rows.append(dict(family='runtime_even_columns',role=role,shape=[2,cols],matched=True))
            if role=='backward':
                try:
                    with program.run(upload(np.ones(16,np.float32)),upload(np.ones(16,np.float32)),
                                     upload(np.array([2,5],np.int64)),upload(np.array([2,4],np.int64))):
                        raise AssertionError('mismatched slice cotangent became readable')
                except RuntimeError as error:assert 'guard failed' in str(error)
                rows.append(dict(family='runtime_slice_bad_cotangent',refused=True))
        finally:
            for pointer in reversed(pointers):device.check(device.free(pointer))
    sources=['python/tessera/compiler/source_control_flow.py','python/tessera/compiler/native_source_state.py',
        'python/tessera/compiler/native_public_result.py','src/transforms/lib/NativeTapeToGPUPass.cpp','src/transforms/lib/AutodiffPairedPass.cpp','tests/fixtures/source_maps/runtime_even_columns.mlir','python/tessera/compiler/trace.py']
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,execution_kind='native_gpu',promotion_eligible=False,
        rows=rows,sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(f'{args.backend}: {len(rows)} mapped-view/exception cases passed')


if __name__=='__main__':main()
