#!/usr/bin/env python3
"""Owning-device source slice, exception-generation and checked VJP evidence."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.trace import trace  # noqa: E402
from tessera.compiler.source_control_flow import to_native_source_ir  # noqa: E402
from tessera.compiler.native_public_result import materialize_source_vjp  # noqa: E402
from tessera.compiler.native_gpu_storage import _run  # noqa: E402


def checked_cube(x):
    if x<x+x:
        return x*x*x
    try:raise ValueError(x+x)
    except ValueError as inner:raise RuntimeError(x*x) from inner


def loop_error(x):
    while x<x+x:
        try:raise ValueError(x+x)
        except ValueError as inner:
            x=x+x
            raise RuntimeError(x*x) from inner
    return x*x


def runtime_slice(x,start,stop,step):
    return x[start:stop:step]


def retained_exception(x):
    limit=x*x*x
    saved=None
    while x<limit:
        try:raise ValueError(x+x)
        except ValueError as error:
            if saved is None:saved=error
        x=x+x
    raise RuntimeError(x*x) from saved


def nested_runtime_slice(x,start,stop,step):
    part=x[start:stop:step]
    return part[::-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=Device(args.backend);chip='sm_120' if device.cuda else 'gfx1151'
    pointers=[];rows=[]
    def upload(array):
        pointer=ct.c_void_p();device.check(device.alloc(ct.byref(pointer),array.nbytes));pointers.append(pointer)
        device.check(device.htod(pointer,array.ctypes.data,array.nbytes) if device.cuda else device.copy(pointer,array.ctypes.data,array.nbytes,1))
        return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=array.shape,typestr=array.dtype.str,data=(pointer.value,False)))
    def read(view,dtype=np.float32):
        result=np.empty(view.__cuda_array_interface__['shape'],dtype)
        if result.nbytes:device.check(device.dtoh(result.ctypes.data,ct.c_void_p(view.__cuda_array_interface__["data"][0]),result.nbytes) if device.cuda else device.copy(result.ctypes.data,ct.c_void_p(view.__cuda_array_interface__["data"][0]),result.nbytes,2))
        return result
    try:
        for fn in (checked_cube,loop_error):
            source=to_native_source_ir(trace(fn,np.ones(1,np.float32),source_control_flow=True,max_steps=2,
                source_error_specs=(((1,),'f32'),)),autodiff='reverse')
            program=materialize_source_vjp(source,compiler=args.compiler,llvm_bin='/usr/lib/llvm-23/bin',backend=args.backend,chip=chip,capacity=4)
            for asynchronous in (False,True):
                for value in (2.,-2.):
                    should_fail=(value<0) if fn is checked_cube else (value>0)
                    x=np.array([value],np.float32)
                    inputs=upload(x);seed=upload(np.ones(1,np.float32))
                    stream=ct.c_void_p();frame=None
                    try:
                        if asynchronous:
                            create=getattr(device.lib,'cuStreamCreate' if device.cuda else 'hipStreamCreateWithFlags')
                            create.argtypes=[ct.POINTER(ct.c_void_p),ct.c_uint];create.restype=ct.c_int
                            device.check(create(ct.byref(stream),1))
                            frame=program.submit(stream.value,inputs,cotangents=(seed,),scoped=True)
                            assert not hasattr(frame,'derivatives')
                            deadline=time.monotonic()+20
                            while not frame.poll():
                                if time.monotonic()>deadline:raise TimeoutError('source VJP pending')
                                time.sleep(.001)
                        else:frame=program.run(inputs,cotangents=(seed,))
                        assert not should_fail
                        from contextlib import nullcontext
                        with frame.read(stream.value) if asynchronous else nullcontext((frame.primals,frame.derivatives)) as (primals,derivatives):
                            np.testing.assert_allclose(read(primals[0]),x**(3 if fn is checked_cube else 2))
                            np.testing.assert_allclose(read(derivatives[0]),3*x*x if fn is checked_cube else 2*x)
                        rows.append(dict(family=fn.__name__,asynchronous=asynchronous,value=value,derivative_matched=True))
                    except RuntimeError as error:
                        assert should_fail and type(error) is RuntimeError,error
                        np.testing.assert_array_equal(error.args[0],x*x*(4 if fn is loop_error else 1))
                        np.testing.assert_array_equal(error.__cause__.args[0],2*x)
                        assert error.__cause__ is error.__context__
                        if asynchronous:
                            assert frame._backward is None and not hasattr(frame,'derivatives')
                            try:frame.poll()
                            except RuntimeError as again:assert again is error
                            else:raise AssertionError('failed VJP became readable')
                        rows.append(dict(family=fn.__name__,asynchronous=asynchronous,value=value,exception_matched=True,derivative_exposed=False))
                    finally:
                        if frame is not None:
                            if asynchronous:
                                frame.retire(stream.value)
                                deadline=time.monotonic()+20
                                while not frame.poll_retired():
                                    if time.monotonic()>deadline:raise TimeoutError('source retirement pending')
                                    time.sleep(.001)
                                assert frame.closed
                            else:frame.close()
                        if stream.value:
                            destroy=getattr(device.lib,'cuStreamDestroy_v2' if device.cuda else 'hipStreamDestroy')
                            destroy.argtypes=[ct.c_void_p];destroy.restype=ct.c_int;device.check(destroy(stream))
        retained=to_native_source_ir(trace(retained_exception,np.ones(1,np.float32),source_control_flow=True,max_steps=2,
            source_error_specs=(((1,),'f32'),)),autodiff='reverse')
        retained_program=materialize_source_vjp(retained,compiler=args.compiler,llvm_bin='/usr/lib/llvm-23/bin',backend=args.backend,chip=chip,capacity=4)
        try:
            with retained_program.run(upload(np.array([2.],np.float32)),cotangents=(upload(np.ones(1,np.float32)),)):
                raise AssertionError('retained exception exposed a derivative')
        except RuntimeError as error:
            np.testing.assert_array_equal(error.args[0],[64])
            np.testing.assert_array_equal(error.__cause__.args[0],[4])
            assert error.__context__ is None
        rows.append(dict(family='retained_exception',earlier_generation_payload=True,derivative_exposed=False))
        source=to_native_source_ir(trace(nested_runtime_slice,np.ones(8,np.float32),*[np.ones(1,np.int64)]*3,source_control_flow=True))
        native=_run(args.compiler,'--tessera-to-linalg',source=source)
        buffered=_run(Path('/usr/lib/llvm-23/bin/mlir-opt'),'--allow-unregistered-dialect',
            '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map',
            '--convert-linalg-to-loops','--canonicalize',source=native)
        from tessera.compiler.native_public_result import _prepare, NativePublicResult
        from tessera.compiler.native_gpu_storage import build_native_gpu_storage
        gpu,_,_=_prepare(buffered,args.compiler,args.backend,8)
        package=build_native_gpu_storage(gpu,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip)
        program=NativePublicResult(buffered,args.compiler,package,8)
        for bounds in ((1,7,2),(-6,-1,2),(-100,100,3),(7,1,1),(0,8,100),(-(1<<63),(1<<63)-1,1),(7,0,-2),(100,-100,-3),(7,0,-(1<<63))):
            data=np.arange(8,dtype=np.float32)
            with program.run(upload(data),*[upload(np.array([n],np.int64)) for n in bounds]) as frame:
                np.testing.assert_array_equal(read(frame.results[0]),data[slice(*bounds)][::-1])
            rows.append(dict(family='nested_runtime_slice',bounds=bounds,matched=True))
        for step in (0,):
            try:
                with program.run(upload(np.ones(8,np.float32)),*[upload(np.array([n],np.int64)) for n in (0,8,step)]):
                    raise AssertionError('invalid step exposed results')
            except RuntimeError as error:assert 'guard failed' in str(error)
            rows.append(dict(family='nested_runtime_slice',step=step,guard_refused=True))
    finally:
        for pointer in reversed(pointers):device.check(device.free(pointer))
    sources=['python/tessera/compiler/source_control_flow.py','python/tessera/compiler/trace.py','python/tessera/compiler/native_public_result.py','python/tessera/compiler/native_source_state.py','src/transforms/lib/NativeTapeToGPUPass.cpp','src/transforms/lib/AutodiffPairedPass.cpp','benchmarks/record_source_scoped_ad_gpu.py']
    packet=dict(backend=args.backend,chip=chip,compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},cases=rows,performance_promotion=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+'\n')
    print(f'{args.backend}: {len(rows)} cases passed')


if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as error:
        print(error.stderr,file=sys.stderr)
        raise
