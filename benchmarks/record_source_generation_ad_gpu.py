#!/usr/bin/env python3
"""Owning-device source slice, exception-generation and checked VJP evidence."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
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
        if result.nbytes:device.check(device.dtoh(result.ctypes.data,view.pointer,result.nbytes) if device.cuda else device.copy(result.ctypes.data,view.pointer,result.nbytes,2))
        return result
    try:
        for fn in (checked_cube,loop_error):
            source=to_native_source_ir(trace(fn,np.ones(1,np.float32),source_control_flow=True,max_steps=2,
                source_error_specs=(((1,),'f32'),)),autodiff='reverse')
            program=materialize_source_vjp(source,compiler=args.compiler,llvm_bin='/usr/lib/llvm-23/bin',backend=args.backend,chip=chip,capacity=4)
            for value in (2.,-2.):
                should_fail=(value<0) if fn is checked_cube else (value>0)
                x=np.array([value],np.float32)
                inputs=upload(x);seed=upload(np.ones(1,np.float32))
                try:
                    with program.run(inputs,cotangents=(seed,)) as frame:
                        assert not should_fail
                        np.testing.assert_allclose(read(frame.primals[0]),x**(3 if fn is checked_cube else 2))
                        np.testing.assert_allclose(read(frame.derivatives[0]),3*x*x if fn is checked_cube else 2*x)
                        rows.append(dict(family=fn.__name__,value=value,derivative_matched=True))
                except RuntimeError as error:
                    assert should_fail and type(error) is RuntimeError,error
                    np.testing.assert_array_equal(error.args[0],x*x*(4 if fn is loop_error else 1))
                    np.testing.assert_array_equal(error.__cause__.args[0],2*x)
                    assert error.__cause__ is error.__context__
                    rows.append(dict(family=fn.__name__,value=value,exception_matched=True,derivative_exposed=False))
        source=to_native_source_ir(trace(runtime_slice,np.ones(8,np.float32),*[np.ones(1,np.int64)]*3,source_control_flow=True))
        native=_run(args.compiler,'--tessera-to-linalg',source=source)
        buffered=_run(Path('/usr/lib/llvm-23/bin/mlir-opt'),'--allow-unregistered-dialect',
            '--one-shot-bufferize=bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map',
            '--convert-linalg-to-loops','--canonicalize',source=native)
        from tessera.compiler.native_public_result import _prepare, NativePublicResult
        from tessera.compiler.native_gpu_storage import build_native_gpu_storage
        gpu,_,_=_prepare(buffered,args.compiler,args.backend,8)
        package=build_native_gpu_storage(gpu,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip)
        program=NativePublicResult(buffered,args.compiler,package,8)
        for bounds in ((1,7,2),(-6,-1,2),(-100,100,3),(7,1,1),(0,8,100),(-(1<<63),(1<<63)-1,1),(7,0,-1)):
            data=np.arange(8,dtype=np.float32)
            with program.run(upload(data),*[upload(np.array([n],np.int64)) for n in bounds]) as frame:
                np.testing.assert_array_equal(read(frame.results[0]),data[slice(*bounds)])
            rows.append(dict(family='runtime_slice',bounds=bounds,matched=True))
        for step in (0,):
            try:
                with program.run(upload(np.ones(8,np.float32)),*[upload(np.array([n],np.int64)) for n in (0,8,step)]):
                    raise AssertionError('invalid step exposed results')
            except RuntimeError as error:assert 'guard failed' in str(error)
            rows.append(dict(family='runtime_slice',step=step,guard_refused=True))
    finally:
        for pointer in reversed(pointers):device.check(device.free(pointer))
    sources=['python/tessera/compiler/source_control_flow.py','python/tessera/compiler/trace.py','python/tessera/compiler/native_public_result.py','src/transforms/lib/NativeTapeToGPUPass.cpp','src/transforms/lib/AutodiffPairedPass.cpp','benchmarks/record_source_generation_ad_gpu.py']
    packet=dict(backend=args.backend,chip=chip,compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},cases=rows,performance_promotion=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(packet,indent=2)+'\n')
    print(f'{args.backend}: {len(rows)} cases passed')


if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as error:
        print(error.stderr,file=sys.stderr)
        raise
