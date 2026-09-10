#!/usr/bin/env python3
"""Prove two scoped readers precede asynchronous derivative pool frees."""
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
from benchmarks.record_native_tape_extensions import data_while_source  # noqa: E402
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); d=Device(args.backend)
    chip='sm_120' if d.cuda else 'gfx1151'
    pair=materialize_persistent_tape(data_while_source(),compiler=args.compiler,
        llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip)
    def bind(cu,hip,types):
        fn=getattr(d.lib,cu if d.cuda else hip);fn.argtypes,fn.restype=types,ct.c_int
        return fn
    P=ct.c_void_p
    create=bind('cuStreamCreate','hipStreamCreateWithFlags',[ct.POINTER(P),ct.c_uint])
    destroy=bind('cuStreamDestroy_v2','hipStreamDestroy',[P])
    copy=bind('cuMemcpyDtoDAsync_v2','hipMemcpyDtoDAsync',[P,P,ct.c_size_t,P])
    streams=[P(),P(),P()]; pointers=[]; rows=[]
    for stream in streams:d.check(create(ct.byref(stream),1))
    def upload(value):
        pointer=P();d.check(d.alloc(ct.byref(pointer),value.nbytes));pointers.append(pointer)
        d.check(d.htod(pointer,value.ctypes.data,value.nbytes) if d.cuda else d.copy(pointer,value.ctypes.data,value.nbytes,1))
        return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,typestr=value.dtype.str,data=(pointer.value,False)))
    try:
        for x0,steps in ((.6,0),(.3,1),(.15,2),(.05,3)):
            x=np.full(4,x0,np.float32); w=np.full(4,2,np.float32)
            inputs=[upload(x),upload(w)];seed=upload(np.ones(4,np.float32))
            destinations=[[upload(np.zeros(4,np.float32)) for _ in range(2)] for _ in range(2)]
            with pair.capture(*inputs) as frame:
                count=len(frame.buffers)
                generation=frame.backward_async(streams[0].value,seed,tracked=True)
                # Fail the test if a healthy tracked path uses a context barrier.
                sync=frame.sync
                def forbidden():raise AssertionError('tracked retirement used a context barrier')
                frame.sync=forbidden
                try:
                    with generation.read_many(*(stream.value for stream in streams[:2])) as readers:
                        for stream,targets in zip(streams[:2],destinations,strict=True):
                            outputs = readers[stream.value]
                            for output,target in zip(outputs,targets,strict=True):
                                src=output.__cuda_array_interface__['data'][0]
                                dst=target.__cuda_array_interface__['data'][0]
                                d.check(copy(P(dst),P(src),16,stream))
                    for outputs in readers.values():
                        try:outputs[0].__cuda_array_interface__
                        except ValueError:pass
                        else:raise AssertionError('borrow survived its scope')
                    generation.retire(streams[2].value)
                    deadline=time.monotonic()+15
                    while not generation.poll():
                        if time.monotonic()>deadline:raise TimeoutError('tracked generation did not retire')
                        time.sleep(.001)
                    assert len(frame.buffers)==count
                finally:frame.sync=sync
                expected=(w**steps,np.zeros_like(w) if steps==0 else steps*x*w**(steps-1))
                for targets in destinations:
                    for target,reference in zip(targets,expected,strict=True):
                        actual=np.empty(4,np.float32);ptr=target.__cuda_array_interface__['data'][0]
                        d.check(d.dtoh(actual.ctypes.data,ptr,16) if d.cuda else d.copy(actual.ctypes.data,ptr,16,2))
                        np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-7)
                rows.append(dict(exit_steps=steps,readers=2,retirement_stream=3,context_waits=0,pool_frees=2))
    finally:
        for pointer in pointers:d.check(d.free(pointer))
        for stream in streams:d.check(destroy(stream))
    names=['python/tessera/compiler/native_reader_retirement.py','python/tessera/compiler/native_device_tape.py',
           'python/tessera/compiler/native_persistent_tape.py','python/tessera/compiler/native_gpu_tensor.py',
           'python/tessera/compiler/native_gpu_storage.py','src/transforms/lib/AutodiffPairedPass.cpp',
           'src/transforms/lib/NativeTapeToGPUPass.cpp']
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,rows=rows,
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        promotion_eligible=False,overlap_proven=False),indent=2)+'\n')
    print('Four native tape generations retired after two readers without a context barrier')


if __name__=='__main__':main()
