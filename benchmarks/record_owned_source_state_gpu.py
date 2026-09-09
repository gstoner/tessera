#!/usr/bin/env python3
"""Exclusive GPU state mutation with scoped readers and stable allocation."""
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
from tessera.compiler.trace import trace  # noqa: E402
from tessera.compiler.source_control_flow import to_native_source_ir  # noqa: E402
from tessera.compiler.native_source_state import materialize_source_state, OwnedSourceGPUState  # noqa: E402


def advance(x):
    x[:]=x+x
    return x*x


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    device=Device(args.backend)
    chip='sm_120' if device.cuda else 'gfx1151'
    source=to_native_source_ir(trace(advance,np.ones(4,np.float32),
        source_control_flow=True,source_state_groups=((0,),)))
    program=materialize_source_state(source,compiler=args.compiler,llvm_bin='/usr/lib/llvm-23/bin',
                                     backend=args.backend,chip=chip,capacity=4)
    x=np.full(4,2,np.float32);pointer=ct.c_void_p()
    device.check(device.alloc(ct.byref(pointer),x.nbytes))
    frames=[]
    try:
        device.check(device.htod(pointer,x.ctypes.data,x.nbytes) if device.cuda else device.copy(pointer,x.ctypes.data,x.nbytes,1))
        view=SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=(4,),typestr='<f4',data=(pointer.value,False)))
        rows=[]
        for value in (2.,4.):
            frame=program.run(view);frames.append(frame)
            observed=[]
            for output in frame.results:
                a=np.empty(4,np.float32)
                device.check(device.dtoh(a.ctypes.data,output.pointer,16) if device.cuda else device.copy(a.ctypes.data,output.pointer,16,2))
                observed.append(a)
            np.testing.assert_array_equal(observed[0],np.full(4,(2*value)**2,np.float32))
            np.testing.assert_array_equal(observed[1],np.full(4,2*value,np.float32))
            rows.append(dict(value=observed[0].tolist(),state=observed[1].tolist()))
            view=frame.results[1]
        original=np.empty_like(x)
        device.check(device.dtoh(original.ctypes.data,pointer,16) if device.cuda else device.copy(original.ctypes.data,pointer,16,2))
        np.testing.assert_array_equal(original,x)
        initial=SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=(4,),typestr='<f4',data=(pointer.value,False)))
        with OwnedSourceGPUState(program,initial) as owner:
            with owner.read() as borrowed:
                before=borrowed.__cuda_array_interface__['data'][0]
                try:owner.step()
                except ValueError as error:assert 'reader scopes' in str(error)
                else:raise AssertionError('mutation admitted a live reader')
            try:_=borrowed.__cuda_array_interface__
            except ValueError as error:assert 'scope ended' in str(error)
            else:raise AssertionError('expired reader exposed its pointer')
            for expected in (8.,16.,32.):
                with owner.step() as result:
                    value=np.empty(4,np.float32)
                    device.check(device.dtoh(value.ctypes.data,result.results[0].pointer,16) if device.cuda else device.copy(value.ctypes.data,result.results[0].pointer,16,2))
                    np.testing.assert_array_equal(value,np.full(4,expected**2,np.float32))
                with owner.read() as state:
                    address=state.__cuda_array_interface__['data'][0]
                    assert address==before
                    actual=np.empty(4,np.float32)
                    device.check(device.dtoh(actual.ctypes.data,address,16) if device.cuda else device.copy(actual.ctypes.data,address,16,2))
                    np.testing.assert_array_equal(actual,np.full(4,expected,np.float32))
                rows.append(dict(owned_state=actual.tolist(),allocation_reused=True))
    finally:
        for frame in reversed(frames):frame.close()
        device.check(device.free(pointer))
    files=['python/tessera/compiler/source_control_flow.py','python/tessera/compiler/trace.py',
           'python/tessera/compiler/native_source_state.py','python/tessera/compiler/native_public_result.py',
           'src/transforms/lib/NativeTapeToGPUPass.cpp']
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,execution_kind='native_gpu',
        rows=rows,in_place_mutation=True,promotion_eligible=False,
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},
        source_ir_sha256=hashlib.sha256(source.encode()).hexdigest(),
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('Three in-place owned steps matched; live and expired readers refused')


if __name__=='__main__':main()
