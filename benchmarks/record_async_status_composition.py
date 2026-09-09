#!/usr/bin/env python3
"""Exact-device truth table for capture plus upstream status composition."""
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
from tessera.compiler.native_persistent_tape import materialize_persistent_tape  # noqa: E402

SOURCE='''module { func.func @square(%x: tensor<4xf32>) -> tensor<4xf32> attributes {tessera.autodiff = "reverse"} {
%y = "tessera.mul"(%x,%x) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>
return %y : tensor<4xf32> } }'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('nvidia','rocm'),required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();device=Device(args.backend);P=ct.c_void_p
    chip='sm_120' if device.cuda else 'gfx1151'
    pair=materialize_persistent_tape(SOURCE,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),backend=args.backend,chip=chip,checked_status=True,gated_input=True,status_inputs=2)
    def bind(cu,hip,types):
        fn=getattr(device.lib,cu if device.cuda else hip);fn.argtypes=types;fn.restype=ct.c_int;return fn
    create=bind('cuStreamCreate','hipStreamCreateWithFlags',[ct.POINTER(P),ct.c_uint])
    destroy=bind('cuStreamDestroy_v2','hipStreamDestroy',[P])
    copy=bind('cuMemcpyDtoDAsync_v2','hipMemcpyDtoDAsync',[P,P,ct.c_size_t,P])
    allocated=[];streams=[];rows=[]
    def upload(value):
        p=P();device.check(device.alloc(ct.byref(p),value.nbytes));allocated.append(p)
        device.check(device.htod(p,value.ctypes.data,value.nbytes) if device.cuda else device.copy(p,value.ctypes.data,value.nbytes,1))
        return SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,typestr=value.dtype.str,data=(p.value,False)))
    def forbidden():raise AssertionError('composition attempted host capture synchronization')
    try:
        for _ in range(3):
            stream=P();device.check(create(ct.byref(stream),1));streams.append(stream)
        x=upload(np.full(4,2,np.float32));y=upload(np.full(4,3,np.float32));seed=upload(np.ones(4,np.float32));failure=upload(np.array([1],np.int64));destination=upload(np.zeros(4,np.float32))
        for capture_failed,upstream_failed in [(False,False),(True,False),(False,True),(True,True)]:
            upstream=pair.capture_async(streams[0].value,x)
            target=pair.capture_async(streams[1].value,y)
            for frame,failed,stream in [(upstream,upstream_failed,streams[0]),(target,capture_failed,streams[1])]:
                frame.poll_capture=forbidden
                frame.sync=forbidden
                for binding in frame._bindings:
                    if binding._bound is not None:binding._bound._sync=forbidden
                if failed:
                    # Fault injection on the owning stream, after capture and
                    # before its consumer. Models a failed device guard.
                    device.check(copy(frame._status[0].pointer,P(failure.__cuda_array_interface__['data'][0]),8,stream))
            first=upstream.backward_async(streams[0].value,seed,tracked=True)
            composed=first.backward_into(target,streams[1].value)
            expected_success=not(capture_failed or upstream_failed)
            try:
                composed.wait_success()
                success=True
            except RuntimeError as error:
                if "persistent GPU product guard failed" not in str(error):
                    raise
                success=False
            assert success==expected_success
            if success:
                with composed.read(streams[1].value) as outputs:
                    device.check(copy(P(destination.__cuda_array_interface__['data'][0]),P(outputs[0].__cuda_array_interface__['data'][0]),16,streams[1]))
            for frame in (target,upstream):
                frame.retire(streams[2].value)
                deadline=time.monotonic()+20
                while not frame.poll_retired():
                    if time.monotonic()>deadline:raise TimeoutError('composition retirement stalled')
                    time.sleep(.001)
                assert not frame.buffers
            if success:
                output=np.empty(4,np.float32);p=destination.__cuda_array_interface__['data'][0]
                device.check(device.dtoh(output.ctypes.data,p,16) if device.cuda else device.copy(output.ctypes.data,p,16,2))
                np.testing.assert_array_equal(output,np.full(4,24,np.float32))
            rows.append(dict(capture_failed=capture_failed,upstream_failed=upstream_failed,success=success,retained_buffers=0))
    finally:
        for p in allocated:device.check(device.free(p))
        for stream in streams:device.check(destroy(stream))
    names=['python/tessera/compiler/native_persistent_tape.py','src/transforms/lib/NativeTapeToGPUPass.cpp','python/tessera/compiler/native_reader_retirement.py','python/tessera/compiler/native_gpu_storage.py','python/tessera/compiler/native_gpu_tensor.py','python/tessera/compiler/native_module_retirement.py','python/tessera/compiler/native_storage_contract.py']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(backend=args.backend,chip=chip,execution_kind='native_gpu',rows=rows,
        forward_binding=pair.forward.binding_digest,backward_binding=pair.backward.binding_digest,
        sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),promotion_eligible=False),indent=2)+'\n')
    print('All four status combinations passed')


if __name__=='__main__':main()
