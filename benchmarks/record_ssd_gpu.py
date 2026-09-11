#!/usr/bin/env python3
"""Owning-device numerical proof for the replay-bound serial SSD baseline."""
import argparse
import os
import uuid
import ctypes as ct
import hashlib
import json
import statistics
import time
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'python')]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402
from tessera.compiler.native_ssd import materialize_ssd  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',choices=['nvidia','rocm'],required=True)
    parser.add_argument('--compiler',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cooperative',action='store_true')
    parser.add_argument('--shape',type=int,nargs=4,default=(5,2,3,2))
    parser.add_argument('--chunk',type=int)
    parser.add_argument('--profile',action='store_true')
    args = parser.parse_args()
    run_id = uuid.uuid4().hex
    device = Device(args.backend)
    rows = []
    T,H,N,P = args.shape
    for chunk in ((args.chunk,) if args.chunk else (1,2,5)):
        logical = lower_scheduled_ssd(T,H,N,P,chunk,compiler=args.compiler)
        program = materialize_ssd(logical,compiler=args.compiler,llvm_bin=Path('/usr/lib/llvm-23/bin'),
                                  backend=args.backend,chip='sm_120' if device.cuda else 'gfx1151',cooperative=args.cooperative)
        rng = np.random.default_rng(740+chunk)
        inputs = [rng.uniform(-.5,.5,shape).astype(np.float32)
                  for shape in [(T,H,P),(T,H),(T,H,N),(T,H,N),(H,N,P)]]
        x,decay,b,c,state = inputs
        state = state.copy()
        y = np.empty_like(x)
        saved = []
        for t in range(T):
            state = decay[t,:,None,None]*state+b[t,:,:,None]*x[t,:,None,:]
            y[t] = (c[t,:,:,None]*state).sum(axis=1)
            if (t+1)%chunk == 0 or t == T-1:
                saved.append(state.copy())
        expected = [y,state,np.array(saved)]
        outputs = [np.full_like(v,np.nan) for v in expected]
        pointers,views = [],[]
        bind_start = time.perf_counter()
        binding = program.bind()
        bind_ms = (time.perf_counter()-bind_start)*1000
        try:
            for value in inputs+outputs:
                pointer = ct.c_void_p()
                device.check(device.alloc(ct.byref(pointer),value.nbytes)); pointers.append(pointer)
                device.check(device.htod(pointer,value.ctypes.data,value.nbytes) if device.cuda
                             else device.copy(pointer,value.ctypes.data,value.nbytes,1))
                views.append(SimpleNamespace(__cuda_array_interface__=dict(version=3,shape=value.shape,
                             typestr=value.dtype.str,data=(pointer.value,False))))
            launch_start = time.perf_counter()
            binding(*views,1)
            checked_call_ms = (time.perf_counter()-launch_start)*1000
            observed = []
            for i,value in enumerate(inputs+outputs):
                result = np.empty_like(value)
                device.check(device.dtoh(result.ctypes.data,pointers[i],result.nbytes) if device.cuda
                             else device.copy(result.ctypes.data,pointers[i],result.nbytes,2))
                if i < 5:
                    np.testing.assert_array_equal(result,value)
                else:
                    np.testing.assert_allclose(result,expected[i-5],rtol=1e-5,atol=1e-6)
                    observed.append(float(np.max(np.abs(result-expected[i-5]))))
            timings = []
            if args.profile:
                # Resident direct launches exclude copies and Python descriptor
                # validation. Device events and end-to-end calls stay separate.
                raw,_,grid,block,_ = binding._resident(*views,1)
                values,shared = binding._bound._launch_size(raw,grid,block)
                argv = (ct.c_void_p*len(values))(*[ct.cast(ct.byref(v),ct.c_void_p) for v in values])
                start,end = ct.c_void_p(),ct.c_void_p()
                device.check(device.event_create(ct.byref(start),0))
                try:
                    device.check(device.event_create(ct.byref(end),0))
                    for _ in range(7):
                        device.check(device.event_record(start,None))
                        for _ in range(100):
                            device.check(device.launch(binding._bound._function,*grid,*block,shared,None,argv,None))
                        device.check(device.event_record(end,None)); device.check(device.event_sync(end))
                        ms = ct.c_float()
                        device.check(device.event_elapsed(ct.byref(ms),start,end))
                        timings.append(ms.value/100)
                finally:
                    if end.value: device.check(device.event_destroy(end))
                    device.check(device.event_destroy(start))
            rows.append(dict(chunk=chunk,binding_ms=bind_ms,checked_call_ms=checked_call_ms,device_event_ms=timings,
                             device_event_median_ms=statistics.median(timings) if timings else None,max_abs_errors=observed,binding_digest=program.package.binding_digest,
                             image_sha256=hashlib.sha256(program.package.image).hexdigest()))
        finally:
            binding.close()
            for pointer in reversed(pointers):
                device.check(device.free(pointer))
    args.output.write_text(json.dumps(dict(schema=1,backend=args.backend,process_id=os.getpid(),run_id=run_id,
        architecture='sm_120' if device.cuda else 'gfx1151',
        compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
        shape=args.shape,clock='CUDA events' if device.cuda else 'HIP events',execution='native_gpu',cooperative=args.cooperative,rows=rows,promotion_eligible=False),indent=2)+'\n')


if __name__ == '__main__':
    main()
