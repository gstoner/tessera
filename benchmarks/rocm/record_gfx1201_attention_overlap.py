"""Same-device HIP-event overlap windows for two independent attention owners.

Events bracket each ordered kernel program (including resets). HIP events alone
are not a kernel-side clock witness, so this WSL evidence cannot promote a
production performance candidate (MASTER_AUDIT 2026-09-25 timing rule).
"""
import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import threading

import numpy as np
from tessera import runtime as rt
from tessera.compiler.resident_rocm_attention import ResidentROCmAttentionTape
from tessera.compiler.scheduled_attention_backward import lower_scheduled_attention_backward
from tessera.compiler.rocm_native import package_scheduled_attention_backward
from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module, _reference


def record():
    if rt._rocm_live_arch() != 'gfx1201':
        raise RuntimeError('requires owning gfx1201 device')
    schedule=lower_scheduled_attention_backward(_module(1,4,2,129,131,64,dtype='fp16',
        dropout_p=0,lse_checkpoint='recompute'),target='rocm_gfx1201')
    program=package_scheduled_attention_backward(schedule,pipeline_name='tessera-lower-to-rocm')
    rng=np.random.default_rng(911)
    q,k,v,do=[(rng.normal(size=s)*0.2).astype(np.float16) for s in
              [(1,4,129,64),(1,2,131,64),(1,2,131,64),(1,4,129,64)]]
    bias=np.zeros((1,4,129,131),np.float32)
    buffers=dict(q=q,key=k,v=v,do=do,bias=bias,dq=np.empty(q.shape,np.float32),
        dk=np.empty(k.shape,np.float32),dv=np.empty(v.shape,np.float32))
    expected=_reference(do,q,k,v,bias,dropout_p=0)
    hip=rt._load_hip_for_launch(); original=rt._load_hip_for_launch
    records=[]; deferred=set(); active=False; lock=threading.Lock()
    class TimingHIP:
        def __getattr__(self,name):
            fn=getattr(hip,name)
            if name=='hipDeviceSynchronize':
                raise RuntimeError('device-wide synchronization would invalidate overlap')
            if name=='hipEventRecord':
                def event_record(event,stream):
                    status=fn(event,stream)
                    if active and status==0:
                        with lock: records.append((event.value,stream.value)); deferred.add(event.value)
                    return status
                return event_record
            if name=='hipEventDestroy':
                def event_destroy(event):
                    return 0 if event.value in deferred else fn(event)
                return event_destroy
            return fn
    rt._load_hip_for_launch=lambda:TimingHIP()
    def check(status):
        if status:
            raise RuntimeError(f"HIP overlap measurement status {status}")
    rows=[]; origin=ct.c_void_p()
    try:
        with ResidentROCmAttentionTape(program,buffers) as first, ResidentROCmAttentionTape(program,buffers) as second:
            first.backward(do);second.backward(do)
            for trial in range(5):
                records.clear();deferred.clear()
                check(hip.hipEventCreate(ct.byref(origin)))
                check(hip.hipEventRecord(origin,None))
                check(hip.hipEventSynchronize(origin))
                active=True
                a=first.submit(do);b=second.submit(do)
                for result in (a.result(),b.result()):
                    for got,want in zip(result['outputs'],expected,strict=True):
                        np.testing.assert_allclose(got,want,rtol=0.04,atol=0.003)
                active=False
                by_stream={}
                for event,stream in records:
                    elapsed=ct.c_float()
                    check(hip.hipEventElapsedTime(ct.byref(elapsed),origin,ct.c_void_p(event)))
                    by_stream.setdefault(str(stream),[]).append(float(elapsed.value))
                if len(by_stream)!=2 or any(len(v)!=2 for v in by_stream.values()):
                    raise RuntimeError('expected two independently recorded program windows')
                windows=list(by_stream.values())
                if any(not all(math.isfinite(x) for x in w) or w[1]<=w[0] for w in windows):
                    raise RuntimeError('invalid device event window; no overlap evidence')
                overlap=max(0.0,min(w[1] for w in windows)-max(w[0] for w in windows))
                rows.append(dict(trial=trial,program_windows_ms=windows,overlap_ms=overlap))
                for event in deferred: check(hip.hipEventDestroy(ct.c_void_p(event)))
                deferred.clear();check(hip.hipEventDestroy(origin));origin=ct.c_void_p()
    finally:
        active=False
        rt._load_hip_for_launch=original
        # Normal completion destroyed all held events. On failure the probe
        # process owns any remaining handles until exit; no unsafe teardown.
    return dict(target='gfx1201',image_digest=program.image.image_digest,rows=rows,
        clock_calibrated=False,performance_eligible=False,
        scope='overlapping ordered-program event windows, not per-kernel counter attribution')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.write_text(json.dumps(record(),indent=2)+'\n')
