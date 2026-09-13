"""Compare scoped reuse and per-call allocation; raw timing, no promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_native import package_scheduled_attention_backward
from tessera.compiler.scheduled_attention_backward import lower_scheduled_attention_backward
from tessera.compiler.resident_rocm_attention import ResidentROCmAttentionTape
from benchmarks.rocm.benchmark_rocm_attention_backward_program import _module, _reference


def record():
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires owning gfx1201 device")
    schedule=lower_scheduled_attention_backward(_module(1,4,2,17,19,64,dtype="fp16",
        dropout_p=0,lse_checkpoint="recompute"),target="rocm_gfx1201")
    program=package_scheduled_attention_backward(schedule,pipeline_name="tessera-lower-to-rocm")
    rng=np.random.default_rng(843)
    q,k,v,do=[(rng.normal(size=s)*0.2).astype(np.float16) for s in
              [(1,4,17,64),(1,2,19,64),(1,2,19,64),(1,4,17,64)]]
    bias=np.zeros((1,4,17,19),np.float32)
    buffers=dict(q=q,key=k,v=v,do=do,bias=bias,dq=np.empty(q.shape,np.float32),
                 dk=np.empty(k.shape,np.float32),dv=np.empty(v.shape,np.float32))
    expected=_reference(do,q,k,v,bias,dropout_p=0)
    rows=[]
    with ResidentROCmAttentionTape(program,buffers) as tape:
        tape.backward(do)
        rt._submit_rocm_gfx1151_attention_backward_program(program,buffers)
        for iteration in range(5):
            for variant in (('per_call','resident') if iteration%2==0 else ('resident','per_call')):
                start=time.perf_counter_ns()
                result=(tape.submit(do).result() if variant=='resident' else
                        rt._submit_rocm_gfx1151_attention_backward_program(program,buffers))
                elapsed=(time.perf_counter_ns()-start)/1e6
                errors=[]
                for got,want in zip(result['outputs'],expected,strict=True):
                    np.testing.assert_allclose(got,want,rtol=0.04,atol=0.003)
                    errors.append(float(np.max(np.abs(got-want))))
                rows.append(dict(variant=variant,iteration=iteration,host_total_ms=elapsed,
                    ordered_program_host_ms=result['kernel_wall_samples_ms'],
                    hip_event_ms=result['device_event_samples_ms'],max_abs_errors=errors))
    return dict(target='gfx1201',image_sha256=hashlib.sha256(program.image.payload).hexdigest(),
                schedule_sha256=schedule.schedule_ir_digest,rows=rows,performance_eligible=False,
                reason='single process, uncalibrated device clocks, no runtime kernel attribution; resident capture and teardown excluded')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.write_text(json.dumps(record(),indent=2)+'\n')
