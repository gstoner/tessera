"""Public tuple-attention owner costs and matched native package windows."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
from benchmarks.nvidia.benchmark_jit_attention_vjp import download
from tests.device.nvidia.test_jit_multiresult_attention_vjp import function, values, reference


def record(samples=5,reps=20):
    gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,uuid,driver_version,compute_cap','--format=csv,noheader'],text=True).strip()
    if len(gpu.splitlines())!=1 or gpu.split(',')[-1].strip()!='12.0':
        raise RuntimeError('requires one selected SM120 GPU: '+gpu)
    rows=[]
    for shape in [(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)]:
      for bias in [False,True]:
       for causal in [False,True]:
        for compact in [False,True]:
            inputs,seeds=values(shape,bias,'mixed')
            expected,grads=reference(inputs,seeds,causal)
            start=time.perf_counter_ns()
            program=function(bias,causal).compile_native_attention_vjp(*inputs,compiler=os.environ['TESSERA_OPT'],compact_gradients=compact)
            compile_ms=(time.perf_counter_ns()-start)/1e6
            program=NativeAttentionVJPProgram.from_json(program.to_json(),expected_digest=program.program_digest)
            pair=program.pair
            timings={key:[] for key in ['capture_wall_ms','backward_wall_ms','paired_wall_ms','forward_device_window_ms','backward_device_window_ms']}
            errors=[]
            with NvidiaDeviceSession() as session:
                resident=[session.upload(x) for x in inputs]
                resident_seeds=tuple(session.upload(x) for x in seeds)
                def check_primal(frame):
                    for value,want in zip(frame.primal,expected,strict=True):
                        np.testing.assert_allclose(download(session,value),want,rtol=4e-5,atol=4e-5)
                def check_grad(actual):
                    for value,role in zip(actual,program.active,strict=True):
                        host=download(session,value)
                        np.testing.assert_allclose(host,grads[role],rtol=4e-5,atol=4e-5)
                        errors.append(float(np.max(np.abs(host-grads[role]))))
                for _ in range(samples):
                    start=time.perf_counter_ns();frame=program.capture(*resident)
                    timings['capture_wall_ms'].append((time.perf_counter_ns()-start)/1e6)
                    try:
                        check_primal(frame)
                        start=time.perf_counter_ns();actual=frame.backward(resident_seeds)
                        timings['backward_wall_ms'].append((time.perf_counter_ns()-start)/1e6)
                        check_grad(actual)
                    finally:frame.close()
                    start=time.perf_counter_ns();frame=program.capture(*resident)
                    try:
                        actual=frame.backward(resident_seeds)
                        timings['paired_wall_ms'].append((time.perf_counter_ns()-start)/1e6)
                        check_primal(frame);check_grad(actual)
                    finally:frame.close()
                scalars=dict(zip(('B','Hq','Hkv','Sq','Sk','D','Dv'),shape,strict=True))
                if bias:scalars.update(dict(zip(('BiasB','BiasH','BiasQ','BiasK'),inputs[3].shape,strict=True)))
                outputs=[session.upload(np.full(x.shape,np.nan,np.float32)) for x in expected]
                physical=pair.backward.descriptor.provenance.get('physical_gradient_roles',list(range(3+int(bias))))
                wanted=[grads[i] if i in program.active else np.zeros_like(grads[i]) for i in physical]
                gradient_buffers=[session.upload(np.full(x.shape,np.nan,np.float32)) for x in wanted]
                fvalues=[*resident,*outputs]
                bvalues=[resident_seeds[0],*resident[:3],outputs[0],*resident[3:],outputs[1],resident_seeds[1],*gradient_buffers]
                for stage,package,buffers,wants,checks in [
                    ('forward',pair.forward,fvalues,expected,outputs),
                    ('backward',pair.backward,bvalues,wanted,gradient_buffers)]:
                    args={**scalars,**dict(zip([x.name for x in package.descriptor.buffers],buffers,strict=True))}
                    for _ in range(samples):
                        timings[stage+'_device_window_ms'].append(rt._nvidia_native_descriptor_resident_device_latency(
                            package.image,package.descriptor,args,stream=session.stream,warmup=2,reps=reps))
                        for value,want in zip(checks,wants,strict=True):
                            np.testing.assert_allclose(download(session,value),want,rtol=4e-5,atol=4e-5)
            rows.append(dict(shape=shape,bias=bias,causal=causal,compact=compact,active=program.active,
                compile_wall_ms=compile_ms,max_abs_error=max(errors),samples=timings,
                medians={key:statistics.median(value) for key,value in timings.items()},
                program_digest=program.program_digest,checkpoint_identity=pair.contract_digest,
                ancestry={stage:package.descriptor.provenance for stage,package in [('forward',pair.forward),('backward',pair.backward)]}))
            print('verified',shape,bias,causal,compact,flush=True)
    sources=['python/tessera/compiler/resident_attention.py','python/tessera/compiler/native_attention_program.py',
        'python/tessera/compiler/scheduled_checkpoint.py','python/tessera/compiler/nvidia_native.py',
        'python/tessera/compiler/jit.py','src/compiler/ir/AttentionADContract.h',
        'src/compiler/ir/AdjointInterface.cpp','src/transforms/lib/AutodiffPairedPass.cpp',
        'benchmarks/nvidia/record_jit_multiresult_owner.py','tests/device/nvidia/test_jit_multiresult_attention_vjp.py']
    return dict(schema='tessera.nvidia.jit_multiresult_owner.v1',gpu=gpu,architecture='sm_120',samples=samples,reps=reps,rows=rows,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        binaries={key:hashlib.sha256(Path(os.environ[key]).read_bytes()).hexdigest()
            for key in ['TESSERA_OPT','TESSERA_NVIDIA_OPT','TESSERA_NVIDIA_PTX_LAUNCH_LIB']},
        timing_scope='Device windows use the same public program packages and include driver gaps. Capture wall includes private copies, module loads and forward. Backward wall includes seed/gradient allocation and synchronization. Pair includes capture/backward; downloads, oracle and frame close excluded. Synchronous owner; no speedup claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=5);parser.add_argument('--reps',type=int,default=20)
    args=parser.parse_args()
    if args.samples<3 or args.reps<20:parser.error('requires >=3 samples and >=20 reps')
    packet=record(args.samples,args.reps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(packet,indent=2)+'\n')
