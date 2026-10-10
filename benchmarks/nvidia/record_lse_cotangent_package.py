"""Correctness-gated seeded package CUDA-event and host end-to-end receipt."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time
import subprocess
import numpy as np
from tessera import runtime as rt
from tests._support.nvidia import nvidia_cuda_host_ready
from tests.device.nvidia.test_lse_cotangent_package import fixture


def run(samples=5,reps=20,*,cases=None,fixture_factory=fixture):
    if not nvidia_cuda_host_ready():
        raise RuntimeError('requires exact SM120 RTX 5070 host and matching toolchain')
    device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap,driver_version","--format=csv,noheader"],text=True).strip()
    if len(device.splitlines())!=1 or device.split(",")[2].strip()!="12.0":
        raise RuntimeError("requires one exact SM120 device")
    rows=[]
    if cases is None:
        cases=[dict(shape=shape,bias=bias,causal=causal,seed_mode=seed_mode)
               for shape in [(1,2,1,3,5,4,3),(1,2,1,5,3,4,3),(2,4,2,7,9,8,6)]
               for bias in [False,True] for causal in [False,True]
               for seed_mode in ['output_only','lse_only','mixed']]
    for case in cases:
        artifact,args,expected,input_count=fixture_factory(**case)
        d=artifact.launch_descriptor;i=artifact.native_image
        outputs=d.buffers[input_count:]
        event=[];wall=[];errors=[]
        def poison():
            for x in outputs:args[x.name].fill(np.nan)
        def check():
            values=[]
            for x,want in zip(outputs,expected,strict=True):
                np.testing.assert_allclose(args[x.name],want,rtol=4e-5,atol=4e-5)
                values.append(float(np.max(np.abs(args[x.name].astype(np.float64)-want))))
            errors.append(values)
        for _ in range(samples):
            poison()
            event.append(rt._nvidia_native_descriptor_device_latency(i,d,args,warmup=2,reps=reps))
            check()
            poison();start=time.perf_counter()
            for _ in range(reps):
                result=rt.launch(artifact,args)
                if not result.get('ok') or result.get('execution_kind')!='native_gpu':
                    raise RuntimeError(result)
            wall.append((time.perf_counter()-start)*1000/reps)
            check()
        rows.append({'case':case,'shape':d.provenance['shape'],'bias':d.provenance['bias'],'causal':d.provenance['causal'],
                     'entry':d.entry_symbol,'abi_id':d.abi_id,'event_ms':event,'host_e2e_ms':wall,
                     'event_median_ms':median(event),'host_e2e_median_ms':median(wall),
                     'max_abs_errors':errors,'samples':samples,'reps':reps,
                     'ancestry':{k:d.provenance[k] for k in ['graph_ir_digest','schedule_digest','schedule_ir_digest','tile_ir_digest','target_ir_digest']}})
    sources=['python/tessera/compiler/nvidia_native.py','python/tessera/compiler/lse_cotangent_contract.py',
             'python/tessera/runtime.py','tests/device/nvidia/test_lse_cotangent_package.py',
             'tests/device/nvidia/test_lse_cotangent_native.py','tests/unit/test_lse_cotangent_schedule.py',
             'benchmarks/nvidia/record_lse_cotangent_package.py',
             'python/tessera/compiler/compact_attention_contract.py',
             'src/compiler/programming_model/lib/NativeCheckpoint.h',
             'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_ptx_launch.cpp',
             fixture_factory.__module__.replace('.', '/')+'.py']
    packet={'schema':'seeded_attention_package_v1','device':device,'rows':rows,
            'timing_domains':{'event':'C++ kernel loop; upload/readback outside event',
                              'host_e2e':'checked serialized package calls including validation/module bridge/copies'},
            'source_sha256':{x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in sources}}
    for env in ['TESSERA_OPT','TESSERA_NVIDIA_OPT','TESSERA_NVIDIA_PTX_LAUNCH_LIB']:
        packet[env+'_sha256']=hashlib.sha256(Path(os.environ[env]).read_bytes()).hexdigest()
    return packet


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=5);parser.add_argument('--reps',type=int,default=20)
    options=parser.parse_args()
    if options.samples<1 or options.reps<1:parser.error('positive samples and reps required')
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(run(options.samples,options.reps),indent=2)+'\n')
