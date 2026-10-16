"""Same-image portable validation overhead and native event timing."""
from pathlib import Path
from contextlib import ExitStack
import os
import hashlib
import json
import statistics
import time
import csv
import io
import subprocess
import argparse
import sys
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler import nvidia_tensor_jvp_runtime as binding
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.nvidia.active_source_jvp_reference import inputs,oracle,product,product_reversed  # noqa: E402
from benchmarks.nvidia import active_source_jvp_reference as reference  # noqa: E402

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--resident-inputs',action='store_true',
                    help='Compare resident-root public/replay calls using the same native owner/images')
parser.add_argument('--include-projection-shapes',action='store_true',
                    help='Also evaluate synthetic M256/N1024/K1536 and K2048 FP16/BF16 projections')
options=parser.parse_args()
probe=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
    '--query-gpu=name,compute_cap,driver_version,uuid','--format=csv,noheader'],
    check=True,text=True,capture_output=True)
devices=list(csv.reader(io.StringIO(probe.stdout)))
if len(devices)!=1 or devices[0][1].strip()!='12.0' or rt._nvidia_device_name()!='sm_120':
    raise RuntimeError('replay benchmark requires one visible exact SM120 GPU')
device_name,compute_capability,driver_version,device_uuid=(v.strip() for v in devices[0])
def source_snapshot():
    paths=[
        'python/tessera/compiler/jit.py',
        'python/tessera/compiler/prepared_nvidia_active_source.py',
        'python/tessera/compiler/nvidia_tensor_jvp_runtime.py',
        'src/compiler/ir/TangentInterface.cpp',
        'src/compiler/ir/AdjointInterface.cpp',
        'src/compiler/ir/include/Tessera/IR/NormalizationAD.h',
        'src/compiler/programming_model/lib/NativeRowScalarizer.h',
        'src/transforms/lib/NativeSM120ActiveSourceProgram.h',
        'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/matmul_prepared.cpp',
        'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/active_source_owner.inc',
        'src/compiler/codegen/tessera_gpu_backend_NVIDIA/runtime/cuda/tessera_nvidia_active_source_program.h',
        'benchmarks/nvidia/benchmark_active_source_jvp.py',
        'benchmarks/nvidia/active_source_jvp_reference.py',
    ]
    tool=Path(rt._tessera_opt_path())
    provider=Path(os.environ['TESSERA_NVIDIA_PTX_LAUNCH_LIB'])
    return {
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),
        'files':{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
        'compiler_sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),
        'runtime_sha256':hashlib.sha256(provider.read_bytes()).hexdigest(),
    }

source_before=source_snapshot()
rows=[]
profiles=[('f16',(17,19,35),False),('bf16',(17,19,35),True),
          ('f16',(129,33,257),False),('bf16',(129,33,257),True)]
if options.include_projection_shapes:
    profiles.extend((storage,(256,1024,k),storage=='bf16')
                    for k in (1536,2048) for storage in ('f16','bf16'))
for storage,shape,permuted in profiles:
    a,b,da=inputs({'shape_mnk':shape,'storage':storage})
    args=(b,a) if permuted else (a,b);roots=(*args,da)
    expected,budgets=oracle(a,b,da)
    fn=ts.jit(target='nvidia_sm120',autodiff='forward',wrt=('a',))(
        product_reversed if permuted else product)
    def check(outputs):
        for actual,reference,budget in zip(outputs,expected,budgets,strict=True):
            assert np.isfinite(actual).all()
            assert np.all(np.abs(actual.astype(np.float64)-reference)<=budget)
    def replay(values=roots):
        result=rt.launch(artifact,values)
        if not result['ok']:raise RuntimeError(str(result))
        return result['output']
    stack=ExitStack()
    try:
        check(fn.native_jvp(*args,tangents=(da,)))
        owner=next(iter(fn._native_prepared_jvp_calls.values()))
        artifact=rt.RuntimeArtifact.from_json(fn.native_jvp_runtime_artifact().to_json())
        check(replay());handle=owner.handle;stats=owner.scratch_stats()
        initial_images=owner.receipt['image_digests']
        samples={'public_jvp':[],'portable_cached_certificate':[],'portable_full_certificate':[]}
        calls={'public_jvp':lambda:fn.native_jvp(*args,tangents=(da,)),
               'portable_cached_certificate':replay,'portable_full_certificate':replay}
        resident_roots=None
        if options.resident_inputs:
            from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
            sessions=[stack.enter_context(NvidiaDeviceSession()) for _ in roots]
            resident_roots=tuple(session.upload(value) for session,value in
                                 zip(sessions,roots,strict=True))
            check(fn.native_jvp(*resident_roots[:2],tangents=(resident_roots[2],)))
            samples.update(resident_public_jvp=[],resident_portable_cached_certificate=[])
            calls.update(
                resident_public_jvp=lambda:fn.native_jvp(*resident_roots[:2],tangents=(resident_roots[2],)),
                resident_portable_cached_certificate=lambda:replay(resident_roots))
        for sample in range(7):
            order=list(samples) if sample%2==0 else list(reversed(samples))
            for mode in order:
                if mode.endswith('cached_certificate'):check(calls[mode]())
                start=time.perf_counter()
                for _ in range(5):
                    if mode=='portable_full_certificate':binding._certificates.clear()
                    pair=calls[mode]()
                samples[mode].append((time.perf_counter()-start)*1e6/5)
                check(pair)
        check(owner(roots))  # Establish the host lease after any resident call.
        pair,events=owner.profile(roots,repeats=100,captured=True);check(pair)
        resident_events=None
        if resident_roots is not None:
            pair,resident_events=owner.profile(resident_roots,repeats=100,captured=True)
            check(pair)
            assert owner.receipt['input_binding']=='cuda_resident_roots'
            assert owner.receipt['root_uploads']==0
        assert owner.handle==handle and owner.scratch_stats()==stats
        assert owner.receipt['image_digests']==initial_images
        rows.append({'storage':storage,'shape_mnk':shape,'permuted':permuted,
            'host_completed_samples_us':samples,
            'host_completed_medians_us':{k:statistics.median(v) for k,v in samples.items()},
            'device_captured_program_ms':events['program_ms'],'device_captured_stage_ms':events['stage_ms'],
            'artifact_hash':artifact.artifact_hash,
            'image_digests':owner.receipt['image_digests'],
            'scratch_capacity_bytes':stats[0],'scratch_allocation_count':stats[1],
            'same_native_owner_and_images':True,'correctness':'independent_fp64_before_after',
            **({'resident_input_device_captured_program_ms':resident_events['program_ms'],
                'resident_input_device_captured_stage_ms':resident_events['stage_ms'],
                'root_uploads_per_call':{'host':3,'resident':0},
                'result_ownership':'two independent completed host f32 arrays'}
               if resident_events is not None else {})})
    finally:
        try:
            fn.close_native_storage()
        finally:
            stack.close()
source_after=source_snapshot()
if source_after!=source_before:
    raise RuntimeError('source, compiler or runtime changed during benchmark')
root=Path(ts.__file__).resolve().parent
packet={'device':device_name,'compute_capability':compute_capability,'driver':driver_version,'device_uuid':device_uuid,
    'compiler_sha256':hashlib.sha256(owner.native.tool.read_bytes()).hexdigest(),
    'compiler_identity':owner.native.compiler_identity,
    'compiler_path':str(owner.native.tool.resolve()),
    'runtime_sha256':hashlib.sha256(Path(owner.lib._name).read_bytes()).hexdigest(),
    'nvidia_codegen_fingerprints':sorted({r['compiler'] for r in owner._members}),
    'benchmark_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'reference_source_sha256':hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
    'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ['compiler/portable_nvidia_active_source.py','compiler/prepared_nvidia_active_source.py',
         'compiler/nvidia_tensor_jvp_runtime.py','compiler/execution_matrix.py','runtime.py','compiler/jit.py']},
    'scope':'same-owner/image public JVP and portable replay; full certificate control clears only pure validation cache',
    'resident_inputs':options.resident_inputs,
    'include_synthetic_projection_shapes':options.include_projection_shapes,
    'source_before':source_before,'source_after':source_after,
    'rows':rows}
options.output.parent.mkdir(parents=True,exist_ok=True)
options.output.write_text(json.dumps(packet,indent=2)+'\n')
for r in rows:print(r['storage'],r['shape_mnk'],r['host_completed_medians_us'])
