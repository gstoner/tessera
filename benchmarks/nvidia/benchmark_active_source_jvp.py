"""Same-image portable validation overhead and native event timing."""
from pathlib import Path
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
options=parser.parse_args()
probe=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi',
    '--query-gpu=name,compute_cap,driver_version,uuid','--format=csv,noheader'],
    check=True,text=True,capture_output=True)
devices=list(csv.reader(io.StringIO(probe.stdout)))
if len(devices)!=1 or devices[0][1].strip()!='12.0' or rt._nvidia_device_name()!='sm_120':
    raise RuntimeError('replay benchmark requires one visible exact SM120 GPU')
device_name,compute_capability,driver_version,device_uuid=(v.strip() for v in devices[0])
rows=[]
for storage,shape,permuted in [('f16',(17,19,35),False),('bf16',(17,19,35),True),
                              ('f16',(129,33,257),False),('bf16',(129,33,257),True)]:
    a,b,da=inputs({'shape_mnk':shape,'storage':storage})
    args=(b,a) if permuted else (a,b);roots=(*args,da)
    expected,budgets=oracle(a,b,da)
    fn=ts.jit(target='nvidia_sm120',autodiff='forward',wrt=('a',))(
        product_reversed if permuted else product)
    def check(outputs):
        for actual,reference,budget in zip(outputs,expected,budgets,strict=True):
            assert np.isfinite(actual).all()
            assert np.all(np.abs(actual.astype(np.float64)-reference)<=budget)
    def replay():
        result=rt.launch(artifact,roots)
        if not result['ok']:raise RuntimeError(str(result))
        return result['output']
    try:
        check(fn.native_jvp(*args,tangents=(da,)))
        owner=next(iter(fn._native_prepared_jvp_calls.values()))
        artifact=rt.RuntimeArtifact.from_json(fn.native_jvp_runtime_artifact().to_json())
        check(replay());handle=owner.handle;stats=owner.scratch_stats()
        samples={'public_jvp':[],'portable_cached_certificate':[],'portable_full_certificate':[]}
        calls={'public_jvp':lambda:fn.native_jvp(*args,tangents=(da,)),
               'portable_cached_certificate':replay,'portable_full_certificate':replay}
        for sample in range(7):
            order=list(samples) if sample%2==0 else list(reversed(samples))
            for mode in order:
                if mode=='portable_cached_certificate':check(replay())
                start=time.perf_counter()
                for _ in range(5):
                    if mode=='portable_full_certificate':binding._certificates.clear()
                    pair=calls[mode]()
                samples[mode].append((time.perf_counter()-start)*1e6/5)
                check(pair)
        pair,events=owner.profile(roots,repeats=100,captured=True);check(pair)
        assert owner.handle==handle and owner.scratch_stats()==stats
        rows.append({'storage':storage,'shape_mnk':shape,'permuted':permuted,
            'host_completed_samples_us':samples,
            'host_completed_medians_us':{k:statistics.median(v) for k,v in samples.items()},
            'device_captured_program_ms':events['program_ms'],'device_captured_stage_ms':events['stage_ms'],
            'artifact_hash':artifact.artifact_hash,
            'image_digests':owner.receipt['image_digests'],
            'scratch_capacity_bytes':stats[0],'scratch_allocation_count':stats[1],
            'same_native_owner_and_images':True,'correctness':'independent_fp64_before_after'})
    finally:fn.close_native_storage()
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
    'rows':rows}
options.output.parent.mkdir(parents=True,exist_ok=True)
options.output.write_text(json.dumps(packet,indent=2)+'\n')
for r in rows:print(r['storage'],r['shape_mnk'],r['host_completed_medians_us'])
