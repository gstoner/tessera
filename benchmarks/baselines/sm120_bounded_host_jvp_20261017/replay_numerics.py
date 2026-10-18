from pathlib import Path
import hashlib
import json
import lzma
import os
import subprocess
from unittest.mock import patch

import argparse
parser=argparse.ArgumentParser(description="Compiler-disabled replay and FP64 checks for this packet")
parser.add_argument("--output",type=Path,required=True)
options=parser.parse_args()
if options.output.exists():
    raise ValueError("choose a fresh replay output")
w=Path(__file__).resolve().parent
packet=json.loads(lzma.decompress((w/'paired-public-forward.json.xz').read_bytes()))
device=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',
    '--query-gpu=name,uuid,compute_cap,driver_version','--format=csv,noheader'],text=True).strip()
assert device==packet['device']
for key in ['TESSERA_OPT','TESSERA_NVIDIA_OPT','TESSERA_LLVM_BIN','TESSERA_NATIVE_STORAGE_LLVM_BIN','CUDA_HOME','CUDACXX']:
    os.environ.pop(key,None)
os.environ['PATH']='/usr/bin:/bin'
import numpy as np
import ml_dtypes
from tessera.compiler.native_attention_jvp_runtime import PreparedAttentionJVP
from tests.device.nvidia.test_public_typed_attention_jvp import reference
from tests.device.nvidia.test_saved_lse_half_backward_kernel import compare

sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
         for p in packet['source_sha256']}
assert sources==packet['source_sha256']
def forbidden(*args,**kwargs):
    raise AssertionError('standalone native replay attempted a subprocess')
owners={};rows=[];finite_differences=[]
try:
 with patch('subprocess.run',forbidden),patch('subprocess.Popen',forbidden):
  for row in packet['rows']:
    sq,sk=row['actual_sq_sk']
    dtype=np.dtype(ml_dtypes.bfloat16 if row['storage']=='bfloat16' else row['storage'])
    rng=np.random.default_rng(20261017+sq*131+sk)
    values={n:rng.normal(0,.3,shape).astype(dtype) for n,shape in {
        'q':(1,2,sq,16),'k':(1,1,sk,16),'v':(1,1,sk,16)}.items()}
    if row['bias']:values['bias']=rng.normal(0,.1,(1,2,1,1)).astype(np.float32)
    seeds={n:rng.normal(0,.1,x.shape).astype(x.dtype) for n,x in values.items()}
    expected=reference(*(values[n] for n in ('q','k','v')),
        *(seeds[n] for n in ('q','k','v')),bias=values.get('bias'),dbias=seeds.get('bias'),causal=True)
    h=1e-4
    high={n:x.astype(np.float64)+h*seeds[n].astype(np.float64) for n,x in values.items()}
    low={n:x.astype(np.float64)-h*seeds[n].astype(np.float64) for n,x in values.items()}
    def primal(v):
        return reference(*(v[n] for n in ('q','k','v')),
            *(seeds[n] for n in ('q','k','v')),bias=v.get('bias'),causal=True)[0]
    plus,minus=primal(high),primal(low)
    fd=[(a-b)/(2*h) for a,b in zip(plus,minus,strict=True)]
    for actual,wanted in zip(fd,expected[1],strict=True):
        np.testing.assert_allclose(actual,wanted,rtol=2e-6,atol=2e-8)
    finite_differences.append({'storage':row['storage'],'bias':row['bias'],'sq_sk':[sq,sk],
        'max_abs_errors':[float(np.max(np.abs(a-b))) for a,b in zip(fd,expected[1],strict=True)]})
    for arm,identity in row['identities'].items():
        digest=identity['program_digest']
        archive=(w/identity['artifact']).read_bytes()
        assert hashlib.sha256(archive).hexdigest()==identity['artifact_sha256']
        if digest not in owners:
            raw=lzma.decompress(archive).decode()
            from tessera.compiler.native_attention_program import NativeAttentionJVPProgram
            program=NativeAttentionJVPProgram.from_json(raw,expected_digest=digest)
            names=[f'primal_{i}' for i in range(len(program.input_indices))]
            names += [f'tangent_{program.input_indices[r]}' for r in program.active]
            metadata={'program_json':raw,'program_digest':digest,'arg_names':names}
            owners[digest]=(PreparedAttentionJVP(metadata),metadata)
        owner,metadata=owners[digest];program=owner.program
        active=[program.input_names[program.input_indices[r]] for r in program.active]
        args=tuple(values[n] for n in program.input_names)+tuple(seeds[n] for n in active)
        result=owner.invoke(metadata,args)
        errors=[]
        for actual_group,target_group in zip(result,expected,strict=True):
            for actual,target in zip(actual_group,target_group,strict=True):
                compare(actual,target)
                a=actual.astype(np.float64);difference=a-target
                errors.append({'dtype':str(actual.dtype),'shape':list(actual.shape),
                    'max_abs_error':float(np.max(np.abs(difference))),
                    'rms_error':float(np.sqrt(np.mean(difference*difference))),
                    'relative_l2':float(np.linalg.norm(difference)/max(np.linalg.norm(target),np.finfo(np.float64).tiny))})
        rows.append({'storage':row['storage'],'bias':row['bias'],'sq_sk':[sq,sk],
            'arm':arm,'program_digest':digest,'outputs_O_LSE_dO_dLSE':errors})
 for p,digest in sources.items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
 result={'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':device,'source_revision':packet['source_revision'],'source_sha256':sources,
    'scope':'fresh-process native replay with compiler environments removed and all subprocesses forbidden; independent FP64 oracle on identical stored values; FP64 finite differences of the unrounded model checked separately',
    'runtime_sha256':hashlib.sha256(Path(os.environ['TESSERA_NVIDIA_PTX_LAUNCH_LIB']).read_bytes()).hexdigest(),
    'replay_rows':rows,'finite_difference_rows':finite_differences,'unique_programs':len(owners)}
 options.output.parent.mkdir(parents=True,exist_ok=True)
 options.output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'replay_rows':len(rows),'finite_difference_rows':len(finite_differences),'unique_programs':len(owners)}))
finally:
 for owner,metadata in owners.values():owner.close()
