"""Exact-device public quantizer package, replay and prepared-owner timing."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.rocm_block_quant_native import PreparedBlockQuantProgram,validate_block_quant_runtime_artifact
from tests.unit.test_rocm_block_quant_frontend import make_quantizer
from tests.device.rocm.test_native_block_quant_jit import oracle,check

root=Path(os.environ.get('TESSERA_K16_QUANT_EVIDENCE_DIR','/home/angstorms/scratch/rocm-native-gate-up-views-20261016/k16-quant'))
assert rt._rocm_live_arch()=='gfx1201'
info=subprocess.check_output(['/opt/rocm/bin/rocminfo'],text=True)
device=re.search(r'Name:\s+gfx1201\s+Uuid:\s+(\S+)\s+Marketing Name:\s+([^\n]+)',info)
assert device
rows=[]
for m,k in [(3,32),(19,12288),(256,12288)]:
 x=np.random.default_rng(1217).normal(size=(m,k)).astype(ml_dtypes.bfloat16)
 expected=oracle(x);fn=ts.jit(target='rocm_gfx1201',native_required=True)(make_quantizer(m,k))
 start=time.perf_counter();actual=fn(x);cold=(time.perf_counter()-start)*1e3;check(actual,expected)
 artifact=fn.runtime_artifact();package=validate_block_quant_runtime_artifact(artifact)
 assert artifact.metadata['compiler_path']=='rocm_block_quant_native_descriptor'
 assert fn.compile_result.program_executable,fn.compile_result.component_blockers
 prefix=root/f'public-quant-{m}-{k}'
 prefix.with_suffix('.json').write_text(artifact.to_json())
 public=[]
 for _ in range(3):
  start=time.perf_counter();out=fn(x);public.append((time.perf_counter()-start)*1e3);check(out,expected)
 with PreparedBlockQuantProgram(package,[x],runtime_library=os.environ['TESSERA_ROCM_MULTI_OUTPUT_LIB']) as owner:
  generation,_=owner.invoke();check(owner.read(generation),expected)
  generation,pilot=owner.profile_members(repeats=8)
  repeats=min(65536,max(8,math.ceil(22/max(pilot[0],1e-6))))
  for _ in range(6):
   generation,elapsed=owner.profile_members(repeats=repeats)
   if elapsed[0]*repeats>=18:break
   repeats=min(65536,max(repeats+1,math.ceil(22/max(elapsed[0],1e-6))))
  else:raise AssertionError('captured timing window not established')
  captured=[];completed=[]
  for _ in range(3):
   generation,elapsed=owner.profile_members(repeats=repeats);assert elapsed[0]*repeats>=10;captured.append(elapsed[0]);check(owner.read(generation),expected)
   start=time.perf_counter();generation,_=owner.invoke();out=owner.read(generation);completed.append((time.perf_counter()-start)*1e3);check(out,expected)
 image=artifact.native_image
 row={'shape':[m,k],'architecture':'gfx1201','device':device.group(2).strip(),'gpu_uuid':device.group(1),
  'route':'Python frontend -> typed Graph -> native Schedule -> typed Tile -> ROCm Target -> ROCDL/LLVM -> HSACO -> checked C++ owner',
  'abi':artifact.launch_descriptor.abi_id,'compiler_path':artifact.metadata['compiler_path'],
  'image_sha256':hashlib.sha256(image.payload).hexdigest(),'cold_public_ms':cold,'public_warm_ms':public,
  'captured_member_ms':captured,'captured_repeats':repeats,'prepared_invoke_read_ms':completed,
  'public_scope':'complete ordinary JIT call, including frontend replay, binding and native prepare/execute/readback',
  'device_scope':'captured native graph dispatch; excludes capture, instantiate, host enqueue and copies',
  'prepared_scope':'retained C++ owner invoke plus two output readbacks; excludes compilation and preparation',
  'recorder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'compiler_sha256':hashlib.sha256(Path(os.environ['TESSERA_OPT']).read_bytes()).hexdigest(),
  'provider_sha256':hashlib.sha256(Path(os.environ['TESSERA_ROCM_MULTI_OUTPUT_LIB']).read_bytes()).hexdigest(),
  'correctness':'bitwise FP8 codes and FP32 scales before all timing samples','program_executable':True,
  'manual_diagnostic_image_equal':hashlib.sha256(image.payload).hexdigest()=='6be7bab64720e6d921fa6d2c8287eba2f3fd7f43a19739bcf7afd2a98afdaa97'}
 rows.append(row);(root/'public-package-device.json').write_text(json.dumps(rows,indent=2)+'\n')
 print(m,k,row['image_sha256'],row['captured_member_ms'],row['public_warm_ms'],flush=True)
