"""Diagnostic Python/native boundary attribution, not performance promotion."""
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import time

import numpy as np
import tessera as ts
from tessera.compiler.prepared_attention_vjp import clear_prepared,prepared

w=Path("/home/angstorms/scratch/sm120-native-row-graph-20261016")
from tests.device.nvidia.test_public_typed_attention_vjp import function
from tests.device.nvidia.test_saved_lse_half_backward_kernel import reference, compare
rng=np.random.default_rng(20261016)
dims=(1,4,2,1,129,32,24)
b,hq,hkv,sq,sk,d,dv=dims
values={name:rng.normal(0,.3,shape).astype(np.float16) for name,shape in {
 "q":(b,hq,sq,d),"k":(b,hkv,sk,d),"v":(b,hkv,sk,dv)}.items()}
values["bias"]=rng.normal(0,.1,(1,hq,1,sk)).astype(np.float32)
do=rng.normal(0,.2,(b,hq,sq,dv)).astype(np.float16)
seed=rng.normal(0,.2,(b,hq,sq)).astype(np.float32)
_,oracle=reference(values["q"],values["k"],values["v"],do,values["bias"],seed)
expected=tuple(oracle[i] for i in (3,2,1,0))
prior=os.environ.get("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION")
records={}
functions={}
try:
 for policy in ("prepass",):
  os.environ["TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION"]=policy
  fn=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=("bias","v","k","q"))(function(True,True))
  functions[policy]=fn
  for _ in range(20):
   outputs=fn.native_backward(**values,out_cotangents=(do,seed))
  for value,wanted in zip(outputs,expected,strict=True):compare(value,wanted)
  owner=prepared(fn.native_backward_runtime_artifact().metadata)
  metadata=fn.native_backward_runtime_artifact().metadata
  ordered=tuple(values[name] for name in fn.arg_names)+(do,seed)
  profile=cProfile.Profile()
  begin=time.perf_counter_ns()
  profile.enable()
  for _ in range(200):
   outputs=fn.native_backward(**values,out_cotangents=(do,seed))
  profile.disable()
  elapsed=(time.perf_counter_ns()-begin)/1e6
  for value,wanted in zip(outputs,expected,strict=True):compare(value,wanted)
  path=w/(policy+"-boundary.prof")
  profile.dump_stats(str(path))
  text=io.StringIO()
  stats=pstats.Stats(profile,stream=text)
  stats.sort_stats("cumulative").print_stats(35)
  stats.sort_stats("tottime").print_stats(25)
  (w/(policy+"-boundary.txt")).write_text(text.getvalue())
  records[policy]={"profiled_calls":200,"profiled_wall_ms":elapsed,
   "profiled_source":"Python cProfile includes instrumentation and native blocking; attribution diagnostic only",
   "program_json_bytes":len(owner.identity[1]),"profile_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
  print(policy,records[policy],flush=True)
  profile2=cProfile.Profile()
  profile2.enable()
  for _ in range(200):
   outputs=owner.invoke(metadata,ordered)
  profile2.disable()
  profile2.dump_stats(str(w/"prepared-boundary.prof"))
  text2=io.StringIO()
  pstats.Stats(profile2,stream=text2).sort_stats("cumulative").print_stats(30)
  (w/"prepared-boundary.txt").write_text(text2.getvalue())
 (w/"boundary-profile.json").write_text(json.dumps({"shape":dims,"storage":"float16","arms":records,
  "not_a_speedup_measurement":True},indent=2)+"\n")
finally:
 for fn in functions.values():fn.close_native_storage()
 clear_prepared()
 if prior is None:os.environ.pop("TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION",None)
 else:os.environ["TESSERA_NVIDIA_ATTENTION_ROW_CORRECTION"]=prior
