from pathlib import Path
import importlib.util,sys,os,json
import numpy as np
import ml_dtypes
root=Path(__file__).parent
os.environ["TESSERA_OPT"]=str(root/"tessera-opt-contract")
name="tessera.compiler.scheduled_kernel"
spec=importlib.util.spec_from_file_location(name,root/"scheduled_kernel.py")
scheduled=importlib.util.module_from_spec(spec);sys.modules[name]=scheduled;spec.loader.exec_module(scheduled)
from benchmarks.nvidia.benchmark_public_softmax_alias import ordinary,safe
rows=[]
for dtype,storage in (("fp16",np.float16),("bf16",ml_dtypes.bfloat16),("fp32",np.float32)):
 source=np.zeros((2,3,4096),storage);artifacts=[]
 for function in (ordinary,safe):
  graph=function._traced_autodiff_module((source,),{})
  before=graph.to_mlir()
  artifact=scheduled.lower_scheduled_kernel(graph,target="nvidia_sm120",schedule="cooperative_128")
  assert graph.to_mlir()==before,"caller-owned Graph mutated"
  assert artifact.schedule=="cooperative_128"
  assert 'schedule = "cooperative_128"' in artifact.tile_ir
  artifacts.append(artifact)
  rows.append(dict(dtype=dtype,frontend=function.__name__,schedule_digest=artifact.schedule_digest,caller_graph="unchanged",stage="frontend_trace_to_native_schedule_tile",execution="not_run"))
 assert artifacts[0].schedule_digest==artifacts[1].schedule_digest
 assert artifacts[0].tile_ir==artifacts[1].tile_ir
(root/"frontend_contract_proof.json").write_text(json.dumps(dict(rows=rows,aliases="identical_tile",device_execution="pending"),indent=2)+"\n")
print(json.dumps(dict(rows=len(rows),aliases="identical_tile",caller_graph="unchanged")))
