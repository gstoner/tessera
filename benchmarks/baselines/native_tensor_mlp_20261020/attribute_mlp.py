import ctypes as ct,json,os
from pathlib import Path
import numpy as np
import tessera as ts
from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame
from tests.unit.test_native_tensor_mlp_program import mlp,mlp_roots
from tests.device.nvidia.test_native_tensor_mlp_program import reference
provider=ct.CDLL(os.environ["TESSERA_NATIVE_FRAME_TEST_LIB"])
rows=[]
for storage,seed in [("f16",120321),("bf16",120322)]:
 fn=ts.jit(target="nvidia_sm120")(mlp);args=mlp_roots(storage,shape=(64,256,128,64),seed=seed)
 try:
  actual=fn(*args);p=fn._native_tensor_last_program
  stages=[]
  for count in (1,2):
   f=p.frame;f["stages"]=f["stages"][:count];f["output"]=2+count;f["buffers"]=f["buffers"][:3+count];f["buffers"][-1]["role"]="output"
   with PreparedNativeTensorFrame(f,p.images[:count],provider=provider) as owner:stages.append(owner(args))
  x,up,down=(v.astype(np.float64) for v in args)
  exact=x@up;rounded=exact.astype(np.float16)
  hidden=stages[0]
  delta=hidden.astype(np.float64)-rounded.astype(np.float64)
  reconstructed=np.maximum(0,hidden).astype(np.float64)@down
  error=np.abs(actual.astype(np.float64)-reference(args,False,"f16"))
  indices=np.argwhere(delta!=0)
  rows.append({"storage":storage,"seed":seed,"producer_different_from_exact_then_round":int(np.count_nonzero(delta)),
    "producer_max_difference":float(np.max(np.abs(delta))),"activation_bitwise_preserves_native_producer_relu":bool(np.array_equal(stages[1].view(np.uint16),np.maximum(0,hidden).view(np.uint16))),
    "consumer_max_error_against_actual_producer":float(np.max(np.abs(actual.astype(np.float64)-reconstructed))),
    "whole_mlp_max_error":float(error.max()),
    "producer_examples":[{"index":i.tolist(),"exact":float(exact[tuple(i)]),"ideal_half":float(rounded[tuple(i)]),"native_half":float(hidden[tuple(i)])} for i in indices[:12]]})
 finally:fn.close_native_storage()
out=Path("/home/angstorms/scratch/w11-program-pipeline-20261019/next-producer-census/mlp-numerical-attribution.json")
out.write_text(json.dumps({"scope":"Diagnostic native member prefix readback; original public program and images unchanged","rows":rows},indent=2)+"\n")
print(out.read_text())
