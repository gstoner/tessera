import hashlib,json,os
from pathlib import Path
import numpy as np
import ml_dtypes
import tessera as ts
from tessera import runtime as rt
from benchmarks.nvidia.benchmark_public_softmax_alias import safe,ordinary

@ts.jit(target="nvidia_sm120")
def matmul(a,b):
    return ts.ops.matmul(a,b,output_dtype="fp32")

rows=[]
saved=[]
rng=np.random.default_rng(5081)
for index,(shape,dtype) in enumerate([
 ((3,1),"fp32"),((17,257),"fp16"),((129,513),"bf16"),
 ((1,1),"bf16"),((257,1025),"fp32"),((3,17),"fp16"),
 ((17,35),"bf16"),((2,3,257),"fp32"),((3,1),"fp16")]):
 storage={"fp32":np.float32,"fp16":np.float16,"bf16":ml_dtypes.bfloat16}[dtype]
 x=rng.uniform(-8,8,shape).astype(storage)
 xf=x.astype(np.float64);ex=np.exp(xf-xf.max(axis=-1,keepdims=True))
 expected=ex/ex.sum(axis=-1,keepdims=True)
 for function in (safe,ordinary):
  actual=function(x)
  assert function.execution_kind=="native_gpu"
  np.testing.assert_allclose(actual.astype(np.float64),expected,rtol=.01,atol=2e-4)
  saved.append((actual,actual.copy()))
  rows.append({"shape":list(shape),"dtype":dtype,"kind":function.__name__,
               "max_abs_error":float(np.max(np.abs(actual.astype(np.float64)-expected)))})
 a=rng.normal(0,.2,(17,35)).astype(np.float16)
 b=rng.normal(0,.2,(35,19)).astype(np.float16)
 out=matmul(a,b)
 assert matmul.execution_kind=="native_gpu"
 np.testing.assert_allclose(out,a.astype(np.float64)@b.astype(np.float64),rtol=.005,atol=.001)
 for actual,previous in saved:np.testing.assert_array_equal(actual,previous)
runtime=Path(rt._load_nvidia_ptx_launch()._name).resolve()
packet={"scope":"owning RTX5070 synchronous grow/shrink and interleaved matmul arena lifetime numerics; no concurrency or performance claim",
        "runtime_path":str(runtime),"runtime_sha256":hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "retained_results":len(saved),"matmul_interleaves":9,"rows":rows}
Path("/home/angstorms/scratch/nvidia-softmax-staging-lifetime-20261008.json").write_text(json.dumps(packet,indent=2)+"\n")
print(json.dumps({"rows":len(rows),"retained_results":len(saved),"matmul_interleaves":9}))
