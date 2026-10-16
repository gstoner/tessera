from pathlib import Path
import hashlib,json,os,traceback
import numpy as np
import ml_dtypes
import tessera as ts
from tessera import runtime as rt
assert rt._nvidia_device_name()=="sm_120"
OUT=Path("/home/angstorms/scratch/sm120-producer-ad-20261015")
OUT.mkdir(parents=True,exist_ok=True)
def product(a,b):
    normalized=ts.ops.rmsnorm(a,eps=1e-5)
    edge=ts.ops.relu(normalized)
    return ts.ops.matmul(edge,b,output_dtype="fp32")
rows=[]
for storage,dtype in (("f16",np.float16),("bf16",ml_dtypes.bfloat16)):
 rng=np.random.default_rng(151061)
 a=rng.uniform(.1,1,(7,32)).astype(dtype)
 a[:,::2]*=-1
 b=rng.uniform(-.25,.25,(32,9)).astype(dtype)
 plain=ts.jit(target="nvidia_sm120")(product)
 value=plain(a,b)
 assert plain.execution_kind=="native_gpu"
 for wrt in (("b",),("a",),("a","b")):
  forward=ts.jit(target="nvidia_sm120",autodiff="forward",wrt=wrt)(product)
  row={"storage":storage,"wrt":list(wrt),"primal_execution":plain.execution_kind}
  try:
   ir=forward.compiled_jvp_ir(a,b)
   filename=f"{storage}_{''.join(wrt)}_paired.mlir"
   (OUT/filename).write_text(ir)
   row["paired_ir"]={"status":"emitted","file":filename,"sha256":hashlib.sha256(ir.encode()).hexdigest()}
  except Exception as exc:
   row["paired_ir"]={"status":"refused","type":type(exc).__name__,"message":str(exc)}
  try:
   seeds=tuple(np.full_like(a if name=="a" else b,.125) for name in wrt)
   primal,tangent=forward.native_jvp(a,b,tangents=seeds)
   row["public_jvp"]={"status":"executed","receipt":forward.last_jvp_execution}
  except Exception as exc:
   row["public_jvp"]={"status":"refused","type":type(exc).__name__,"message":str(exc)}
  reverse=ts.jit(target="nvidia_sm120",autodiff="reverse",wrt=wrt)(product)
  try:
   gradients=reverse.native_backward(a,b,out_cotangents=np.ones_like(value))
   row["public_vjp"]={"status":"executed","receipt":reverse.last_backward_execution}
  except Exception as exc:
   row["public_vjp"]={"status":"refused","type":type(exc).__name__,"message":str(exc)}
  rows.append(row)
  print(json.dumps(row),flush=True)
files=["src/compiler/ir/TesseraOps.td","src/compiler/ir/TangentInterface.cpp","src/compiler/ir/AdjointInterface.cpp","python/tessera/compiler/jit.py","python/tessera/compiler/native_jvp_plugins.py"]
packet={"scope":"current-source named RMSNorm->ReLU->matmul public AD route census; no new AD execution claim","cases":rows,"source_sha256":{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},"compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),"script_source":Path(__file__).read_text()}
(OUT/"route-census.json").write_text(json.dumps(packet,indent=2)+"\n")
