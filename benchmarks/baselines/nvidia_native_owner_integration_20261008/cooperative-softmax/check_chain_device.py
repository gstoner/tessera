from pathlib import Path
import ast, __future__, os, sys, json, hashlib, subprocess
import numpy as np
import ml_dtypes
root=Path(__file__).parent
exec((root/"check_package_device.py").read_text().split("rows=[]")[0])
os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]=str(root/"libtessera_nvidia_chain_cooperative.so")
from tessera.compiler import nvidia_tensor_lhs as lhs, prepared_nvidia_lhs as prepared
for module,file,names in ((lhs,"nvidia_tensor_lhs.py",{"package_traced_lhs"}),(prepared,"prepared_nvidia_lhs.py",{"PreparedLhsCall"})):
 tree=ast.parse((root/file).read_text())
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/file),"exec",flags=__future__.annotations.compiler_flag),module.__dict__)
from benchmarks.nvidia.benchmark_native_producer_chain import chain,three_chain
from tests.device.nvidia.test_lhs_tensor_jit import softmax_lhs
rows=[]
for dtype,storage in (("fp16",np.float16),("bf16",ml_dtypes.bfloat16)):
 for shape in ((17,35,19),(128,4096,64)):
  m,k,n=shape
  for count,function in ((1,softmax_lhs),(2,chain),(3,three_chain)):
   rng=np.random.default_rng(5070128);x=rng.normal(0,.2,(m,k)).astype(storage)
   rhs=np.array(rng.normal(0,.2,(k,n)),dtype=storage,order="F")
   graph=function._traced_autodiff_module((x,rhs),{})
   graph=lhs.project_rhs_storage(graph,[x,rhs])
   before=graph.to_mlir()
   program=lhs.package_traced_lhs(graph,softmax_schedule="cooperative_128")
   assert graph.to_mlir()==before,"caller Graph mutated"
   packages=program.producer_chain or (program.edge.producer,)
   soft=packages[-1]
   assert soft.descriptor.provenance["schedule"]=="cooperative_128"
   assert soft.descriptor.entry_symbol.endswith("_cooperative_128")
   def oracle(source):
    value=source.astype(np.float64)
    if count==3:
     centered=value-value.mean(axis=-1,keepdims=True)
     value=(centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+1e-5)).astype(storage).astype(np.float64)
    if count>=2:
     value=(value/np.sqrt(np.mean(value*value,axis=-1,keepdims=True)+1e-5)).astype(storage).astype(np.float64)
    exp=np.exp(value-value.max(axis=-1,keepdims=True));value=(exp/exp.sum(axis=-1,keepdims=True)).astype(storage)
    return value.astype(np.float64)@rhs.astype(np.float64)
   owner=prepared.PreparedLhsCall(program)
   try:
    result,receipt=owner([x,rhs])
    np.testing.assert_allclose(result,oracle(x),rtol=.015,atol=.002)
    saved=result.copy()
    changed=-x
    second,receipt2=owner([changed,rhs])
    np.testing.assert_allclose(second,oracle(changed),rtol=.015,atol=.002)
    np.testing.assert_array_equal(result,saved)
    assert receipt["execution_kind"]==receipt2["execution_kind"]=="native_gpu"
    rows.append(dict(dtype=dtype,shape_mkn=list(shape),producer_count=count,max_abs_error=float(np.max(np.abs(second.astype(np.float64)-oracle(changed)))),consumer_route=program.edge.consumer.descriptor.provenance["physical_route"],changed_input="passed",retained_output="unchanged",plan_digest=hashlib.sha256(program.native_plan_json.encode()).hexdigest()))
    print(json.dumps(rows[-1]),flush=True)
   finally:owner.close()
packet=dict(device=device,rows=rows,runtime_sha256=hashlib.sha256((root/"libtessera_nvidia_chain_cooperative.so").read_bytes()).hexdigest(),compiler_sha256=hashlib.sha256((root/"tessera-opt-contract").read_bytes()).hexdigest(),candidate_sources={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ("prepared_nvidia_lhs.py","nvidia_tensor_lhs.py","matmul_prepared.cpp")},timing="not_run",dynamic_multi_guard="unchanged")
(root/"chain_device_proof.json").write_text(json.dumps(packet,indent=2)+"\n")

