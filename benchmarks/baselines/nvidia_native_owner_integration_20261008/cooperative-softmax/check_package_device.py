from pathlib import Path
import ast, importlib.util, sys, os, json, hashlib, subprocess
import numpy as np
import ml_dtypes
root=Path(__file__).parent
os.environ["TESSERA_OPT"]=str(root/"tessera-opt-contract")
os.environ["TESSERA_NVIDIA_OPT"]=str(root/"tessera-opt-contract")
os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]=str(root/"libtessera_nvidia_ptx_launch.so")
import tessera.compiler
name="tessera.compiler.scheduled_kernel"
spec=importlib.util.spec_from_file_location(name,root/"scheduled_kernel.py")
scheduled=importlib.util.module_from_spec(spec);sys.modules[name]=scheduled;spec.loader.exec_module(scheduled)
tessera.compiler.scheduled_kernel=scheduled
from tessera.compiler import nvidia_native
from tessera import runtime as rt
for module,file,function in ((nvidia_native,"nvidia_native.py","package_scheduled_kernel"),(rt,"runtime.py","_submit_nvidia_sm120_native")):
 tree=ast.parse((root/file).read_text())
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/file),"exec"),module.__dict__)
from benchmarks.nvidia.benchmark_public_softmax_alias import ordinary,safe
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,compute_cap","--format=csv,noheader"],text=True).strip()
assert "12.0" in device,device
rows=[]
for dtype,storage in (("fp16",np.float16),("bf16",ml_dtypes.bfloat16),("fp32",np.float32)):
 for shape in ((3,1),(3,17),(2,3,257),(3,4096),(3,4097)):
  x=np.random.default_rng(5070128).uniform(-20,20,shape).astype(storage)
  x.reshape(-1,shape[-1])[0]=storage(1000)
  graph=ordinary._traced_autodiff_module((x,),{})
  before=graph.to_mlir()
  artifact=scheduled.lower_scheduled_kernel(graph,target="nvidia_sm120",schedule="cooperative_128")
  assert before==graph.to_mlir()
  package=nvidia_native.package_scheduled_kernel(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
  assert package.descriptor.entry_symbol.endswith("_cooperative_128")
  native=rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},native_image=package.image,launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
  bindings=sorted(package.descriptor.buffers,key=lambda b:b.ordinal)
  source_name,output_name=bindings[0].name,bindings[1].name
  scalars={"Rows":int(np.prod(shape[:-1])),"K":shape[-1]}
  probes=[("finite",x)]
  if shape[-1]>1:
   for label,value in (("nan",np.nan),("positive_infinity",np.inf),("negative_infinity",-np.inf)):
    probe=x.copy();probe.reshape(-1,shape[-1])[1]=value;probes.append((label,probe))
  for label,probe in probes:
   with np.errstate(invalid="ignore"):
    xf=probe.astype(np.float64);ex=np.exp(xf-xf.max(axis=-1,keepdims=True));oracle=ex/ex.sum(axis=-1,keepdims=True)
   out=np.empty_like(probe)
   result=rt.launch(native,{source_name:probe,output_name:out,**scalars})
   assert result.get("ok"),result
   np.testing.assert_allclose(out.astype(np.float64),oracle,rtol=.01 if dtype!="fp32" else 3e-5,atol=2e-4 if dtype!="fp32" else 2e-6,equal_nan=True)
   session=NvidiaDeviceSession()
   try:
    args={source_name:session.upload(probe),output_name:session.empty(shape,storage),**scalars}
    result=rt.launch(native,args,stream=session.stream);assert result.get("ok"),result
    resident=session.download(args[output_name])
    np.testing.assert_allclose(resident.astype(np.float64),oracle,rtol=.01 if dtype!="fp32" else 3e-5,atol=2e-4 if dtype!="fp32" else 2e-6,equal_nan=True)
   finally: session.close()
   finite=np.isfinite(oracle)
   rows.append(dict(dtype=dtype,shape=list(shape),probe=label,host="passed",resident="passed",max_abs_error=float(np.max(np.abs(out.astype(np.float64)[finite]-oracle[finite]))) if finite.any() else None,entry=package.descriptor.entry_symbol,schedule_digest=artifact.schedule_digest))
   print(json.dumps(rows[-1]),flush=True)
packet=dict(device=device,compiler_sha256=hashlib.sha256((root/"tessera-opt-contract").read_bytes()).hexdigest(),runtime_sha256=hashlib.sha256((root/"libtessera_nvidia_ptx_launch.so").read_bytes()).hexdigest(),source_sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ("scheduled_kernel.py","nvidia_native.py","runtime.py","tessera_nvidia_ptx_launch.cpp","PMPasses.cpp","TileOps.cpp","NVIDIALowering.cpp")},rows=rows,timing="not_run",integration="isolated_candidate")
(root/"package_device_proof.json").write_text(json.dumps(packet,indent=2)+"\n")
print(json.dumps(dict(profiles=len(rows),host_and_resident_checks=2*len(rows))),flush=True)

