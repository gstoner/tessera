from pathlib import Path
import time
from statistics import median
root=Path(__file__).parent
exec((root/"check_package_device.py").read_text().split("rows=[]")[0])
# No timing while another validation or graph extraction process is active.
live=subprocess.check_output(["ps","-eo","args"],text=True).splitlines()
assert not any(("pytest" in p or "graphify update" in p) and not "check_softmax_ab.py" in p for p in live), "validation/graph job is active"
rows=[]
for dtype,storage in (("fp16",np.float16),("bf16",ml_dtypes.bfloat16),("fp32",np.float32)):
 for shape in ((3,17),(128,4096),(128,4097)):
  x=np.random.default_rng(5070128).uniform(-20,20,shape).astype(storage)
  xf=x.astype(np.float64);ex=np.exp(xf-xf.max(axis=-1,keepdims=True));oracle=ex/ex.sum(axis=-1,keepdims=True)
  graph=ordinary._traced_autodiff_module((x,),{})
  packages={}
  for mode in ("serial","cooperative_128"):
   artifact=scheduled.lower_scheduled_kernel(graph,target="nvidia_sm120",schedule=mode)
   packages[mode]=nvidia_native.package_scheduled_kernel(artifact,pipeline_name="tessera-nvidia-pipeline-sm120")
  session=NvidiaDeviceSession()
  try:
   samples={m:dict(device=[],host=[]) for m in packages}; launchers={}
   for mode,package in packages.items():
    bindings=sorted(package.descriptor.buffers,key=lambda b:b.ordinal)
    scalars={"Rows":shape[0],"K":shape[-1]}
    resident={bindings[0].name:session.upload(x),bindings[1].name:session.empty(shape,storage),**scalars}
    out=np.empty_like(x)
    host={bindings[0].name:x,bindings[1].name:out,**scalars}
    native=rt.RuntimeArtifact(metadata={"target":"nvidia_sm120"},native_image=package.image,launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
    launchers[mode]=(native,resident,host,out)
    result=rt.launch(native,host);assert result.get("ok"),result
    np.testing.assert_allclose(out.astype(np.float64),oracle,rtol=.01 if dtype!="fp32" else 3e-5,atol=2e-4 if dtype!="fp32" else 2e-6)
   for window in range(5):
    for mode in (("serial","cooperative_128") if window%2==0 else ("cooperative_128","serial")):
     native,resident,host,out=launchers[mode];package=packages[mode]
     samples[mode]["device"].append(rt._nvidia_native_descriptor_resident_device_latency(package.image,package.descriptor,resident,stream=session.stream,reps=100,warmup=10))
     start=time.perf_counter()
     for _ in range(11):
      result=rt.launch(native,host);assert result.get("ok"),result
     samples[mode]["host"].append((time.perf_counter()-start)*1000/11)
     bindings=sorted(package.descriptor.buffers,key=lambda b:b.ordinal)
     np.testing.assert_allclose(session.download(resident[bindings[1].name]).astype(np.float64),oracle,rtol=.01 if dtype!="fp32" else 3e-5,atol=2e-4 if dtype!="fp32" else 2e-6)
     np.testing.assert_allclose(out.astype(np.float64),oracle,rtol=.01 if dtype!="fp32" else 3e-5,atol=2e-4 if dtype!="fp32" else 2e-6)
   row=dict(dtype=dtype,shape=list(shape),samples=samples,device_speedup=median(samples["serial"]["device"])/median(samples["cooperative_128"]["device"]),host_speedup=median(samples["serial"]["host"])/median(samples["cooperative_128"]["host"]),images={m:p.image.image_digest for m,p in packages.items()},correctness="before_and_after_timing")
   rows.append(row);print(json.dumps(row),flush=True)
  finally:session.close()
base=json.loads((root/"package_device_proof.json").read_text())
packet={k:v for k,v in base.items() if k not in ("rows","timing")}
packet.update(rows=rows,timing="alternating five windows; CUDA-event resident kernel loop and separate host calls; same compiler/runtime",promotion=False)
(root/"softmax_ab.json").write_text(json.dumps(packet,indent=2)+"\n")

