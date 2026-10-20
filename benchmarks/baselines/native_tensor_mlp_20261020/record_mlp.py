import hashlib,json,os,time,subprocess
from pathlib import Path
from statistics import median
import numpy as np
import tessera as ts
from tests.unit.test_native_tensor_mlp_program import mlp,default_mlp,mlp_roots
from tests.device.nvidia.test_native_tensor_mlp_program import reference
out=Path("/home/angstorms/scratch/w11-program-pipeline-20261019/next-producer-census")
tool=Path(os.environ["TESSERA_OPT"])
identity=json.loads((tool.parent/"source-identity.json").read_text())
assert identity["compiler_sha256"]==hashlib.sha256(tool.read_bytes()).hexdigest()
gpu=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True).strip()
assert len(gpu.splitlines())==1 and gpu.split(",")[-1].strip()=="12.0"
rows=[]
for shape in [(17,32,48,11),(17,35,29,19)]:
 for storage in ("f16","bf16"):
  for intermediate in ("f16","f32"):
   fn=ts.jit(target="nvidia_sm120")(mlp if intermediate=="f16" else default_mlp)
   args=mlp_roots(storage,shape=shape,intermediate=intermediate)
   try:
    start=time.perf_counter_ns(); actual=fn(*args);cold=(time.perf_counter_ns()-start)/1e6
    expected=reference(args,False,intermediate)
    np.testing.assert_allclose(actual,expected,rtol=.004,atol=2e-5)
    assert fn.execution_kind=="native_gpu"
    program=fn._native_tensor_last_program;owner=fn._native_tensor_last_owner
    before=owner.stats()
    windows={}
    for captured in (False,True):
     raw=owner.profile(64,captured=captured)
     if "output" in raw:np.testing.assert_allclose(raw["output"],expected,rtol=.004,atol=2e-5)
     windows["captured" if captured else "enqueued"]={k:v for k,v in raw.items() if k!="output"}
    samples=[]
    for _ in range(7):
     start=time.perf_counter_ns()
     for _ in range(8):actual=fn(*args)
     samples.append((time.perf_counter_ns()-start)/8e6)
    np.testing.assert_allclose(actual,expected,rtol=.004,atol=2e-5)
    after=owner.stats()
    assert before["allocations"]==after["allocations"]==1
    rows.append({"shape_mkhn":list(shape),"root_profile":storage,"intermediate":intermediate,
     "correctness":"passed_before_and_after_timing","cold_call_ms":cold,
     "completed_public_samples_ms":samples,"completed_public_median_ms":median(samples),
     "device_windows":windows,"frame":program.frame,
     "image_sha256":[hashlib.sha256(x).hexdigest() for x in program.images],
     "max_abs_error":float(np.max(np.abs(actual.astype(np.float64)-expected)))})
    print(storage,intermediate,shape,rows[-1]["completed_public_median_ms"],flush=True)
    (out/"mlp-timing.json").write_text(json.dumps({"gpu":gpu,"compiler_source":identity,"rows":rows,"scope":"Distinct FP16/FP32 precision profiles; no cross-precision speedup claim; stage windows are measured separately from completed public latency"},indent=2)+"\n")
   finally:fn.close_native_storage()
