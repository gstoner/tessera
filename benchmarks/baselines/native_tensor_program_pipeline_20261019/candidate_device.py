import ctypes as ct
import hashlib
import json
import subprocess
from pathlib import Path
import ml_dtypes
import numpy as np
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tests.device.nvidia.test_scalar_adjoint_contraction import driver

p=Path(__file__).parent
images=p/"candidate";images.mkdir(exist_ok=True)
subprocess.run(["tar","-xzf",str(p/"candidate-exact-source-images.tar.gz"),"-C",str(images)],check=True)
device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
 "--query-gpu=name,compute_cap,uuid,driver_version","--format=csv,noheader"],text=True).strip()
assert "RTX 5070, 12.0" in device
index=json.loads((images/"index.json").read_text())
lib=driver()
records=[]
rng=np.random.default_rng(20261019)
for case in index["cases"]:
 name=case["name"]
 frame=json.loads((images/case["frame"]).read_text())
 dtype=np.float16 if name.endswith("-f16") else ml_dtypes.bfloat16
 unit=2.0**(-11 if dtype is np.float16 else -8)
 for mode in ("counterexample","changed_input"):
  a=(np.full((17,32),-.5) if mode=="counterexample" else rng.normal(size=(17,32))).astype(dtype)
  b=(np.ones((32,11)) if mode=="counterexample" else rng.normal(size=(32,11))).astype(dtype)
  lhs=np.maximum(a.astype(np.float64),0).astype(dtype)
  rhs=b
  if name.startswith("norm"):
   wide=a.astype(np.float64)
   lhs=(wide/np.sqrt(np.mean(wide*wide,axis=-1,keepdims=True)+float(np.float32(1e-5)))).astype(dtype)
  if name.startswith("two"):
   wide=b.astype(np.float64);wide-=wide.max(axis=-1,keepdims=True)
   rhs=(np.exp(wide)/np.exp(wide).sum(axis=-1,keepdims=True)).astype(dtype)
  expected={}
  expected[frame["roots"][0]]=a
  expected[frame["roots"][1]]=b
  unary=[lhs,rhs] if name.startswith("two") else [lhs]
  for stage,value in zip(frame["stages"][:-1],unary):
   expected[stage["output"]]=value
  expected[frame["output"]]=lhs.astype(np.float64)@rhs.astype(np.float64)
  magnitudes=np.abs(lhs.astype(np.float64))@np.abs(rhs.astype(np.float64))
  errors=[]
  with NvidiaDeviceSession() as session:
   buffers={}
   originals={}
   for spec in frame["buffers"]:
    id=spec["id"];dt={"f16":np.float16,"bf16":ml_dtypes.bfloat16,"f32":np.float32}[spec["storage"]]
    size=int(np.prod(spec["shape"]))
    host=np.full(size+16,123,dt)
    if spec["role"]=="input": host[:size]=expected[id].ravel()
    buffers[id]=session.upload(host)
    originals[id]=host
   for stage in frame["stages"]:
    text=(images/stage["ptx_file"]).read_bytes()
    assert hashlib.sha256(text).hexdigest()==stage["ptx_sha256"]
    module,function=ct.c_void_p(),ct.c_void_p()
    encoded=ct.create_string_buffer(text)
    assert lib.cuModuleLoadData(ct.byref(module),encoded)==0
    try:
     assert lib.cuModuleGetFunction(ct.byref(function),module,stage["entry"].encode())==0
     values=[ct.c_uint64(buffers[id].ptr) for id in [*stage["inputs"],stage["output"]]]
     values += [ct.c_int64(value) for value in stage["scalars"]]
     params=(ct.c_void_p*len(values))(*(ct.cast(ct.pointer(value),ct.c_void_p) for value in values))
     gx,gy,gz,bx,by,bz=stage["geometry"]
     assert lib.cuLaunchKernel(function,gx,gy,gz,bx,by,bz,0,ct.c_void_p(session.stream),params,None)==0
     assert session.synchronize()==0
    finally:
     assert lib.cuModuleUnload(module)==0
    spec=frame["buffers"][stage["output"]]
    raw=session.download(buffers[stage["output"]])
    size=int(np.prod(spec["shape"]))
    np.testing.assert_array_equal(raw[size:],123)
    actual=raw[:size].reshape(spec["shape"]).astype(np.float64)
    oracle=expected[stage["output"]].astype(np.float64)
    delta=np.abs(actual-oracle)
    if stage["output"]==frame["output"]:
     if mode=="counterexample": np.testing.assert_array_equal(actual,oracle)
     else:
      bound=4*unit*magnitudes+(32*2**-24/(1-32*2**-24))*magnitudes+1e-6
      assert np.all(delta<=bound), (name,mode,delta.max(),bound.max())
    else:
     np.testing.assert_allclose(actual,oracle,rtol=2*unit,atol=1e-6)
    errors.append(float(delta.max()))
   for spec in frame["buffers"]:
    id=spec["id"];raw=session.download(buffers[id]);size=int(np.prod(spec["shape"]))
    np.testing.assert_array_equal(raw[size:],originals[id][size:])
    if spec["role"]=="input": np.testing.assert_array_equal(raw.view(np.uint8),originals[id].view(np.uint8))
  result={"name":name,"mode":mode,"stages":len(frame["stages"]),
          "max_abs_error_per_stage":errors,"all_guards_preserved":True,"roots_unchanged":True,
          "status":"passed","source_sha256":frame["source_sha256"],"program_sha256":frame["program_sha256"]}
  records.append(result)
  print(json.dumps(result),flush=True)
(p/"candidate-device.json").write_text(json.dumps({
 "scope":"diagnostic generic CUDA-driver replay of compiler-projected frame and member images; not retained runtime/public package proof",
 "device":device,"compiler_sha256":index["compiler_sha256"],"cases":records},indent=2)+"\n")
