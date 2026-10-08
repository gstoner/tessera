from pathlib import Path
import base64,json,hashlib,time,statistics,subprocess
import numpy as np
from tessera.compiler.native_scaled_program import NativeScaledProgram,PreparedScaledProgram
root=Path("/home/angstorms/scratch/scaled-matmul-member-images-20261007")
raw=json.loads((root/"projected-package.json").read_text())
package=NativeScaledProgram(raw["program_json"],tuple(raw["members_json"]),tuple(base64.b64decode(v,validate=True) for v in raw["images"]),raw["graph_ir"],raw["target"])
package.validate()
# Replay must not invoke either a compiler or Python numerical kernel builder.
def forbidden(*args,**kwargs):raise AssertionError("compiler invoked during package replay")
subprocess.run=forbidden
rng=np.random.default_rng(20261007)
codes=np.array([0,0x28,0x30,0x38,0x40,0xa8,0xb0,0xb8,0xc0],dtype=np.uint8)
values=np.array([0,.25,.5,1,2,-.25,-.5,-1,-2],dtype=np.float64)
ai=rng.integers(0,len(codes),(17,256));bi=rng.integers(0,len(codes),(256,19))
sa=rng.uniform(.2,1,(17,2)).astype(np.float32);sb=rng.uniform(.2,1,(2,1)).astype(np.float32)
da=rng.uniform(-.3,.3,(17,2)).astype(np.float32);db=rng.uniform(-.3,.3,(2,1)).astype(np.float32)
inputs=[codes[ai],codes[bi],sa,sb,da,db]
ab=[values[ai][:,g*128:(g+1)*128]@values[bi][g*128:(g+1)*128,:] for g in range(2)]
def oracle(x,y):return sum(ab[g]*x[:,g,None].astype(np.float64)*y[g,0].astype(np.float64) for g in range(2))
def expected():return oracle(sa,sb),oracle(da,sb)+oracle(sa,db)
runtime="/home/angstorms/scratch/next-five-compiler-slices-rocm/.build-gfx1201-current/src/compiler/codegen/Tessera_ROCM_Backend/runtime/hip/libtessera_rocm_native_movement.so"
device=[];host=[];updated=[]
with PreparedScaledProgram(package,inputs,runtime_library=runtime) as owner:
 generation,_=owner.invoke()
 out=owner.read(generation)
 for a,w in zip(out,expected()):np.testing.assert_allclose(a,w,rtol=2e-5,atol=2e-4)
 da*=np.float32(-.5);db*=np.float32(2)
 owner.update(inputs)
 try:owner.read(generation)
 except RuntimeError as e:assert "status 10" in str(e)
 else:raise AssertionError("upload did not invalidate old output")
 for _ in range(11):
  generation,ms=owner.invoke(repeats=100,timed=True);device.append(ms)
  start=time.perf_counter();generation,_=owner.invoke();out=owner.read(generation);host.append((time.perf_counter()-start)*1e3)
  start=time.perf_counter();owner.update(inputs);generation,_=owner.invoke();out=owner.read(generation);updated.append((time.perf_counter()-start)*1e3)
  for a,w in zip(out,expected()):np.testing.assert_allclose(a,w,rtol=2e-5,atol=2e-4)
eps=1e-5
fd=(oracle(sa.astype(np.float64)+eps*da,sb.astype(np.float64)+eps*db)-oracle(sa.astype(np.float64)-eps*da,sb.astype(np.float64)-eps*db))/(2*eps)
np.testing.assert_allclose(out[1],fd,rtol=2e-5,atol=2e-4)
import ctypes as c
hip=c.CDLL("/opt/rocm/lib/libamdhip64.so");name=c.create_string_buffer(256)
assert hip.hipDeviceGetName(name,256,0)==0
packet={"architecture":"gfx1201","device":name.value.decode(),"shape_mnk":[17,19,256],"execution":"compiler_owned_member_ABI_to_native_CPP_program_owner","compiler_free_replay":"passed_subprocess_refusal","ordinary_jit_AD":"not_yet_integrated","numerical_max_abs_error":[float(np.max(np.abs(a-w))) for a,w in zip(out,expected())],"finite_difference_max_abs_error":float(np.max(np.abs(out[1]-fd))),"device_native_launch_window_ms":device,"device_native_launch_window_median_ms":statistics.median(device),"warm_invoke_two_readbacks_ms":host,"warm_invoke_two_readbacks_median_ms":statistics.median(host),"update_invoke_two_readbacks_ms":updated,"update_invoke_two_readbacks_median_ms":statistics.median(updated),"timing_note":"device event window includes native C++ enqueue gaps; host windows exclude compilation/preparation","runtime_sha256":hashlib.sha256(Path(runtime).read_bytes()).hexdigest(),"package_sha256":hashlib.sha256((root/"projected-package.json").read_bytes()).hexdigest(),"image_sha256":[hashlib.sha256(v).hexdigest() for v in package.images]}
(root/"projected-package-device.json").write_text(json.dumps(packet,indent=2)+"\n")
print(json.dumps(packet,indent=2))
