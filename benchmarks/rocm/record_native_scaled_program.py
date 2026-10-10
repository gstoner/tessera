"""Diagnostic plan admission, numerical checks and native-owner measurements.
This does not implement the production compiler/package projection.
"""
from pathlib import Path
import ctypes as c, hashlib, json, re, time, statistics, os
import numpy as np
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--images",type=Path,required=True)
parser.add_argument("--runtime",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
options=parser.parse_args()
root=options.images
class Buffer(c.Structure):
 _fields_=[("bytes",c.c_uint64),("elements",c.c_uint64),("first_write",c.c_int64),("last_read",c.c_int64),("ownership",c.c_uint32),("reserved",c.c_uint32)]
class Step(c.Structure):
 _fields_=[("image",c.c_void_p),("image_bytes",c.c_uint64),("entry",c.c_char_p),("input_count",c.c_uint32),("scalar_count",c.c_uint32),("inputs",c.c_uint32*6),("output",c.c_uint32),("geometry",c.c_uint32*6),("scalars",c.c_int64*8)]
lib=c.CDLL(str(options.runtime))
prepare=lib.tessera_rocm_program_prepare
prepare.argtypes=[c.c_char_p,c.c_uint32,c.c_uint32,c.POINTER(Buffer),c.c_uint32,c.POINTER(Step),c.POINTER(c.c_void_p),c.POINTER(c.c_uint64),c.POINTER(c.c_uint64)]
invoke=lib.tessera_rocm_program_invoke
invoke.argtypes=[c.c_uint64,c.c_uint32,c.POINTER(c.c_uint64),c.POINTER(c.c_float)]
read=lib.tessera_rocm_program_read
read.argtypes=[c.c_uint64,c.c_uint32,c.c_uint64,c.c_void_p,c.c_uint64]
update=lib.tessera_rocm_program_update
update.argtypes=[c.c_uint64,c.POINTER(c.c_void_p),c.POINTER(c.c_uint64)]
close=lib.tessera_rocm_program_close;close.argtypes=[c.c_uint64]
def check(rc):
 if rc:raise RuntimeError(f"native program status {rc}")
rng=np.random.default_rng(20261007)
codes=np.array([0,0x28,0x30,0x38,0x40,0xa8,0xb0,0xb8,0xc0],dtype=np.uint8)
values=np.array([0,.25,.5,1,2,-.25,-.5,-1,-2],dtype=np.float64)
ai=rng.integers(0,len(codes),(17,256));bi=rng.integers(0,len(codes),(256,19))
sa=rng.uniform(.2,1,(17,2)).astype(np.float32);sb=rng.uniform(.2,1,(2,1)).astype(np.float32)
da=rng.uniform(-.3,.3,(17,2)).astype(np.float32);db=rng.uniform(-.3,.3,(2,1)).astype(np.float32)
host=[codes[ai],codes[bi],sa,sb,da,db]
out=[np.empty((17,19),np.float32) for _ in range(2)]
ab=[values[ai][:,g*128:(g+1)*128]@values[bi][g*128:(g+1)*128,:] for g in range(2)]
def oracle(x,y):return sum(ab[g]*x[:,g,None].astype(np.float64)*y[g,0].astype(np.float64) for g in range(2))
expected=[oracle(sa,sb),oracle(da,sb)+oracle(sa,db)]
inputs=(c.c_void_p*6)(*(a.ctypes.data for a in host))
sizes=(c.c_uint64*6)(*(a.nbytes for a in host))
buffers=(Buffer*10)()
last=[2,2,2,1,1,2,4,3,3,4]
for i in range(10):
 a=host[i] if i<6 else out[0]
 buffers[i]=Buffer(a.nbytes,a.size,-1 if i<6 else i-6,last[i],0 if i<6 else 2 if i in (6,9) else 1,0)
plan=(Step*4)();blobs=[];entries=[]
for i,slots in enumerate(([0,1,2,3],[0,1,4,3],[0,1,2,5],[7,8])):
 blob=c.create_string_buffer((root/f"member-{i}.hsaco").read_bytes());blobs.append(blob)
 entry=re.search(r'gpu.kernel_metadata<"([^"]+)"',(root/f"member-{i}.mlir").read_text()).group(1).encode();entries.append(entry)
 geometry=(2,2,1,32,1,1) if i<3 else (2,1,1,256,1,1)
 scalars=[17,19,256] if i<3 else [323]
 plan[i]=Step(c.cast(blob,c.c_void_p),len(blob.raw)-1,entry,len(slots),len(scalars),(c.c_uint32*6)(*slots),i+6,(c.c_uint32*6)(*geometry),(c.c_int64*8)(*scalars))
handle=c.c_uint64()
# Reject a future input, forged lifetime and wrong byte size before allocations.
rejections={}
original=plan[3].inputs[0];plan[3].inputs[0]=9
rejections["future_input"]=prepare(b"gfx1201",6,10,buffers,4,plan,inputs,sizes,c.byref(handle));assert rejections["future_input"]==1 and handle.value==0
plan[3].inputs[0]=original
buffers[7].last_read=2
rejections["forged_lifetime"]=prepare(b"gfx1201",6,10,buffers,4,plan,inputs,sizes,c.byref(handle));assert rejections["forged_lifetime"]==1 and handle.value==0
buffers[7].last_read=3
sizes[0]-=1
rejections["wrong_bytes"]=prepare(b"gfx1201",6,10,buffers,4,plan,inputs,sizes,c.byref(handle));assert rejections["wrong_bytes"]==1 and handle.value==0
sizes[0]+=1
rejections["wrong_architecture"]=prepare(b"gfx1151",6,10,buffers,4,plan,inputs,sizes,c.byref(handle));assert rejections["wrong_architecture"]==2 and handle.value==0
original_entry=plan[3].entry
plan[3].entry=b"missing_scaled_program_symbol"
rejections["missing_symbol"]=prepare(b"gfx1201",6,10,buffers,4,plan,inputs,sizes,c.byref(handle))
assert rejections["missing_symbol"]==3 and handle.value!=0
check(close(handle))
plan[3].entry=original_entry
check(prepare(b"gfx1201",6,10,buffers,4,plan,inputs,sizes,c.byref(handle)))
try:
 generation=c.c_uint64()
 assert read(handle,6,0,c.c_void_p(out[0].ctypes.data),out[0].nbytes)==10
 # Single native call submits and completes all four products/sum.
 check(invoke(handle,1,c.byref(generation),None))
 for slot,a,want in zip((6,9),out,expected):
  check(read(handle,slot,generation.value,c.c_void_p(a.ctypes.data),a.nbytes))
  np.testing.assert_allclose(a,want,rtol=2e-5,atol=2e-4)
 rejections["private_scratch_read"]=read(handle,7,generation.value,c.c_void_p(out[0].ctypes.data),out[0].nbytes);assert rejections["private_scratch_read"]==1
 old=generation.value
 check(invoke(handle,1,c.byref(generation),None))
 rejections["stale_generation"]=read(handle,6,old,c.c_void_p(out[0].ctypes.data),out[0].nbytes);assert rejections["stale_generation"]==10
 sizes[0]-=1
 rejections["invalid_update"]=update(handle,inputs,sizes);assert rejections["invalid_update"]==1
 sizes[0]+=1
 # Rejected updates preserve previous completed output.
 check(read(handle,6,generation.value,c.c_void_p(out[0].ctypes.data),out[0].nbytes))
 pid=os.fork()
 if pid==0:
  os._exit(0 if invoke(handle,1,c.byref(generation),None)==2 else 1)
 _,status=os.waitpid(pid,0);assert os.waitstatus_to_exitcode(status)==0
 rejections["fork_identity"]="passed"
 # Changed tangent inputs must invalidate old outputs and flow through native
 # private uploads, not retained device values from the previous generation.
 da *= np.float32(-.5);db *= np.float32(2)
 expected=[oracle(sa,sb),oracle(da,sb)+oracle(sa,db)]
 check(update(handle,inputs,sizes))
 rejections["read_after_update"]=read(handle,9,generation.value,c.c_void_p(out[1].ctypes.data),out[1].nbytes)
 assert rejections["read_after_update"]==10
 check(invoke(handle,1,c.byref(generation),None))
 for slot,a,want in zip((6,9),out,expected):
  check(read(handle,slot,generation.value,c.c_void_p(a.ctypes.data),a.nbytes))
  np.testing.assert_allclose(a,want,rtol=2e-5,atol=2e-4)
 samples=[];warm=[];updated=[]
 for _ in range(11):
  elapsed=c.c_float()
  check(invoke(handle,100,c.byref(generation),c.byref(elapsed)));samples.append(float(elapsed.value))
  start=time.perf_counter();check(invoke(handle,1,c.byref(generation),None))
  for slot,a in zip((6,9),out):check(read(handle,slot,generation.value,c.c_void_p(a.ctypes.data),a.nbytes))
  warm.append((time.perf_counter()-start)*1e3)
  start=time.perf_counter();check(update(handle,inputs,sizes));check(invoke(handle,1,c.byref(generation),None))
  for slot,a in zip((6,9),out):check(read(handle,slot,generation.value,c.c_void_p(a.ctypes.data),a.nbytes))
  updated.append((time.perf_counter()-start)*1e3)
  for a,want in zip(out,expected):np.testing.assert_allclose(a,want,rtol=2e-5,atol=2e-4)
 eps=1e-5
 fd=(oracle(sa.astype(np.float64)+eps*da,sb.astype(np.float64)+eps*db)-oracle(sa.astype(np.float64)-eps*da,sb.astype(np.float64)-eps*db))/(2*eps)
 np.testing.assert_allclose(out[1],fd,rtol=2e-5,atol=2e-4)
 hip=c.CDLL("/opt/rocm/lib/libamdhip64.so")
 device_name=c.create_string_buffer(256);check(hip.hipDeviceGetName(device_name,256,0))
 packet={"architecture":"gfx1201","device":device_name.value.decode(),"shape_mnk":[17,19,256],"execution":"native_cpp_owned_sequence","compiler_package_projection":"not_yet_integrated","numerical_max_abs_error":[float(np.max(np.abs(a-w))) for a,w in zip(out,expected)],"finite_difference_max_abs_error":float(np.max(np.abs(out[1]-fd))),"negative_contract_checks":rejections,"device_native_launch_window_ms":samples,"device_native_launch_window_median_ms":statistics.median(samples),"warm_invoke_and_two_readbacks_ms":warm,"warm_invoke_and_two_readbacks_median_ms":statistics.median(warm),"update_invoke_two_readbacks_ms":updated,"update_invoke_two_readbacks_median_ms":statistics.median(updated),"timing_note":"HIP event interval includes native C++ enqueue gaps across four kernels. Host intervals exclude prepare/compilation.","runtime_sha256":hashlib.sha256(options.runtime.read_bytes()).hexdigest(),"member_image_sha256":[hashlib.sha256((root/f"member-{i}.hsaco").read_bytes()).hexdigest() for i in range(4)],"buffer_contract":[{"id":i,"bytes":b.bytes,"elements":b.elements,"ownership":b.ownership,"first_write":b.first_write,"last_read":b.last_read} for i,b in enumerate(buffers)],"step_contract":[{"entry":s.entry.decode(),"inputs":list(s.inputs)[:s.input_count],"output":s.output,"geometry":list(s.geometry),"scalars":list(s.scalars)[:s.scalar_count]} for s in plan]}
 options.output.write_text(json.dumps(packet,indent=2)+"\n");print(json.dumps(packet,indent=2))
finally:
 check(close(handle))
assert close(handle)==1
