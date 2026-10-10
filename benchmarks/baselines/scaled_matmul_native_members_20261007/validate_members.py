from pathlib import Path
import ctypes as c, json, re, hashlib
import numpy as np
root=Path(__file__).parent
hip=c.CDLL("/opt/rocm/lib/libamdhip64.so")
def check(rc):
 if rc: raise RuntimeError(f"HIP status {rc}")
hip.hipMalloc.argtypes=[c.POINTER(c.c_void_p),c.c_size_t]
hip.hipMemcpy.argtypes=[c.c_void_p,c.c_void_p,c.c_size_t,c.c_int]
hip.hipModuleLoadData.argtypes=[c.POINTER(c.c_void_p),c.c_void_p]
hip.hipModuleGetFunction.argtypes=[c.POINTER(c.c_void_p),c.c_void_p,c.c_char_p]
hip.hipModuleLaunchKernel.argtypes=[c.c_void_p]+[c.c_uint]*7+[c.c_void_p,c.POINTER(c.c_void_p),c.c_void_p]
hip.hipFree.argtypes=[c.c_void_p]
hip.hipModuleUnload.argtypes=[c.c_void_p]
check(hip.hipInit(0))
name=c.create_string_buffer(256)
check(hip.hipDeviceGetName(name,256,0))
rng=np.random.default_rng(20261007)
codes=np.array([0,0x28,0x30,0x38,0x40,0xa8,0xb0,0xb8,0xc0],dtype=np.uint8)
vals=np.array([0,.25,.5,1,2,-.25,-.5,-1,-2],dtype=np.float64)
ai=rng.integers(0,len(codes),(17,256)); bi=rng.integers(0,len(codes),(256,19))
sa=rng.uniform(.2,1,(17,2)).astype(np.float32); sb=rng.uniform(.2,1,(2,1)).astype(np.float32)
da=rng.uniform(-.3,.3,(17,2)).astype(np.float32); db=rng.uniform(-.3,.3,(2,1)).astype(np.float32)
host=[codes[ai],codes[bi],sa,sb,da,db]+[np.full((17,19),np.nan,np.float32) for _ in range(4)]
buffers=[]; modules=[]
try:
 for a in host:
  ptr=c.c_void_p();check(hip.hipMalloc(c.byref(ptr),a.nbytes));buffers.append(ptr)
  check(hip.hipMemcpy(ptr,c.c_void_p(a.ctypes.data),a.nbytes,1))
 results=[]
 for i,slots in enumerate(([0,1,2,3,6],[0,1,4,3,7],[0,1,2,5,8],[7,8,9])):
  image=(root/f"member-{i}.hsaco").read_bytes();blob=c.create_string_buffer(image)
  mod=c.c_void_p();check(hip.hipModuleLoadData(c.byref(mod),blob));modules.append(mod)
  text=(root/f"member-{i}.mlir").read_text()
  entry=re.search(r'gpu.kernel_metadata<"([^"]+)"',text).group(1)
  fn=c.c_void_p();check(hip.hipModuleGetFunction(c.byref(fn),mod,entry.encode()))
  args=[]
  for slot in slots:
   args.extend([c.c_void_p(buffers[slot].value),c.c_void_p(buffers[slot].value),c.c_int64(0),c.c_int64(host[slot].size),c.c_int64(1)])
  args.extend(c.c_int64(v) for v in ([17,19,256] if i<3 else [323]))
  argv=(c.c_void_p*len(args))(*(c.addressof(a) for a in args))
  geo=(2,2,1,32,1,1) if i<3 else (2,1,1,256,1,1)
  check(hip.hipModuleLaunchKernel(fn,*geo,0,None,argv,None))
  check(hip.hipDeviceSynchronize())
  results.append({"step":i,"entry":entry,"geometry":geo,"image_sha256":hashlib.sha256(image).hexdigest()})
 for slot in range(6,10):check(hip.hipMemcpy(c.c_void_p(host[slot].ctypes.data),buffers[slot],host[slot].nbytes,2))
 ab=[vals[ai][:,g*128:(g+1)*128]@vals[bi][g*128:(g+1)*128,:] for g in range(2)]
 def oracle(x,y):return sum(ab[g]*x[:,g,None].astype(np.float64)*y[g,0].astype(np.float64) for g in range(2))
 expected=[oracle(sa,sb),oracle(da,sb),oracle(sa,db),oracle(da,sb)+oracle(sa,db)]
 for row,actual,want in zip(results,host[6:],expected):
  row["max_abs_error"]=float(np.max(np.abs(actual-want)))
  np.testing.assert_allclose(actual,want,rtol=2e-5,atol=2e-4)
 eps=1e-5
 fd=(oracle(sa.astype(np.float64)+eps*da,sb.astype(np.float64)+eps*db)-oracle(sa.astype(np.float64)-eps*da,sb.astype(np.float64)-eps*db))/(2*eps)
 np.testing.assert_allclose(host[9],fd,rtol=2e-5,atol=2e-4)
 packet={"device":name.value.decode(),"shape_mnk":[17,19,256],"driver":"diagnostic_member_launch_loop_not_production_program","correctness":"passed","finite_difference_max_abs_error":float(np.max(np.abs(host[9]-fd))),"members":results}
 (root/"numerics.json").write_text(json.dumps(packet,indent=2)+"\n")
 print(json.dumps(packet,indent=2))
finally:
 hip.hipDeviceSynchronize()
 for ptr in buffers:check(hip.hipFree(ptr))
 for mod in modules:check(hip.hipModuleUnload(mod))
