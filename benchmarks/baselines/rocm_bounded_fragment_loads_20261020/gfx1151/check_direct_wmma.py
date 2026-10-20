import ctypes as c,json,hashlib
from pathlib import Path
import numpy as np,ml_dtypes
from tests.unit.test_rocm_bounded_fragment_pack import SHAPES
p=Path(__file__).resolve().parent
library=c.CDLL(str(p/'direct_wmma.so'));f=library.direct_run
f.argtypes=[c.c_void_p,c.c_void_p,c.c_void_p,c.c_int];f.restype=c.c_int
compiled=np.load(p/'gfx1151-floating-control-candidate.npz')
results=[];arrays={}
for dtype,bf in [('f16',0),('bf16',1)]:
 for m,n,k in SHAPES:
  rng=np.random.default_rng(m*1307+n*31+k)
  ty=np.float16 if not bf else ml_dtypes.bfloat16
  a=rng.integers(-3,4,(16,16)).astype(ty);b=rng.integers(-3,4,(16,16)).astype(ty)
  a[m:,:]=0;a[:,k:]=0;b[k:,:]=0;b[:,n:]=0
  o=np.empty((16,16),np.float32)
  err=f(a.ctypes.data,b.ctypes.data,o.ctypes.data,bf);assert not err,err
  name=f'{dtype}_0_{m}_{n}_{k}'
  expected=compiled[f'expected_{name}'];candidate=compiled[f'candidate_{name}']
  equal=np.array_equal(o.view(np.uint32),candidate.view(np.uint32))
  exact=a.astype(np.float64)@b.astype(np.float64)
  abs_sum=np.abs(a.astype(np.float64))@np.abs(b.astype(np.float64))
  unit=2.**-24;gamma=16*unit/(1-16*unit);bound=gamma*abs_sum
  assert np.all(np.abs(o.astype(np.float64)-exact)<=bound)
  results.append(dict(dtype=dtype,shape=[m,n,k],bitwise_equal_to_tessera=bool(equal),max_abs_error=float(np.max(np.abs(o-exact))),
    nonzero_reference_zero_error=int(np.count_nonzero((exact==0)&(o!=0))),bound='gamma16(fp32) * sum(abs(A*B))',violations=0))
  arrays[name]=o
np.savez_compressed(p/'direct-wmma-outputs.npz',**arrays)
receipt=dict(architecture='gfx1151',implementation='independent hipcc builtins; no Tessera compiler or runtime',cases=results,
 all_bitwise_equal=all(r['bitwise_equal_to_tessera'] for r in results),
 source_sha256=hashlib.sha256((p/'direct_wmma.hip').read_bytes()).hexdigest(),library_sha256=hashlib.sha256((p/'direct_wmma.so').read_bytes()).hexdigest())
(p/'direct-wmma-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(cases=len(results),all_bitwise_equal=receipt['all_bitwise_equal'],max_abs_error=max(r['max_abs_error'] for r in results))))
