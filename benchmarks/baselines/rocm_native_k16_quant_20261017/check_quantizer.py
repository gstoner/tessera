"""Diagnostic exact-gfx1201 source-derived quantizer; no public JIT claim."""
import ctypes as C
import hashlib
import json
import math
import os
import time
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import PreparedScaledProgram,_BoundScaledABI,_Buffer,_StorageSpec
from tessera.compiler.rocm_nvfp4_expansion_program import _MultiOutputStep
ROOT=Path(os.environ.get('TESSERA_K16_QUANT_EVIDENCE_DIR','/home/angstorms/scratch/rocm-native-gate-up-views-20261016/k16-quant'))
assert rt._rocm_live_arch()=='gfx1201'
info=subprocess.check_output(['/opt/rocm/bin/rocminfo'],text=True)
device=re.search(r'Name:\s+gfx1201\s+Uuid:\s+(\S+)\s+Marketing Name:\s+([^\n]+)',info)
assert device, 'exact GPU identity missing'
hardware={'device':device.group(2).strip(),'gpu_uuid':device.group(1),'architecture':'gfx1201'}
class QuantOwner(PreparedScaledProgram):
 _prepare_symbol='tessera_rocm_program_prepare_multi_output'
 _step_type=_MultiOutputStep
 def _bind_abi(self,package):
  record=package.record;manifest=record['member'];frame=json.loads(__import__('base64').b64decode(manifest['program_base64']))
  payload=package.payload
  assert hashlib.sha256(payload).hexdigest()==record['image_sha256']
  blob=C.create_string_buffer(payload);entry=manifest['entry'].encode();steps=(_MultiOutputStep*1)();step=steps[0]
  step.image=C.cast(blob,C.c_void_p);step.image_bytes=len(payload);step.entry=entry
  step.input_count=1;step.output_count=2;step.scalar_count=1
  step.inputs[0]=0;step.outputs[0]=1;step.outputs[1]=2;step.scalars[0]=manifest['scalars'][0]
  step.geometry[:]=manifest['geometry']
  buffers=(_Buffer*3)(*[_Buffer(b['bytes'],b['elements'],b['first_write'],b['last_read'],b['ownership'],0) for b in frame['buffers']])
  specs=tuple(_StorageSpec(tuple(b['shape']),b['storage'],b['bytes'],np.dtype(ml_dtypes.bfloat16) if i==0 else np.dtype('uint8') if i==1 else np.dtype('float32')) for i,b in enumerate(frame['buffers']))
  return _BoundScaledABI(buffers,steps,(blob,),(entry,),1,specs,(1,2),())

def inputs(shape):
 n=math.prod(shape);rng=np.random.default_rng(120117+n)
 bits=rng.integers(0,65536,n,dtype=np.uint16)
 # Full BF16 domain both as uniform groups and heterogeneous groups.
 allbits=np.arange(65536,dtype=np.uint16)
 if n>=allbits.size:bits[:allbits.size]=allbits
 repeated=np.repeat(allbits,16)
 if n>=65536+repeated.size:bits[65536:65536+repeated.size]=repeated
 special=np.array([0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x8080,0x7f80,0xff80,0x7fc1,0xffc1,0x3f80,0xbf80,0x7f7f,0xff7f],np.uint16)
 bits[-16:]=special
 # Anchor each rounding-boundary group at amax=448 so scale is exactly 1.
 # BF16 can represent each FP8 midpoint and its neighboring BF16 values.
 represented=np.arange(256,dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
 finite=np.unique(represented[np.isfinite(represented)])
 ties=((finite[:-1]+finite[1:])*np.float32(0.5)).astype(ml_dtypes.bfloat16)
 tie_groups=np.zeros((len(ties),16),dtype=ml_dtypes.bfloat16)
 tie_groups[:,0]=448;tie_groups[:,1]=ties
 tie_groups[:,2]=np.nextafter(ties,ml_dtypes.bfloat16(-np.inf))
 tie_groups[:,3]=np.nextafter(ties,ml_dtypes.bfloat16(np.inf))
 offset=65536+repeated.size if n>=65536+repeated.size else 65536
 if offset+tie_groups.size<=n-16:
  bits[offset:offset+tie_groups.size]=tie_groups.reshape(-1).view(np.uint16)
 # All-zero, signed-zero, and minimum-subnormal groups precede mixed specials.
 for g,value in enumerate([0,0x8000,1,0x8001,0x007f]):
  if (g+1)*16<=n-16:bits[g*16:(g+1)*16]=value
 return np.ascontiguousarray(bits.view(ml_dtypes.bfloat16).reshape(shape))

def oracle(x):
 wide=x.astype(np.float32).reshape(-1,16)
 with np.errstate(all='ignore'):
  scale=np.maximum(np.max(np.abs(wide),axis=1,keepdims=True)/np.float32(448),np.float32(np.finfo(np.float32).tiny))
  codes=(wide/scale).astype(ml_dtypes.float8_e4m3fn).reshape(x.shape)
 return codes,scale.reshape(x.shape[0],x.shape[1]//16)

def compare(actual,expected):
 a,s=actual;q,t=expected
 ab=a.view(np.uint8);qb=q.view(np.uint8)
 qbad=(((ab&127)==127)!=((qb&127)==127))|(((qb&127)!=127)&(ab!=qb))
 sbad=(np.isnan(s)!=np.isnan(t))|(~np.isnan(t)&(s.view(np.uint32)!=t.view(np.uint32)))
 return {'code_mismatches':int(np.count_nonzero(qbad)),'scale_mismatches':int(np.count_nonzero(sbad)),
  'code_examples':[{'index':int(i),'got':int(ab.flat[i]),'wanted':int(qb.flat[i])} for i in np.flatnonzero(qbad)[:12]],
  'scale_examples':[{'index':int(i),'got':float(s.flat[i]),'wanted':float(t.flat[i]),'got_bits':int(s.view(np.uint32).flat[i]),'wanted_bits':int(t.view(np.uint32).flat[i])} for i in np.flatnonzero(sbad)[:12]]}

records=[]
for shape in [(3,32),(19,12288),(256,12288)]:
 m,k=shape;prefix=ROOT/f'quant-{m}-{k}';record=json.loads(prefix.with_suffix('.json').read_text())
 package=SimpleNamespace(record=record,payload=prefix.with_suffix('.hsaco').read_bytes(),program_json='{}',graph_ir='')
 x=inputs(shape);wanted=oracle(x)
 with QuantOwner(package,[x],runtime_library=os.environ['TESSERA_ROCM_MULTI_OUTPUT_LIB']) as owner:
  generation,_=owner.invoke();actual=owner.read(generation);comparison=compare(actual,wanted)
  result={**hardware,'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'provider_sha256':hashlib.sha256(Path(os.environ['TESSERA_ROCM_MULTI_OUTPUT_LIB']).read_bytes()).hexdigest(),'shape':list(shape),'architecture':rt._rocm_live_arch(),'identity':record,'comparison':comparison,'scope':'diagnostic native C++ ownership; public package/JIT still required'}
  result.update({"uniform_bf16_patterns":65536 if math.prod(shape)>=65536+65536*16 else 0,"rounding_boundary_groups":252 if math.prod(shape)>=65536+252*16+16 else 0})
  records.append(result);(ROOT/'device-quantizer.json').write_text(json.dumps(records,indent=2)+'\n')
  print(shape,comparison,flush=True)
  if comparison['code_mismatches'] or comparison['scale_mismatches']:continue
  held=tuple(a.copy() for a in actual);changed=np.ascontiguousarray(x[::-1]);owner.update([changed]);generation,_=owner.invoke()
  assert not any(compare(owner.read(generation),oracle(changed))[key] for key in ('code_mismatches','scale_mismatches'))
  for a,b in zip(actual,held,strict=True):np.testing.assert_array_equal(a.view(np.uint8),b.view(np.uint8))
  generation,pilot=owner.profile_members(repeats=8);reps=min(65536,max(8,math.ceil(22/max(pilot[0],1e-6))))
  for _ in range(6):
   generation,times=owner.profile_members(repeats=reps)
   if times[0]*reps>=18:break
   reps=min(65536,max(reps+1,math.ceil(22/max(times[0],1e-6))))
  else:raise AssertionError("could not establish captured duration")
  samples=[]
  for _ in range(3):
   generation,times=owner.profile_members(repeats=reps);assert times[0]*reps>=10;samples.append(times[0])
  host=[]
  for _ in range(3):
   start=time.perf_counter();generation,_=owner.invoke();owner.read(generation);host.append((time.perf_counter()-start)*1e3)
  result.update({'changed_input_and_held_outputs':'passed','captured_repeats':reps,'captured_member_ms':samples,'completed_invoke_read_ms':host,
    'device_scope':'captured native graph dispatch; excludes host enqueue/copy/capture','host_scope':'prepared invoke plus two output readbacks; excludes compilation and preparation'})
  (ROOT/'device-quantizer.json').write_text(json.dumps(records,indent=2)+'\n')
assert all(not r['comparison']['code_mismatches'] and not r['comparison']['scale_mismatches'] for r in records)
