"""Exact gfx1201 candidate proof through the existing checked native owner.

This is an engineering diagnostic, not public JIT/plugin closure.
"""
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import patch
import base64,hashlib,json,os,time,statistics,ctypes as ct
import numpy as np
import tessera.runtime as rt
from tessera.compiler.native_scaled_program import PreparedScaledProgram

v=Path('/home/angstorms/scratch/rocm-native-gate-up-views-20261016')
arch=rt._rocm_live_arch()
assert arch=='gfx1201',arch
hip=rt._load_hip_from_toolkit();assert hip is not None
ordinal=ct.c_int();hip.hipGetDevice.argtypes=[ct.POINTER(ct.c_int)]
assert hip.hipGetDevice(ct.byref(ordinal))==0
name_buffer=ct.create_string_buffer(256);hip.hipDeviceGetName.argtypes=[ct.c_void_p,ct.c_int,ct.c_int]
assert hip.hipDeviceGetName(name_buffer,256,ordinal.value)==0
name=name_buffer.value.decode();assert name and name.lower()!='unknown',name
uuid=(ct.c_ubyte*16)();hip.hipDeviceGetUuid.argtypes=[ct.c_void_p,ct.c_int]
assert hip.hipDeviceGetUuid(ct.byref(uuid),ordinal.value)==0
uuid_hex=bytes(uuid).hex()

library='/home/angstorms/scratch/rocm-native-silu-mlp-20261016/runtime-candidate/libtessera_rocm_native_movement.full.so'
os.environ['TESSERA_OPT']='/no/compiler';os.environ['TESSERA_ROCM_OPT']='/no/compiler'

@dataclass(frozen=True)
class Candidate:
 program_json:str
 members_json:tuple[str,...]
 images:tuple[bytes,...]
 graph_ir:str
 def validate(self):
  p=json.loads(self.program_json);m=json.loads(self.members_json[0]);b=p['buffers']
  assert p['schema']==1 and p['kind']=='parent_view' and p['argument_count']==1
  assert len(self.members_json)==len(self.images)==len(p['steps'])==1 and len(b)==2
  assert hashlib.sha256(p['root_ir'].encode()).hexdigest()==p['root_ir_sha256']
  assert json.loads(base64.b64decode(m['program_base64']))==p
  assert m['architecture']=='gfx1201' and m['profile']=='gfx1201_parent_view_f32_v1'
  assert m['abi']=='flattened_rank1_memref_then_i64' and m['inputs']==[0] and m['output']==1
  rows,width=m['output_shape'];n=rows*width
  assert m['parent_shape']==[rows,2*width] and m['scalars']==[n]
  assert m['geometry']==[(n+255)//256,1,1,256,1,1]
  assert p['outputs']==[1] and [x['shape'] for x in b]==[m['parent_shape'],m['output_shape']]
  assert [x['storage'] for x in b]==['f32','f32']
  assert [x['elements'] for x in b]==[2*n,n] and [x['bytes'] for x in b]==[8*n,4*n]
  assert [x['ownership'] for x in b]==[0,2] and [x['first_write'] for x in b]==[-1,0]
  assert [x['last_read'] for x in b]==[0,1]
  assert self.images[0].startswith(b'\x7fELF')

def forbidden(*args,**kwargs):raise AssertionError('candidate invoked compiler or eager arithmetic')

def oracle(parent,width,reverse):
 left,right=parent[:,:width].astype(np.float64),parent[:,width:].astype(np.float64)
 gate,up=(right,left) if reverse else (left,right)
 return ((gate/(1+np.exp(-gate)))*up).astype(np.float32)

index=json.loads((v/'native-view-candidate-index.json').read_text())
assert index['rows'][0]['image_sha256']==index['rows'][2]['image_sha256']
results=[];rng=np.random.default_rng(20261017)
for row in index['rows']:
 d=json.loads((v/row['packet']).read_text());m,width=d['shape'];reverse=d['reverse']
 images=tuple(base64.b64decode(x,validate=True) for x in d['images'])
 assert hashlib.sha256(images[0]).hexdigest()==d['image_sha256']==row['image_sha256']
 package=Candidate(d['program_json'],tuple(d['members_json']),images,d['graph_ir']);package.validate()
 parent=rng.uniform(-5,5,(m,2*width)).astype(np.float32);original=parent.copy();expected=oracle(parent,width,reverse)
 with patch('numpy.exp',forbidden),patch('subprocess.run',forbidden):
  owner=PreparedScaledProgram(package,[parent],runtime_library=library)
 try:
  with patch('numpy.exp',forbidden),patch('subprocess.run',forbidden):
   generation,_=owner.invoke();actual=owner.read(generation)[0]
  np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=2e-7)
  error=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
  np.testing.assert_array_equal(parent,original);held=actual.copy()
  try:owner.update([parent[:,:-1]])
  except ValueError:pass
  else:raise AssertionError('malformed input shape was accepted')
  np.testing.assert_array_equal(owner.read(generation)[0],actual)
  changed=(parent*0.75).astype(np.float32);changed_expected=oracle(changed,width,reverse)
  with patch('numpy.exp',forbidden),patch('subprocess.run',forbidden):
   owner.update([changed]);generation,_=owner.invoke();updated=owner.read(generation)[0]
  np.testing.assert_allclose(updated,changed_expected,rtol=3e-6,atol=2e-7)
  np.testing.assert_array_equal(actual,held)
  samples=[];repeats=8192
  for trial in range(5):
   generation,elapsed=owner.invoke(repeats=repeats,timed=True)
   assert elapsed>0 and elapsed*repeats>=10,(elapsed,repeats)
   samples.append(elapsed)
  np.testing.assert_allclose(owner.read(generation)[0],changed_expected,rtol=3e-6,atol=2e-7)
  host=[]
  for trial in range(5):
   start=time.perf_counter_ns();owner.update([changed]);generation,_=owner.invoke();completed=owner.read(generation)[0];host.append((time.perf_counter_ns()-start)/1e6)
   np.testing.assert_allclose(completed,changed_expected,rtol=3e-6,atol=2e-7)
  result={'shape_m_i':[m,width],'reverse':reverse,'image_sha256':row['image_sha256'],'entry':row['entry'],'max_abs_error_vs_f64_then_f32':error,'correctness':'passed_before_and_after_timing','changed_inputs':'passed','malformed_shape_preserves_result':'passed','retained_host_output':'passed','parent_unchanged':True,'compiler_and_eager_calls_forbidden':True,'activation_device_launch_window_samples_ms':samples,'activation_device_launch_window_median_ms':statistics.median(samples),'repeats':repeats,'completed_update_invoke_read_samples_ms':host,'completed_update_invoke_read_median_ms':statistics.median(host)}
  results.append(result);print(json.dumps(result),flush=True)
 finally:owner.close()
packet={'architecture':arch,'device':name,'device_ordinal':ordinal.value,'device_uuid':uuid_hex,'scope':'candidate source->native Schedule->typed Tile->ROCm Target->ROCDL/LLVM->HSACO->checked C++ HIP owner; public JIT/plugin, resident matmul producer chain, low-storage and full MLP/model quality remain open','runtime_library':library,'runtime_library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest(),'image_reuse_across_source_shapes':True,'timing_boundary':'device launch windows include GPU dispatch; host samples include upload, invoke, completion and readback, excluding preparation/compilation','rows':results}
(v/'native-view-device-candidate.json').write_text(json.dumps(packet,indent=2)+'\n')
print('All five exact gfx1201 candidate rows passed',flush=True)
