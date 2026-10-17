"""Diagnostic exact-device typed cast; not a public Graph quantizer."""
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler.native_scaled_program import (
    PreparedScaledProgram, _BoundScaledABI, _Buffer, _StorageSpec, _status,
)
from tessera.compiler.rocm_nvfp4_expansion_program import _MultiOutputStep

root=Path("/home/angstorms/scratch/rocm-native-gate-up-views-20261016/block-quant-probe")
identity=json.loads((root/"image-identity.json").read_text())
payload=(root/"fp8-cast.hsaco").read_bytes()
assert hashlib.sha256(payload).hexdigest()==identity["image_sha256"]
assert rt._rocm_live_arch()=="gfx1201"
rng=np.random.default_rng(120117)
with np.errstate(invalid="ignore",over="ignore"):
    represented=np.arange(256,dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    finite=np.unique(represented[np.isfinite(represented)])
    ties=((finite[:-1].astype(np.float64)+finite[1:].astype(np.float64))/2).astype(np.float32)
    half=np.arange(65536,dtype=np.uint16).view(np.float16).astype(np.float32)
values=np.ascontiguousarray(np.concatenate([
    represented,ties,np.nextafter(ties,np.float32(-np.inf)),np.nextafter(ties,np.float32(np.inf)),
    half,rng.integers(0,2**32,size=100000,dtype=np.uint32).view(np.float32),
    rng.uniform(-500,500,100000).astype(np.float32),
    np.array([0.,-0.,448.,-448.,464.,-464.,465.,-465.,np.inf,-np.inf,np.nan,-np.nan],np.float32),
]),dtype=np.float32)
with np.errstate(invalid="ignore",over="ignore",under="ignore"):
    expected=values.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
n=values.size

class CastOwner(PreparedScaledProgram):
    _prepare_symbol="tessera_rocm_program_prepare_multi_output"
    _step_type=_MultiOutputStep

    def _bind_abi(self, package):
        blob=C.create_string_buffer(payload)
        entry=b"fp8_cast"
        steps=(_MultiOutputStep*1)()
        step=steps[0]
        step.image=C.cast(blob,C.c_void_p)
        step.image_bytes=len(payload)
        step.entry=entry
        step.input_count=step.output_count=step.scalar_count=1
        step.inputs[0]=0
        step.outputs[0]=1
        step.scalars[0]=n
        step.geometry[:]=((n+255)//256,1,1,256,1,1)
        buffers=(_Buffer*2)(_Buffer(n*4,n,-1,0,0,0),_Buffer(n,n,0,1,2,0))
        specs=(_StorageSpec((n,),"f32",n*4,np.dtype("float32")),
               _StorageSpec((n,),"f8E4M3FN",n,np.dtype("uint8")))
        return _BoundScaledABI(buffers,steps,(blob,),(entry,),1,specs,(1,),())

    def read_codes(self,generation):
        output=np.empty(n,np.uint8)
        _status(self.lib.tessera_rocm_program_read(
            self.handle,1,generation,C.c_void_p(output.ctypes.data),output.nbytes))
        return output

def compare(actual,expected,inputs):
    wanted_nan=(expected&127)==127
    actual_nan=(actual&127)==127
    mismatch=(wanted_nan!=actual_nan)|(~wanted_nan&(actual!=expected))
    count=int(np.count_nonzero(mismatch))
    rows=[{"value":float(inputs[i]),"input_bits":int(inputs.view(np.uint32)[i]),
           "expected":int(expected[i]),"actual":int(actual[i])}
          for i in np.flatnonzero(mismatch)[:24]]
    return count,rows

record={"architecture":rt._rocm_live_arch(),"scope":"diagnostic typed scalar conversion only",
        "identity":identity,"input_count":int(n),"half_bit_patterns":65536,
        "rounding_boundary_values":int(ties.size*3),
        "random_f32_bit_patterns":100000,"uniform_f32_values":100000,
        "nan_comparison":"semantic NaN; finite encodings, including signed zero, compare bitwise",
        "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provider_sha256":hashlib.sha256(Path(os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]).read_bytes()).hexdigest()}
with CastOwner(SimpleNamespace(program_json="{}",graph_ir=""),[values],
               runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
    generation,_=owner.invoke()
    actual=owner.read_codes(generation)
    count,samples=compare(actual,expected,values)
    record.update({"mismatches":count,"mismatch_examples":samples})
    (root/"device-result.json").write_text(json.dumps(record,indent=2)+"\n")
    print("checked",n,"values; mismatches",count,flush=True)
    assert count==0,samples
    held=actual.copy()
    changed=np.ascontiguousarray(values[::-1])
    owner.update([changed])
    generation,_=owner.invoke()
    second=owner.read_codes(generation)
    assert compare(second,expected[::-1],changed)[0]==0
    np.testing.assert_array_equal(actual,held)
    generation,pilot=owner.profile_members(repeats=64)
    assert compare(owner.read_codes(generation),expected[::-1],changed)[0]==0
    repeats=min(65536,max(64,math.ceil(22/max(pilot[0],1e-6))))
    for _ in range(5):
        generation,times=owner.profile_members(repeats=repeats)
        assert compare(owner.read_codes(generation),expected[::-1],changed)[0]==0
        if times[0]*repeats>=18:break
        repeats=min(65536,max(repeats+1,math.ceil(repeats*22/max(times[0]*repeats,1e-6))))
    else:raise AssertionError("could not establish duration")
    samples=[]
    for _ in range(3):
        generation,times=owner.profile_members(repeats=repeats)
        assert times[0]*repeats>=10
        assert compare(owner.read_codes(generation),expected[::-1],changed)[0]==0
        samples.append(times[0])
    record.update({"captured_repeats":repeats,"captured_samples_ms":samples,
                   "captured_scope":"device graph dispatch; capture/instantiation/CPU enqueue excluded",
                   "changed_input_and_retained_output":"passed"})
(root/"device-result.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record),flush=True)
