"""Representative original-checkpoint bounded source validation."""
from pathlib import Path
import ctypes as C
import hashlib
import json
import os
import statistics
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy
from benchmarks.rocm.checkpoint_activations import load_capture
from benchmarks.rocm import benchmark_gfx1201_three_formats as formats
from benchmarks.rocm.benchmark_rocm_nvfp4_checkpoint import _relative_rms

w=Path("/home/angstorms/scratch/nvfp4-source-bounded-20261016")
data=w/"original-checkpoint"
pool,activation_source=load_capture(
    "/home/angstorms/scratch/nvfp4-model-quality-20261015/layer0_mlp_inputs.npz",[1,19,127,200,256])
sources=[json.loads((data/(role+".json")).read_text()) for role in ("gate_proj","up_proj")]
baseline=json.loads(Path("benchmarks/baselines/rocm_nvfp4_public_source_20261015/original-composed-256.json").read_text())
assert sources==baseline["checkpoint_sources"], "pinned checkpoint identities changed"
items=[np.load(data/(role+".npz"),allow_pickle=False) for role in ("gate_proj","up_proj")]
for item,source in zip(items,sources,strict=True):
    for key,field in (("codes","nvfp4_tensor_sha256"),("scale_bits","nvfp4_scale_sha256"),("bf16_bits","bf16_tensor_sha256")):
        assert hashlib.sha256(item[key].tobytes()).hexdigest()==source[field],field
    raw=np.array(item["global_scale"],np.float32).tobytes()
    assert hashlib.sha256(raw).hexdigest()==source["nvfp4_global_scale_sha256"]
codes=np.ascontiguousarray(np.concatenate([x["codes"] for x in items]))
scales=np.ascontiguousarray(np.concatenate([x["scale_bits"] for x in items])).view(ml_dtypes.float8_e4m3fn)
globals_=np.array([float(x["global_scale"]) for x in items],np.float64)
base=np.concatenate([x["bf16_bits"].view(ml_dtypes.bfloat16).astype(np.float32) for x in items])
offsets=[0,items[0]["codes"].shape[0],codes.shape[0]]
n,packed_k=codes.shape;k=packed_k*2
a=np.ascontiguousarray(pool[:256],dtype=np.float32)
qa,_,encoded,da,_=formats.quantize_fp8(a,np.zeros((k,1),np.float32),16,1)
sa=encoded[0]
levels=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],np.float64)
weights=np.empty((n,k),np.float64)
weights[:,0::2],weights[:,1::2]=levels[codes&15],levels[codes>>4]
scale64=scales.astype(np.float64)*np.where(np.arange(n)<offsets[1],globals_[0],globals_[1])[:,None]
represented=weights*np.repeat(scale64.astype(np.float32).astype(np.float64),16,axis=1)
expected=da@represented.T
magnitude=np.abs(da)@np.abs(represented).T
original=a.astype(np.float64)@base.astype(np.float64).T
gamma=(k+8*(k//16)+16)*2.**-24;gamma/=1-gamma
bound=gamma*magnitude+2.**-126
del weights,represented,base,magnitude
print("source hashes and independent projection oracle ready",flush=True)

@ts.jit(target="rocm_gfx1201")
def product(codes,scales,globals_,activation,activation_scales):
    b,sb=ts.ops.nvfp4_expand(codes,scales,globals_,row_offsets=offsets,
                           numeric_policy=nvfp4_expansion_policy())
    return ts.ops.scaled_matmul(activation,b,activation_scales,sb,transposeB=True,
        numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
        scale_layout={"granularity":"block","block":[1,16],"format":"fp32"})

assert rt._rocm_live_arch()=="gfx1201"
program=product.compile_native_nvfp4_program(codes,scales,globals_,qa[:19],sa[:19],m_bound=256)
package=program.native
assert package.consumer.descriptor.provenance["staging"]=="lds"
assert package.consumer.descriptor.provenance["arbitrary_m"] is True
assert package.consumer.descriptor.provenance["lds_whole_m"] is False
lib=rt._load_rocm_native_movement_runtime()
if lib is None:raise RuntimeError("missing native provider")
hip=rt._load_hip_for_launch()
ordinal,name=C.c_int(),C.create_string_buffer(256)
assert hip.hipGetDevice(C.byref(ordinal))==0
assert hip.hipDeviceGetName(name,len(name),ordinal.value)==0
record={"architecture":rt._rocm_live_arch(),"device":name.value.decode(),
 "schema":"tessera.nvfp4_original_bounded_source.v1","capacity_m":256,"shape_nk":[n,k],
 "checkpoint_sources":sources,"activation_source":activation_source,
 "images":[package.producer.native.image.image_digest,package.consumer.image.image_digest],
 "binding_digest":package.binding_digest,"provider_sha256":hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
 "compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
 "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 "full_model_quality_acceptance":False,"strategy_promotion":False,
 "timing_scope":"grouped member graph dispatch; program events may include host enqueue gaps; completed prepared activation-update/invoke-consumer/read includes transfer. No pure-kernel or public-serving claim.",
 "rows":[]}

def verify(value,m):
    assert value.shape==(m,n) and value.dtype==np.float32
    assert np.isfinite(value).all()
    difference=np.abs(value.astype(np.float64)-expected[:m])
    violations=int(np.count_nonzero(difference>bound[:m]))
    assert violations==0,(m,violations)
    return {"arithmetic_bound_violations":violations,
            "native_vs_decoded_fp8":_relative_rms(expected[:m],value),
            "native_vs_bf16":_relative_rms(original[:m],value)}

with package.prepare([codes,scales,globals_,qa[:19],sa[:19]],runtime_library=lib._name) as owner:
    initial=owner.rows_info()
    owner.import_weights()
    retained=[]
    for m in (19,1,256,200,127,19):
        owner.update_activations(qa[:m],sa[:m])
        generation,_=owner.invoke_consumer()
        actual=owner.read(generation)[0]
        quality=verify(actual,m)
        retained.append((actual,actual.copy()))
        assert owner.rows_info()==(256,m,*initial[2:])
        members,program_times,host_times=[],[],[]
        for window in range(7):
            generation,times=owner.profile_members(repeats=128)
            assert min(times)*128>=10.,times
            verify(owner.read(generation)[0],m)
            members.append(times)
            generation,event=owner.invoke(repeats=128,timed=True)
            assert event*128>=10.
            verify(owner.read(generation)[0],m)
            program_times.append(event)
            begin=time.perf_counter_ns()
            for _ in range(8):
                owner.update_activations(qa[:m],sa[:m])
                generation,_=owner.invoke_consumer()
                actual=owner.read(generation)[0]
            elapsed=(time.perf_counter_ns()-begin)/1e6
            assert elapsed>=10.
            verify(actual,m)
            host_times.append(elapsed/8)
        row={"active_m":m,**quality,"allocation_bytes":initial[2],"allocation_count":initial[3],
             "device_repeats":128,"host_repeats":8,
             "member_graph_samples_ms":members,"program_event_samples_ms":program_times,
             "prepared_update_consumer_read_samples_ms":host_times,
             "member_median_ms":[statistics.median(x[i] for x in members) for i in (0,1)],
             "program_median_ms":statistics.median(program_times),"prepared_host_median_ms":statistics.median(host_times)}
        record["rows"].append(row)
        (w/"original-bounded-source.json").write_text(json.dumps(record,indent=2)+"\n")
        print(json.dumps({"active_m":m,"quality":quality,"member_median_ms":row["member_median_ms"],
                          "prepared_host_median_ms":row["prepared_host_median_ms"]}),flush=True)
    for value,old in retained:np.testing.assert_array_equal(value,old)
# Source values are consumed as-is; only benchmark input preparation quantizes activation.
print("representative bounded source proof completed",flush=True)


from tessera.compiler.rocm_nvfp4_program import runtime_artifact
from tessera.compiler import nvfp4_expansion, reference_typed_scaled_matmul
import subprocess
def forbidden(*args, **kwargs):
    raise AssertionError("representative public/portable replay invoked compiler or eager oracle")
artifact=rt.RuntimeArtifact.from_json(runtime_artifact(program).to_json())
product._fn=forbidden
product._trace_frontend_capture=forbidden
nvfp4_expansion.reference_nvfp4_expand=forbidden
reference_typed_scaled_matmul.reference_typed_scaled_matmul=forbidden
subprocess.run=forbidden
record["public_portable_checks"]=[]
for m in (1,19,200,256,1):
    roots=[codes,scales,globals_,qa[:m],sa[:m]]
    begin=time.perf_counter_ns()
    actual=product(*roots)
    public_ms=(time.perf_counter_ns()-begin)/1e6
    verify(actual,m)
    assert product.execution_kind=="native_gpu"
    begin=time.perf_counter_ns()
    receipt=rt.launch(artifact,roots)
    portable_ms=(time.perf_counter_ns()-begin)/1e6
    assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
    verify(receipt["output"],m)
    np.testing.assert_array_equal(actual,receipt["output"])
    record["public_portable_checks"].append({"active_m":m,"public_completed_sample_ms":public_ms,
        "portable_completed_sample_ms":portable_ms,"arithmetic_bound_violations":0,
        "sample_kind":"single correctness-call wall characterization, not a benchmark median"})
    print("original public/portable bounded replay",m,"passed",flush=True)
record["recorder_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(w/"original-bounded-source.json").write_text(json.dumps(record,indent=2)+"\n")
