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


# This input preparation is a benchmark oracle/control, not a compiled activation
# quantizer or a claim that the complete MLP is already native.
import subprocess
import math
outdir=Path("/home/angstorms/scratch/rocm-native-gate-up-views-20261016/original-store-source")
outdir.mkdir(exist_ok=True)
width=n//2
assert offsets==[0,width,n]
parent_low=(expected-bound).astype(ml_dtypes.bfloat16).astype(np.float64)
parent_high=(expected+bound).astype(ml_dtypes.bfloat16).astype(np.float64)
def silu64(x):
    with np.errstate(over="ignore",under="ignore"):
        return x/(1+np.exp(-x))
critical=-1.2784645427610738
left_low,left_high=parent_low[:,:width],parent_high[:,:width]
gate_low=np.minimum(silu64(left_low),silu64(left_high))
gate_high=np.maximum(silu64(left_low),silu64(left_high))
contains=(left_low<=critical)&(left_high>=critical)
gate_low=np.where(contains,np.minimum(gate_low,silu64(np.array(critical))),gate_low)
right_low,right_high=parent_low[:,width:],parent_high[:,width:]
products=np.stack([gate_low*right_low,gate_low*right_high,
                   gate_high*right_low,gate_high*right_high])
lower=products.min(axis=0).astype(ml_dtypes.bfloat16)
upper=products.max(axis=0).astype(ml_dtypes.bfloat16)
# One outward BF16 ULP budgets activation FP32 exp/divide/multiply rounding.
lower=np.nextafter(lower,np.array(-np.inf,dtype=ml_dtypes.bfloat16)).astype(np.float64)
upper=np.nextafter(upper,np.array(np.inf,dtype=ml_dtypes.bfloat16)).astype(np.float64)
rounded=expected.astype(ml_dtypes.bfloat16).astype(np.float64)
reference=(silu64(rounded[:,:width])*rounded[:,width:]).astype(ml_dtypes.bfloat16).astype(np.float64)
bf16_activation=(silu64(original[:,:width])*original[:,width:]).astype(ml_dtypes.bfloat16).astype(np.float64)
del parent_low,parent_high,products,rounded,left_low,left_high,gate_low,gate_high,right_low,right_high
record={"schema":"tessera.gfx1201.original_nvfp4_bf16_activation_source.v1",
    "architecture":rt._rocm_live_arch(),"checkpoint_sources":sources,"activation_source":activation_source,
    "shape_nk":[n,k],"source_rows":list(offsets),
    "full_model_quality_acceptance":False,"strategy_promotion":False,
    "validation_contract":"independent f64 decoded-weight projection with FP32 accumulation gamma bound; BF16 parent interval; exact SiLU extrema; product interval; outward one-BF16-ULP activation allowance",
    "input_preparation":"captured BF16 activations quantized to E4M3/K16 scales on host for this control only; native activation quantizer/down projection remain open",
    "timing_scope":{"members":"captured grouped member graph dispatch, excludes capture/instantiate/CPU enqueue",
                    "program":"resident product+activation HIP event range may include CPU enqueue gaps",
                    "host":"retained owner update/upload + resident product/activation + read; compilation/prepare excluded"},
    "compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
    "provider_sha256":hashlib.sha256(Path(os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]).read_bytes()).hexdigest(),
    "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"rows":[]}
def function(m):
    policy=nvfp4_expansion_policy()
    def composed(codes,scales,globals_,activation,activation_scales):
        weight,weight_scales=ts.ops.nvfp4_expand(codes,scales,globals_,
            row_offsets=offsets,numeric_policy=policy)
        accumulator=ts.ops.scaled_matmul(activation,weight,activation_scales,weight_scales,
            transposeB=True,numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"},
            scale_layout={"granularity":"block","block":[1,16],"format":"fp32"})
        parent=ts.ops.cast(accumulator,"bf16")
        gate=ts.ops.slice(parent,[0,0],[m,width])
        up=ts.ops.slice(parent,[0,width],[m,width])
        gate_f32=ts.ops.cast(gate,"fp32")
        up_f32=ts.ops.cast(up,"fp32")
        activation=ts.ops.silu_mul(gate_f32,up_f32)
        return ts.ops.cast(activation,"bf16")
    return composed
def verify(value,m):
    assert value.shape==(m,width) and value.dtype==np.dtype(ml_dtypes.bfloat16)
    f64=value.astype(np.float64)
    assert np.isfinite(f64).all()
    violations=int(np.count_nonzero((f64<lower[:m])|(f64>upper[:m])))
    assert violations==0,(m,violations,float(np.max(np.maximum(lower[:m]-f64,f64-upper[:m]))))
    return {"interval_violations":violations,
        "relative_rms_vs_decoded_nvfp4":_relative_rms(reference[:m],f64),
        "relative_rms_vs_original_bf16":_relative_rms(bf16_activation[:m],f64)}
def forbidden(*a,**kw):
    raise AssertionError("prepared/portable original-model execution invoked compiler or eager oracle")
assert rt._rocm_live_arch()=="gfx1201"
for m in (19,256):
    roots=[codes,scales,globals_,qa[:m],sa[:m]]
    compiled=ts.jit(target="rocm_gfx1201")(function(m))
    start=time.perf_counter()
    program=compiled.compile_native_nvfp4_program(*roots)
    package=program.native
    compile_ms=(time.perf_counter()-start)*1000
    assert len(package.native_stages)==3
    plan=json.loads(package.program_json)
    assert len(plan["buffers"])==9 and plan["buffers"][7]["storage"]=="bf16"
    assert plan["buffers"][7]["ownership"]=="private_scratch" and plan["output"]==8
    (outdir/f"source-{m}.mlir").write_text(program.graph_ir)
    with package.prepare(roots,runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as owner:
        owner.import_weights()
        generation,_=owner.invoke_consumer()
        value=owner.read(generation)[0]
        quality=verify(value,m)
        held=value.copy()
        generation,pilot=owner.profile_members(repeats=8)
        verify(owner.read(generation)[0],m)
        repeats=min(65536,max(8,math.ceil(22/max(min(pilot),1e-6))))
        for _ in range(5):
            generation,calibration=owner.profile_members(repeats=repeats)
            verify(owner.read(generation)[0],m)
            window=repeats*min(calibration)
            if window>=18:break
            repeats=min(65536,max(repeats+1,math.ceil(repeats*22/max(window,1e-6))))
        else:raise AssertionError("member calibration failed")
        member_samples=[];program_samples=[];host_samples=[]
        program_repeats=max(4,math.ceil(22/max(sum(calibration[1:]),1e-6)))
        old_run=subprocess.run
        subprocess.run=forbidden
        try:
            for _ in range(3):
                generation,times=owner.profile_members(repeats=repeats)
                assert repeats*min(times)>=10.,times
                verify(owner.read(generation)[0],m)
                member_samples.append(list(times))
                generation,elapsed=owner.invoke_consumer(repeats=program_repeats,timed=True)
                assert elapsed*program_repeats>=10.
                verify(owner.read(generation)[0],m)
                program_samples.append(elapsed)
                begin=time.perf_counter()
                for _ in range(8):
                    owner.update_activations(roots[3],roots[4])
                    generation,_=owner.invoke_consumer()
                    value=owner.read(generation)[0]
                completed=(time.perf_counter()-begin)*1000
                verify(value,m)
                assert completed>=10.
                host_samples.append(completed/8)
            # Compiler-free portable replay of this exact model-shaped source.
            from tessera.compiler.rocm_nvfp4_program import runtime_artifact
            artifact=rt.RuntimeArtifact.from_json(runtime_artifact(program).to_json())
            receipt=rt.launch(artifact,roots)
            assert receipt["ok"] and receipt["execution_kind"]=="native_gpu",receipt
            verify(receipt["output"],m)
        finally:
            subprocess.run=old_run
        np.testing.assert_array_equal(held,value)
    row={"m":m,"width":width,"k":k,**quality,"compile_package_ms":compile_ms,
        "binding_digest":package.binding_digest,"program_sha256":hashlib.sha256(package.program_json.encode()).hexdigest(),
        "image_digests":[p.image.image_digest for p in package.native_stages],
        "physical_buffers":plan["buffers"],"member_repeats":repeats,
        "member_graph_samples_ms":member_samples,"member_median_ms":[statistics.median(x[i] for x in member_samples) for i in range(3)],
        "program_repeats":program_repeats,"product_activation_event_samples_ms":program_samples,
        "completed_update_invoke_read_samples_ms":host_samples,"completed_host_median_ms":statistics.median(host_samples),
        "portable_compiler_disabled":"passed"}
    record["rows"].append(row)
    (outdir/"gfx1201.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"m":m,"quality":quality,"member_medians_ms":row["member_median_ms"],
        "host_ms":row["completed_host_median_ms"]}),flush=True)
print("original gate/up BF16 activation composition complete",flush=True)
