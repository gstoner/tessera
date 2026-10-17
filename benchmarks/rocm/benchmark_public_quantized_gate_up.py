"""Authored native quantizer/weight expansion/BF16-store/gate-up Graph."""
from __future__ import annotations
import ml_dtypes
import numpy as np
import tessera as ts
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy,reference_nvfp4_expand


def make_gate_up(m=3,width=17,k=48,reverse=False):
    groups=m*k//16
    divisor=np.full((groups,1),448,np.float32);divisor.flags.writeable=False
    floor=np.full((groups,1),np.finfo(np.float32).tiny,np.float32);floor.flags.writeable=False
    def gate_up(x,packed,weight_scales,globals_):
        blocks=ts.ops.reshape(x,shape=(groups,16))
        wide=ts.ops.cast(blocks,"fp32")
        absolute=ts.ops.absolute(wide)
        peak=ts.ops.reduce(absolute,op="max",axis=1,keepdims=True)
        raw=ts.ops.div(peak,divisor)
        scale=ts.ops.maximum(raw,floor)
        expanded=ts.ops.expand(scale,shape=(groups,16))
        normalized=ts.ops.div(wide,expanded)
        encoded=ts.ops.cast(normalized,"fp8_e4m3")
        codes=ts.ops.reshape(encoded,shape=(m,k))
        scales=ts.ops.reshape(scale,shape=(m,k//16))
        b,sb=ts.ops.nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,width,2*width],
            numeric_policy=nvfp4_expansion_policy())
        accumulator=ts.ops.scaled_matmul(codes,b,scales,sb,transposeB=True,
            scale_layout={"granularity":"block","block":[1,16],"format":"fp32"},
            numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"})
        parent=ts.ops.cast(accumulator,"bf16")
        left=ts.ops.slice(parent,start_indices=[0,0],slice_sizes=[m,width])
        right=ts.ops.slice(parent,start_indices=[0,width],slice_sizes=[m,width])
        left_f32=ts.ops.cast(left,"fp32")
        right_f32=ts.ops.cast(right,"fp32")
        gated=ts.ops.silu_mul(right_f32,left_f32) if reverse else ts.ops.silu_mul(left_f32,right_f32)
        return ts.ops.cast(gated,"bf16")
    return gate_up


def inputs(m,width,k,seed=731):
    rng=np.random.default_rng(seed)
    return [rng.normal(size=(m,k)).astype(ml_dtypes.bfloat16),
            rng.integers(0,256,size=(2*width,k//2),dtype=np.uint8),
            rng.uniform(.125,2,size=(2*width,k//16)).astype(ml_dtypes.float8_e4m3fn),
            np.array([.0123456789,.007654321],np.float64)]


def oracle(values,reverse=False):
    x,packed,weight_scales,globals_=values;m,k=x.shape;width=packed.shape[0]//2
    blocks=x.astype(np.float32).reshape(m,k//16,16)
    sa=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/sa[...,None]).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    b,sb=reference_nvfp4_expand(packed,weight_scales,globals_,row_offsets=[0,width,2*width],numeric_policy=nvfp4_expansion_policy())
    accum=np.einsum("mgj,ngj,mg,gn->mn",q,b.astype(np.float64).reshape(2*width,k//16,16),sa.astype(np.float64),sb.astype(np.float64))
    parent=accum.astype(np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)
    gate,up=(parent[:,width:],parent[:,:width]) if reverse else (parent[:,:width],parent[:,width:])
    return (gate/(np.float32(1)+np.exp(-gate))*up).astype(ml_dtypes.bfloat16)


def run(m,n,k,output_dir,repeats):
    import hashlib
    import json
    import os
    import re
    import subprocess
    import time
    from pathlib import Path
    from statistics import median
    from tessera import runtime as rt
    if rt._rocm_live_arch()!="gfx1201":
        raise RuntimeError("requires owning gfx1201")
    info=subprocess.check_output(["/opt/rocm/bin/rocminfo"],text=True)
    identity=re.search(r"Name:\s+gfx1201\s+Uuid:\s+(GPU-\S+)\s+Marketing Name:\s+([^\n]+)",info)
    if identity is None:
        raise RuntimeError("owning device identity unavailable")
    roots=inputs(m,n,k);expected=oracle(roots)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_gate_up(m,n,k))
    start=time.perf_counter();first=fn(*roots);cold=(time.perf_counter()-start)*1000
    np.testing.assert_allclose(first,expected,rtol=.015625,atol=.00002)
    if fn.execution_kind!="native_gpu":
        raise RuntimeError("public call did not execute native program")
    artifact=fn.runtime_artifact()
    if artifact.metadata["compiler_path"] != "canonical_rocm_quantized_gate_up_program":
        raise RuntimeError("wrong compiler route")
    program=fn._rocm_quantized_product_last_program.native
    record={"shape_mhk":[m,n,k],"architecture":rt._rocm_live_arch(),
        "device_uuid":identity[1],"device_name":identity[2].strip(),
        "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "correctness":"passed_before_timing","max_abs_error":float(np.max(np.abs(first.astype(np.float32)-expected.astype(np.float32)))),
        "route":"authored Graph SSA -> native quantizer/expansion/BF16-store-product/gate-up partitions -> Schedule -> Tile -> ROCm Target -> LLVM/HSACO -> checked multi-output HIP owner",
        "native_schema":json.loads(program.program_json)["schema"],
        "compiler_sha256":hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256":hashlib.sha256(Path(rt._load_rocm_native_movement_runtime()._name).read_bytes()).hexdigest(),
        "binding_digest":program.binding_digest,"component_receipts":list(fn._rocm_quantized_product_last_receipts),
        "public_cold_ms":cold,"public_warm_samples_ms":[],"prepared_invoke_read_samples_ms":[],
        "ordinary_program_event_samples_ms":[],"captured_member_samples_ms":[]}
    for _ in range(7):
        start=time.perf_counter()
        for _ in range(repeats):result=fn(*roots)
        record["public_warm_samples_ms"].append((time.perf_counter()-start)*1000/repeats)
    np.testing.assert_allclose(result,expected,rtol=.015625,atol=.00002)
    lib=rt._load_rocm_native_movement_runtime()
    with program.prepare(roots,runtime_library=lib._name) as owner:
        generation,_=owner.invoke();np.testing.assert_allclose(owner.read(generation)[0],expected,rtol=.015625,atol=.00002)
        for _ in range(7):
            _,members=owner.profile_members(repeats=128)
            record["captured_member_samples_ms"].append(list(members))
            generation,event=owner.invoke(repeats=50,timed=True)
            record["ordinary_program_event_samples_ms"].append(event)
            start=time.perf_counter()
            generation,_=owner.invoke();held=owner.read(generation)[0]
            record["prepared_invoke_read_samples_ms"].append((time.perf_counter()-start)*1000)
        np.testing.assert_allclose(held,expected,rtol=.015625,atol=.00002)
    for label in ("public_warm","prepared_invoke_read","ordinary_program_event"):
        record[label+"_median_ms"]=median(record[label+"_samples_ms"])
    record["captured_member_median_ms"]=[median(row[i] for row in record["captured_member_samples_ms"]) for i in range(4)]
    record["timing_scope"]={"public":"warm frontend, input validation, root copies, owner prepare/invoke/read",
        "prepared":"invoke and returned readback; excludes prepare, compilation and root upload",
        "ordinary_event":"per-program HIP event window; native API already averages repeats; includes enqueue gaps",
        "captured_members":"quantizer, weight expansion, BF16-store product, gate/up SiLU; includes graph dispatch; excludes capture/instantiation/copies"}
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    stem=f"public-{m}-{n}-{k}"
    payload=artifact.to_json();(output_dir/(stem+"-artifact.json")).write_text(payload)
    record["artifact_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    (output_dir/(stem+".json")).write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({key:record[key] for key in ("shape_mhk","public_warm_median_ms","captured_member_median_ms","ordinary_program_event_median_ms","max_abs_error")}),flush=True)


if __name__=="__main__":
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=7)
    args=parser.parse_args()
    if not 1<=args.repeats<=1024:parser.error("repeats must be between 1 and 1024")
    for shape in ((1,1,16),(3,17,48),(256,128,12288)):
        run(*shape,args.output_dir,args.repeats)
