"""Public authored BF16 quantizer -> scaled-product JIT benchmark and artifact export."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt

def make_product(m=3,n=17,k=32):
    groups=m*k//16
    divisor=np.full((groups,1),448,np.float32);divisor.flags.writeable=False
    floor=np.full((groups,1),np.finfo(np.float32).tiny,np.float32);floor.flags.writeable=False
    def product(x,b,sb):
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
        return ts.ops.scaled_matmul(codes,b,scales,sb,transposeB=True,
            scale_layout={"granularity":"block","block":[1,16],"format":"fp32"},
            numeric_policy={"accum":"fp32","execution_mode":"exact_per_block"})
    return product


def inputs(m,n,k,seed=731):
    rng=np.random.default_rng(seed)
    return [rng.normal(size=(m,k)).astype(ml_dtypes.bfloat16),
            rng.normal(size=(n,k)).astype(ml_dtypes.float8_e4m3fn),
            rng.uniform(.02,.2,size=(k//16,n)).astype(np.float32)]


def oracle(values):
    x,b,sb=values;m,k=x.shape;n=b.shape[0]
    blocks=x.astype(np.float32).reshape(m,k//16,16)
    scales=np.maximum(np.max(np.abs(blocks),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
    q=(blocks/scales[...,None]).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
    return np.einsum("mgj,ngj,mg,gn->mn",q,b.astype(np.float64).reshape(n,k//16,16),scales.astype(np.float64),sb.astype(np.float64))


def run(m,n,k,output_dir,repeats):
    if rt._rocm_live_arch()!="gfx1201":
        raise RuntimeError("requires owning gfx1201")
    info=subprocess.check_output(["/opt/rocm/bin/rocminfo"],text=True)
    identity=re.search(r"Name:\s+gfx1201\s+Uuid:\s+(GPU-\S+)\s+Marketing Name:\s+([^\n]+)",info)
    if identity is None:
        raise RuntimeError("owning device identity unavailable")
    roots=inputs(m,n,k)
    expected=oracle(roots)
    fn=ts.jit(target="rocm_gfx1201",native_required=True)(make_product(m,n,k))
    begin=time.perf_counter();result=fn(*roots);cold=(time.perf_counter()-begin)*1000
    np.testing.assert_allclose(result,expected,rtol=.0002,atol=.0002)
    artifact=fn.runtime_artifact()
    if artifact.metadata["compiler_path"]!="canonical_rocm_quantized_product_program":
        raise AssertionError("public call did not select the compiled program")
    error=float(np.max(np.abs(result-expected)))
    changed=inputs(m,n,k,seed=901);expected2=oracle(changed)
    restored=rt.RuntimeArtifact.from_json(artifact.to_json())
    receipt=rt.launch(restored,changed)
    if not receipt.get("ok") or receipt.get("execution_kind")!="native_gpu":
        raise AssertionError(receipt)
    np.testing.assert_allclose(receipt["output"],expected2,rtol=.0002,atol=.0002)
    warm=[]
    for _ in range(7):
        begin=time.perf_counter()
        for _ in range(repeats):fn(*roots)
        warm.append((time.perf_counter()-begin)*1000/repeats)
    # Same compiled members, exercising the former artifact/portable boundary.
    # This control omits frontend dispatch; it is not a kernel-time comparison.
    check=rt.launch(fn._build_runtime_artifact(),roots)
    if not check.get("ok"):
        raise AssertionError(check)
    np.testing.assert_allclose(check["output"],expected,rtol=.0002,atol=.0002)
    serialized=[]
    for _ in range(7):
        begin=time.perf_counter()
        for _ in range(repeats):
            control=rt.launch(fn._build_runtime_artifact(),roots)
            if not control.get("ok"):
                raise AssertionError(control)
        serialized.append((time.perf_counter()-begin)*1000/repeats)
    output_dir.mkdir(parents=True,exist_ok=True)
    name=f"public-{m}-{n}-{k}"
    payload=artifact.to_json();(output_dir/(name+"-artifact.json")).write_text(payload)
    record={"device":{"uuid":identity[1],"name":identity[2].strip()},"architecture":"gfx1201",
        "shape_mnk":[m,n,k],"correctness":"passed_before_timing","max_abs_error":error,
        "changed_roots_portable_replay":"passed","component_receipts":receipt["component_receipts"],
        "cold_call_ms":cold,"warm_public_samples_ms":warm,"warm_public_median_ms":median(warm),
        "serialized_dispatch_control_samples_ms":serialized,"serialized_dispatch_control_median_ms":median(serialized),
        "control_scope":"fresh compiler artifact construction plus portable launch using the same native members; excludes frontend dispatch",
        "artifact_sha256":hashlib.sha256(payload.encode()).hexdigest(),
        "recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "route":"Python -> typed Graph -> native partition -> Schedule -> Tile -> ROCm Target -> ROCDL/LLVM/HSACO -> checked C++ HIP owner",
        "timing_scope":"cold includes tracing/compilation/preparation/invoke/readback but excludes decorator construction; warm includes current host-root copies, checked owner preparation, invoke and readback; no kernel-time or speedup claim"}
    (output_dir/(name+".json")).write_text(json.dumps(record,indent=2)+"\n")
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=5)
    args=parser.parse_args()
    if not 1<=args.repeats<=1000:
        raise ValueError("repeats must be in [1,1000]")
    for m,n,k in ((3,17,32),(19,33,64),(256,128,12288)):
        result=run(m,n,k,args.output_dir,args.repeats)
        print(result["shape_mnk"],result["warm_public_median_ms"],flush=True)
