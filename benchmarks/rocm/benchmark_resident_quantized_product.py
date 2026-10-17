"""Measure an authored native quantizer/product Graph; no Python Graph lowering.

Consumes --source MLIR and the compiler's SSA/lifetime plan. This recorder is
the production package/owner API; this is not a public JIT performance promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import time
import subprocess

import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.rocm_quantized_product_program import package_quantized_product_graph
from tessera.compiler.scheduled_matmul import find_tessera_opt


def run(graph, repeats):
    arch = rt._rocm_live_arch()
    if arch != "gfx1201":
        raise RuntimeError(f"requires live gfx1201, found {arch}")
    device_info = subprocess.check_output(["/opt/rocm/bin/rocminfo"],text=True)
    identity = re.search(r"Name:\s+gfx1201\s+Uuid:\s+(GPU-\S+)\s+Marketing Name:\s+([^\n]+)",device_info)
    if not identity:
        raise RuntimeError("live gfx1201 identity is unavailable")
    device = {"uuid":identity[1],"name":identity[2].strip()}
    tool = find_tessera_opt()
    if tool is None:
        raise RuntimeError("requires matching native compiler")
    bundle = package_quantized_product_graph(graph)
    plan = json.loads(bundle.program_json)
    m,k = plan["buffers"][plan["steps"][0]["inputs"][0]]["shape"]
    n = plan["buffers"][plan["output"]]["shape"][1]
    out = plan["buffers"][plan["output"]]["storage"]
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise RuntimeError("requires native multi-output HIP owner")
    rng = np.random.default_rng(731)
    def inputs():
        values = [None]*3
        values[plan["steps"][0]["inputs"][0]] = rng.normal(size=(m,k)).astype(ml_dtypes.bfloat16)
        values[plan["steps"][1]["inputs"][1]] = rng.normal(size=(n,k)).astype(ml_dtypes.float8_e4m3fn)
        values[plan["steps"][1]["inputs"][3]] = rng.uniform(.02,.2,size=(k//16,n)).astype(np.float32)
        return values
    def oracle(values):
        x = values[plan["steps"][0]["inputs"][0]].astype(np.float32).reshape(m,k//16,16)
        scale = np.maximum(np.max(np.abs(x),axis=-1)/np.float32(448),np.finfo(np.float32).tiny)
        quant = (x/scale[...,None]).astype(ml_dtypes.float8_e4m3fn).astype(np.float64)
        weights = values[plan["steps"][1]["inputs"][1]].astype(np.float64).reshape(n,k//16,16)
        rhs_scale = values[plan["steps"][1]["inputs"][3]].astype(np.float64)
        return np.einsum("mgj,ngj,mg,gn->mn",quant,weights,scale.astype(np.float64),rhs_scale)
    values = inputs()
    errors=[]
    with bundle.prepare(values,runtime_library=lib._name) as owner:
        try:
            owner.invoke_range(1,1)
        except RuntimeError as error:
            if "status 10" not in str(error):
                raise
        else:
            raise AssertionError("consumer executed before its private producer outputs")
        held=None
        for _ in range(3):
            owner.update(values)
            generation,_ = owner.invoke()
            result = owner.read(generation)[0]
            expected=oracle(values)
            np.testing.assert_allclose(result.astype(np.float64),expected,rtol=.005 if out=="bf16" else .0002,atol=.01 if out=="bf16" else .0002)
            errors.append(float(np.max(np.abs(result.astype(np.float64)-expected))))
            if held is not None:
                np.testing.assert_array_equal(held[0],held[1])
            held=(result,result.copy());values=inputs()
        producer_times=[owner.invoke_range(0,1,repeats)[1] for _ in range(7)]
        consumer_times=[owner.invoke_range(1,1,repeats)[1] for _ in range(7)]
        composed=[owner.invoke(repeats=repeats,timed=True)[1] for _ in range(7)]
        captured=[owner.profile_members(repeats=min(repeats,65536))[1] for _ in range(7)]
        prepared=[]
        for _ in range(7):
            start=time.perf_counter();generation,_=owner.invoke();owner.read(generation)
            prepared.append((time.perf_counter()-start)*1e3)
    return {"architecture":arch,"device":device,"rocminfo":device_info,"compiler_sha256":hashlib.sha256(tool.read_bytes()).hexdigest(),"runtime_sha256":hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),"recorder_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"shape_mnk":[m,n,k],"output":out,"correctness":"passed_before_timing","max_abs_errors":errors,
        "changed_input_generations":3,"consumer_before_producer":"rejected","retained_outputs":"passed",
        "producer_event_samples_ms":producer_times,"consumer_event_samples_ms":consumer_times,
        "composed_event_samples_ms":composed,"captured_member_samples_ms":captured,"prepared_invoke_read_samples_ms":prepared,
        "medians_ms":{"producer":median(producer_times),"consumer":median(consumer_times),"composed":median(composed),"captured_producer":median(v[0] for v in captured),"captured_consumer":median(v[1] for v in captured),"prepared_invoke_read":median(prepared)},
        "image_sha256":[hashlib.sha256(p.image.payload).hexdigest() for p in bundle.native_stages],
        "graph_sha256":hashlib.sha256(graph.encode()).hexdigest(),"plan":plan,
        "timing_domains":{"event":"ordinary HIP event windows include launch gaps; native API already divides by repeats","captured":"grouped captured member device windows include graph dispatch, exclude capture/instantiation/copies","prepared":"invoke plus returned-output readback; excludes package compilation/preparation"},
        "scope":"native authored Graph partition plus prepared C++ owner; no public JIT or speedup claim"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--repeats",type=int,default=50)
    args=parser.parse_args()
    if not 1 <= args.repeats <= 1048576:
        raise ValueError("repeats outside native owner limit")
    args.output.write_text(json.dumps(run(args.source.read_text(),args.repeats),indent=2)+"\n")
