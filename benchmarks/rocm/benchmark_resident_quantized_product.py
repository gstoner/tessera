"""Measure an authored native quantizer/product Graph; no Python Graph lowering.

Consumes --source MLIR and the compiler's SSA/lifetime plan. This recorder is
an engineering harness, not a public JIT integration or performance promotion.
"""
from __future__ import annotations
import argparse
import base64
import ctypes as C
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import time
import subprocess

import ml_dtypes
import numpy as np
from tessera import runtime as rt
from tessera.compiler.native_scaled_program import (
    PreparedScaledProgram, _BoundScaledABI, _Buffer, _StorageSpec, _status,
    _storage_numpy_dtype,
)
from tessera.compiler.rocm_block_quant_native import lower_block_quant_text, package_block_quant_recipe, _frame
from tessera.compiler.rocm_fp8_blockscale import BlockScaleProgram, BlockScaleShape, package_blockscale, verify_blockscale_schedule
from tessera.compiler.rocm_nvfp4_expansion_program import _MultiOutputStep
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

OPTION = "--tessera-autodiff-forward=export-scaled-primal=true"


def export(tool, graph):
    outlined = run_tessera_opt(tool, graph, OPTION)
    encoded = re.search(r'tessera.native.quantized_product_json = "([^"\n]+)"', outlined)[1]
    return json.loads(base64.b64decode(encoded)), outlined


@dataclass
class Bundle:
    graph: str
    plan: dict
    native: tuple
    tool: Path

    @property
    def program_json(self):
        return json.dumps(self.plan)

    def validate(self):
        replay, _ = export(self.tool, self.graph)
        if replay != self.plan:
            raise ValueError("native composed plan changed before preparation")
        if any(p.image.architecture != "gfx1201" or not p.image.payload.startswith(b"\x7fELF") for p in self.native):
            raise ValueError("resident product requires gfx1201 HSACO members")


class Owner(PreparedScaledProgram):
    _prepare_symbol = "tessera_rocm_program_prepare_multi_output"
    _step_type = _MultiOutputStep

    def _bind_abi(self, package):
        package.validate()
        plan, native = package.plan, package.native
        rows = plan["buffers"]
        blobs = tuple(C.create_string_buffer(p.image.payload) for p in native)
        entries = tuple(p.descriptor.entry_symbol.encode() for p in native)
        steps = (_MultiOutputStep * len(native))()
        quant_input = plan["steps"][0]["inputs"][0]
        m,k = rows[quant_input]["shape"]
        n = rows[plan["output"]]["shape"][1]
        for i,p in enumerate(native):
            step,spec = steps[i],plan["steps"][i]
            step.image,step.image_bytes,step.entry = C.cast(blobs[i],C.c_void_p),len(p.image.payload),entries[i]
            step.input_count,step.output_count = len(spec["inputs"]),len(spec["outputs"])
            step.inputs[:step.input_count],step.outputs[:step.output_count] = spec["inputs"],spec["outputs"]
            if i == 0:
                _,member = _frame(p.target_ir)
                scalars,geometry = member["scalars"],member["geometry"]
            else:
                bm,bn = p.descriptor.provenance["macro_tile"]
                scalars = [m,n,k]
                geometry = [(n+bn-1)//bn,(m+bm-1)//bm,1,*p.descriptor.provenance["workgroup"]]
            step.scalar_count = len(scalars)
            step.scalars[:step.scalar_count],step.geometry[:] = scalars,geometry
        ownership = {"readonly_input":0,"private_scratch":1,"returned_output":2}
        buffers = (_Buffer * len(rows))(*[
            _Buffer(b["bytes"],math.prod(b["shape"]),b["first_write"],b["last_read"],ownership[b["ownership"]],0)
            for b in rows])
        storage = tuple(_StorageSpec(tuple(b["shape"]),b["storage"],b["bytes"],_storage_numpy_dtype(b["storage"])) for b in rows)
        return _BoundScaledABI(buffers,steps,blobs,entries,3,storage,(plan["output"],),())

    def invoke_range(self, first, count, repeats=1):
        invoke = self.lib.tessera_rocm_program_invoke_range
        invoke.argtypes = [C.c_uint64,C.c_uint32,C.c_uint32,C.c_uint32,C.POINTER(C.c_uint64),C.POINTER(C.c_float)]
        invoke.restype = C.c_int
        generation,elapsed = C.c_uint64(),C.c_float()
        _status(invoke(self.handle,first,count,repeats,C.byref(generation),C.byref(elapsed)))
        return generation.value,float(elapsed.value)


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
    plan,outlined = export(tool,graph)
    producer = package_block_quant_recipe(lower_block_quant_text(plan["member_graphs"][0]))
    m,k = plan["buffers"][plan["steps"][0]["inputs"][0]]["shape"]
    n = plan["buffers"][plan["output"]]["shape"][1]
    out = plan["buffers"][plan["output"]]["storage"]
    shape = BlockScaleShape(m,n,k,16,1,"nk",out)
    child = plan["member_graphs"][1]
    schedule = run_tessera_opt(tool,child,"--tessera-graph-to-schedule")
    verify_blockscale_schedule(shape,schedule)
    tile = run_tessera_opt(tool,schedule,"--tessera-schedule-to-tile")
    consumer = package_blockscale(BlockScaleProgram(shape,"quantized_product",child,schedule,tile))
    bundle = Bundle(graph,plan,(producer,consumer),tool)
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
    with Owner(bundle,values,runtime_library=lib._name) as owner:
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
        "image_sha256":[hashlib.sha256(p.image.payload).hexdigest() for p in bundle.native],
        "graph_sha256":hashlib.sha256(graph.encode()).hexdigest(),"plan":plan,"outlined_ir":outlined,
        "timing_domains":{"event":"ordinary HIP event windows include launch gaps; native API already divides by repeats","captured":"grouped captured member device windows include graph dispatch, exclude capture/instantiation/copies","prepared":"invoke plus returned-output readback; excludes package compilation/preparation"},
        "scope":"native authored Graph partition plus prepared C++ owner; no public JIT or speedup claim"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--repeats",type=int,default=50)
    args=parser.parse_args()
    if not 1 <= args.repeats <= 1048576:
        raise ValueError("repeats outside native owner limit")
    args.output.write_text(json.dumps(run(args.source.read_text(),args.repeats),indent=2)+"\n")
