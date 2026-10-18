"""Correctness-gated four-stage SM120 source reverse timing packet.

Device events measure ordered launch windows, not isolated instruction time.
Public/JIT dispatch and compiler-free portable replay are separate future gates.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.prepared_source_attention_reverse import PreparedSourceAttentionReverse
from tests.device.nvidia.test_source_attention_reverse import oracle, source


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(storage, wide, repeats, samples):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    sq, sk, d, dv = (17, 19, 257, 13) if wide else (3, 5, 4, 3)
    rng = np.random.default_rng(20261018)
    shapes = ((1, 2, sq, d), (1, 1, sk, d), (1, 1, sk, dv),
              (1, 2, sq, dv), (1, 2, sq))
    values = tuple(rng.normal(0, .3, shape).astype(np.float32 if i == 4 else dtype)
                   for i, shape in enumerate(shapes))
    before = time.perf_counter()
    program = export_native_source_attention_reverse(source(values[:3], "normal", (0, 1, 2)))
    export_ms = (time.perf_counter() - before) * 1000
    before = time.perf_counter()
    owner = PreparedSourceAttentionReverse(program)
    compile_ms = (time.perf_counter() - before) * 1000
    expected = oracle(*values)
    result = dict(storage=storage, shape_b_hq_hkv_sq_sk_d_dv=list(owner._dims),
                  source_graph_sha256=program.manifest["source_graph_sha256"],
                  program_graph_sha256=program.manifest["program_graph_sha256"],
                  export_ms=export_ms, member_compile_ms=compile_ms,
                  entries=[m.entry for m in owner._members],
                  images=[m.image_digest for m in owner._members],
                  schedule_sha256=[hashlib.sha256(m.schedule_ir.encode()).hexdigest() for m in owner._members])
    try:
        actual = owner(*values)
        for got, ref in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32), rtol=.03, atol=.002)
        result["max_abs_error"] = [float(np.max(np.abs(a.astype(np.float32)-b.astype(np.float32))))
                                   for a, b in zip(actual, expected, strict=True)]
        result["native_prepare_ms"] = owner.receipt["native_prepare_ms"]
        result["correctness"] = "independent_float64_oracle_with_explicit_storage_edges"
        with ExitStack() as stack:
            sessions = tuple(stack.enter_context(NvidiaDeviceSession()) for _ in values)
            resident = tuple(s.upload(x) for s, x in zip(sessions, values, strict=True))
            got = owner(*resident)
            for a, b in zip(got, actual, strict=True):
                np.testing.assert_array_equal(a, b)
            records = {"host": [], "resident": []}
            # Alternate placement order to limit one-sided drift. Values/images
            # are identical; resident execution still snapshots private buffers.
            for index in range(samples):
                for placement in (("host", "resident") if index % 2 == 0 else ("resident", "host")):
                    inputs = values if placement == "host" else resident
                    stage, device, completed = [], [], []
                    start = time.perf_counter()
                    for _ in range(repeats):
                        owner(*inputs)
                        stage.append(owner.receipt["stage_device_ms"])
                        device.append(owner.receipt["program_device_ms"])
                        completed.append(owner.receipt["completed_host_ms"])
                    records[placement].append(dict(
                        end_to_end_ms=(time.perf_counter()-start)*1000/repeats,
                        native_completed_host_ms=median(completed),
                        program_device_window_ms=median(device),
                        stage_device_window_ms=[median(row[i] for row in stage) for i in range(4)]))
            result["timings"] = records
            result["medians"] = {placement: {
                key: median(row[key] for row in rows) for key in (
                    "end_to_end_ms", "native_completed_host_ms", "program_device_window_ms")}
                for placement, rows in records.items()}
    finally:
        owner.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 3 or args.repeats < 1:
        parser.error("at least three samples and one repeat required")
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("owning SM120 required")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
    packet = dict(schema="tessera.source_attention_reverse_timing.v1", device=gpu,
                  architecture="sm_120", samples=args.samples, repeats=args.repeats,
                  compiler_sha256=digest(os.environ["TESSERA_OPT"]),
                  provider_sha256=digest(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]),
                  scope="prepared_native_owner; public_jit_and_portable_route_not_yet_integrated",
                  stage_order=["rmsnorm", "saved_lse_forward", "attention_backward", "source_pullback"],
                  device_timing="CUDA event launch windows; includes possible dispatch gaps; excludes copies",
                  rows=[run(storage, wide, args.repeats, args.samples)
                        for storage in ("f16", "bf16") for wide in (False, True)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
