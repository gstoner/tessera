"""Correctness-gated same-image A/B for public source-attention VJP dispatch."""
from dataclasses import replace
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts

from tessera import runtime as rt
from tessera.compiler import native_vjp_composites as composites
from tests.device.nvidia.test_source_attention_reverse import oracle
from tests.unit.test_native_attention_prefix_program import prefix_attention


def run(storage, wide, control, samples, repeats):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    sq, sk, d, dv = (17, 19, 257, 13) if wide else (3, 5, 4, 3)
    rng = np.random.default_rng(20261019)
    shapes = ((1, 2, sq, d), (1, 1, sk, d), (1, 1, sk, dv), (1, 2, sq, dv), (1, 2, sq))
    values = tuple(rng.normal(0, .3, shape).astype(np.float32 if i == 4 else dtype)
                   for i, shape in enumerate(shapes))
    expected = oracle(*values)
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse")(prefix_attention)
    key = "source_normalization_attention"
    entry = composites._COMPOSITES[key]
    arms = {"retained_native_owner": entry,
            "serialized_runtime_control": replace(entry, execute=control)}
    records = {name: [] for name in arms}
    def invoke():
        return fn.native_backward(*values[:3], out_cotangents=values[3:])
    def check(actual):
        for got, ref in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32), rtol=.03, atol=.002)
    before = time.perf_counter()
    check(invoke())
    cold_ms = (time.perf_counter() - before) * 1000
    images = fn.last_backward_execution["image_digests"]
    try:
        for arm in arms.values():
            composites._COMPOSITES[key] = arm
            for _ in range(3):
                check(invoke())
                assert fn.last_backward_execution["image_digests"] == images
        for sample in range(samples):
            order = tuple(arms) if sample % 2 == 0 else tuple(reversed(arms))
            for name in order:
                composites._COMPOSITES[key] = arms[name]
                device, completed, stages = [], [], []
                start = time.perf_counter()
                for _ in range(repeats):
                    outputs = invoke()
                    receipt = fn.last_backward_execution
                    device.append(receipt["program_device_ms"])
                    completed.append(receipt["completed_host_ms"])
                    stages.append(receipt["stage_device_ms"])
                elapsed = (time.perf_counter() - start) * 1000 / repeats
                check(outputs)
                assert receipt["image_digests"] == images
                records[name].append(dict(public_completed_ms=elapsed,
                    native_completed_host_ms=median(completed), program_device_window_ms=median(device),
                    stage_device_window_ms=[median(row[i] for row in stages) for i in range(4)]))
        owner = next(iter(fn._native_prepared_vjp_calls.values()))
        exported = owner.manifest()
        medians = {name: {field: median(row[field] for row in rows) for field in (
            "public_completed_ms", "native_completed_host_ms", "program_device_window_ms")}
            for name, rows in records.items()}
        return dict(storage=storage, dims=list(owner._dims), cold_public_ms=cold_ms,
            images=list(images), source_graph_sha256=exported["plan"]["source_graph_sha256"],
            program_graph_sha256=exported["plan"]["program_graph_sha256"],
            member_stages=[record["provenance"] for record in exported["members"]],
            correctness="float64_storage_boundary_oracle_before_and_after_each_sample",
            samples=records, medians=medians,
            public_speedup=medians["serialized_runtime_control"]["public_completed_ms"] /
                           medians["retained_native_owner"]["public_completed_ms"])
    finally:
        composites._COMPOSITES[key] = entry
        for owner in fn._native_prepared_vjp_calls.values():
            owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 3 or args.repeats < 1:
        parser.error("at least three samples and one repeat required")
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("owning SM120 required")
    spec = importlib.util.spec_from_file_location("tessera.compiler._source_reverse_control", args.control)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packet = dict(schema="tessera.public_source_attention_reverse_ab.v1", architecture="sm_120",
        device=subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version",
                                         "--format=csv,noheader"], text=True).strip(),
        control_sha256=hashlib.sha256(args.control.read_bytes()).hexdigest(),
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        provider_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        samples=args.samples, repeats=args.repeats, scope="public_dispatch_same_inputs_images_and_native_owner",
        timing="completed host calls; CUDA event launch windows include possible dispatch gaps",
        rows=[run(storage, wide, module.execute, args.samples, args.repeats)
              for storage in ("f16", "bf16") for wide in (False, True)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
