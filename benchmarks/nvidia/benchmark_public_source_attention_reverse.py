"""Correctness-gated public source-VJP A/B with independent native owners."""
from dataclasses import replace
import argparse
import gzip
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
from benchmarks.nvidia.benchmark_source_attention_rows import PROFILES
from tests.device.nvidia.test_source_attention_reverse import oracle
from tests.unit.test_native_attention_prefix_program import prefix_attention


def run(storage, profile, control, samples, repeats, order_offset, directory):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    b, hq, hkv, sq, sk, d, dv = PROFILES[profile]
    rng = np.random.default_rng(20261019)
    shapes = ((b, hq, sq, d), (b, hkv, sk, d), (b, hkv, sk, dv), (b, hq, sq, dv), (b, hq, sq))
    values = tuple(rng.normal(0, .3, shape).astype(np.float32 if i == 4 else dtype)
                   for i, shape in enumerate(shapes))
    snapshots = tuple(value.copy() for value in values)
    expected = oracle(*values)
    fn = ts.jit(target="nvidia_sm120", autodiff="reverse")(prefix_attention)
    key = "source_normalization_attention"
    entry = composites._COMPOSITES[key]
    arms = {"native_candidate": entry, "frozen_control": replace(entry, execute=control)}
    records = {name: [] for name in arms}
    pins, cold = {}, {}
    def invoke():
        return fn.native_backward(*values[:3], out_cotangents=values[3:])
    def check(actual):
        for got, ref in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32),
                                       rtol=.03 if storage == "bf16" else .006, atol=.002)
    try:
        for name in (tuple(arms) if order_offset == 0 else tuple(reversed(arms))):
            composites._COMPOSITES[key] = arms[name]
            start = time.perf_counter()
            check(invoke())
            cold[name] = (time.perf_counter() - start) * 1000
            for _ in range(2):
                check(invoke())
            receipt = fn.last_backward_execution
            owners = [owner for owner in fn._native_prepared_vjp_calls.values()
                      if owner._source_graph_digest == receipt["source_graph_ir_digest"]]
            assert len(owners) == 1, "public arm has no unique retained source owner"
            manifest = owners[0].manifest()
            archive = f"{storage}-{profile}-{name}.json.gz"
            encoded = json.dumps(manifest, sort_keys=True).encode()
            (directory / archive).write_bytes(gzip.compress(encoded, mtime=0))
            pins[name] = dict(images=list(receipt["image_digests"]), program_schema=manifest["schema"],
                source_graph_sha256=manifest["plan"]["source_graph_sha256"],
                program_graph_sha256=manifest["plan"]["program_graph_sha256"],
                contract_digest=manifest["contract_digest"], archive=archive,
                archive_sha256=hashlib.sha256((directory / archive).read_bytes()).hexdigest())
        prefix_equal = pins["native_candidate"]["images"][:2] == pins["frozen_control"]["images"][:2]
        assert prefix_equal, "normalization/forward images differ between the public arms"
        old_run, old_popen, old_fn = subprocess.run, subprocess.Popen, fn._fn
        def forbidden(*args, **kwargs):
            raise AssertionError("warm public A/B invoked a compiler or external process")
        try:
            subprocess.run = subprocess.Popen = forbidden
            fn._fn = forbidden
            for sample in range(samples):
                order = tuple(arms) if (sample + order_offset) % 2 == 0 else tuple(reversed(arms))
                for name in order:
                    composites._COMPOSITES[key] = arms[name]
                    device, completed, stages = [], [], []
                    start = time.perf_counter()
                    for _ in range(repeats):
                        outputs = invoke()
                        receipt = fn.last_backward_execution
                        assert receipt["native_prepare_ms"] == 0
                        device.append(receipt["program_device_ms"])
                        completed.append(receipt["completed_host_ms"])
                        stages.append(receipt["stage_device_ms"])
                    elapsed = (time.perf_counter() - start) * 1000 / repeats
                    check(outputs)
                    assert list(receipt["image_digests"]) == pins[name]["images"]
                    records[name].append(dict(sample=sample, public_completed_ms=elapsed,
                        native_completed_host_ms=median(completed), program_device_window_ms=median(device),
                        stage_device_window_ms=[median(row[i] for row in stages) for i in range(len(stages[0]))]))
        finally:
            subprocess.run, subprocess.Popen, fn._fn = old_run, old_popen, old_fn
        for actual, original in zip(values, snapshots, strict=True):
            np.testing.assert_array_equal(actual, original)
        medians = {name: {field: median(row[field] for row in rows) for field in (
            "public_completed_ms", "native_completed_host_ms", "program_device_window_ms")}
            for name, rows in records.items()}
        return dict(storage=storage, profile=profile, dims=list(PROFILES[profile]),
            cold_public_ms=cold, cold_scope="sequential warmup of shared frontend; no cold-speedup claim",
            pins=pins, prefix_and_forward_images_identical=prefix_equal,
            complete_images_identical=pins["native_candidate"]["images"] == pins["frozen_control"]["images"],
            correctness="independent float64 storage-boundary oracle before/after samples; warm subprocesses forbidden",
            samples=records, medians=medians,
            public_speedup=medians["frozen_control"]["public_completed_ms"] /
                           medians["native_candidate"]["public_completed_ms"])
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
    parser.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=("small", "wide"))
    args = parser.parse_args()
    if args.samples < 3 or args.repeats < 1:
        parser.error("at least three samples and one repeat required")
    if rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("owning SM120 required")
    spec = importlib.util.spec_from_file_location("tessera.compiler._source_reverse_control", args.control)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.parent / (args.output.stem + "-programs")
    directory.mkdir(exist_ok=False)
    packet = dict(schema="tessera.public_source_attention_reverse_ab.v2", architecture="sm_120",
        device=subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"], text=True).strip(),
        control_sha256=hashlib.sha256(args.control.read_bytes()).hexdigest(),
        recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        provider_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
        samples=args.samples, repeats=args.repeats, order_offset=args.order_offset,
        scope="same mathematical inputs; independently retained compiler-selected native owners",
        timing="completed public host calls; CUDA event launch windows may include enqueue gaps",
        program_directory=directory.name, rows=[])
    for storage in ("f16", "bf16"):
        for profile in args.profiles:
            row = run(storage, profile, module.execute, args.samples, args.repeats, args.order_offset, directory)
            packet["rows"].append(row)
            args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + "\n")
            print(storage, profile, row["public_speedup"], row["medians"], flush=True)


if __name__ == "__main__":
    main()
