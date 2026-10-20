"""Paired static/bounded public source-VJP calls; event and host scopes stay separate."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests.device.nvidia.test_source_attention_reverse import oracle


def source(q, k, v):
    normalized = ts.ops.rmsnorm(q, eps=1e-5, numeric_policy={"accum": "fp32"})
    return ts.ops.flash_attn(normalized, k, v, causal=False, lse_checkpoint="saved")


def measure(storage, profile, samples, reps, reverse):
    sq_cap, sk_cap, d, dv = profile
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    calls = {label: ts.jit(target="nvidia_sm120", autodiff="reverse")(source) for label in ("static", "bounded")}
    rows, preparations = [], []
    frames = [(1, 1), (sq_cap, sk_cap), (3, 2)]
    # Always trace at one for bounded, independently of measurement order.
    if reverse:
        frames = [frames[0], *reversed(frames[1:])]
    try:
        for frame, (sq, sk) in enumerate(frames):
            rng = np.random.default_rng(20261020 + sq * 37 + sk)
            shapes = ((1, 2, sq, d), (1, 1, sk, d), (1, 1, sk, dv), (1, 2, sq, dv), (1, 2, sq))
            values = tuple(rng.normal(0, .25, s).astype(np.float32 if i == 4 else dtype) for i, s in enumerate(shapes))
            expected = oracle(*values)
            def invoke(label):
                return calls[label].native_backward(*values[:3], out_cotangents=values[3:],
                    **({"sequence_bounds": (sq_cap, sk_cap)} if label == "bounded" else {}))
            def check(label):
                result = invoke(label)
                errors = []
                for got, ref in zip(result, expected, strict=True):
                    np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32), rtol=.03, atol=.002)
                    errors.append(float(np.max(np.abs(got.astype(np.float32) - ref.astype(np.float32)))))
                return errors
            records = {}
            for label in (("bounded", "static") if reverse else ("static", "bounded")):
                started = time.perf_counter()
                errors = check(label)
                first_ms = (time.perf_counter() - started) * 1000
                preparations.append({"storage": storage, "profile": profile, "frame": [sq, sk],
                    "arm": label, "first_frame_completed_host_ms": first_ms,
                    "includes_compilation_prepare": label == "static" or frame == 0})
                receipt = calls[label].last_backward_execution
                records[label] = {"completed_public_host_samples_ms": [], "program_event_samples_ms": [],
                    "stage_event_samples_ms": [], "image_digests": receipt["image_digests"],
                    "max_abs_error": errors, "stages": len(receipt["stage_device_ms"])}
            for trial in range(samples):
                order = ("static", "bounded") if (trial % 2 == 0) != reverse else ("bounded", "static")
                for label in order:
                    started = time.perf_counter()
                    for _ in range(reps):
                        invoke(label)
                    records[label]["completed_public_host_samples_ms"].append((time.perf_counter() - started) * 1000 / reps)
                    # Last completed call per sample; not an average over repetitions.
                    receipt = calls[label].last_backward_execution
                    records[label]["program_event_samples_ms"].append(receipt["program_device_ms"])
                    records[label]["stage_event_samples_ms"].append(receipt["stage_device_ms"])
            for label, record in records.items():
                check(label)
                record["completed_public_host_median_ms"] = median(record["completed_public_host_samples_ms"])
                record["program_event_median_ms"] = median(record["program_event_samples_ms"])
                record["retained_owners"] = len(calls[label]._native_prepared_vjp_calls)
            rows.append({"storage": storage, "capacity": profile, "runtime_sq_sk": [sq, sk],
                "input_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in values],
                "correctness": "independent_source_vjp_oracle_before_and_after_timing", "arms": records})
            print(storage, profile, sq, sk, {k: v["completed_public_host_median_ms"] for k, v in records.items()}, flush=True)
    finally:
        for fn in calls.values():
            for owner in fn._native_prepared_vjp_calls.values():
                owner.close()
    return rows, preparations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=21)
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--reverse", action="store_true")
    args = p.parse_args()
    if args.samples < 3 or args.reps < 1:
        p.error("requires at least three samples and positive repetitions")
    device = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,uuid", "--format=csv,noheader"], text=True).strip()
    if len(device.splitlines()) != 1 or device.split(",")[1].strip() != "12.0" or rt._nvidia_device_name() != "sm_120":
        raise RuntimeError("requires owning SM120 GPU")
    compiler = find_tessera_opt()
    if compiler is None:
        raise RuntimeError("requires matching native compiler")
    profiles = [(s, cap) for s in ("f16", "bf16") for cap in ((7, 11, 4, 3), (17, 19, 257, 13))]
    if args.reverse:
        profiles.reverse()
    packet = {"schema": "tessera.bounded_public_source_reverse_measurement.v1", "device": device,
        "samples": args.samples, "reps": args.reps, "reverse": args.reverse,
        "scope": "completed public native_backward calls and separately last-call native event windows; not isolated kernel time",
        "compiler_sha256": hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
        "rows": [], "first_frame_preparations": []}
    for storage, profile in profiles:
        rows, preparations = measure(storage, profile, args.samples, args.reps, args.reverse)
        packet["rows"].extend(rows)
        packet["first_frame_preparations"].extend(preparations)
    packet["provider_sha256"] = hashlib.sha256(Path(rt._load_nvidia_ptx_launch()._name).read_bytes()).hexdigest()
    packet["source_sha256"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in (
        "python/tessera/compiler/source_attention_reverse_frontend.py", "python/tessera/compiler/jit.py",
        "python/tessera/compiler/prepared_source_attention_reverse.py", "python/tessera/compiler/portable_source_attention_reverse.py",
        "benchmarks/nvidia/benchmark_bounded_public_source_reverse.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2) + "\n")


if __name__ == "__main__":
    main()
