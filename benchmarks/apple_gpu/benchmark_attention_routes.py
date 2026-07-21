"""Paired Apple forward-attention route characterization.

The selector-eligible corpus compares the production online-softmax MSL route
with the owned-command-buffer MPSGraph BSMM route for equal-dtype MHA.  A
separate device-resident domain measures the variant-capable scalar-resident
and cooperative-SIMD-group candidates, including native f16/bf16 storage.  All
routes share the same oracle; unavailable timing is retained as unavailable
rather than represented by synthetic values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np

from tessera._apple_gpu_dispatch import (
    clear_dispatch_telemetry,
    read_dispatch_telemetry,
    read_profiling_capabilities,
    set_dispatch_telemetry_enabled,
)
from tessera.compiler.apple_route_selector import ROUTE_REPORT_SCHEMA_VERSION
from tessera.compiler.apple_route_selector import live_apple_route_context


MSL_ROUTE = "online_msl_variant"
MPSGRAPH_ROUTE = "mpsgraph_bsmm"
RESIDENT_ROUTE = "resident_command_buffer"
COOPERATIVE_ROUTE = "cooperative_simdgroup"


def _reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, q_heads: int,
               kv_heads: int, scale: float, causal: bool,
               bias: np.ndarray | None = None, window: int = 0,
               softcap: float = 0.0) -> np.ndarray:
    outer, _, sq, dim = q.shape
    sk = k.shape[-2]
    out = np.empty(q.shape, dtype=np.float32)
    group = q_heads // kv_heads
    offset = max(sk - sq, 0)
    for b in range(outer):
        for h in range(q_heads):
            kh = h // group
            score = q[b, h].astype(np.float32) @ k[b, kh].astype(np.float32).T
            score *= scale
            if bias is not None:
                score += np.broadcast_to(bias, (outer, q_heads, sq, sk))[b, h]
            if softcap:
                score = softcap * np.tanh(score / softcap)
            qpos = np.arange(sq)[:, None] + offset
            kpos = np.arange(sk)[None, :]
            mask = np.zeros((sq, sk), dtype=bool)
            if causal:
                mask |= kpos > qpos
            if window:
                if causal:
                    mask |= kpos <= qpos - window
                else:
                    half = window // 2
                    mask |= (kpos < qpos - half) | (kpos > qpos + half)
            score = np.where(mask, -np.inf, score)
            weight = np.exp(score - score.max(axis=-1, keepdims=True))
            weight /= weight.sum(axis=-1, keepdims=True)
            out[b, h] = weight @ v[b, kh].astype(np.float32)
    return out


def _dispatch(call: Callable[[], Any]) -> tuple[np.ndarray | None, dict[str, Any], int]:
    clear_dispatch_telemetry()
    started = time.perf_counter_ns()
    value = call()
    elapsed = time.perf_counter_ns() - started
    return (None if value is None else np.asarray(value)), read_dispatch_telemetry(), elapsed


def _trial(call: Callable[[], Any], reps: int) -> tuple[np.ndarray | None, int,
                                                        int | None, dict[str, Any]]:
    e2e: list[int] = []
    device: list[int] = []
    output: np.ndarray | None = None
    record: dict[str, Any] = {}
    for _ in range(reps):
        output, record, elapsed = _dispatch(call)
        e2e.append(elapsed)
        measured = record.get("device_time_ns")
        if isinstance(measured, int) and measured > 0:
            device.append(measured)
    return (output, int(statistics.median(e2e)),
            int(statistics.median(device)) if len(device) == reps else None,
            record)


def _paired_rows(*, op: str, shape: str, dtype: str, device: str,
                 reference: np.ndarray, incumbent: Callable[[], Any],
                 candidate: Callable[[], Any], logical_io_bytes: int,
                 reps: int, trials: int
                 ) -> list[dict[str, Any]]:
    incumbent()
    candidate()
    values: dict[str, list[tuple[np.ndarray | None, int, int | None,
                                 dict[str, Any]]]] = {
        MSL_ROUTE: [], MPSGRAPH_ROUTE: []}
    for trial in range(trials):
        order = ((MSL_ROUTE, incumbent), (MPSGRAPH_ROUTE, candidate))
        if trial % 2:
            order = tuple(reversed(order))
        for route, call in order:
            values[route].append(_trial(call, reps))
    rows: list[dict[str, Any]] = []
    for route in (MSL_ROUTE, MPSGRAPH_ROUTE):
        route_values = values[route]
        output, _, _, last_record = route_values[-1]
        e2e = [item[1] for item in route_values]
        device_times = [item[2] for item in route_values]
        valid = output is not None and np.allclose(
            output.astype(np.float32), reference, rtol=4e-2 if dtype == "f16" else 3e-3,
            atol=4e-2 if dtype == "f16" else 3e-3)
        rows.append({
            "backend": "apple_gpu", "op": op, "shape": shape,
            "dtype": dtype, "device": device, "route": route,
            "latency_ms": statistics.median(e2e) / 1e6,
            "stdev_ms": statistics.stdev(e2e) / 1e6,
            "reps": reps * trials, "trials": trials,
            "native_dispatched": output is not None,
            "numerically_validated": bool(valid),
            "memory": {
                "logical_input_output_bytes": logical_io_bytes,
                "input_residency": "host_input_host_output",
                "intermediate_materialization": (
                    "none_for_online_kernel" if route == MSL_ROUTE
                    else "framework_opaque"),
            },
            "telemetry": {
                "end_to_end_median_ns": int(statistics.median(e2e)),
                "device_time_median_ns": (
                    int(statistics.median(device_times))
                    if all(value is not None for value in device_times) else None),
                "device_time_samples": reps * trials if all(
                    value is not None for value in device_times) else 0,
                "device_time_coverage": 1.0 if all(
                    value is not None for value in device_times) else 0.0,
                "timing_source": last_record.get("timing_source"),
                "paired_trial_end_to_end_medians_ns": e2e,
                "paired_trial_device_medians_ns": device_times,
                "resources": {
                    "route": route,
                    "pipeline": last_record.get("resources"),
                },
            },
        })
    return rows


def _resident_candidate_row(*, route: str, q: np.ndarray, k: np.ndarray,
                            v: np.ndarray, bias: np.ndarray | None,
                            reference: np.ndarray, dtype: str, shape: str,
                            q_heads: int, kv_heads: int, causal: bool,
                            window: int, softcap: float, reps: int,
                            trials: int) -> dict[str, Any]:
    from tessera import apple_gpu_batched as agpu

    outer, _, sq, dim = q.shape
    sk = k.shape[-2]
    storage = q.dtype
    q_flat = np.ascontiguousarray(q.reshape(outer * q_heads, sq, dim))
    k_flat = np.ascontiguousarray(k.reshape(outer * kv_heads, sk, dim))
    v_flat = np.ascontiguousarray(v.reshape(outer * kv_heads, sk, dim))
    bias_flat = (None if bias is None else np.ascontiguousarray(
        np.broadcast_to(bias, (outer, q_heads, sq, sk)).reshape(
            outer * q_heads, sq, sk)))
    q_dev, k_dev, v_dev = (agpu.device_tensor(q_flat),
                           agpu.device_tensor(k_flat),
                           agpu.device_tensor(v_flat))
    bias_dev = agpu.device_tensor(bias_flat) if bias_flat is not None else None

    def call() -> np.ndarray:
        with agpu.batched_session() as session:
            fn = (agpu.flash_attn_variant_enc if route == RESIDENT_ROUTE
                  else agpu.flash_attn_cooperative_enc)
            out_dev = fn(
                session, q_dev, k_dev, v_dev, bias_dev, dtype=dtype,
                B=outer * q_heads, q_heads=q_heads, kv_heads=kv_heads,
                Sq=sq, Sk=sk, D=dim, scale=dim ** -0.5, causal=causal,
                window_size=window, logit_softcap=softcap)
        try:
            return out_dev.download(storage, q_flat.shape).reshape(q.shape)
        finally:
            out_dev.free()

    try:
        call()
        samples = [_trial(call, reps) for _ in range(trials)]
    finally:
        if bias_dev is not None:
            bias_dev.free()
        v_dev.free()
        k_dev.free()
        q_dev.free()
    output, _, _, record = samples[-1]
    e2e = [item[1] for item in samples]
    device_times = [item[2] for item in samples]
    tolerance = 6e-2 if dtype == "bf16" else (4e-2 if dtype == "f16" else 3e-3)
    return {
        "shape": shape, "dtype": dtype, "route": route,
        "input_residency_domain": "device_input_host_output",
        "native_dispatched": output is not None,
        "numerically_validated": bool(
            output is not None and np.allclose(
                output.astype(np.float32), reference, rtol=tolerance,
                atol=tolerance)),
        "memory": {
            "logical_input_output_bytes": int(
                q.nbytes + k.nbytes + v.nbytes + q.nbytes
                + (0 if bias is None else bias.nbytes)),
            "input_residency": "device_input_host_output",
            "host_staging_inside_attention_abi": False,
            "intermediate_storage": (
                "f32_gpu_cast_buffers" if dtype in {"f16", "bf16"}
                else "none"),
        },
        "end_to_end_median_ns": int(statistics.median(e2e)),
        "device_time_median_ns": (
            int(statistics.median(device_times))
            if all(value is not None for value in device_times) else None),
        "device_time_samples": reps * trials if all(
            value is not None for value in device_times) else 0,
        "device_time_coverage": 1.0 if all(
            value is not None for value in device_times) else 0.0,
        "timing_source": record.get("timing_source"),
        "resources": record.get("resources"),
    }


def characterize(*, reps: int, trials: int) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.apple_target import probe_apple_runtime_limits

    if not rt.DeviceTensor.is_metal():
        return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION, "runs": [],
                "skipped_apple_gpu": "Apple Metal device unavailable"}
    limits = probe_apple_runtime_limits()
    family = limits.apple_gpu_family if limits is not None else -1
    device = (f"apple{family - 1000}" if 1001 <= family <= 1099
              else "apple_silicon_metal_unknown_family")
    rng = np.random.default_rng(2501)
    runs: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    resident_rows: list[dict[str, Any]] = []
    set_dispatch_telemetry_enabled(True)
    try:
        for dtype, storage in (("f32", np.float32), ("f16", np.float16)):
            for outer, heads, sq, sk, dim in (
                    (1, 4, 64, 64, 64),
                    (1, 4, 65, 67, 128),
                    (2, 8, 64, 257, 128),
                    (1, 16, 16, 1025, 256)):
                q = np.ascontiguousarray((rng.normal(size=(outer, heads, sq, dim)) * .2).astype(storage))
                k = np.ascontiguousarray((rng.normal(size=(outer, heads, sk, dim)) * .2).astype(storage))
                v = np.ascontiguousarray((rng.normal(size=(outer, heads, sk, dim)) * .2).astype(storage))
                scale = dim ** -0.5
                ref = _reference(q, k, v, q_heads=heads, kv_heads=heads,
                                 scale=scale, causal=False)
                shape = f"b{outer}_h{heads}_sq{sq}_sk{sk}_d{dim}"
                runs.extend(_paired_rows(
                    op="flash_attn_mha", shape=shape, dtype=dtype, device=device,
                    reference=ref,
                    incumbent=lambda q=q, k=k, v=v, heads=heads, scale=scale:
                        rt._apple_gpu_dispatch_gqa(q, k, v, heads, heads, np,
                                                   scale=scale,
                                                   route_override=MSL_ROUTE),
                    candidate=lambda q=q, k=k, v=v, scale=scale:
                        rt._apple_gpu_dispatch_batched_attention(q, k, v, np,
                                                                 scale=scale),
                    logical_io_bytes=int(q.nbytes + k.nbytes + v.nbytes
                                         + q.nbytes),
                    reps=reps, trials=trials))

        try:
            import ml_dtypes
            variant_dtypes = (("f32", np.float32), ("f16", np.float16),
                              ("bf16", ml_dtypes.bfloat16))
        except ImportError:
            variant_dtypes = (("f32", np.float32), ("f16", np.float16))
        variant_specs = (
            ("gqa_bias_window_softcap", 1, 8, 2, 17, 129, 128, True, 33, 3.0),
            ("mha_ragged", 2, 8, 8, 65, 67, 64, False, 0, 0.0),
            ("mqa_long_context", 2, 8, 1, 4, 2049, 128, True, 256, 0.0),
        )
        for dtype, storage in variant_dtypes:
            for (name, outer, hq, hkv, sq, sk, dim, causal, window,
                 softcap) in variant_specs:
                q = np.ascontiguousarray(
                    (rng.normal(size=(outer, hq, sq, dim)) * .2).astype(storage))
                k = np.ascontiguousarray(
                    (rng.normal(size=(outer, hkv, sk, dim)) * .2).astype(storage))
                v = np.ascontiguousarray(
                    (rng.normal(size=(outer, hkv, sk, dim)) * .2).astype(storage))
                bias = (np.ascontiguousarray(
                    (rng.normal(size=(outer, hq, sq, sk)) * .1).astype(storage))
                        if "bias" in name else None)
                output, record, elapsed = _dispatch(
                    lambda q=q, k=k, v=v, hq=hq, hkv=hkv, causal=causal,
                           bias=bias, window=window, softcap=softcap:
                    rt._apple_gpu_dispatch_gqa(
                        q, k, v, hq, hkv, np, causal=causal, attn_bias=bias,
                        window_size=window, logit_softcap=softcap))
                ref = _reference(q, k, v, q_heads=hq, kv_heads=hkv,
                                 scale=dim ** -0.5, causal=causal, bias=bias,
                                 window=window, softcap=softcap)
                shape = (f"b{outer}_hq{hq}_hkv{hkv}_sq{sq}_sk{sk}_d{dim}")
                tolerance = (6e-2 if dtype == "bf16" else
                             (4e-2 if dtype == "f16" else 3e-3))
                coverage.append({
                    "variant": name, "shape": shape, "dtype": dtype,
                    "route": MSL_ROUTE, "native_dispatched": output is not None,
                    "numerically_validated": bool(
                        output is not None and np.allclose(
                            output.astype(np.float32), ref, rtol=tolerance,
                            atol=tolerance)),
                    "end_to_end_ns": elapsed,
                    "device_time_ns": record.get("device_time_ns"),
                    "resources": record.get("resources"),
                })
                for route in (RESIDENT_ROUTE, COOPERATIVE_ROUTE):
                    resident_rows.append(_resident_candidate_row(
                        route=route, q=q, k=k, v=v, bias=bias,
                        reference=ref, dtype=dtype, shape=shape,
                        q_heads=hq, kv_heads=hkv, causal=causal,
                        window=window, softcap=softcap, reps=reps,
                        trials=trials))
    finally:
        set_dispatch_telemetry_enabled(False)
    return {
        "schema_version": ROUTE_REPORT_SCHEMA_VERSION,
        "context": live_apple_route_context().as_mapping(),
        "device": device,
        "paired_trials": trials,
        "runs": runs,
        "variant_coverage": coverage,
        "resident_comparison": resident_rows,
        "candidate_capabilities": {
            MSL_ROUTE: "bias/window/softcap + MHA/GQA/MQA",
            MPSGRAPH_ROUTE: "plain equal-dtype MHA only",
            "resident_command_buffer": (
                "bias/window/softcap + MHA/GQA/MQA; f32 accumulation with "
                "native f32/f16/bf16 device storage"),
            COOPERATIVE_ROUTE: (
                "one SIMD group per query row; bias/window/softcap + "
                "MHA/GQA/MQA; D<=256; native f32/f16/bf16 device storage"),
            "cooperative_matrix": (
                "unavailable: no attention-specific cooperative-matrix ABI; "
                "the measured cooperative candidate is SIMD-group MSL"),
        },
        "profiling_capabilities": read_profiling_capabilities(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.reps < 2 or args.trials < 2:
        parser.error("reps and trials must both be at least 2")
    report = characterize(reps=args.reps, trials=args.trials)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
