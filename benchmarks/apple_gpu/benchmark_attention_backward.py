"""Paired exact-device characterization for Apple attention backward.

Serial recompute, relaxed atomic accumulation, and deterministic two-way
split-reduction consume the same MHA/GQA/MQA semantic fixtures. Each report
retains native placement, oracle errors, workspace/determinism policy,
resources, and separate device and end-to-end timing domains.
"""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from tessera._apple_gpu_dispatch import (
    bind_registered,
    clear_dispatch_telemetry,
    read_dispatch_telemetry,
    read_profiling_capabilities,
    set_dispatch_telemetry_enabled,
)
from tessera.compiler.apple_attention_backward import (
    ROUTE_IDS, ROUTES, resolve_route, selector_shape_key)
from tessera.compiler.apple_route_selector import ROUTE_REPORT_SCHEMA_VERSION
from tessera.compiler.apple_route_selector import live_apple_route_context


def _shape(spec: str) -> tuple[int, int, int, int]:
    try:
        dims = tuple(int(part) for part in spec.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"shape must be BxSqxSkxD: {spec!r}") from exc
    if len(dims) != 4 or any(dim <= 0 for dim in dims):
        raise ValueError(
            f"shape must be BxSqxSkxD with positive dimensions: {spec!r}")
    return dims  # type: ignore[return-value]


def _reference(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, do: np.ndarray, *,
    q_heads: int = 1, kv_heads: int = 1, scale: float, causal: bool,
    bias: np.ndarray | None = None, window: int = 0, softcap: float = 0.0,
) -> tuple[np.ndarray, ...]:
    dq = np.zeros_like(q, dtype=np.float32)
    dk = np.zeros_like(k, dtype=np.float32)
    dv = np.zeros_like(v, dtype=np.float32)
    sq, sk = q.shape[1], k.shape[1]
    group = q_heads // kv_heads
    for qb in range(q.shape[0]):
        outer, q_head = divmod(qb, q_heads)
        kvb = outer * kv_heads + q_head // group
        raw = (q[qb] @ k[kvb].T) * scale
        if bias is not None:
            raw = raw + bias[qb]
        keep = np.ones((sq, sk), dtype=bool)
        for qi in range(sq):
            qpos = qi + max(sk - sq, 0)
            if causal:
                keep[qi, qpos + 1:] = False
            if window:
                if causal:
                    keep[qi, :max(qpos - window + 1, 0)] = False
                else:
                    half = window // 2
                    keep[qi, :max(qpos - half, 0)] = False
                    keep[qi, min(qpos + half + 1, sk):] = False
        score = softcap * np.tanh(raw / softcap) if softcap > 0 else raw
        score = np.where(keep, score, -np.inf)
        probability = np.exp(score - score.max(axis=-1, keepdims=True))
        probability /= probability.sum(axis=-1, keepdims=True)
        dp = do[qb] @ v[kvb].T
        ds = probability * (
            dp - (probability * dp).sum(axis=-1, keepdims=True))
        if softcap > 0:
            tanh_raw = np.tanh(raw / softcap)
            ds *= 1.0 - tanh_raw * tanh_raw
        dq[qb] = scale * (ds @ k[kvb])
        dk[kvb] += scale * (ds.T @ q[qb])
        dv[kvb] += probability.T @ do[qb]
    return dq, dk, dv


def _case_key(case: dict[str, Any]) -> str:
    return selector_shape_key(
        outer=case["outer"], q_heads=case["q_heads"],
        kv_heads=case["kv_heads"], sq=case["sq"], sk=case["sk"],
        dim=case["dim"], causal=case["causal"], window=case["window"],
        bias=case["bias"], softcap=case["softcap"])


def _closure_cases() -> list[dict[str, Any]]:
    return [
        {"outer": 1, "q_heads": 4, "kv_heads": 4, "sq": 16, "sk": 16,
         "dim": 16, "causal": False, "window": 0, "softcap": 0.0,
         "bias": False, "dtype": "f32"},
        {"outer": 1, "q_heads": 4, "kv_heads": 4, "sq": 17, "sk": 19,
         "dim": 64, "causal": True, "window": 0, "softcap": 0.0,
         "bias": False, "dtype": "f32"},
        {"outer": 1, "q_heads": 8, "kv_heads": 2, "sq": 9, "sk": 33,
         "dim": 64, "causal": True, "window": 17, "softcap": 2.5,
         "bias": True, "dtype": "f32"},
        {"outer": 1, "q_heads": 8, "kv_heads": 1, "sq": 8, "sk": 65,
         "dim": 64, "causal": False, "window": 17, "softcap": 0.0,
         "bias": False, "dtype": "f16"},
        {"outer": 2, "q_heads": 4, "kv_heads": 2, "sq": 9, "sk": 33,
         "dim": 64, "causal": True, "window": 0, "softcap": 1.5,
         "bias": True, "dtype": "bf16"},
        {"outer": 1, "q_heads": 2, "kv_heads": 2, "sq": 4, "sk": 1025,
         "dim": 64, "causal": True, "window": 0, "softcap": 0.0,
         "bias": False, "dtype": "f32"},
    ]


def _stored(value: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    if dtype == "f32":
        stored = np.ascontiguousarray(value.astype(np.float32))
        return stored, stored
    if dtype == "f16":
        typed = np.ascontiguousarray(value.astype(np.float16))
    elif dtype == "bf16":
        import ml_dtypes
        typed = np.ascontiguousarray(value.astype(ml_dtypes.bfloat16))
    else:
        raise ValueError(f"unsupported attention-backward dtype {dtype!r}")
    return typed.view(np.uint16), typed.astype(np.float32)


def _native_call(fn: Any, *, dtype: str, arrays: tuple[np.ndarray, ...],
                 outputs: tuple[np.ndarray, ...], case: dict[str, Any],
                 route: str) -> int:
    q, k, v, do, bias = arrays
    dq, dk, dv = outputs
    input_type = ctypes.c_float if dtype == "f32" else ctypes.c_uint16
    input_pointer = ctypes.POINTER(input_type)
    output_pointer = ctypes.POINTER(ctypes.c_float)
    b = case["outer"] * case["q_heads"]
    return int(fn(
        q.ctypes.data_as(input_pointer), k.ctypes.data_as(input_pointer),
        v.ctypes.data_as(input_pointer), do.ctypes.data_as(input_pointer),
        bias.ctypes.data_as(input_pointer) if case["bias"] else None,
        dq.ctypes.data_as(output_pointer), dk.ctypes.data_as(output_pointer),
        dv.ctypes.data_as(output_pointer), b, case["q_heads"],
        case["kv_heads"], case["sq"], case["sk"], case["dim"],
        case["dim"] ** -.5, int(case["causal"]), case["window"],
        case["softcap"], ROUTE_IDS[route]))


def _characterize_case(*, case: dict[str, Any], routes: tuple[str, ...],
                       reps: int, trials: int, device: str,
                       rng: np.random.Generator) -> list[dict[str, Any]]:
    dtype = case["dtype"]
    b = case["outer"] * case["q_heads"]
    kv_outer = case["outer"] * case["kv_heads"]
    sq, sk, dim = case["sq"], case["sk"], case["dim"]
    q, qf = _stored((rng.normal(size=(b, sq, dim)) * .2).astype("f4"), dtype)
    k, kf = _stored(
        (rng.normal(size=(kv_outer, sk, dim)) * .2).astype("f4"), dtype)
    v, vf = _stored(
        (rng.normal(size=(kv_outer, sk, dim)) * .2).astype("f4"), dtype)
    do, dof = _stored((rng.normal(size=(b, sq, dim)) * .2).astype("f4"), dtype)
    bias_values = (rng.normal(size=(b, sq, sk)) * .05).astype("f4")
    bias, biasf = _stored(bias_values, dtype)
    expected = _reference(
        qf, kf, vf, dof, q_heads=case["q_heads"],
        kv_heads=case["kv_heads"], scale=dim ** -.5,
        causal=case["causal"], bias=biasf if case["bias"] else None,
        window=case["window"], softcap=case["softcap"])
    fn = bind_registered(
        f"tessera_apple_gpu_flash_attn_bwd_variant_{dtype}_status")
    if fn is None:
        raise RuntimeError(f"attention backward {dtype} ABI unavailable")
    state = {route: {"e2e": [], "device": [], "output": None, "record": {}}
             for route in routes}
    for trial in range(trials + 1):
        ordered = routes[trial % len(routes):] + routes[:trial % len(routes)]
        for route in ordered:
            e2e_samples: list[int] = []
            device_samples: list[int] = []
            for _ in range(reps):
                outputs = (np.empty_like(qf), np.empty_like(kf),
                           np.empty_like(vf))
                clear_dispatch_telemetry()
                started = time.perf_counter_ns()
                status = _native_call(
                    fn, dtype=dtype, arrays=(q, k, v, do, bias),
                    outputs=outputs, case=case, route=route)
                e2e_samples.append(time.perf_counter_ns() - started)
                if status != 1:
                    raise RuntimeError(
                        f"native Apple attention backward failed: {dtype}/{route}")
                record = read_dispatch_telemetry()
                measured = record.get("device_time_ns")
                if isinstance(measured, int) and measured > 0:
                    device_samples.append(measured)
                state[route]["output"] = outputs
                state[route]["record"] = record
            if trial:
                state[route]["e2e"].append(
                    int(statistics.median(e2e_samples)))
                state[route]["device"].append(
                    int(statistics.median(device_samples))
                    if len(device_samples) == reps else None)
    rows: list[dict[str, Any]] = []
    for route in routes:
        route_state = state[route]
        output = route_state["output"]
        assert output is not None
        errors = [float(np.max(np.abs(got - ref)))
                  for got, ref in zip(output, expected)]
        device_trials = route_state["device"]
        complete = all(value is not None for value in device_trials)
        policy = resolve_route(kf.shape, vf.shape, route=route)
        input_bytes = q.nbytes + k.nbytes + v.nbytes + do.nbytes
        if case["bias"]:
            input_bytes += bias.nbytes
        output_bytes = sum(value.nbytes for value in output)
        rows.append({
            "backend": "apple_gpu", "op": "flash_attn_bwd",
            "shape": _case_key(case), "dtype": dtype, "device": device,
            "route": route, "native_dispatched": True,
            "numerically_validated": max(errors) <= 1e-4,
            "max_abs_error": {"dQ": errors[0], "dK": errors[1],
                              "dV": errors[2]},
            "deterministic": policy.deterministic,
            "workspace_bytes": policy.workspace_bytes,
            "logical_input_bytes": input_bytes,
            "logical_output_bytes": output_bytes,
            "residency": "host_input_host_output_single_command_buffer",
            "selector_eligible": False,
            "latency_ms": statistics.median(route_state["e2e"]) / 1e6,
            "reps": reps * trials, "trials": trials,
            "semantics": {key: case[key] for key in (
                "outer", "q_heads", "kv_heads", "sq", "sk", "dim",
                "causal", "window", "softcap", "bias")},
            "telemetry": {
                "end_to_end_median_ns": int(statistics.median(route_state["e2e"])),
                "device_time_median_ns": (
                    int(statistics.median(value for value in device_trials
                                          if value is not None))
                    if complete else None),
                "device_time_coverage": 1.0 if complete else 0.0,
                "timing_source": route_state["record"].get("timing_source"),
                "resources": route_state["record"].get("resources"),
                "paired_trial_end_to_end_medians_ns": route_state["e2e"],
                "paired_trial_device_medians_ns": device_trials,
            },
        })
    return rows


def characterize(*, shapes: list[tuple[int, int, int, int]] | None = None,
                 reps: int, trials: int, routes: tuple[str, ...] = ROUTES,
                 closure_matrix: bool = False) -> dict[str, Any]:
    from tessera import runtime as rt
    from tessera.compiler.apple_target import probe_apple_runtime_limits

    if not rt.DeviceTensor.is_metal():
        return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION, "runs": [],
                "skipped_apple_gpu": "Apple Metal device unavailable"}
    limits = probe_apple_runtime_limits()
    family = limits.apple_gpu_family if limits is not None else -1
    device = (f"apple{family - 1000}" if 1001 <= family <= 1099
              else "apple_silicon_metal_unknown_family")
    cases = _closure_cases() if closure_matrix else []
    for b, sq, sk, dim in shapes or []:
        cases.append({
            "outer": b, "q_heads": 1, "kv_heads": 1, "sq": sq, "sk": sk,
            "dim": dim, "causal": False, "window": 0, "softcap": 0.0,
            "bias": False, "dtype": "f32"})
        cases.append({**cases[-1], "causal": True})
    rng = np.random.default_rng(1909)
    rows: list[dict[str, Any]] = []
    set_dispatch_telemetry_enabled(True)
    try:
        for case in cases:
            rows.extend(_characterize_case(
                case=case, routes=routes, reps=reps, trials=trials,
                device=device, rng=rng))
    finally:
        set_dispatch_telemetry_enabled(False)
    return {"schema_version": ROUTE_REPORT_SCHEMA_VERSION,
            "context": live_apple_route_context().as_mapping(), "device": device,
            "runs": rows, "profiling_capabilities": read_profiling_capabilities(),
            "paired_route_order_rotated": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="*", default=[])
    parser.add_argument("--closure-matrix", action="store_true")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--routes", nargs="+", choices=ROUTES,
                        default=list(ROUTES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.closure_matrix and not args.shapes:
        parser.error("provide --closure-matrix or at least one --shapes value")
    if args.reps < 2 or args.trials < 3:
        parser.error("reps must be >=2 and trials must be >=3")
    payload = characterize(
        shapes=[_shape(spec) for spec in args.shapes], reps=args.reps,
        trials=args.trials, routes=tuple(args.routes),
        closure_matrix=args.closure_matrix)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
