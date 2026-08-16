#!/usr/bin/env python3
"""Measure the Apple scheduled Graph->Schedule->Tile consumers (E2E-REAL-3/5/5A/5B).

What this measures, and what it deliberately does not
----------------------------------------------------
Each row times one **complete** `runtime.launch` of a package produced by the
shared scheduled boundary, on the exact Apple device, after warmup, reporting a
repeated median. Every row also re-checks its numerical oracle so a fast wrong
answer cannot be recorded as a win.

This is a **characterization**, not a selector decision. Per the Apple plan a
production route change requires two independently produced reports, native
placement, retained resource/counter evidence, and a stable paired win in the
intended timing domain. One process's medians are none of those things, so
`selector_eligible` is `false` in the emitted JSON and no committed ledger is
written.

Timing domain is **end-to-end host wall** around `launch`. The MPS matmul route
exposes no command-buffer device timer (APPLE-DEVICE-EVENT-1), so mixing
domains across these families would compare unlike things; the honest common
denominator is the complete call.

Usage:
    python3 benchmarks/apple_gpu/benchmark_scheduled_boundary.py \
        --reps 30 --warmup 5 --output /tmp/apple-scheduled-boundary.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from tessera import runtime as rt  # noqa: E402
from tessera.compiler import apple_native  # noqa: E402
from tessera.compiler.driver import compile_graph_module  # noqa: E402
from tessera.compiler.graph_ir import (  # noqa: E402
    GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
)
from tessera.compiler.scheduled_attention_backward import (  # noqa: E402
    lower_scheduled_attention_backward,
)
from tessera.compiler.scheduled_matmul import find_tessera_opt  # noqa: E402

_SPELLING = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}


def _tensor(shape: tuple[int, ...], dtype: str = "fp32") -> IRType:
    text = "tensor<" + "x".join(map(str, shape)) + "x" + _SPELLING[dtype] + ">"
    return IRType(text, tuple(map(str, shape)), dtype)


def _matmul_module(m: int, k: int, n: int, dtype: str) -> GraphIRModule:
    out_dtype = "fp32"
    a, b = _tensor((m, k), dtype), _tensor((k, n), dtype)
    o = _tensor((m, n), out_dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="bench_matmul", args=[IRArg("a", a), IRArg("b", b)],
        result_types=[o],
        body=[IROp(result="o", op_name="tessera.matmul", operands=["%a", "%b"],
                   operand_types=[str(a), str(b)], result_type=str(o), kwargs={})],
        return_values=["%o"])])


def _softmax_module(shape: tuple[int, ...]) -> GraphIRModule:
    x = _tensor(shape)
    return GraphIRModule(functions=[GraphIRFunction(
        name="bench_softmax", args=[IRArg("x", x)], result_types=[x],
        body=[IROp(result="o", op_name="tessera.softmax", operands=["%x"],
                   operand_types=[str(x)], result_type=str(x),
                   kwargs={"axis": -1})],
        return_values=["%o"])])


def _attention_module(b, hq, hkv, sq, sk, d, causal) -> GraphIRModule:
    q, k = _tensor((b, hq, sq, d)), _tensor((b, hkv, sk, d))
    v, o = _tensor((b, hkv, sk, d)), _tensor((b, hq, sq, d))
    return GraphIRModule(functions=[GraphIRFunction(
        name="bench_attention",
        args=[IRArg("q", q), IRArg("k", k), IRArg("v", v)], result_types=[o],
        body=[IROp(result="o", op_name="tessera.flash_attn",
                   operands=["%q", "%k", "%v"],
                   operand_types=[str(q), str(k), str(v)], result_type=str(o),
                   kwargs={"scale": 0.25, "causal": causal, "window_left": -1,
                           "window_right": -1, "softcap": 0.0,
                           "dropout_p": 0.0, "dropout_seed": 0})],
        return_values=["%o"])])


def _attention_backward_module(b, hq, hkv, sq, sk, d, causal) -> GraphIRModule:
    q, k = _tensor((b, hq, sq, d)), _tensor((b, hkv, sk, d))
    v, do = _tensor((b, hkv, sk, d)), _tensor((b, hq, sq, d))
    return GraphIRModule(functions=[GraphIRFunction(
        name="bench_attention_backward",
        args=[IRArg("do", do), IRArg("q", q), IRArg("k", k), IRArg("v", v)],
        result_types=[q, k, v],
        body=[IROp(result="dq,dk,dv", op_name="tessera.flash_attn_bwd",
                   operands=["%do", "%q", "%k", "%v"],
                   operand_types=[str(do), str(q), str(k), str(v)],
                   result_type=f"({q}, {k}, {v})",
                   kwargs={"scale": 0.25, "causal": causal, "window_left": -1,
                           "window_right": -1, "softcap": 0.0, "dropout_p": 0.0,
                           "dropout_seed": 0, "query_block": 16,
                           "key_block": 16, "split_count": 2})],
        return_values=["%dq", "%dk", "%dv"])])


def _artifact_from_driver(module: GraphIRModule) -> tuple[Any, dict[str, Any]]:
    bundle = compile_graph_module(
        module, source_origin="apple-scheduled-boundary-benchmark",
        target="apple_gpu", options={"package_native": True},
        enable_tool_validation=False,
    )
    if bundle.native_image is None or bundle.launch_descriptor is None:
        raise RuntimeError("driver produced no native Apple package")
    artifact = rt.RuntimeArtifact(
        metadata={"target": "apple_gpu"}, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text if bundle.tile else "",
        target_ir=bundle.target_ir.text if bundle.target_ir else "",
    )
    return artifact, dict(bundle.launch_descriptor.provenance)


def _time(fn, *, reps: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "p90_ms": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
    }


def _row(name: str, provenance: dict, timing: dict, error: float,
         *, native: bool, shape: list[int]) -> dict:
    return {
        "case": name,
        "route": provenance.get("route"),
        "work_item": provenance.get("work_item"),
        "storage": provenance.get("storage"),
        "shape": shape,
        "native_gpu": native,
        "max_abs_error": error,
        "timing_domain": "end_to_end_host_wall",
        "device_time_promotion": provenance.get("device_time_promotion", "n/a"),
        **timing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if platform.system() != "Darwin":
        print("Apple scheduled-boundary benchmark requires a Darwin host", file=sys.stderr)
        return 2
    if find_tessera_opt() is None:
        print("requires a built tessera-opt (the shared artifact is produced by it)",
              file=sys.stderr)
        return 2
    if not apple_native.tools_available():
        print("requires a fresh Apple GPU runtime dylib", file=sys.stderr)
        return 2

    rng = np.random.default_rng(20260816)
    rows: list[dict] = []

    # --- matmul: f32 (delegated MPS BMM) vs f16 (compiler-emitted simdgroup) ---
    for dtype, tol in (("fp32", 3e-5), ("fp16", 2e-2)):
        for (m, k, n) in ((64, 64, 64), (256, 256, 256)):
            artifact, provenance = _artifact_from_driver(_matmul_module(m, k, n, dtype))
            np_dtype = np.float32 if dtype == "fp32" else np.float16
            a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np_dtype)
            b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np_dtype)
            out = np.zeros((m, n), dtype=np.float32)
            result = rt.launch(artifact, {"a": a, "b": b, "o": out})
            native = result.get("execution_kind") == "native_gpu"
            error = float(np.abs(out - a.astype(np.float32) @ b.astype(np.float32)).max())
            if error > tol:
                raise RuntimeError(f"matmul {dtype} {m}x{k}x{n} failed its oracle: {error}")
            timing = _time(lambda: rt.launch(artifact, {"a": a, "b": b, "o": out}),
                           reps=args.reps, warmup=args.warmup)
            rows.append(_row(f"matmul.{dtype}", provenance, timing, error,
                             native=native, shape=[m, n, k]))

    # --- softmax (native MSL), including a rank-3 logical launch ---
    for shape in ((64, 256), (2, 32, 128)):
        artifact, provenance = _artifact_from_driver(_softmax_module(shape))
        x = np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
        out = np.zeros_like(x)
        result = rt.launch(artifact, {"x": x, "o": out})
        native = result.get("execution_kind") == "native_gpu"
        shifted = x - x.max(axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        error = float(np.abs(out - expected).max())
        if error > 3e-5:
            raise RuntimeError(f"softmax {shape} failed its oracle: {error}")
        timing = _time(lambda: rt.launch(artifact, {"x": x, "o": out}),
                       reps=args.reps, warmup=args.warmup)
        rows.append(_row(f"softmax.rank{len(shape)}", provenance, timing, error,
                         native=native, shape=list(shape)))

    # --- attention forward (status-returning GQA MSL) ---
    for (b_, hq, hkv, sq, sk, d, causal) in ((1, 4, 2, 64, 64, 64, False),
                                             (1, 4, 2, 128, 128, 64, True)):
        module = _attention_module(b_, hq, hkv, sq, sk, d, causal)
        artifact, provenance = _artifact_from_driver(module)
        q = np.ascontiguousarray(rng.standard_normal((b_, hq, sq, d)), dtype=np.float32)
        k = np.ascontiguousarray(rng.standard_normal((b_, hkv, sk, d)), dtype=np.float32)
        v = np.ascontiguousarray(rng.standard_normal((b_, hkv, sk, d)), dtype=np.float32)
        out = np.zeros((b_, hq, sq, d), dtype=np.float32)
        result = rt.launch(artifact, {"q": q, "k": k, "v": v, "o": out})
        native = result.get("execution_kind") == "native_gpu"
        group = hq // hkv
        kx, vx = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)
        scores = np.einsum("bhqd,bhkd->bhqk", q, kx) * 0.25
        if causal:
            offset = max(sk - sq, 0)
            scores = np.where(np.arange(sk)[None, :] <=
                              np.arange(sq)[:, None] + offset, scores, -np.inf)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        error = float(np.abs(out - np.einsum("bhqk,bhkd->bhqd", probs, vx)).max())
        if error > 1e-4:
            raise RuntimeError(f"attention failed its oracle: {error}")
        timing = _time(lambda: rt.launch(artifact, {"q": q, "k": k, "v": v, "o": out}),
                       reps=args.reps, warmup=args.warmup)
        rows.append(_row(f"attention.fwd.causal{int(causal)}", provenance, timing,
                         error, native=native, shape=[b_, hq, hkv, sq, sk, d]))

    # --- attention backward (deterministic split VJP) ---
    for (b_, hq, hkv, sq, sk, d, causal) in ((1, 4, 2, 64, 64, 64, False),
                                             (1, 4, 2, 128, 128, 64, True)):
        module = _attention_backward_module(b_, hq, hkv, sq, sk, d, causal)
        backward = lower_scheduled_attention_backward(module, target="apple_gpu")
        package = apple_native.package_scheduled_attention_backward(
            backward, pipeline_name="tessera-lower-to-apple_gpu")
        artifact = rt.RuntimeArtifact(
            metadata={"target": "apple_gpu"}, native_image=package.image,
            launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
            target_ir=package.target_ir)
        provenance = dict(package.descriptor.provenance)
        q = np.ascontiguousarray(rng.standard_normal((b_, hq, sq, d)), dtype=np.float32)
        k = np.ascontiguousarray(rng.standard_normal((b_, hkv, sk, d)), dtype=np.float32)
        v = np.ascontiguousarray(rng.standard_normal((b_, hkv, sk, d)), dtype=np.float32)
        do = np.ascontiguousarray(rng.standard_normal((b_, hq, sq, d)), dtype=np.float32)
        grads = {"dq": np.zeros_like(q), "dk": np.zeros_like(k), "dv": np.zeros_like(v)}
        feed = {"do": do, "q": q, "k": k, "v": v, **grads}
        result = rt.launch(artifact, feed)
        native = result.get("execution_kind") == "native_gpu"
        group = hq // hkv
        kx = np.repeat(k, group, axis=1).astype(np.float64)
        vx = np.repeat(v, group, axis=1).astype(np.float64)
        scores = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64), kx) * 0.25
        if causal:
            offset = max(sk - sq, 0)
            scores = np.where(np.arange(sk)[None, :] <=
                              np.arange(sq)[:, None] + offset, scores, -np.inf)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        d_p = np.einsum("bhqd,bhkd->bhqk", do.astype(np.float64), vx)
        d_s = probs * (d_p - (d_p * probs).sum(-1, keepdims=True))
        expected_dq = np.einsum("bhqk,bhkd->bhqd", d_s, kx) * 0.25
        error = float(np.abs(grads["dq"] - expected_dq).max())
        if error > 1e-3:
            raise RuntimeError(f"attention backward failed its dQ oracle: {error}")
        timing = _time(lambda: rt.launch(artifact, feed),
                       reps=args.reps, warmup=args.warmup)
        rows.append(_row(f"attention.bwd.causal{int(causal)}", provenance, timing,
                         error, native=native, shape=[b_, hq, hkv, sq, sk, d]))

    report = {
        "schema": "tessera.apple.scheduled_boundary.characterization.v1",
        # Not a selector corpus: one process, one timing domain, no paired
        # interleaving and no retained counter evidence.
        "selector_eligible": False,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "reps": args.reps,
        "warmup": args.warmup,
        "timing_domain": "end_to_end_host_wall",
        "rows": rows,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    width = max(len(r["case"]) for r in rows)
    print(f"\n{'case':<{width}}  {'route':<38} {'median_ms':>10} {'p90_ms':>9} "
          f"{'err':>9}  native")
    for row in rows:
        print(f"{row['case']:<{width}}  {str(row['route'])[:38]:<38} "
              f"{row['median_ms']:>10.4f} {row['p90_ms']:>9.4f} "
              f"{row['max_abs_error']:>9.2e}  {row['native_gpu']}")
    print("\nselector_eligible=False — characterization only "
          "(one process, end-to-end domain, no paired interleaving).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
