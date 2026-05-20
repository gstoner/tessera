#!/usr/bin/env python3
"""Portable Tessera operator benchmark bridge.

The C++ harness uses this helper for hardware-free ``tessera-runtime`` and
generated-artifact validation. It intentionally goes through the public Python
JIT/runtime path so benchmark rows exercise the same Graph->Schedule->Tile->
Target bundle contract as developer code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[3]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import tessera as ts  # noqa: E402


BRIDGE_OPS = {
    "matmul",
    "conv2d",
    "flash_attention",
    "reduce",
    "elementwise",
    "softmax_layernorm",
    "transpose_gather",
}


def _jit_source(name: str, args: tuple[str, ...], return_expr: str):
    source = f"def {name}({', '.join(args)}):\n    return {return_expr}\n"
    namespace = {"ts": ts}
    exec(source, namespace)
    return ts.jit(namespace[name], source=source)


def _kernel_for(ns: argparse.Namespace):
    if ns.op == "matmul":
        return _jit_source("kernel_matmul", ("a", "b"), "ts.ops.matmul(a, b)")
    if ns.op == "conv2d":
        return _jit_source(
            "kernel_conv2d",
            ("x", "w"),
            f"ts.ops.conv2d(x, w, stride=({ns.stride_h}, {ns.stride_w}), padding=({ns.pad_h}, {ns.pad_w}))",
        )
    if ns.op == "flash_attention":
        return _jit_source("kernel_flash_attention", ("q", "k", "v"), "ts.ops.flash_attn(q, k, v, causal=True)")
    if ns.op == "reduce":
        axis = "None" if _axis(ns.axis) is None else str(_axis(ns.axis))
        return _jit_source("kernel_reduce", ("x",), f'ts.ops.reduce(x, op="sum", axis={axis})')
    if ns.op == "elementwise":
        return _jit_source("kernel_elementwise", ("x",), "ts.ops.tanh(x) + 0.1 * x")
    if ns.op == "softmax_layernorm":
        return _jit_source("kernel_softmax_layernorm", ("x",), "ts.ops.layer_norm(ts.ops.softmax(x, axis=1))")
    if ns.op == "transpose_gather":
        return _jit_source("kernel_transpose_gather", ("x",), "ts.ops.transpose(x, axes=(0, 2, 1))")
    raise KeyError(ns.op)


def _args_for(ns: argparse.Namespace) -> tuple[tuple[Any, ...], dict[str, Any], np.ndarray]:
    rng = np.random.default_rng(ns.seed)
    dtype = _dtype(ns.dtype)
    if ns.op == "matmul":
        k2 = ns.k2 if ns.k2 is not None else ns.k
        a = rng.normal(size=(ns.m, ns.k)).astype(dtype)
        b = rng.normal(size=(k2, ns.n)).astype(dtype)
        expected = np.matmul(a, b)
        return (a, b), {}, expected
    if ns.op == "conv2d":
        if ns.stride_h <= 0 or ns.stride_w <= 0:
            raise ValueError("conv2d stride must be positive")
        if ns.pad_h < 0 or ns.pad_w < 0:
            raise ValueError("conv2d padding must be non-negative")
        x = rng.normal(size=(ns.Nn, ns.H, ns.W, ns.C)).astype(dtype)
        w = rng.normal(size=(ns.R, ns.S, ns.C, ns.Kc)).astype(dtype)
        stride = (ns.stride_h, ns.stride_w)
        padding = (ns.pad_h, ns.pad_w)
        return (x, w), {}, _conv2d_nhwc(x, w, stride=stride, padding=padding)
    if ns.op == "flash_attention":
        if ns.dim <= 0 or ns.seq <= 0 or ns.B <= 0 or ns.heads <= 0:
            raise ValueError("flash_attention dimensions must be positive")
        q = rng.normal(size=(ns.B, ns.heads, ns.seq, ns.dim)).astype(dtype)
        k = rng.normal(size=(ns.B, ns.heads, ns.seq, ns.dim)).astype(dtype)
        v = rng.normal(size=(ns.B, ns.heads, ns.seq, ns.dim)).astype(dtype)
        return (q, k, v), {}, _flash_attn(q, k, v, causal=True)
    if ns.op in {"reduce", "elementwise"}:
        x = rng.normal(size=(ns.m, ns.n, ns.k)).astype(dtype).reshape(ns.m, ns.n * ns.k)
        if ns.op == "reduce":
            axis = _axis(ns.axis)
            expected = np.sum(x, axis=axis)
            return (x,), {}, expected
        return (x,), {}, np.tanh(x) + 0.1 * x
    if ns.op == "softmax_layernorm":
        x = rng.normal(size=(ns.m, ns.n, ns.k)).astype(dtype)
        expected = _layer_norm(_softmax(x, axis=1))
        return (x,), {}, expected
    if ns.op == "transpose_gather":
        x = rng.normal(size=(ns.m, ns.n, ns.k)).astype(dtype)
        return (x,), {}, np.transpose(x, (0, 2, 1))
    raise KeyError(ns.op)


def _dtype(name: str) -> Any:
    if name in {"f32", "float32"}:
        return np.float32
    raise ValueError(f"unsupported dtype {name!r}; bridge currently supports f32 only")


def _axis(text: str | None) -> int | None:
    if text in {None, "", "none", "None", "null"}:
        return None
    return int(text)


def _conv2d_nhwc(x: np.ndarray, w: np.ndarray, *, stride: tuple[int, int], padding: tuple[int, int]) -> np.ndarray:
    sh, sw = stride
    ph, pw = padding
    x_pad = np.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    batch, in_h, in_w, _ = x_pad.shape
    kh, kw, _, out_c = w.shape
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    out = np.zeros((batch, out_h, out_w, out_c), dtype=np.result_type(x, w))
    for i in range(out_h):
        for j in range(out_w):
            window = x_pad[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]
            out[:, i, j, :] = np.tensordot(window, w, axes=([1, 2, 3], [0, 1, 2]))
    return out


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _flash_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, causal: bool) -> np.ndarray:
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool), k=1 + max(k_len - q_len, 0))
        scores = np.where(mask, -np.inf, scores)
    return np.matmul(_softmax(scores, axis=-1), v)


def _artifact_flags(artifact: ts.RuntimeArtifact) -> dict[str, Any]:
    return {
        "graph": bool(artifact.graph_ir),
        "schedule": bool(artifact.schedule_ir),
        "tile": bool(artifact.tile_ir),
        "target": bool(artifact.target_ir),
        "artifact_hash": artifact.artifact_hash,
        "graph_hash": _hash_text(artifact.graph_ir),
        "schedule_hash": _hash_text(artifact.schedule_ir),
        "tile_hash": _hash_text(artifact.tile_ir),
        "target_hash": _hash_text(artifact.target_ir),
    }


def _hash_text(text: str) -> str | None:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


def _memory_bytes(inputs: tuple[Any, ...], output: Any | None) -> int:
    total = 0
    for value in inputs:
        total += int(np.asarray(value).nbytes)
    if output is not None:
        total += int(np.asarray(output).nbytes)
    return total


def _work_estimate(ns: argparse.Namespace) -> float:
    if ns.op == "matmul":
        return float(2 * ns.m * ns.n * ns.k)
    if ns.op == "conv2d":
        out_h = (ns.H + 2 * ns.pad_h - ns.R) // ns.stride_h + 1
        out_w = (ns.W + 2 * ns.pad_w - ns.S) // ns.stride_w + 1
        return float(2 * ns.Nn * out_h * out_w * ns.Kc * ns.R * ns.S * ns.C)
    if ns.op == "flash_attention":
        return float(4 * ns.B * ns.heads * ns.seq * ns.seq * ns.dim)
    return float(max(ns.m * ns.n * ns.k, 1))


def _row(
    ns: argparse.Namespace,
    *,
    artifact: ts.RuntimeArtifact | None,
    runtime_status: str,
    compiler_path: str,
    reason: str,
    output: Any | None,
    expected: Any | None,
    elapsed_ms: float | None,
    runtime_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    flags = _artifact_flags(artifact) if artifact is not None else {
        "graph": False,
        "schedule": False,
        "tile": False,
        "target": False,
        "artifact_hash": None,
        "graph_hash": None,
        "schedule_hash": None,
        "tile_hash": None,
        "target_hash": None,
    }
    max_error = None
    rel_error = None
    passed = None
    if output is not None and expected is not None:
        out = np.asarray(output)
        ref = np.asarray(expected)
        max_error = float(np.max(np.abs(out - ref))) if out.size else 0.0
        denom = float(np.max(np.abs(ref))) if ref.size else 0.0
        rel_error = max_error / max(denom, 1e-12)
        passed = bool(np.allclose(out, ref, atol=1e-4, rtol=1e-4))
    telemetry_status = "ok" if runtime_status == "success" else (
        "unmeasured" if runtime_status == "artifact_only" else runtime_status
    )
    latency = elapsed_ms if elapsed_ms is not None else 0.0
    tflops = (_work_estimate(ns) / max(latency, 1e-9)) / 1e9 if runtime_status == "success" else 0.0
    inputs = getattr(ns, "_bridge_inputs", ())
    memory_bytes = _memory_bytes(inputs, output) if inputs else None
    gbps = (memory_bytes / max(latency, 1e-9)) / 1e6 if memory_bytes and runtime_status == "success" else 0.0
    telemetry = (runtime_result or {}).get("telemetry") or {
        "schema": "tessera.telemetry.v1",
        "name": ns.op,
        "source": "tessera_operator_bench",
        "op": ns.op,
        "dtype": ns.dtype,
        "arch": "cpu",
        "latency_ms": latency,
        "tflops": tflops,
        "bandwidth_gbps": gbps,
        "status": telemetry_status,
        "counters": {},
        "metadata": {},
    }
    telemetry.setdefault("schema", "tessera.telemetry.v1")
    telemetry.setdefault("source", "tessera_operator_bench")
    telemetry["op"] = ns.op
    execution_kind = "artifact_only" if runtime_status == "artifact_only" else (
        "reference" if runtime_status == "success" else "unknown"
    )
    telemetry.setdefault("status", telemetry_status)
    telemetry.setdefault("metadata", {})
    telemetry["metadata"].update({
        "backend": ns.backend,
        "runtime": ns.runtime,
        "compiler_path": compiler_path,
        "runtime_status": runtime_status,
        "execution_kind": execution_kind,
        "artifact_hash": flags["artifact_hash"],
        "graph_hash": flags["graph_hash"],
        "schedule_hash": flags["schedule_hash"],
        "tile_hash": flags["tile_hash"],
        "target_hash": flags["target_hash"],
    })
    return {
        "operator": {"name": ns.op, "dtype": ns.dtype, "shape": "cli", "target": "cpu"},
        "compiler_path": compiler_path,
        "runtime_status": runtime_status,
        "execution_kind": execution_kind,
        "artifact_levels": flags,
        "correctness": {
            "max_error": max_error,
            "relative_error": rel_error,
            "tolerance": 1e-4 if output is not None else None,
            "passed": passed,
        },
        "profile": {
            "cpu_wall_ms": latency,
            "kernel_elapsed_ms": None,
            "memory_bytes": memory_bytes,
            "launch_overhead_ms": (runtime_result or {}).get("profile", {}).get("launch_overhead_ms"),
        },
        "metrics": {"backend": ns.backend, "gflops": tflops * 1000.0, "gbps": gbps},
        "telemetry": telemetry,
        "reason": reason,
    }


def _emit_error(ns: argparse.Namespace, status: str, reason: str) -> int:
    print(json.dumps(_row(
        ns,
        artifact=None,
        runtime_status=status,
        compiler_path="runtime_unavailable",
        reason=reason,
        output=None,
        expected=None,
        elapsed_ms=0.0,
    )))
    return 0


def run(ns: argparse.Namespace) -> int:
    if ns.op not in BRIDGE_OPS:
        return _emit_error(ns, "unsupported", f"unknown operator {ns.op!r}")
    if ns.runtime == "native":
        return _emit_error(ns, "backend_unavailable", "Native C ABI operator launch is pending; use --runtime bridge")
    try:
        inputs, kwargs, expected = _args_for(ns)
    except Exception as exc:
        return _emit_error(ns, "invalid_argument", str(exc))
    ns._bridge_inputs = inputs
    try:
        kernel = _kernel_for(ns)
        artifact = kernel.runtime_artifact()
    except Exception as exc:
        return _emit_error(ns, "invalid_artifact", str(exc))
    if ns.mode == "artifact":
        print(json.dumps(_row(
            ns,
            artifact=artifact,
            runtime_status="artifact_only",
            compiler_path=str((artifact.metadata or {}).get("compiler_path", "artifact_only")),
            reason="generated Graph/Schedule/Tile/Target artifact bundle validated; runtime execution skipped",
            output=None,
            expected=None,
            elapsed_ms=0.0,
        )))
        return 0

    start = time.perf_counter()
    result = ts.launch(artifact, inputs)
    elapsed_ms = float((time.perf_counter() - start) * 1000.0)
    output = result.get("output") if result.get("ok") else None
    print(json.dumps(_row(
        ns,
        artifact=artifact,
        runtime_status=str(result.get("runtime_status", "backend_unavailable")),
        compiler_path=str(result.get("compiler_path", (artifact.metadata or {}).get("compiler_path", "artifact_only"))),
        reason=str(result.get("reason", "")),
        output=output,
        expected=expected if result.get("ok") else None,
        elapsed_ms=float(result.get("profile", {}).get("cpu_wall_ms", elapsed_ms)),
        runtime_result=result,
    )))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="runtime", choices=["runtime", "artifact"])
    ap.add_argument("--backend", default="tessera-runtime")
    ap.add_argument("--runtime", default="bridge", choices=["bridge", "native"])
    ap.add_argument("--op", required=True)
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--k2", type=int, default=None)
    ap.add_argument("--axis", default=None)
    ap.add_argument("--Nn", type=int, default=1)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--Kc", type=int, default=4)
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--stride_h", type=int, default=1)
    ap.add_argument("--stride_w", type=int, default=1)
    ap.add_argument("--pad_h", type=int, default=0)
    ap.add_argument("--pad_w", type=int, default=0)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--dim", type=int, default=8)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
