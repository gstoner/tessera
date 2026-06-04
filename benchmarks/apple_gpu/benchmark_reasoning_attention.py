"""Sprint 10 (2026-06-03) — Apple GPU reasoning-attention benchmark runway.

This is a *benchmark-facing* example that reports five things as SEPARATE,
non-conflated fields for each reasoning-model attention variant:

    route        — how the variant compiles on apple_gpu (fused kernel vs
                   compiler-visible-only Graph IR op)
    target       — always "apple_gpu" here
    executor     — the runtime executor id if (and only if) the variant is
                   executable; otherwise None
    correctness  — max relative error vs a numpy reference, measured ONLY when
                   the variant actually ran; otherwise None ("not executed")
    timing_ms    — measured wall-clock per call, ONLY when it actually ran;
                   otherwise None

HONESTY CONTRACT (Decisions #21 / #25, VALUE_TARGET_IR_CONTRACT.md):
  * Exactly one strict executable envelope is promoted: the MLA-style attention
    block `matmul -> softmax -> matmul` (+ MPS projections). On apple_gpu the
    inner block fuses into the `matmul_softmax_matmul_f32` MSL kernel
    (Phase 8.4.5) and the projections dispatch through MPS — both real runtime
    kernels. Its correctness + timing are measured against a numpy reference.
  * Every other reasoning variant (DeepSeek MLA-decode-fused, DeepSeek NSA,
    Lightning, Gated DeltaNet, Kimi-Delta, hybrid) is COMPILER-VISIBLE ONLY:
    the Sprint 10 reasoning prologue recognizes/positions it in Graph IR, but
    there is no Apple runtime kernel yet, so executor=None, correctness=None,
    timing_ms=None. We never fabricate a number for an op we did not run.
  * The executable/non-executable split is GROUNDED in the runtime envelope
    (`tessera.compiler.driver._APPLE_GPU_RUNTIME_OPS`) — not hard-coded here —
    so this example can never drift into over-claiming.

On non-Darwin hosts (or when Metal is inactive) the static route/target/executor
classification is still reported (it's a compile-time fact); correctness and
timing for the executable envelope are reported as None with an explicit
``skip_reason``. The script always exits 0.

Usage:
    python benchmarks/apple_gpu/benchmark_reasoning_attention.py \\
        --shapes 8x16x2 16x32x4 32x64x8 --reps 30 \\
        --output apple_gpu_reasoning.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from typing import Any

import numpy as np

import tessera as ts
from tessera.compiler import driver as _driver


# ── The one strict executable envelope + the compiler-visible-only variants ──
#
# `graph_op` is the canonical Graph IR op name whose runtime-executability we
# look up in the driver envelope. The MLA-style block is built from executable
# primitives (matmul / softmax / matmul), so we probe `tessera.matmul`.
EXECUTABLE_ENVELOPE = {
    "name": "mla_style_attention",
    "graph_op": "tessera.matmul",
    "route": "fused matmul_softmax_matmul (MSL) + MPS projections",
    "description": "MLA-flavored single-layer MHA decode step "
                   "(3 projections + per-head matmul->softmax->matmul + "
                   "output projection)",
}

COMPILER_VISIBLE_ONLY = [
    {"name": "mla_decode_fused", "graph_op": "tessera.mla_decode_fused",
     "description": "DeepSeek MLA decode chain fused by -tessera-mla-fusion"},
    {"name": "deepseek_native_sparse_attn",
     "graph_op": "tessera.native_sparse_attn_fused",
     "description": "DeepSeek NSA 3-branch fused by "
                    "-tessera-native-sparse-attn-fusion"},
    {"name": "lightning_attention", "graph_op": "tessera.lightning_attention",
     "description": "MiniMax-M1 Lightning linear attention"},
    {"name": "gated_deltanet", "graph_op": "tessera.gated_deltanet",
     "description": "Gated DeltaNet recurrence"},
    {"name": "kimi_delta_attention", "graph_op": "tessera.kimi_delta_attention",
     "description": "Kimi-Dev delta attention"},
    {"name": "hybrid_attention", "graph_op": "tessera.hybrid_attention",
     "description": "Ling/Kimi named hybrid attention policy"},
]


def _executor_for(graph_op: str) -> str | None:
    """Resolve the apple_gpu runtime executor for a Graph IR op, grounded in
    the driver envelope. Returns None if the op has no real runtime kernel."""
    if graph_op in _driver._APPLE_GPU_MPS_OPS:
        return "apple_gpu_mps"
    if graph_op in _driver._APPLE_GPU_MSL_OPS:
        return "apple_gpu_msl"
    if graph_op in _driver._APPLE_GPU_MPSGRAPH_OPS:
        return "apple_gpu_mpsgraph"
    if graph_op in getattr(_driver, "_APPLE_GPU_RUNTIME_OPS", frozenset()):
        return "apple_gpu_runtime"
    return None


def _is_executable(graph_op: str) -> bool:
    return graph_op in getattr(_driver, "_APPLE_GPU_RUNTIME_OPS", frozenset())


# ── numpy reference for the executable MLA-style block ──────────────────────

def _np_mha_reference(x, Wq, Wk, Wv, Wo, num_heads):
    T, D = x.shape
    Dh = D // num_heads
    Q, K, V = x @ Wq, x @ Wk, x @ Wv

    def split(M):
        return M.reshape(T, num_heads, Dh).transpose(1, 0, 2)

    Qh, Kh, Vh = split(Q), split(K), split(V)
    scale = 1.0 / math.sqrt(Dh)
    out = np.empty_like(Qh)
    for h in range(num_heads):
        s = (Qh[h] @ Kh[h].T) * scale
        s = s - s.max(axis=-1, keepdims=True)
        e = np.exp(s)
        out[h] = (e / e.sum(axis=-1, keepdims=True)) @ Vh[h]
    return out.transpose(1, 0, 2).reshape(T, D) @ Wo


def _build_mla_callables():
    @ts.jit(target="apple_gpu")
    def project(x, W):
        return ts.ops.matmul(x, W)

    @ts.jit(target="apple_gpu")
    def per_head_attn(qh, kh_t, vh):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(qh, kh_t)), vh)

    return project, per_head_attn


def _run_mla_once(project, per_head_attn, x, Wq, Wk, Wv, Wo, H):
    T, D = x.shape
    Dh = D // H
    Q = np.asarray(project(x, Wq))
    K = np.asarray(project(x, Wk))
    V = np.asarray(project(x, Wv))
    scale = 1.0 / math.sqrt(Dh)

    def split(M):
        return M.reshape(T, H, Dh).transpose(1, 0, 2)

    Qh, Kh, Vh = split(Q) * scale, split(K), split(V)
    out = np.empty_like(Qh)
    for h in range(H):
        out[h] = np.asarray(per_head_attn(Qh[h], Kh[h].T.copy(), Vh[h]))
    concat = out.transpose(1, 0, 2).reshape(T, D)
    return np.asarray(project(concat, Wo))


def _metal_active() -> bool:
    """True iff the apple_gpu runtime can actually execute (Darwin + Metal)."""
    if sys.platform != "darwin":
        return False
    try:
        from tessera._apple_gpu_dispatch import apple_gpu_skip_reason
        return apple_gpu_skip_reason() is None
    except Exception:
        return False


def _measure_executable_envelope(shapes, reps):
    """Run the executable MLA-style block; return one report row per shape with
    measured correctness + timing. If Metal is inactive, correctness/timing are
    None with an explicit skip_reason (route/target/executor still reported)."""
    graph_op = EXECUTABLE_ENVELOPE["graph_op"]
    executor = _executor_for(graph_op)
    executable = _is_executable(graph_op)
    rows = []
    active = _metal_active()
    project = per_head_attn = None
    if active:
        project, per_head_attn = _build_mla_callables()

    for (T, D, H) in shapes:
        row: dict[str, Any] = {
            "name": EXECUTABLE_ENVELOPE["name"],
            "variant_kind": "executable_envelope",
            "shape": f"T{T}_D{D}_H{H}",
            "route": EXECUTABLE_ENVELOPE["route"],
            "target": "apple_gpu",
            "executor": executor,
            "executable": executable,
            "correctness_max_rel_err": None,
            "timing_ms": None,
            "skip_reason": None,
        }
        if not active:
            row["skip_reason"] = "metal runtime inactive (non-Darwin or no GPU)"
            rows.append(row)
            continue

        rng = np.random.RandomState(2026_06_03 + T)
        f = lambda *s: (rng.randn(*s) * 0.1).astype(np.float32)
        x = f(T, D)
        Wq, Wk, Wv, Wo = f(D, D), f(D, D), f(D, D), f(D, D)

        got = _run_mla_once(project, per_head_attn, x, Wq, Wk, Wv, Wo, H)
        ref = _np_mha_reference(x, Wq, Wk, Wv, Wo, H)
        denom = float(np.abs(ref).max()) + 1e-9
        row["correctness_max_rel_err"] = float(
            np.abs(got - ref).max() / denom)

        times = []
        for _ in range(max(1, reps)):
            t0 = time.perf_counter()
            _run_mla_once(project, per_head_attn, x, Wq, Wk, Wv, Wo, H)
            times.append((time.perf_counter() - t0) * 1e3)
        row["timing_ms"] = float(statistics.median(times))
        rows.append(row)
    return rows


def _compiler_visible_rows():
    """One row per compiler-visible-only reasoning variant. These are NOT
    executed — executor/correctness/timing are all None by construction."""
    rows = []
    for v in COMPILER_VISIBLE_ONLY:
        graph_op = v["graph_op"]
        rows.append({
            "name": v["name"],
            "variant_kind": "compiler_visible_only",
            "shape": "-",
            "route": "compiler_visible (Graph IR recognized + reasoning "
                     "prologue positioned; no Apple runtime kernel)",
            "target": "apple_gpu",
            "executor": _executor_for(graph_op),   # expected None
            "executable": _is_executable(graph_op),  # expected False
            "correctness_max_rel_err": None,
            "timing_ms": None,
            "skip_reason": "not executed — no apple_gpu runtime kernel "
                           "(compiler-visible only)",
        })
    return rows


def build_report(shapes, reps) -> dict[str, Any]:
    rows = _measure_executable_envelope(shapes, reps) + _compiler_visible_rows()
    return {
        "benchmark": "apple_gpu_reasoning_attention",
        "sprint": "S10",
        "tessera_version": getattr(ts, "__version__", "unknown"),
        "metal_active": _metal_active(),
        "rows": rows,
    }


def _print_table(report):
    hdr = f"{'variant':28} {'kind':22} {'shape':12} {'exec':5} " \
          f"{'executor':18} {'rel_err':>10} {'ms':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in report["rows"]:
        rel = (f"{r['correctness_max_rel_err']:.2e}"
               if r["correctness_max_rel_err"] is not None else "n/a")
        ms = (f"{r['timing_ms']:.3f}"
              if r["timing_ms"] is not None else "n/a")
        print(f"{r['name']:28} {r['variant_kind']:22} {r['shape']:12} "
              f"{str(r['executable']):5} {str(r['executor']):18} "
              f"{rel:>10} {ms:>8}")


def _parse_shape(spec: str) -> tuple[int, int, int]:
    parts = spec.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape must be TxDxH, got {spec!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", nargs="+", default=["8x16x2", "16x32x4",
                                                    "32x64x8"],
                    help="TxDxH triples (T=batch*seq, D=model dim, H=heads)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args(argv)

    shapes = [_parse_shape(s) for s in args.shapes]
    report = build_report(shapes, args.reps)
    _print_table(report)
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
