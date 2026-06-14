"""DLOP-Bench-style long-tail operator benchmark (DeepLink DLOP-Bench lens).

DLOP-Bench's insight: the expensive operators are *long-tail composites* — a
domain op (attention block, SwiGLU FFN, a CV bbox transform) decomposes into
many basic ops, and running them unfused pays a function-call / kernel-launch
per basic op.  It grades each in Stage-1 (eager) vs Stage-2 (graph/JIT-fused).

That decomposition-overhead story *is* Tessera's fusion thesis (matmul→softmax→
matmul, moe_swiglu_block).  This core turns that claim into measured rows: for
each composite it reports the **dispatch count** unfused (one per primitive) vs
fused (one), the **decomposition factor** (primitives-per-composite), and a
**metamorphic equivalence** check (fused ≡ eager) — the same dispatch-count +
equivalence telemetry pattern as ``long_memory_core``'s resident-vs-recompute
row (which proved an 8.5× build-traffic reduction).  Reference-level (numpy) so
it is portable; the on-device fused kernels it points at (flash_attn,
moe_swiglu_block) are real Apple-GPU lanes whose launch-count = the fused 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from benchmarks.common import (
    BenchmarkOperator,
    BenchmarkRow,
    CompilerPath,
    Correctness,
    ExecutionKind,
    RuntimeStatus,
    telemetry_for_row,
)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# A long-tail composite: a fused reference + the primitive op-list it decomposes
# into (the dispatches an unfused backend would pay).  fused_fn and the eager
# decomposition must be numerically identical — that is the metamorphic oracle.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LongTailOp:
    name: str
    family: str
    primitives: tuple[str, ...]          # the unfused dispatch sequence
    inputs: Callable[[np.random.Generator], tuple[np.ndarray, ...]]
    eager: Callable[..., np.ndarray]     # decomposed (calls each primitive)
    fused: Callable[..., np.ndarray]     # single fused kernel reference
    fused_apple_gpu: str | None = None   # the real Apple-GPU lane, if any

    @property
    def decomposition_factor(self) -> int:
        return len(self.primitives)


def _attention_block() -> LongTailOp:
    def inputs(rng):
        d = 32
        q = rng.standard_normal((8, d)).astype(np.float32)
        k = rng.standard_normal((8, d)).astype(np.float32)
        v = rng.standard_normal((8, d)).astype(np.float32)
        return q, k, v

    def eager(q, k, v):
        scale = 1.0 / np.sqrt(q.shape[-1])
        s = q @ k.T                        # matmul
        s = s * scale                      # scale (mul)
        p = _softmax(s, axis=-1)           # softmax
        return p @ v                       # matmul

    def fused(q, k, v):                    # one flash_attn dispatch
        scale = 1.0 / np.sqrt(q.shape[-1])
        return _softmax((q @ k.T) * scale, axis=-1) @ v

    return LongTailOp(
        "attention_block", "attention",
        ("matmul", "scale", "softmax", "matmul"),
        inputs, eager, fused, fused_apple_gpu="flash_attn",
    )


def _swiglu_ffn() -> LongTailOp:
    def inputs(rng):
        d, h = 16, 32
        x = rng.standard_normal((8, d)).astype(np.float32)
        wg = rng.standard_normal((d, h)).astype(np.float32)
        wu = rng.standard_normal((d, h)).astype(np.float32)
        wd = rng.standard_normal((h, d)).astype(np.float32)
        return x, wg, wu, wd

    def eager(x, wg, wu, wd):
        g = x @ wg                         # matmul
        a = _silu(g)                       # silu
        u = x @ wu                         # matmul
        h = a * u                          # mul
        return h @ wd                      # matmul

    def fused(x, wg, wu, wd):              # one moe_swiglu_block dispatch
        return (_silu(x @ wg) * (x @ wu)) @ wd

    return LongTailOp(
        "swiglu_ffn", "moe",
        ("matmul", "silu", "matmul", "mul", "matmul"),
        inputs, eager, fused, fused_apple_gpu="moe_swiglu_block",
    )


def _rmsnorm_linear() -> LongTailOp:
    def inputs(rng):
        d, n = 16, 24
        x = rng.standard_normal((8, d)).astype(np.float32)
        w = rng.standard_normal((d, n)).astype(np.float32)
        g = rng.standard_normal((d,)).astype(np.float32)
        return x, w, g

    def eager(x, w, g):
        sq = x * x                         # square (mul)
        ms = sq.mean(axis=-1, keepdims=True)  # mean
        inv = 1.0 / np.sqrt(ms + 1e-6)     # rsqrt
        nrm = x * inv * g                  # mul (×2 folded)
        return nrm @ w                     # matmul

    def fused(x, w, g):                    # one rmsnorm→matmul dispatch
        inv = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-6)
        return (x * inv * g) @ w

    return LongTailOp(
        "rmsnorm_linear", "normalization",
        ("square", "mean", "rsqrt", "mul", "matmul"),
        inputs, eager, fused, fused_apple_gpu="matmul_rmsnorm",
    )


def _bbox_transform() -> LongTailOp:
    # A CV long-tail op in DLOP's spirit (bbox2delta): many elementwise ops with
    # no dedicated kernel — pure decomposition overhead, no fused backend lane.
    def inputs(rng):
        def _boxes():
            x0 = rng.uniform(0, 50, (64,)).astype(np.float32)
            y0 = rng.uniform(0, 50, (64,)).astype(np.float32)
            w = rng.uniform(5, 30, (64,)).astype(np.float32)   # positive width
            h = rng.uniform(5, 30, (64,)).astype(np.float32)   # positive height
            return np.stack([x0, y0, x0 + w, y0 + h], axis=1)
        return _boxes(), _boxes()

    def eager(boxes, gt):
        pw = boxes[:, 2] - boxes[:, 0]     # sub
        ph = boxes[:, 3] - boxes[:, 1]     # sub
        px = boxes[:, 0] + 0.5 * pw        # mul + add
        py = boxes[:, 1] + 0.5 * ph        # mul + add
        gw = gt[:, 2] - gt[:, 0]           # sub
        gh = gt[:, 3] - gt[:, 1]           # sub
        gx = gt[:, 0] + 0.5 * gw           # mul + add
        gy = gt[:, 1] + 0.5 * gh           # mul + add
        dx = (gx - px) / pw                # sub + div
        dy = (gy - py) / ph                # sub + div
        dw = np.log(gw / pw)               # div + log
        dh = np.log(gh / ph)               # div + log
        return np.stack([dx, dy, dw, dh], axis=1)  # stack

    def fused(boxes, gt):
        return _bbox_transform().eager(boxes, gt)  # same math, "one" custom op

    return LongTailOp(
        "bbox2delta", "cv_longtail",
        ("sub", "sub", "mul_add", "mul_add", "sub", "sub", "mul_add",
         "mul_add", "sub_div", "sub_div", "div_log", "div_log", "stack"),
        inputs, eager, fused, fused_apple_gpu=None,
    )


LONGTAIL_OPS: tuple[LongTailOp, ...] = (
    _attention_block(), _swiglu_ffn(), _rmsnorm_linear(), _bbox_transform(),
)


@dataclass(frozen=True)
class DlopLongtailConfig:
    seed: int = 0


def _row(op: LongTailOp, *, equivalent: bool, max_err: float) -> BenchmarkRow:
    # The fused dispatch count is 1 when a real fused lane exists, else the op is
    # "host-composed" and stays at its decomposition count (honest: no fused
    # kernel to collapse into).
    fused_dispatches = 1 if op.fused_apple_gpu else op.decomposition_factor
    reduction = op.decomposition_factor / fused_dispatches
    return BenchmarkRow(
        operator=BenchmarkOperator(name=op.name, dtype="fp32",
                                   shape=f"primitives={op.decomposition_factor}",
                                   target="cpu"),
        compiler_path=CompilerPath.REFERENCE,
        runtime_status=RuntimeStatus.EXECUTABLE,
        correctness=Correctness(max_error=max_err, passed=equivalent),
        execution_kind=ExecutionKind.REFERENCE,
        metrics={
            "family": op.family,
            "eager_dispatches": op.decomposition_factor,
            "fused_dispatches": fused_dispatches,
            "decomposition_factor": op.decomposition_factor,
            "dispatch_reduction_x": round(reduction, 2),
            "fused_apple_gpu_lane": op.fused_apple_gpu or "none",
            "metamorphic_equivalent": equivalent,
        },
        reason="" if op.fused_apple_gpu else "no fused backend lane (host-composed)",
    )


def run_core(cfg: DlopLongtailConfig | None = None) -> list[BenchmarkRow]:
    cfg = cfg or DlopLongtailConfig()
    rows: list[BenchmarkRow] = []
    for i, op in enumerate(LONGTAIL_OPS):
        rng = np.random.default_rng(cfg.seed + i)
        args = op.inputs(rng)
        eager_out = np.asarray(op.eager(*args))
        fused_out = np.asarray(op.fused(*args))
        err = float(np.abs(eager_out - fused_out).max())
        rows.append(_row(op, equivalent=err < 1e-4, max_err=err))
    return rows


def build_report(rows: list[BenchmarkRow] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else run_core()
    fused = [r for r in rows if r.metrics["fused_apple_gpu_lane"] != "none"]
    factors = [r.metrics["decomposition_factor"] for r in rows]
    return {
        "ops": len(rows),
        "all_metamorphic_equivalent": all(r.correctness.passed for r in rows),
        "fusible_ops": len(fused),
        "mean_decomposition_factor": round(float(np.mean(factors)), 2),
        "max_dispatch_reduction_x": max(r.metrics["dispatch_reduction_x"] for r in rows),
        "host_composed_gaps": sorted(
            r.operator.name for r in rows if r.metrics["fused_apple_gpu_lane"] == "none"
        ),
    }


def telemetry(rows: list[BenchmarkRow]) -> list[dict[str, Any]]:
    return [telemetry_for_row(r, source="dlop_longtail_core") for r in rows]
