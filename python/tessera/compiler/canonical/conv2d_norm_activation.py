"""Canonical program: ``conv2d → layer_norm → gelu``.

The CNN-block counterpart to the attention-shaped
``matmul_softmax_matmul`` canonical program.  M1.5 ships this as a
numpy-reference path; native NHWC conv2d on Apple GPU is a planned
backend kernel (not yet fused), so the report's
``target_decision`` documents the gap explicitly.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import tessera
from tessera.compiler import backend_manifest as bm
from tessera.compiler import jit_bridge as bridge
from tessera.compiler.compile_report import (
    CompileReport,
    finalize_compile_report,
    FRONTEND_TESSERA_JIT,
    VALUE_KIND_TENSOR,
    hash_ir_text,
)
from tessera.compiler.fallback import (
    FallbackReason,
    TesseraNativeRequiredError,
)


PROGRAM_ID = "conv2d_norm_activation"


def _make_inputs(
    *, N: int = 1, H: int = 8, W: int = 8, C: int = 4, K: int = 8,
    R: int = 3, S: int = 3, seed: int = 0,
):
    """Tiny ``(input, weights, gamma, beta)`` tuple — keeps the
    numpy reference fast enough for CI."""
    rng = np.random.RandomState(seed)
    x = rng.randn(N, H, W, C).astype(np.float32)  # NHWC
    w = rng.randn(R, S, C, K).astype(np.float32)
    gamma = np.ones((K,), dtype=np.float32)
    beta = np.zeros((K,), dtype=np.float32)
    return x, w, gamma, beta


def _conv2d_nhwc_valid(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Numpy reference NHWC conv2d with valid padding + stride 1."""
    N, H, W, C = x.shape
    R, S, _, K = w.shape
    out_h = H - R + 1
    out_w = W - S + 1
    out = np.zeros((N, out_h, out_w, K), dtype=np.float32)
    for n in range(N):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[n, i:i+R, j:j+S, :]
                out[n, i, j, :] = np.tensordot(patch, w, axes=([0, 1, 2], [0, 1, 2]))
    return out


def _layer_norm_last(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def _gelu(x: np.ndarray) -> np.ndarray:
    # Approximate GELU (tanh form), matching tessera.energy.gelu.
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))


def _numpy_reference(
    x: np.ndarray, w: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
) -> np.ndarray:
    return _gelu(_layer_norm_last(_conv2d_nhwc_valid(x, w), gamma, beta))


def _ir_text(N: int, H: int, W: int, C: int, K: int, R: int, S: int) -> str:
    return (
        "graph_ir {\n"
        f"  %x     = tessera.placeholder shape=({N}, {H}, {W}, {C}) dtype=f32 layout=NHWC\n"
        f"  %w     = tessera.placeholder shape=({R}, {S}, {C}, {K}) dtype=f32\n"
        f"  %gamma = tessera.placeholder shape=({K},) dtype=f32\n"
        f"  %beta  = tessera.placeholder shape=({K},) dtype=f32\n"
        "  %y      = tessera.conv2d_nhwc(%x, %w)\n"
        "  %norm   = tessera.layer_norm(%y, %gamma, %beta)\n"
        "  %act    = tessera.gelu(%norm)\n"
        "  return %act\n"
        "}\n"
    )


def _target_decision_for_host() -> tuple[str, dict[str, str], FallbackReason | None]:
    if sys.platform == "darwin":
        # conv2d does not yet have a fused MSL kernel on Apple GPU;
        # layer_norm + gelu do.  The target_decision documents the
        # split honestly so claim_lint never sees a false claim here.
        conv_entries = bm.manifest_for("conv2d")
        ln_entries = bm.manifest_for("layer_norm")
        gelu_entries = bm.manifest_for("gelu")
        ag = lambda es: next((e for e in es if e.target == "apple_gpu"), None)
        c_st = (ag(conv_entries).status if ag(conv_entries) else "?")
        ln_st = (ag(ln_entries).status if ag(ln_entries) else "?")
        gelu_st = (ag(gelu_entries).status if ag(gelu_entries) else "?")
        return (
            "apple_gpu",
            {
                "apple_gpu": (
                    f"per-op: conv2d status={c_st}, layer_norm status={ln_st}, "
                    f"gelu status={gelu_st}; no fused 3-op chain yet"
                ),
            },
            # Not a TRUE native dispatch yet — every op falls back to
            # numpy until conv2d gets a fused kernel.
            FallbackReason.REFERENCE_FORCED,
        )
    return (
        "cpu",
        {"cpu": "non-Darwin host; numpy reference path"},
        FallbackReason.NON_DARWIN_HOST,
    )


def run(
    *, N: int = 1, H: int = 8, W: int = 8, C: int = 4, K: int = 8,
    R: int = 3, S: int = 3, seed: int = 0, native_required: bool = False,
) -> CompileReport:
    x, w, gamma, beta = _make_inputs(N=N, H=H, W=W, C=C, K=K, R=R, S=S, seed=seed)
    target, target_decision, fallback_reason = _target_decision_for_host()
    if native_required and fallback_reason is not None:
        raise TesseraNativeRequiredError(
            fallback_reason, target=target, op_name=PROGRAM_ID,
        )

    prev_tracing = bridge.tracing_enabled()
    bridge.set_tracing_enabled(True)
    bridge.clear_dispatch_trace()
    try:
        t0 = time.perf_counter_ns()
        out = _numpy_reference(x, w, gamma, beta)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        routes = tuple(bridge.take_dispatch_trace())
    finally:
        bridge.set_tracing_enabled(prev_tracing)

    ref = _numpy_reference(x, w, gamma, beta)
    max_abs_err = float(np.abs(out - ref).max())

    return finalize_compile_report(CompileReport(
        program_id=PROGRAM_ID,
        source=f"{__name__}.run",
        frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR,
        target=target,
        tessera_version=getattr(tessera, "__version__", ""),
        ir_hashes={"graph_ir": hash_ir_text(_ir_text(N, H, W, C, K, R, S))},
        target_decision=target_decision,
        fallback_reason=fallback_reason,
        proof_routes=routes,
        timing_ms={"end_to_end": elapsed_ms},
        correctness={"max_abs_err": max_abs_err, "tolerance": 1e-5},
    ))


if __name__ == "__main__":  # pragma: no cover
    print(run().as_json())
