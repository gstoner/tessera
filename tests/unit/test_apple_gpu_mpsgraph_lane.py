"""Apple GPU MetalPerformanceShadersGraph (MPSGraph) lane (2026-05-29).

Covers the Tier-1 activation / normalization ops that now execute on the
Apple GPU through a single parametrized MPSGraph runner per shape class
(see ``apple_gpu_runtime.mm`` and the dispatchers in ``runtime.py``):

  * unary elementwise: relu / sigmoid / tanh / softplus / silu / exp / log /
    sqrt / rsqrt / neg / abs
  * row ops over the last axis: layer_norm / rmsnorm / rmsnorm_safe /
    log_softmax
  * binary: silu_mul (silu(a) * b)

It also covers the f16/bf16 (+ large-N) completion of the fused MLP /
attention chains, which now compose the GPU matmul with an MPSGraph
epilogue instead of falling back to host numpy.

These ops adopt MPSGraph rather than hand-written MSL so the long tail is
covered by one lane. Compute is fp32 internally (inputs cast up, outputs
cast down) matching the backend's fp16-I/O + fp32-accumulator convention;
bf16 upcasts host-side. Where Metal is unavailable the dispatchers degrade
to a numpy reference, so the correctness assertions hold on any platform.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as R

DARWIN = sys.platform == "darwin"


# ── numpy references ──────────────────────────────────────────────────────
def _silu(x):
    return x / (1.0 + np.exp(-x))


_UNARY_REFS = {
    "relu": lambda v: np.maximum(0.0, v),
    "sigmoid": lambda v: 1.0 / (1.0 + np.exp(-v)),
    "tanh": np.tanh,
    "softplus": lambda v: np.maximum(v, 0.0) + np.log1p(np.exp(-np.abs(v))),
    "silu": _silu,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "rsqrt": lambda v: 1.0 / np.sqrt(v),
    "neg": lambda v: -v,
    "abs": np.abs,
}
# ops needing positive inputs (log / sqrt / rsqrt)
_POSITIVE = {"log", "sqrt", "rsqrt"}


@pytest.mark.parametrize("op_name", sorted(_UNARY_REFS))
def test_mpsgraph_unary_matches_reference(op_name):
    rng = np.random.RandomState(0)
    # N=300 (>256) proves the per-thread N<=256 MSL limit does not apply.
    x = rng.randn(4, 300).astype(np.float32)
    if op_name in _POSITIVE:
        x = np.abs(x) + 0.1
    out = R._apple_gpu_dispatch_unary(f"tessera.{op_name}", [x], np)
    np.testing.assert_allclose(np.asarray(out), _UNARY_REFS[op_name](x),
                               rtol=1e-5, atol=1e-5)


def test_mpsgraph_unary_f16():
    rng = np.random.RandomState(1)
    x = (rng.randn(8, 128) * 0.5).astype(np.float16)
    out = np.asarray(R._apple_gpu_dispatch_unary("tessera.silu", [x], np))
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32),
                               _silu(x.astype(np.float32)), rtol=2e-2, atol=2e-3)


def test_mpsgraph_unary_high_rank_preserves_shape():
    rng = np.random.RandomState(2)
    x = rng.randn(2, 3, 64).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_unary("tessera.relu", [x], np))
    assert out.shape == (2, 3, 64)
    np.testing.assert_allclose(out, np.maximum(0.0, x), rtol=1e-6, atol=1e-6)


def test_mpsgraph_layer_norm():
    rng = np.random.RandomState(3)
    x = rng.randn(8, 512).astype(np.float32)  # N>256
    out = np.asarray(R._apple_gpu_dispatch_rowop("tessera.layer_norm", [x], {}, np))
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    np.testing.assert_allclose(out, (x - mu) / np.sqrt(var + 1e-5),
                               rtol=1e-4, atol=1e-5)


def test_mpsgraph_rmsnorm_and_safe_eps():
    rng = np.random.RandomState(4)
    x = rng.randn(8, 384).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_rowop("tessera.rmsnorm", [x], {}, np))
    np.testing.assert_allclose(
        out, x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5),
        rtol=1e-4, atol=1e-5)
    # rmsnorm_safe defaults to eps=1e-6
    out_safe = np.asarray(
        R._apple_gpu_dispatch_rowop("tessera.rmsnorm_safe", [x], {}, np))
    np.testing.assert_allclose(
        out_safe, x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6),
        rtol=1e-4, atol=1e-5)


def test_mpsgraph_log_softmax():
    rng = np.random.RandomState(5)
    x = rng.randn(8, 500).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_rowop("tessera.log_softmax", [x], {}, np))
    m = x.max(-1, keepdims=True)
    ref = (x - m) - np.log(np.exp(x - m).sum(-1, keepdims=True))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_mpsgraph_silu_mul_semantics():
    # silu_mul(a, b) must be silu(a) * b (NOT a * silu(b)).
    rng = np.random.RandomState(6)
    a = rng.randn(8, 300).astype(np.float32)
    b = rng.randn(8, 300).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_silu_mul([a, b], np))
    np.testing.assert_allclose(out, _silu(a) * b, rtol=1e-5, atol=1e-5)


# ── fused-chain f16/bf16 + large-N completion (composes GPU matmul + MPSGraph) ──
def _gelu(s):
    return 0.5 * s * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (s + 0.044715 * s ** 3)))


def test_matmul_gelu_f16_large_n_no_nan():
    rng = np.random.RandomState(7)
    a = (rng.randn(16, 64) * 0.5).astype(np.float16)
    b = (rng.randn(64, 512) * 0.5).astype(np.float16)  # N=512 > 256
    out = np.asarray(R._apple_gpu_dispatch_matmul_gelu([a, b], np))
    assert out.dtype == np.float16
    assert not np.isnan(out.astype(np.float32)).any()
    ref = _gelu(a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=5e-2, atol=5e-2)


def test_matmul_gelu_f32_large_activation_no_nan():
    # Regression guard: the MPSGraph gelu epilogue must not overflow on large
    # matmul outputs (the hand-written MSL gelu kernel does — see the
    # spawned follow-up). N=2048 forces the compose path.
    rng = np.random.RandomState(8)
    a = rng.randn(16, 64).astype(np.float32)
    b = rng.randn(64, 2048).astype(np.float32)
    out = np.asarray(R._apple_gpu_dispatch_matmul_gelu([a, b], np))
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, _gelu(a @ b), rtol=1e-4, atol=1e-4)


def test_matmul_rmsnorm_f16_large_n():
    rng = np.random.RandomState(9)
    a = (rng.randn(16, 64) * 0.5).astype(np.float16)
    b = (rng.randn(64, 512) * 0.5).astype(np.float16)
    out = np.asarray(R._apple_gpu_dispatch_matmul_rmsnorm([a, b], 1e-5, np))
    s = a.astype(np.float32) @ b.astype(np.float32)
    ref = s / np.sqrt((s * s).mean(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=5e-2, atol=5e-2)


def test_matmul_softmax_f16_large_n():
    rng = np.random.RandomState(10)
    a = (rng.randn(16, 64) * 0.5).astype(np.float16)
    b = (rng.randn(64, 512) * 0.5).astype(np.float16)  # >256: native tiled f16
    out = np.asarray(R._apple_gpu_dispatch_matmul_softmax([a, b], np))
    s = a.astype(np.float32) @ b.astype(np.float32)
    e = np.exp(s - s.max(-1, keepdims=True))
    np.testing.assert_allclose(out.astype(np.float32), e / e.sum(-1, keepdims=True),
                               rtol=5e-2, atol=5e-3)


# ── native single-kernel f16/bf16 fused chains (N<=256) ─────────────────────
def test_matmul_gelu_f16_small_n_native_fused():
    # N=128 <= 256: routes through tessera_apple_gpu_matmul_gelu_f16 (one
    # dispatch) rather than composing GPU matmul + MPSGraph gelu.
    rng = np.random.RandomState(20)
    a = (rng.randn(8, 32) * 0.5).astype(np.float16)
    b = (rng.randn(32, 128) * 0.5).astype(np.float16)
    out = np.asarray(R._apple_gpu_dispatch_matmul_gelu([a, b], np))
    assert out.dtype == np.float16
    assert not np.isnan(out.astype(np.float32)).any()
    ref = _gelu(a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=5e-2, atol=5e-2)


def test_matmul_rmsnorm_f16_small_n_native_fused():
    rng = np.random.RandomState(21)
    a = (rng.randn(8, 32) * 0.5).astype(np.float16)
    b = (rng.randn(32, 128) * 0.5).astype(np.float16)
    out = np.asarray(R._apple_gpu_dispatch_matmul_rmsnorm([a, b], 1e-5, np))
    assert out.dtype == np.float16
    s = a.astype(np.float32) @ b.astype(np.float32)
    ref = s / np.sqrt((s * s).mean(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=5e-2, atol=5e-2)


def test_matmul_gelu_bf16_small_n_native_fused():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    rng = np.random.RandomState(22)
    a = (rng.randn(8, 32) * 0.5).astype(bf16)
    b = (rng.randn(32, 128) * 0.5).astype(bf16)
    out = np.asarray(R._apple_gpu_dispatch_matmul_gelu([a, b], np))
    assert out.dtype == bf16
    ref = _gelu(a.astype(np.float32) @ b.astype(np.float32))
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=8e-2, atol=8e-2)


def test_matmul_rmsnorm_bf16_small_n_native_fused():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    rng = np.random.RandomState(23)
    a = (rng.randn(8, 32) * 0.5).astype(bf16)
    b = (rng.randn(32, 128) * 0.5).astype(bf16)
    out = np.asarray(R._apple_gpu_dispatch_matmul_rmsnorm([a, b], 1e-5, np))
    assert out.dtype == bf16
    s = a.astype(np.float32) @ b.astype(np.float32)
    ref = s / np.sqrt((s * s).mean(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=8e-2, atol=8e-2)


def test_matmul_softmax_bf16_large_n_native_tiled():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    bf16 = ml_dtypes.bfloat16
    rng = np.random.RandomState(24)
    a = (rng.randn(16, 64) * 0.5).astype(bf16)
    b = (rng.randn(64, 512) * 0.5).astype(bf16)  # >256: native tiled bf16
    out = np.asarray(R._apple_gpu_dispatch_matmul_softmax([a, b], np))
    assert out.dtype == bf16
    s = a.astype(np.float32) @ b.astype(np.float32)
    e = np.exp(s - s.max(-1, keepdims=True))
    np.testing.assert_allclose(out.astype(np.float32), e / e.sum(-1, keepdims=True),
                               rtol=8e-2, atol=8e-3)


# ── on-device dispatch gate via @jit (source must be a real module) ─────────
@ts.jit(target="apple_gpu")
def _jit_silu(x):
    return ts.ops.silu(x)


@ts.jit(target="apple_gpu")
def _jit_layer_norm(x):
    return ts.ops.layer_norm(x)


def test_jit_tier1_ops_are_runtime_executable():
    for fn in (_jit_silu, _jit_layer_norm):
        meta = fn.runtime_artifact().metadata
        assert meta["compiler_path"] == "apple_gpu_mps"
        assert meta["runtime_status"] == "ready"
        assert meta["execution_mode"] in ("metal_runtime", "metal_artifact")


@pytest.mark.skipif(not DARWIN, reason="metal_runtime dispatch is Darwin-only")
def test_jit_tier1_ops_metal_runtime_on_darwin():
    x = np.random.RandomState(11).randn(8, 64).astype(np.float32)
    out = np.asarray(_jit_silu(x))
    np.testing.assert_allclose(out, _silu(x), rtol=1e-5, atol=1e-5)
    assert _jit_silu.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


def test_synthesized_epilogue_symbols_are_exported():
    """Catalog retirement (Optimizing-Compiler Plan F2): the per-kernel f16/bf16
    matmul_{gelu,rmsnorm,softmax}(_tiled) symbols are RETIRED — the synthesized
    epilogue symbols (f32 stack + tiled + native f16) subsume the whole family.
    The dispatchers now prefer the synthesizer over the compose path."""
    rt = R._load_apple_gpu_runtime()
    for sym in (
        "tessera_apple_gpu_synth_matmul_epilogue_f32",
        "tessera_apple_gpu_synth_matmul_epilogue_tiled_f32",
        "tessera_apple_gpu_synth_matmul_epilogue_f16",
    ):
        assert hasattr(rt, sym), sym
    for retired in (
        "tessera_apple_gpu_matmul_softmax_tiled_f16",
        "tessera_apple_gpu_matmul_gelu_f16",
        "tessera_apple_gpu_matmul_rmsnorm_bf16",
    ):
        assert not hasattr(rt, retired), f"retired kernel still present: {retired}"


@pytest.mark.skipif(not DARWIN, reason="Metal device required")
def test_runtime_reports_metal_available():
    rt = R._load_apple_gpu_runtime()
    assert rt.tessera_apple_gpu_runtime_has_metal() == 1
    # The new MPSGraph symbols must be present in the compiled runtime.
    for sym in ("tessera_apple_gpu_mpsgraph_unary_f32",
                "tessera_apple_gpu_layer_norm_f32",
                "tessera_apple_gpu_rmsnorm_gpu_f32",
                "tessera_apple_gpu_log_softmax_f32",
                "tessera_apple_gpu_mpsgraph_softmax_f32",
                "tessera_apple_gpu_mpsgraph_binary_f32",
                "tessera_apple_gpu_mpsgraph_cache_size"):
        assert hasattr(rt, sym), sym


# ── MSL gelu large-activation regression (the spawned fix) ──────────────────
@pytest.mark.parametrize("scale", [1.0, 10.0, 50.0, 200.0])
def test_gelu_no_nan_for_large_activations(scale):
    # The hand-written MSL gelu kernel used to NaN for |x| >~ 16 (tanh
    # overflow); the argument is now clamped. Exercise the single-op gelu
    # dispatch (which uses the MSL kernel on Darwin).
    rng = np.random.RandomState(12)
    x = (rng.randn(8, 64).astype(np.float32)) * scale
    out = np.asarray(R._apple_gpu_dispatch_gelu("tessera.gelu", [x], np))
    assert not np.isnan(out).any(), f"gelu NaN at scale={scale}"
    # tanh saturates, so the clamped result matches the clamped reference.
    t = np.clip(0.7978845608028654 * (x + 0.044715 * x ** 3), -30.0, 30.0)
    ref = 0.5 * x * (1.0 + np.tanh(t))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# ── MPSGraph executable caching ─────────────────────────────────────────────
@pytest.mark.skipif(not DARWIN, reason="MPSGraph cache is Darwin-only")
def test_mpsgraph_graph_cache_reuses_across_calls():
    """Repeated dispatches with the same (op, dtype, shape) signature reuse a
    single cached MPSGraph; new signatures add one entry each."""
    rt = R._load_apple_gpu_runtime()
    cache_size = rt.tessera_apple_gpu_mpsgraph_cache_size
    cache_size.restype = ctypes.c_int32

    rng = np.random.RandomState(13)
    # Distinctive flat sizes no other test uses, so the (op, dtype, shape)
    # signatures are guaranteed fresh and the deltas are exact.
    x = rng.randn(1, 771).astype(np.float32)

    # Warm once so the signature is definitely present, then measure that
    # repeats add nothing.
    R._apple_gpu_dispatch_unary("tessera.silu", [x], np)
    base = cache_size()
    for _ in range(6):
        R._apple_gpu_dispatch_unary("tessera.silu", [x], np)
    assert cache_size() == base, "repeated same-signature calls must reuse the graph"

    # A different op at the same shape adds exactly one graph.
    for _ in range(3):
        R._apple_gpu_dispatch_unary("tessera.relu", [x], np)
    assert cache_size() == base + 1

    # A different shape adds exactly one more graph.
    x2 = rng.randn(1, 387).astype(np.float32)
    for _ in range(3):
        R._apple_gpu_dispatch_unary("tessera.silu", [x2], np)
    assert cache_size() == base + 2
