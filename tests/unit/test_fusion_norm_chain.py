"""Apple GPU codegen M2 — norm_chain synthesis (non-matmul-rooted).

`norm(x [+ residual])` — the pre-norm transformer pattern (`rmsnorm(x + attn_out)`)
— fused into ONE synthesized MSL kernel instead of a separate elementwise add +
a separate norm dispatch. The reduction blocks (rmsnorm/layer_norm) are reused
verbatim from REDUCTION_OPS; only the row materialization differs from the
matmul-epilogue synthesizer. See docs/audit/backend/apple/APPLE_GPU_CODEGEN_PLAN.md
(M2).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import fusion as F


# ── synthesizer ─────────────────────────────────────────────────────────────
def test_layer_norm_added_to_reduction_ops():
    assert "layer_norm" in F.REDUCTION_OPS


@pytest.mark.parametrize("dtype,io", [("f32", "float"), ("f16", "half"),
                                      ("bf16", "bfloat")])
def test_synthesizer_emits_expected_io_type(dtype, io):
    region = F.NormChainRegion("rmsnorm", add_residual=True)
    src = F.synthesize_norm_chain_msl(region, dtype=dtype)
    assert f"device const {io}* X" in src
    assert "residual" in src                 # the add was fused in
    assert "float scores" in src             # fp32 accumulator regardless of io


def test_synthesizer_omits_residual_when_not_fused():
    src = F.synthesize_norm_chain_msl(F.NormChainRegion("layer_norm"))
    assert "residual" not in src


# ── runner (executes on Metal when available, else numpy reference) ──────────
_RNG = np.random.default_rng(3)


@pytest.mark.parametrize("norm", ["rmsnorm", "layer_norm"])
@pytest.mark.parametrize("residual", [False, True])
def test_run_norm_chain_matches_reference(norm, residual):
    region = F.NormChainRegion(norm, add_residual=residual)
    X = _RNG.standard_normal((8, 48)).astype(np.float32)
    R = _RNG.standard_normal((8, 48)).astype(np.float32) if residual else None
    out, ex = F.run_norm_chain_region(region, X, R)
    assert ex in ("metal_runtime", "reference")
    np.testing.assert_allclose(np.asarray(out), region.reference(X, R),
                               rtol=1e-5, atol=1e-5)


def test_unknown_norm_rejected():
    with pytest.raises(ValueError):
        F.NormChainRegion("not_a_norm")


# ── post-norm affine (γ weight + β bias — the real transformer norm) ─────────
@pytest.mark.parametrize("norm,weight,bias", [
    ("rmsnorm", True, False),       # Llama RMSNorm
    ("rmsnorm", True, True),
    ("layer_norm", True, True),     # LayerNorm with affine
])
@pytest.mark.parametrize("residual", [False, True])
def test_affine_norm_matches_reference(norm, weight, bias, residual):
    region = F.NormChainRegion(norm, add_residual=residual,
                               weight=weight, bias=bias)
    N = 48
    X = _RNG.standard_normal((8, N)).astype(np.float32)
    R = _RNG.standard_normal((8, N)).astype(np.float32) if residual else None
    G = _RNG.standard_normal((N,)).astype(np.float32) if weight else None
    B = _RNG.standard_normal((N,)).astype(np.float32) if bias else None
    out, ex = F.run_norm_chain_region(region, X, residual=R, gamma=G, beta=B)
    assert ex in ("metal_runtime", "reference")
    np.testing.assert_allclose(np.asarray(out),
                               region.reference(X, R, G, B), rtol=1e-5, atol=1e-5)


def test_msl_compile_error_is_surfaced():
    # The runtime used to swallow `newLibraryWithSource` failures (silent
    # fallback); now compile_msl_kernel records them via the last-error channel.
    import ctypes
    from tessera.compiler.fusion import _synth_norm_chain_symbol
    from tessera.runtime import _load_apple_gpu_runtime
    # Metal-gate: only meaningful when the GPU runtime is live.
    probe = F.NormChainRegion("rmsnorm", add_residual=True)
    X = _RNG.standard_normal((4, 8)).astype(np.float32)
    if F.run_norm_chain_region(probe, X, X)[1] != "metal_runtime":
        pytest.skip("Metal runtime not available")
    rt = _load_apple_gpu_runtime()
    rt.tessera_apple_gpu_clear_last_error()
    rt.tessera_apple_gpu_last_error_message.restype = ctypes.c_char_p
    sym = _synth_norm_chain_symbol()
    bad = b"#include <metal_stdlib>\nkernel void synth_norm_chain() { @@@ not msl }\n"
    fp = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out = np.zeros((4, 8), np.float32)
    rc = sym(bad, b"synth_norm_chain", fp(X), None, fp(out), 4, 8, 0,
             None, None, 0, 0)
    assert rc == 0
    msg = rt.tessera_apple_gpu_last_error_message().decode()
    assert "msl_compile" in msg and "error" in msg.lower(), msg


def test_affine_synthesizer_emits_gamma_beta_buffers():
    src = F.synthesize_norm_chain_msl(
        F.NormChainRegion("layer_norm", weight=True, bias=True))
    assert "gamma [[buffer(5)]]" in src
    assert "beta [[buffer(6)]]" in src
    assert "o *= float(gamma[n])" in src and "o += float(beta[n])" in src


# ── dtype breadth (f16 native; bf16 native-or-correct-reference) ────────────
def test_f16_norm_chain_runs_native_and_matches():
    ml = pytest.importorskip("ml_dtypes")  # noqa: F841 - parity with bf16 import
    region = F.NormChainRegion("rmsnorm", add_residual=True)
    X = _RNG.standard_normal((8, 32)).astype(np.float32)
    R = _RNG.standard_normal((8, 32)).astype(np.float32)
    out, ex = F.run_norm_chain_region(region, X.astype(np.float16),
                                      R.astype(np.float16))
    assert ex == "metal_runtime"               # native MSL `half`
    assert np.asarray(out).dtype == np.float16
    # f32-accumulated reduction → half-precision accurate vs the f32 reference.
    np.testing.assert_allclose(np.asarray(out).astype(np.float32),
                               region.reference(X, R), rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("norm", ["rmsnorm", "layer_norm"])
def test_bf16_norm_chain_is_native_and_correct(norm):
    # Native `bfloat` kernel (MSL 3.1, Apple7) — the store goes through an
    # explicit `ST(...)` cast since `bfloat` rejects implicit float→bfloat
    # assignment (the bug the surfaced MSL compile error revealed). Where Metal
    # is unavailable it falls back to the (correct) f32 reference cast to bf16.
    bf16 = pytest.importorskip("ml_dtypes").bfloat16
    region = F.NormChainRegion(norm, add_residual=True)
    X = _RNG.standard_normal((8, 32)).astype(np.float32)
    R = _RNG.standard_normal((8, 32)).astype(np.float32)
    # If the f32 kernel runs on Metal here, the bf16 one must too (native, not a
    # silent f32-emulation fallback) — that's the fix.
    _f32_out, f32_ex = F.run_norm_chain_region(region, X, R)
    out, ex = F.run_norm_chain_region(region, X.astype(bf16), R.astype(bf16))
    if f32_ex == "metal_runtime":
        assert ex == "metal_runtime", "bf16 norm_chain must run the native bfloat kernel"
    assert np.asarray(out).dtype == bf16
    np.testing.assert_allclose(np.asarray(out).astype(np.float32),
                               region.reference(X, R), rtol=8e-2, atol=8e-2)


# ── discoverer ──────────────────────────────────────────────────────────────
def test_discover_add_then_rmsnorm():
    ops = [F._Op("tessera.add", ("x", "r"), "h"),
           F._Op("tessera.rmsnorm", ("h",), "o")]
    (regs,) = F.discover_norm_chain_regions(ops)
    idxs, region, operands, out_v = regs
    assert idxs == [0, 1]
    assert region.norm == "rmsnorm" and region.add_residual and not region.weight
    assert operands == {"x": "x", "residual": "r"}
    assert out_v == "o"


def test_discover_post_norm_affine_weight_and_bias():
    # rmsnorm(x + r) * gamma + beta — the full transformer norm.
    ops = [F._Op("tessera.add", ("x", "r"), "h"),
           F._Op("tessera.rmsnorm", ("h",), "n"),
           F._Op("tessera.mul", ("n", "g"), "w"),
           F._Op("tessera.add", ("w", "b"), "o")]
    (regs,) = F.discover_norm_chain_regions(ops)
    idxs, region, operands, out_v = regs
    assert idxs == [0, 1, 2, 3]
    assert region.add_residual and region.weight and region.bias
    assert operands == {"x": "x", "residual": "r", "gamma": "g", "beta": "b"}
    assert out_v == "o"


def test_discover_weighted_norm_without_residual():
    # rmsnorm(x) * gamma — Llama RMSNorm (no residual); the weight is the win.
    ops = [F._Op("tessera.rmsnorm", ("x",), "n"),
           F._Op("tessera.mul", ("n", "g"), "o")]
    (regs,) = F.discover_norm_chain_regions(ops)
    _idxs, region, operands, _out = regs
    assert region.weight and not region.add_residual and not region.bias
    assert operands == {"x": "x", "gamma": "g"}


def test_discover_layer_norm_and_rmsnorm_safe_eps():
    ln = F.discover_norm_chain_regions(
        [F._Op("tessera.add", ("x", "r"), "h"),
         F._Op("tessera.layer_norm", ("h",), "o")])
    assert ln[0][1].norm == "layer_norm"
    safe = F.discover_norm_chain_regions(
        [F._Op("tessera.add", ("x", "r"), "h"),
         F._Op("tessera.rmsnorm_safe", ("h",), "o")])
    assert safe[0][1].norm == "rmsnorm" and safe[0][1].eps == 1e-6


def test_standalone_norm_is_not_discovered():
    # No preceding add and no post weight → left to the MPSGraph rowop lane.
    assert F.discover_norm_chain_regions([F._Op("tessera.rmsnorm", ("x",), "o")]) == []


def test_add_with_extra_consumer_not_fused():
    # The add result feeds the norm AND another op → not single-use → no fuse.
    ops = [F._Op("tessera.add", ("x", "r"), "h"),
           F._Op("tessera.rmsnorm", ("h",), "o"),
           F._Op("tessera.gelu", ("h",), "g")]
    assert F.discover_norm_chain_regions(ops) == []


def test_skip_excludes_matmul_claimed_ops():
    ops = [F._Op("tessera.add", ("x", "r"), "h"),
           F._Op("tessera.rmsnorm", ("h",), "o")]
    assert F.discover_norm_chain_regions(ops, skip={1}) == []


# ── end-to-end via @jit(apple_gpu) ──────────────────────────────────────────
def _prenorm_rms(x, r):
    return ts.ops.rmsnorm(ts.ops.add(x, r))


def _prenorm_ln(x, r):
    return ts.ops.layer_norm(ts.ops.add(x, r))


@pytest.mark.parametrize("fn,ref", [
    (_prenorm_rms, lambda h: h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-5)),
    (_prenorm_ln, lambda h: (h - h.mean(-1, keepdims=True)) /
        np.sqrt(((h - h.mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5)),
])
def test_jit_prenorm_matches_numpy(fn, ref):
    jfn = ts.jit(target="apple_gpu")(fn)
    x = _RNG.standard_normal((8, 32)).astype(np.float32)
    r = _RNG.standard_normal((8, 32)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(jfn(x, r)), ref(x + r),
                               rtol=1e-5, atol=1e-5)


def _llama_norm(x, r, g):
    # rmsnorm(x + residual) * weight — a Llama-style pre-norm with affine.
    return ts.ops.mul(ts.ops.rmsnorm(ts.ops.add(x, r)), g)


def test_jit_affine_prenorm_matches_numpy():
    jfn = ts.jit(target="apple_gpu")(_llama_norm)
    x = _RNG.standard_normal((8, 32)).astype(np.float32)
    r = _RNG.standard_normal((8, 32)).astype(np.float32)
    g = _RNG.standard_normal((32,)).astype(np.float32)
    h = x + r
    ref = (h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-5)) * g
    np.testing.assert_allclose(np.asarray(jfn(x, r, g)), ref, rtol=1e-4, atol=1e-4)
