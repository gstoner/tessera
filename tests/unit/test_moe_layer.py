"""Single-device top-k SwiGLU MoE layer (MegaMoE precursor, no comm).

nn.functional.moe_layer ties the existing router (route_tokens, top-k gating) to
the local SwiGLU expert-FFN block (ops.moe_swiglu_block) with the permute /
scatter-combine glue:

    router → top-k → gather tokens into expert order → moe_swiglu_block → combine

The heavy expert compute flows through the GPU grouped-GEMM lanes; routing /
permute / combine are host-side index math (data-dependent, like argmax).
"""

import numpy as np
import pytest

import tessera as ts
from tessera.nn import functional as F


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _ref_moe_layer(x, Wr, Wg, Wu, Wd, top_k, normalize):
    """Independent f64 reference: explicit per-token top-k expert combine."""
    x = x.astype(np.float64)
    T, N = x.shape[0], Wd.shape[2]
    probs = _softmax(x @ Wr.astype(np.float64))
    idx = np.argsort(probs, axis=1)[:, -top_k:][:, ::-1]            # (T, k) desc
    w = np.take_along_axis(probs, idx, axis=1)
    if normalize and top_k > 1:
        w = w / w.sum(axis=1, keepdims=True)
    out = np.zeros((T, N), np.float64)
    for t in range(T):
        for j in range(top_k):
            e = int(idx[t, j])
            h = _silu(x[t] @ Wg[e].astype(np.float64)) * (x[t] @ Wu[e].astype(np.float64))
            out[t] += w[t, j] * (h @ Wd[e].astype(np.float64))
    return out


def _inputs(seed, T=24, K=16, E=4, Fdim=20, N=12):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((T, K)).astype(np.float32),     # x
        rng.standard_normal((K, E)).astype(np.float32),     # W_router
        rng.standard_normal((E, K, Fdim)).astype(np.float32),  # W_gate
        rng.standard_normal((E, K, Fdim)).astype(np.float32),  # W_up
        rng.standard_normal((E, Fdim, N)).astype(np.float32),  # W_down
    )


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_matches_reference(top_k):
    x, Wr, Wg, Wu, Wd = _inputs(top_k)
    got = np.asarray(F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=top_k))
    ref = _ref_moe_layer(x, Wr, Wg, Wu, Wd, top_k, normalize=True)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_top1_no_normalize_matches_reference():
    x, Wr, Wg, Wu, Wd = _inputs(7)
    got = np.asarray(F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=2, normalize_weights=False))
    ref = _ref_moe_layer(x, Wr, Wg, Wu, Wd, top_k=2, normalize=False)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_single_expert_reduces_to_dense_swiglu():
    # E=1: every token routes to the one expert with weight 1 → dense SwiGLU.
    x, Wr, Wg, Wu, Wd = _inputs(3, E=1)
    got = np.asarray(F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=1))
    dense = np.asarray(ts.ops.swiglu(x, Wg[0], Wu[0], Wd[0]))
    np.testing.assert_allclose(got, dense, rtol=1e-5, atol=1e-5)


def test_output_shape():
    x, Wr, Wg, Wu, Wd = _inputs(11, T=30, N=12)
    out = np.asarray(F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=2))
    assert out.shape == (30, 12)


def test_capacity_drop_zeroes_overflow_tokens():
    # Tiny capacity_factor forces overflow drops; dropped slots contribute 0,
    # so the layer still produces a finite (T, N) output without error.
    x, Wr, Wg, Wu, Wd = _inputs(5, T=32, E=4)
    out = np.asarray(F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=2, capacity_factor=1.0))
    assert out.shape == (32, 12)
    assert np.isfinite(out).all()


def test_quantized_layer_within_budget():
    x, Wr, Wg, Wu, Wd = _inputs(9, K=64, Fdim=32, N=16)
    ref = _ref_moe_layer(x, Wr, Wg, Wu, Wd, top_k=2, normalize=True)
    got = np.asarray(F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=2, quant="fp8_e4m3"))
    rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9)
    assert rel < 0.15, f"fp8 moe_layer rel {rel:.4f}"


@pytest.mark.parametrize("kind", ["masked", "k_grouped"])
def test_rejects_unsupported_kinds(kind):
    x, Wr, Wg, Wu, Wd = _inputs(13)
    with pytest.raises(NotImplementedError):
        F.moe_layer(x, Wr, Wg, Wu, Wd, top_k=2, kind=kind)


def test_exposed_on_nn_namespace():
    assert ts.nn.moe_layer is F.moe_layer
    assert "moe_layer" in F.__all__
