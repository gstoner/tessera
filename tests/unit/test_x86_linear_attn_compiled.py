"""Linear-attention backbone on x86 AVX-512 (P10 scan-family partner of
S_SERIES_GAP_CLOSURE_PLAN) — linear_attn / power_attn / retention via the
QUADRATIC-PARALLEL form O = (φ(Q)·φ(K)ᵀ ⊙ causal ⊙ decay) @ V: two batched
GEMMs on the AVX-512 f32 GEMM, with the feature map / mask / decay on the
host. The AVX-512 partner to the ROCm linear_attn lane. Reachable via
`compiler_path="x86_linear_attn_compiled"`. Validated vs the numpy references
(tessera.ops.linear_attn / power_attn / retention). Skip-clean:
libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_linear_attn_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["q", "k", "v"],
                 "kwargs": kwargs}]})


def _run(rt, op, q, k, v, **kwargs):
    res = rt.launch(_art(rt, op, kwargs), (q, k, v))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_linear_attn_compiled"
    return np.asarray(res["output"])


_RNG = np.random.default_rng(53)


def _qkv(B, H, S, Dqk, Dv):
    return (_RNG.standard_normal((B, H, S, Dqk)).astype(np.float32),
            _RNG.standard_normal((B, H, S, Dqk)).astype(np.float32),
            _RNG.standard_normal((B, H, S, Dv)).astype(np.float32))


@pytest.mark.parametrize("fmap", ["elu", "relu", "identity", "polynomial_2"])
def test_linear_attn_causal_feature_maps(fmap):
    rt = _rt_or_skip()
    q, k, v = _qkv(2, 2, 10, 8, 6)
    got = _run(rt, "tessera.linear_attn", q, k, v,
               feature_map=fmap, causal=True)
    ref = tessera.ops.linear_attn(q, k, v, feature_map=fmap, causal=True)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-3, atol=1e-3)


def test_linear_attn_non_causal():
    rt = _rt_or_skip()
    q, k, v = _qkv(2, 3, 9, 8, 8)
    got = _run(rt, "tessera.linear_attn", q, k, v,
               feature_map="elu", causal=False)
    ref = tessera.ops.linear_attn(q, k, v, feature_map="elu", causal=False)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-3, atol=1e-3)


def test_linear_attn_with_decay():
    rt = _rt_or_skip()
    q, k, v = _qkv(2, 2, 8, 6, 6)
    decay = (0.85 + 0.1 * _RNG.random((2, 2, 8))).astype(np.float32)  # (0,1)
    # decay/log_g ride in the op kwargs (JSON-hashed for the artifact key) —
    # pass plain nested lists, not numpy arrays.
    got = _run(rt, "tessera.linear_attn", q, k, v,
               feature_map="identity", causal=True, decay=decay.tolist())
    ref = tessera.ops.linear_attn(q, k, v, feature_map="identity",
                                  causal=True, decay=decay)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=2e-3, atol=2e-3)


def test_linear_attn_decay_long_context_finite():
    """A long run of a small decay underflows cumprod to 0; the c[t]/c[r] ratio
    would then be inf/NaN and the causal mask's inf*0 would contaminate valid
    rows. The log-space decay matrix must stay finite (regression)."""
    rt = _rt_or_skip()
    s = 1200  # 0.5**1200 underflows float64 cumprod to 0
    q, k, v = _qkv(1, 1, s, 4, 4)
    decay = np.full((1, 1, s), 0.5, np.float32)
    got = _run(rt, "tessera.linear_attn", q, k, v,
               feature_map="identity", causal=True, decay=decay.tolist())
    assert np.all(np.isfinite(got)), "decay matrix underflowed to NaN/inf"


def test_linear_attn_non_causal_ignores_decay():
    """The reference non-causal branch folds to a single global φQ @ Σ(φKᵀV)
    and IGNORES decay — the x86 lane must return the same as no-decay, not
    silently apply a c[t]/c[r] matrix to the full (unmasked) score matrix."""
    rt = _rt_or_skip()
    q, k, v = _qkv(2, 2, 9, 6, 6)
    decay = (0.7 + 0.2 * _RNG.random((2, 2, 9))).astype(np.float32)
    with_decay = _run(rt, "tessera.linear_attn", q, k, v,
                      feature_map="elu", causal=False, decay=decay.tolist())
    no_decay = _run(rt, "tessera.linear_attn", q, k, v,
                    feature_map="elu", causal=False)
    np.testing.assert_allclose(with_decay, no_decay, rtol=1e-6, atol=1e-6)
    ref = tessera.ops.linear_attn(q, k, v, feature_map="elu", causal=False,
                                  decay=decay)
    np.testing.assert_allclose(with_decay, np.asarray(ref),
                               rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("deg", [2, 3])
def test_power_attn(deg):
    rt = _rt_or_skip()
    q, k, v = _qkv(1, 2, 8, 6, 6)
    got = _run(rt, "tessera.power_attn", q, k, v, deg=deg, causal=True)
    ref = tessera.ops.power_attn(q, k, v, deg=deg, causal=True)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=2e-3, atol=2e-3)


def test_retention_no_decay():
    rt = _rt_or_skip()
    q, k, v = _qkv(2, 2, 8, 6, 6)
    got = _run(rt, "tessera.retention", q, k, v, deg=2, causal=True)
    ref = tessera.ops.retention(q, k, v, deg=2, causal=True)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=2e-3, atol=2e-3)


def test_retention_with_log_g():
    rt = _rt_or_skip()
    q, k, v = _qkv(2, 2, 10, 6, 6)
    log_g = (-0.1 * _RNG.random((2, 2, 10))).astype(np.float32)  # <= 0
    got = _run(rt, "tessera.retention", q, k, v, deg=2,
               log_g=log_g.tolist(), causal=True)
    ref = tessera.ops.retention(q, k, v, deg=2, log_g=log_g, causal=True)
    np.testing.assert_allclose(got, np.asarray(ref), rtol=2e-3, atol=2e-3)
