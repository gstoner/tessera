"""Phase 3 Sprint 3.3 — Metal-4 resident-weight MLP decode session
(docs/spec/PRODUCTION_COMPILER_PLAN.md).

`Mtl4MlpSession` keeps the weight `W[K,N]` resident on the GPU and runs
`Y = act(X @ W + bias)` per decode step — the per-call cost is just the dispatch
(re-uploading W is what otherwise kills small-M decode). X is f16/bf16, Y is f32.

Oracle: an f16-rounded, f32-accumulate numpy reference (matches the kernel's
half-input / float-accumulator convention).

Needs macOS 26+ / the MTL4 ML matrix-unit lane; skips cleanly otherwise.
"""

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb

pytestmark = pytest.mark.skipif(
    not agb.mtl4_mlp_available(),
    reason="MTL4 ML lane unavailable (needs macOS 26+)",
)


def _act_ref(z, act):
    if act == "none":
        return z
    if act == "relu":
        return np.maximum(z, 0.0)
    if act == "silu":
        return z / (1.0 + np.exp(-z))
    if act == "gelu":  # tanh approximation, matching the kernel epilogue
        return 0.5 * z * (1.0 + np.tanh(0.7978845608028654 * (z + 0.044715 * z**3)))
    raise ValueError(act)


def _oracle(x, w, bias, act):
    xf = x.astype(np.float16).astype(np.float32)
    wf = w.astype(np.float16).astype(np.float32)
    z = xf @ wf + (0.0 if bias is None else bias)
    return _act_ref(z, act)


@pytest.mark.parametrize("act", ["none", "relu", "silu", "gelu"])
@pytest.mark.parametrize("M,K,N", [(4, 16, 8), (1, 32, 32), (8, 64, 16)])
def test_mlp_session_matches_oracle(act, M, K, N):
    rng = np.random.default_rng(hash((act, M, K, N)) & 0xFFFF)
    w = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
    bias = rng.standard_normal((N,)).astype(np.float32)
    x = rng.standard_normal((M, K)).astype(np.float32)
    with agb.Mtl4MlpSession(w, bias=bias, act=act) as s:
        assert s.shape == (K, N)
        y = s.run(x)
    ref = _oracle(x, w, bias, act)
    tol = 2e-2 if act == "gelu" else 3e-3
    np.testing.assert_allclose(y, ref, rtol=tol, atol=tol)


def test_mlp_session_no_bias():
    rng = np.random.default_rng(11)
    K, N = 16, 8
    w = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
    x = rng.standard_normal((4, K)).astype(np.float32)
    with agb.Mtl4MlpSession(w, act="relu") as s:
        y = s.run(x)
    np.testing.assert_allclose(y, _oracle(x, w, None, "relu"), rtol=3e-3, atol=3e-3)


def test_mlp_session_resident_weight_reused_across_decode_steps():
    """The weight stays resident: many decode steps (varying M, varying X) reuse
    the same session and each matches its own oracle."""
    rng = np.random.default_rng(7)
    K, N = 32, 16
    w = (rng.standard_normal((K, N)) / np.sqrt(K)).astype(np.float32)
    bias = rng.standard_normal((N,)).astype(np.float32)
    with agb.Mtl4MlpSession(w, bias=bias, act="silu") as s:
        for step_m in (1, 1, 4, 1, 8):
            x = rng.standard_normal((step_m, K)).astype(np.float32)
            y = s.run(x)
            np.testing.assert_allclose(
                y, _oracle(x, w, bias, "silu"), rtol=3e-3, atol=3e-3)


def test_mlp_session_envelope():
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((16, 8))).astype(np.float32)
    with pytest.raises(agb.AppleGpuError):
        agb.Mtl4MlpSession(w, act="not_an_act")  # bad activation
    with agb.Mtl4MlpSession(w, act="none") as s:
        with pytest.raises(agb.AppleGpuError):
            s.run(np.ones((4, 7), np.float32))  # wrong inner dim (K=16 expected)
