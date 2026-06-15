"""Track L (L1.1) — the genuine gated delta rule on Metal, DESIL-gated.

`tessera_apple_gpu_gated_delta_rule_f32` (recurrent, with the `(v_t − α·v̂_t)`
erase) must equal the numpy reference `gated_delta_rule_recurrent`.  This is the
*true* DeltaNet recurrence — unlike linear attention it carries an erase term and
cannot be written as a masked `(QKᵀ⊙mask)@V`, so it runs as a per-(b,h)
sequential MSL scan, not a composed bmm.

Conditioning note: the delta rule is only well-behaved when keys are
L2-normalized, so `β·k·kᵀ` has eigenvalue `β < 1` and `(I − β k kᵀ)` is a
contraction.  With unnormalized keys (`‖k‖²≫1`) the recurrence expands and f32
diverges from f64 — that is genuine ill-conditioning, not a kernel defect, so
these oracles use normalized keys (the regime real models use).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera.stdlib import delta_rule as dr

_GPU = agb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime unavailable")

_B, _H, _S, _D = 2, 3, 16, 16


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _qkv(seed=0, dv=_D):
    rng = np.random.default_rng(seed)
    # L2-normalized Q/K (the contraction regime the rule is used in).
    Q = _normalize(rng.standard_normal((_B, _H, _S, _D))).astype(np.float32)
    K = _normalize(rng.standard_normal((_B, _H, _S, _D))).astype(np.float32)
    V = rng.standard_normal((_B, _H, _S, dv)).astype(np.float32)
    return Q, K, V


def _sig(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def test_symbol_is_registered():
    """L1.1 lock: the genuine-delta kernel is in the runtime ABI."""
    assert hasattr(agb, "gpu_gated_delta_rule")


@gpu
def test_metal_equals_numpy_true_delta():
    """Headline DESIL oracle: Metal genuine delta rule ≡ numpy reference."""
    Q, K, V = _qkv(1)
    o_gpu = dr.gated_delta_rule_recurrent(Q, K, V, backend="apple_gpu")
    o_ref = dr.gated_delta_rule_recurrent(Q, K, V)
    np.testing.assert_allclose(np.asarray(o_gpu), o_ref, rtol=1e-4, atol=1e-4)


@gpu
def test_metal_equals_numpy_with_beta_and_decay():
    Q, K, V = _qkv(2)
    beta = _sig(np.random.default_rng(3).standard_normal((_B, _H, _S)))
    decay = _sig(np.random.default_rng(4).standard_normal((_B, _H, _S)) + 2.0)
    o_gpu = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay,
                                          backend="apple_gpu")
    o_ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay)
    np.testing.assert_allclose(np.asarray(o_gpu), o_ref, rtol=1e-4, atol=1e-4)


@gpu
def test_metal_equals_numpy_with_output_gate():
    Q, K, V = _qkv(5)
    gate = np.random.default_rng(6).standard_normal((_B, _H, _S, _D)).astype(np.float32)
    o_gpu = dr.gated_delta_rule_recurrent(Q, K, V, gate=gate, backend="apple_gpu")
    o_ref = dr.gated_delta_rule_recurrent(Q, K, V, gate=gate)
    np.testing.assert_allclose(np.asarray(o_gpu), o_ref, rtol=1e-4, atol=1e-4)


@gpu
def test_metal_erase_off_equals_existing_linear_reference():
    """erase=False on Metal ≡ the shipped (linear-attention) gated_deltanet."""
    from tessera import ops
    Q, K, V = _qkv(7)
    beta = _sig(np.random.default_rng(8).standard_normal((_B, _H, _S)))
    o_gpu = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, erase=False,
                                          backend="apple_gpu")
    o_existing = np.asarray(ops.gated_deltanet(Q, K, V, beta=beta), np.float64)
    np.testing.assert_allclose(np.asarray(o_gpu), o_existing, rtol=1e-4, atol=1e-4)


@gpu
def test_metal_equals_chunked_prefill():
    """End-to-end Track L closure: Metal recurrent ≡ the L2 chunked UT-transform
    (both are the genuine rule, reached by independent routes)."""
    Q, K, V = _qkv(9)
    beta = _sig(np.random.default_rng(10).standard_normal((_B, _H, _S)))
    decay = _sig(np.random.default_rng(11).standard_normal((_B, _H, _S)) + 2.0)
    o_gpu = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay,
                                          backend="apple_gpu")
    o_chunk = dr.gated_delta_rule_chunked(Q, K, V, beta=beta, decay=decay, chunk_size=8)
    np.testing.assert_allclose(np.asarray(o_gpu), o_chunk, rtol=1e-4, atol=1e-4)


@gpu
def test_metal_non_square_head_dims():
    """D_qk ≠ D_v (state is rectangular)."""
    Q, K, V = _qkv(12, dv=8)
    o_gpu = dr.gated_delta_rule_recurrent(Q, K, V, backend="apple_gpu")
    o_ref = dr.gated_delta_rule_recurrent(Q, K, V)
    np.testing.assert_allclose(np.asarray(o_gpu), o_ref, rtol=1e-4, atol=1e-4)


# ── L2.1 — chunk-parallel UT-transform on Metal ──────────────────────────────
def test_chunked_symbol_is_registered():
    assert hasattr(agb, "gpu_gated_delta_rule_chunked")


@gpu
@pytest.mark.parametrize("chunk", [1, 4, 8, 16, 32])
def test_metal_chunked_equals_numpy(chunk):
    """L2.1 headline: the on-device chunk UT-transform ≡ numpy recurrent, across
    chunk sizes (S=20 exercises a partial last chunk)."""
    rng = np.random.default_rng(20)
    B, H, S, D = 2, 3, 20, 16
    Q = _normalize(rng.standard_normal((B, H, S, D))).astype(np.float32)
    K = _normalize(rng.standard_normal((B, H, S, D))).astype(np.float32)
    V = rng.standard_normal((B, H, S, D)).astype(np.float32)
    beta = _sig(rng.standard_normal((B, H, S)))
    decay = _sig(rng.standard_normal((B, H, S)) + 2.0)
    o_gpu = dr.gated_delta_rule_chunked(Q, K, V, beta=beta, decay=decay,
                                        chunk_size=chunk, backend="apple_gpu")
    o_ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay)
    np.testing.assert_allclose(np.asarray(o_gpu), o_ref, rtol=1e-4, atol=1e-4)


@gpu
def test_metal_chunked_equals_metal_recurrent():
    """The two Metal kernels (sequential L1.1 vs chunked L2.1) agree — same rule,
    fully independent on-device routes."""
    Q, K, V = _qkv(21)
    beta = _sig(np.random.default_rng(22).standard_normal((_B, _H, _S)))
    decay = _sig(np.random.default_rng(23).standard_normal((_B, _H, _S)) + 2.0)
    o_rec = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay, backend="apple_gpu")
    o_chk = dr.gated_delta_rule_chunked(Q, K, V, beta=beta, decay=decay,
                                        chunk_size=8, backend="apple_gpu")
    np.testing.assert_allclose(np.asarray(o_rec), np.asarray(o_chk), rtol=1e-4, atol=1e-4)


@gpu
def test_metal_chunked_coop_equals_lane0_equals_numpy():
    """L2.2: the cooperative kernel (column-parallel solve + cell-parallel state
    carry, default) and the L2.1 lane-0 form are both exact vs numpy."""
    Q, K, V = _qkv(27)
    beta = _sig(np.random.default_rng(28).standard_normal((_B, _H, _S)))
    decay = _sig(np.random.default_rng(29).standard_normal((_B, _H, _S)) + 2.0)
    ref = dr.gated_delta_rule_recurrent(Q, K, V, beta=beta, decay=decay)
    coop = agb.gpu_gated_delta_rule_chunked(Q, K, V, beta, decay, chunk=8, coop=True)
    lane0 = agb.gpu_gated_delta_rule_chunked(Q, K, V, beta, decay, chunk=8, coop=False)
    np.testing.assert_allclose(np.asarray(coop), ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(lane0), ref, rtol=1e-4, atol=1e-4)


@gpu
def test_metal_chunked_with_output_gate_and_erase_off():
    Q, K, V = _qkv(24)
    gate = np.random.default_rng(25).standard_normal((_B, _H, _S, _D)).astype(np.float32)
    o_gpu = dr.gated_delta_rule_chunked(Q, K, V, gate=gate, chunk_size=8, backend="apple_gpu")
    o_ref = dr.gated_delta_rule_recurrent(Q, K, V, gate=gate)
    np.testing.assert_allclose(np.asarray(o_gpu), o_ref, rtol=1e-4, atol=1e-4)
    # erase=False chunked ≡ shipped linear reference.
    from tessera import ops
    beta = _sig(np.random.default_rng(26).standard_normal((_B, _H, _S)))
    o_lin = dr.gated_delta_rule_chunked(Q, K, V, beta=beta, erase=False,
                                        chunk_size=8, backend="apple_gpu")
    o_exist = np.asarray(ops.gated_deltanet(Q, K, V, beta=beta), np.float64)
    np.testing.assert_allclose(np.asarray(o_lin), o_exist, rtol=1e-4, atol=1e-4)
