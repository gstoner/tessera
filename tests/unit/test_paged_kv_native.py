"""Workstream A follow-on (#8) — native Apple-GPU paged attention + heterogeneous
KV kinds.

Proves: (1) paged_attention runs the gathered KV through the shipped fused Metal
kernel and reports honest provenance; (2) the native cross-path oracle earns the
rung only when Metal actually fired and agrees with the reference; (3) the LATENT
(MLA expand) and QUANTIZED_TAIL kinds route through the same consumer.

The Metal path is skipped (not failed) when the runtime isn't loadable, so the
suite stays green off-Apple — but the provenance gate means a fallback can never
masquerade as a native pass.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Workstream A follow-on).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import (KVCacheHandle, LatentKVCacheHandle, paged_attention,
                           latent_paged_kv, quantized_tail_paged_kv, KVKind)
from tessera.compiler.evaluator import paged_kv_native_equivalence


def _rng(s=0):
    return np.random.default_rng(s)


def _fill(num_heads, head_dim, n, *, seed=0, page_size=8):
    h = KVCacheHandle(num_heads=num_heads, head_dim=head_dim, max_seq=n + 8,
                      page_size=page_size)
    rng = _rng(seed)
    k = rng.standard_normal((n, num_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((n, num_heads, head_dim)).astype(np.float32)
    h.append(k, v)
    return h, k, v


def _metal_available() -> bool:
    """True iff the fused-attention kernel runs natively here."""
    from tessera.compiler.fusion import run_fused_attention, AttentionRegion
    Q = np.zeros((4, 8), np.float32)
    K = np.zeros((16, 8), np.float32)
    V = np.zeros((16, 8), np.float32)
    _, exe = run_fused_attention(AttentionRegion(scale=1.0), Q, K, V)
    return exe == "metal_runtime"


# ── native execution path + provenance honesty ───────────────────────────────


def test_apple_gpu_backend_matches_reference():
    h, _, _ = _fill(4, 32, 48)
    Q = _rng(1).standard_normal((4, 4, 32)).astype(np.float32)
    ref = paged_attention(Q, h, backend="reference")
    gpu, exe = paged_attention(Q, h, backend="apple_gpu", return_execution=True)
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-4)
    assert exe in {"metal_runtime", "reference"}  # honest provenance


def test_unknown_backend_raises():
    h, _, _ = _fill(2, 8, 8)
    Q = np.zeros((2, 1, 8), np.float32)
    with pytest.raises(ValueError):
        paged_attention(Q, h, backend="cuda")


# ── native cross-path oracle (the promoted rung) ─────────────────────────────


@pytest.mark.hardware_apple_gpu
def test_native_equivalence_oracle():
    h, _, _ = _fill(4, 32, 64)
    Q = _rng(2).standard_normal((4, 4, 32)).astype(np.float32)
    verdict = paged_kv_native_equivalence(h, Q)
    assert _metal_available(), "Metal runtime unavailable on the hardware test host"
    assert verdict.relation == "equivalent", verdict.detail
    assert "apple_gpu:metal_runtime" in verdict.paths


def test_native_oracle_is_inconclusive_without_metal(monkeypatch):
    # Force the apple_gpu path to report a fallback → the oracle must NOT claim
    # equivalence (provenance gate), proving a silent fallback can't earn the rung.
    import tessera.cache.paged_kv as pk

    def fake_gpu(Q, K, V, scale, causal):
        return pk._reference_attention(Q, K, V, scale, causal), "reference"

    monkeypatch.setattr(pk, "_paged_attention_apple_gpu", fake_gpu)
    h, _, _ = _fill(2, 16, 16)
    Q = _rng(3).standard_normal((2, 2, 16)).astype(np.float32)
    verdict = paged_kv_native_equivalence(h, Q)
    assert verdict.relation == "inconclusive"


# ── LATENT (MLA) kind ─────────────────────────────────────────────────────────


def test_latent_kind_equals_expanded_full_kv():
    H, hd, latent_dim, S = 4, 16, 12, 20
    rng = _rng(4)
    latents = rng.standard_normal((S, latent_dim)).astype(np.float32)
    Wk = rng.standard_normal((latent_dim, H * hd)).astype(np.float32)
    Wv = rng.standard_normal((latent_dim, H * hd)).astype(np.float32)

    # Latent cache + expand projections.
    lh = LatentKVCacheHandle(latent_dim=latent_dim, max_seq=S + 8)
    lh.append(latents)
    lstate = latent_paged_kv(lh, Wk, Wv, num_heads=H, head_dim=hd)
    assert lstate.kind is KVKind.LATENT

    # Reference: explicitly expand to full per-head K/V in a normal cache.
    K_full = (latents @ Wk).reshape(S, H, hd)
    V_full = (latents @ Wv).reshape(S, H, hd)
    fh = KVCacheHandle(num_heads=H, head_dim=hd, max_seq=S + 8)
    fh.append(K_full, V_full)

    Q = rng.standard_normal((H, 3, hd)).astype(np.float32)
    np.testing.assert_allclose(
        paged_attention(Q, lstate), paged_attention(Q, fh), rtol=1e-4, atol=1e-4)


# ── QUANTIZED_TAIL kind ───────────────────────────────────────────────────────


def test_quantized_tail_approximates_full_fp():
    H, hd, S = 4, 16, 32
    _, k, v = _fill(H, hd, S, seed=5)
    full = KVCacheHandle(num_heads=H, head_dim=hd, max_seq=S + 8)
    full.append(k, v)

    qt = quantized_tail_paged_kv(k, v, hot_window=8)
    assert qt.kind is KVKind.QUANTIZED_TAIL
    assert qt.quant_bits() == 8
    assert qt.seq_len() == S

    Q = _rng(6).standard_normal((H, 4, hd)).astype(np.float32)
    out_qt = paged_attention(Q, qt)
    out_full = paged_attention(Q, full)
    # int8 cold tail → a small but bounded error vs full fp.
    rel = np.max(np.abs(out_qt - out_full)) / (np.max(np.abs(out_full)) + 1e-9)
    assert rel < 0.05


def test_quantized_tail_hot_window_is_exact():
    # Querying only the hot (fp) window must be bit-exact vs full fp.
    H, hd, S = 2, 8, 16
    _, k, v = _fill(H, hd, S, seed=7)
    full = KVCacheHandle(num_heads=H, head_dim=hd, max_seq=S + 8)
    full.append(k, v)
    qt = quantized_tail_paged_kv(k, v, hot_window=4)
    Q = _rng(8).standard_normal((H, 1, hd)).astype(np.float32)
    hot_idx = list(range(S - 4, S))
    np.testing.assert_allclose(
        paged_attention(Q, qt, token_indices=hot_idx),
        paged_attention(Q, full, token_indices=hot_idx), rtol=1e-5, atol=1e-5)
