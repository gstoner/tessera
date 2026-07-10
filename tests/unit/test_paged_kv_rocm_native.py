"""ROCm gfx1151 native paged attention (#8 follow-on, ROCm lane).

Proves the KV-cache → attention fusion on the compiled ROCm FA-2 lane: the paged
KV state is *gathered* (the staging stage), then the dense per-head K/V feeds the
compiler-generated WMMA flash-attn forward kernel in a single launch (the folded
``(num_heads, S, head_dim)`` batch is exactly the lane's ``[..., S, D]`` contract).

Mirrors ``test_paged_kv_native.py`` (Apple), adapted to the hardware-gated ROCm
lane. The native path is *skipped* (not failed) when no gfx1151 GPU / tessera-opt
is present, so the suite stays green off-box — but the provenance gate (only a
genuine ``native_gpu`` launch earns the rung) means a fallback can never
masquerade as a native pass. f16 WMMA storage → looser tolerances than the numpy
reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import KVCacheHandle, paged_attention
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


def _rocm_available() -> bool:
    """True iff the compiled ROCm FA-2 forward lane can launch here."""
    from tessera import runtime as rt
    return rt._rocm_compiled_flash_attn_available()


# ── native execution path + provenance honesty ───────────────────────────────


def test_rocm_backend_matches_reference():
    # head_dim a multiple of 16 (WMMA tile) so the native lane is eligible.
    h, _, _ = _fill(4, 32, 48)
    Q = _rng(1).standard_normal((4, 4, 32)).astype(np.float32)
    ref = paged_attention(Q, h, backend="reference")
    gpu, exe = paged_attention(Q, h, backend="rocm", return_execution=True)
    assert exe in {"native_gpu", "reference"}  # honest provenance
    # f16 WMMA storage → bounded, not bit-exact, agreement vs the f32 reference.
    tol = 2e-2 if exe == "native_gpu" else 1e-4
    np.testing.assert_allclose(gpu, ref, rtol=tol, atol=tol)


def test_rocm_backend_demotes_on_unsupported_head_dim():
    # head_dim=8 is not a multiple of the 16-wide WMMA tile — the lane cannot run
    # it, so the call demotes to the (correct) reference. GPU-free: always runs.
    h, _, _ = _fill(2, 8, 16)
    Q = _rng(9).standard_normal((2, 2, 8)).astype(np.float32)
    ref = paged_attention(Q, h, backend="reference")
    out, exe = paged_attention(Q, h, backend="rocm", return_execution=True)
    assert exe == "reference"
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


# ── native cross-path oracle (the promoted rung) ─────────────────────────────


def test_rocm_native_equivalence_oracle():
    h, _, _ = _fill(4, 32, 64)
    Q = _rng(2).standard_normal((4, 4, 32)).astype(np.float32)
    verdict = paged_kv_native_equivalence(h, Q, backend="rocm",
                                          rtol=2e-2, atol=2e-2)
    if not _rocm_available():
        # Honest: no ROCm lane → inconclusive, never a false pass.
        assert verdict.relation == "inconclusive"
        pytest.skip("compiled ROCm FA lane not available — rung not earnable here")
    assert verdict.relation == "equivalent", verdict.detail
    assert "rocm:native_gpu" in verdict.paths


def test_rocm_native_oracle_is_inconclusive_without_gpu(monkeypatch):
    # Force the rocm path to report a fallback → the oracle must NOT claim
    # equivalence (provenance gate): a silent fallback can't earn the native rung.
    import tessera.cache.paged_kv as pk

    def fake_rocm(Q, K, V, scale, causal):
        return pk._reference_attention(Q, K, V, scale, causal), "reference"

    monkeypatch.setattr(pk, "_paged_attention_rocm", fake_rocm)
    h, _, _ = _fill(2, 16, 16)
    Q = _rng(3).standard_normal((2, 2, 16)).astype(np.float32)
    verdict = paged_kv_native_equivalence(h, Q, backend="rocm")
    assert verdict.relation == "inconclusive"


def test_oracle_rejects_unknown_backend():
    h, _, _ = _fill(2, 16, 16)
    Q = _rng(3).standard_normal((2, 2, 16)).astype(np.float32)
    with pytest.raises(ValueError):
        paged_kv_native_equivalence(h, Q, backend="cuda")
