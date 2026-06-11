"""Gap 4 — fused single-dispatch GPU kernel for LSA footprint attention.

`tessera_apple_gpu_lookahead_sparse_attn_f32` collapses the host-select
bmm + mask-add + softmax + bmm (4 GPU dispatches) into ONE MSL dispatch:
per (head, query) it computes softmax(scale·Q·Kᵀ + mask)·V over the padded
footprint. Validated against the numpy oracle and against the multi-dispatch
fallback path. See `docs/audit/domain/archive/lsa_scope.md` (Gap 4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import lsa
from tessera import runtime as R

REPO = Path(__file__).resolve().parents[2]
MM = REPO / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm"
STUB = REPO / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime_stub.cpp"

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")

_OP = "tessera.lookahead_sparse_attention"


def _qkv(seed, B=2, H=3, S=16, D=16):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((B, H, S, D)).astype(np.float32),
            rng.standard_normal((B, H, S, D)).astype(np.float32),
            rng.standard_normal((B, H, S, D)).astype(np.float32))


def test_fused_kernel_and_stub_parity_in_source():
    mm = MM.read_text()
    assert "kernel void lookahead_sparse_attn_f32" in mm
    assert "tessera_apple_gpu_lookahead_sparse_attn_f32" in mm
    assert "reference_lookahead_sparse_attn_f32" in mm  # host fallback
    stub = STUB.read_text()
    assert "tessera_apple_gpu_lookahead_sparse_attn_f32" in stub  # non-Darwin parity


def test_fused_is_freshness_sentinel():
    # The fused symbol gates dylib freshness so a stale prebuilt runtime is
    # rebuilt with the kernel present.
    rt = R.__file__
    assert 'getattr(lib, "tessera_apple_gpu_lookahead_sparse_attn_f32")' in Path(rt).read_text()


@gpu
def test_fused_symbol_resolves():
    assert R._apple_gpu_lookahead_sparse_attn_f32() is not None


@gpu
@pytest.mark.parametrize("threshold,window_size", [(0.5, 6), (0.0, 4), (0.8, 2)])
def test_fused_matches_oracle(threshold, window_size):
    Q, K, V = _qkv(seed=int(threshold * 100) + window_size)
    kw = {"window_size": window_size, "block_size": 4, "threshold": threshold, "causal": True}
    out = R._apple_gpu_dispatch_sparse_attn(_OP, [Q, K, V], kw, np)
    ref = lsa.lookahead_sparse_attention(Q, K, V, **kw)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)


@gpu
def test_fused_matches_multi_dispatch_fallback(monkeypatch):
    # Forcing the fused symbol to be unavailable selects the bmm + mask-add +
    # softmax + bmm fallback; both paths must agree (same math, fewer dispatches).
    Q, K, V = _qkv(seed=99)
    kw = {"window_size": 5, "block_size": 4, "threshold": 0.5, "causal": True}
    fused_out = R._apple_gpu_dispatch_sparse_attn(_OP, [Q, K, V], kw, np)
    monkeypatch.setattr(R, "_apple_gpu_lookahead_sparse_attn_f32", lambda: None)
    fallback_out = R._apple_gpu_dispatch_sparse_attn(_OP, [Q, K, V], kw, np)
    np.testing.assert_allclose(np.asarray(fused_out), np.asarray(fallback_out), atol=1e-4)
