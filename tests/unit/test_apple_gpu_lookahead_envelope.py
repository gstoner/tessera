"""LSA on Apple GPU — envelope parity + host-select/GPU-attention numerics.

Mirrors ``test_apple_gpu_sparse_attn.py``: the sigmoid-threshold block selection
runs on the host (data-dependent), the per-query footprint attention runs on the
GPU. See ``docs/audit/domain/archive/lsa_scope.md`` (D4).
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import lsa
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver
from tessera.compiler.apple_gpu_envelope import lane_for

R = _runtime
_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")

_OP = "tessera.lookahead_sparse_attention"
_B, _H, _S, _D = 2, 3, 16, 16


def test_in_envelope_and_lane():
    assert _OP in _driver._APPLE_GPU_RUNTIME_OPS
    assert _OP in _runtime._APPLE_GPU_RUNTIME_OPS
    assert _OP in _driver._APPLE_GPU_SPARSE_ATTN_OPS
    assert lane_for(_OP) == "sparse_attn"


def test_driver_runtime_sparse_envelopes_agree():
    assert _driver._APPLE_GPU_SPARSE_ATTN_OPS == _runtime._APPLE_GPU_SPARSE_ATTN_OPS


@gpu
@pytest.mark.parametrize("threshold,window_size,block_size", [
    (0.5, 6, 4), (0.0, 4, 4), (0.8, 2, 8),
])
def test_dispatch_matches_oracle(threshold, window_size, block_size):
    rng = np.random.default_rng(int(threshold * 100) + window_size + block_size)
    Q = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    K = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    V = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    kw = {"window_size": window_size, "block_size": block_size,
          "threshold": threshold, "causal": True}
    out = R._apple_gpu_dispatch_sparse_attn(_OP, [Q, K, V], kw, np)
    ref = lsa.lookahead_sparse_attention(Q, K, V, **kw)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), atol=1e-4)


@gpu
def test_jit_reports_metal_runtime():
    rng = np.random.default_rng(11)
    Q = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    K = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)
    V = rng.standard_normal((_B, _H, _S, _D)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(q, k, v):
        return ts.ops.lookahead_sparse_attention(
            q, k, v, window_size=6, block_size=4, threshold=0.5, causal=True)

    ref = ts.ops.lookahead_sparse_attention(
        Q, K, V, window_size=6, block_size=4, threshold=0.5, causal=True)
    np.testing.assert_allclose(np.asarray(f(Q, K, V)), np.asarray(ref), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
