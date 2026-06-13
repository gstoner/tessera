"""MSA Phase 3 — Apple GPU host-select runtime lane.

`@jit(target="apple_gpu")` on `msa_sparse_attention` runs the Index Branch
scoring + Top-k block selection on the host (data-dependent, like
`attn_top_k_blocks`) and the exact block-sparse Main Branch attention on the
GPU (two GPU bmms + GPU softmax, with token-level causal masking + block dedup
folded into a per-row additive mask). The GPU result must match the numpy
reference `ops.msa_sparse_attention` bit-for-bit (within fp32 tolerance).

Skips cleanly when the Apple GPU runtime can't be loaded (non-Darwin / no Metal).
See docs/msa.md §6.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera import runtime as R
from tessera.compiler.apple_gpu_envelope import (
    APPLE_GPU_LANE_BY_OP,
    _APPLE_GPU_SPARSE_ATTN_OPS,
)


def _apple_gpu_available() -> bool:
    try:
        return R._load_apple_gpu_runtime() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _apple_gpu_available(), reason="Apple GPU runtime unavailable (non-Darwin / no Metal)"
)


def _qkv(B=1, Hq=4, Hkv=2, Sq=16, Sk=16, D=8, Dv=8, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(B, Hq, Sq, D)).astype(np.float32)
    K = rng.normal(size=(B, Hkv, Sk, D)).astype(np.float32)
    V = rng.normal(size=(B, Hkv, Sk, Dv)).astype(np.float32)
    return Q, K, V


# ── envelope wiring ──────────────────────────────────────────────────────────

def test_msa_is_in_the_sparse_attn_lane():
    assert "tessera.msa_sparse_attention" in _APPLE_GPU_SPARSE_ATTN_OPS
    assert APPLE_GPU_LANE_BY_OP.get("tessera.msa_sparse_attention") == "sparse_attn"


# ── direct dispatcher vs reference ───────────────────────────────────────────

@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("top_k", [2, 4])
def test_gpu_dispatch_matches_reference(causal, top_k):
    Q, K, V = _qkv(seed=top_k + int(causal))
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=top_k, causal=causal)
    gpu = R._apple_gpu_dispatch_sparse_attn(
        "tessera.msa_sparse_attention", [Q, K, V],
        {"block_size": 4, "top_k": top_k, "causal": causal, "force_local_block": True}, np,
    )
    assert gpu is not None
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-4)


def test_gpu_dispatch_gqa_shape_matches_reference():
    # Hq=6, Hkv=2 → group size 3 (group-shared selection).
    Q, K, V = _qkv(B=2, Hq=6, Hkv=2, Sq=16, Sk=16, D=8, Dv=8, seed=9)
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2, causal=True)
    gpu = R._apple_gpu_dispatch_sparse_attn(
        "tessera.msa_sparse_attention", [Q, K, V],
        {"block_size": 4, "top_k": 2, "causal": True, "force_local_block": True}, np,
    )
    assert gpu is not None
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-4)


def test_gpu_dense_equivalence_when_topk_equals_num_blocks():
    # top_k == num_blocks → exact dense GQA, on the GPU.
    Q, K, V = _qkv(seed=3)
    num_blocks = 16 // 4
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=num_blocks, causal=True)
    gpu = R._apple_gpu_dispatch_sparse_attn(
        "tessera.msa_sparse_attention", [Q, K, V],
        {"block_size": 4, "top_k": num_blocks, "causal": True, "force_local_block": True}, np,
    )
    assert gpu is not None
    np.testing.assert_allclose(gpu, ref, rtol=1e-4, atol=1e-4)


# ── end-to-end @jit(target="apple_gpu") ──────────────────────────────────────

def test_jit_apple_gpu_reports_metal_runtime():
    @ts.jit(target="apple_gpu")
    def msa(Q, K, V):
        return ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2, causal=True)

    Q, K, V = _qkv(seed=1)
    md = msa.runtime_artifact().metadata
    assert md["compiler_path"] == "apple_gpu_mps"
    # Host-select + GPU exact attention reports metal_runtime, consistent with
    # the rest of the sparse-attn lane (e.g. deepseek_sparse_attention): the
    # attention FLOPs run on Metal; block selection is host-side.
    assert md["execution_mode"] == "metal_runtime"
    assert md["runtime_status"] == "ready"


def test_jit_apple_gpu_launch_matches_reference():
    from tessera.runtime import launch

    @ts.jit(target="apple_gpu")
    def msa(Q, K, V):
        return ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2, causal=True)

    Q, K, V = _qkv(seed=2)
    res = launch(msa.runtime_artifact(), args=(Q, K, V))
    assert res["ok"] is True
    assert res["runtime_status"] == "success"
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2, causal=True)
    np.testing.assert_allclose(res["output"], ref, rtol=1e-4, atol=1e-4)
