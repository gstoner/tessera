"""MSA Phase 3 — Apple GPU host-select runtime lane.

`@jit(target="apple_gpu")` on `msa_sparse_attention` runs the Index Branch
scoring + Top-k block selection on the host (data-dependent, like
`attn_top_k_blocks`) and the exact block-sparse Main Branch attention on the
GPU (two GPU bmms + GPU softmax, with token-level causal masking + block dedup
folded into a per-row additive mask). The GPU result must match the numpy
reference `ops.msa_sparse_attention` bit-for-bit (within fp32 tolerance).

Skips cleanly when the Apple GPU runtime can't be loaded (non-Darwin / no Metal).
See docs/architecture/workloads/msa.md §6.
"""

from __future__ import annotations

import ctypes

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


# ── Phase 6: native fused block-sparse flash kernel ──────────────────────────

def test_native_kernel_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_msa_block_sparse_f32")
    assert R._apple_gpu_msa_block_sparse_f32() is not None


def _call_native(Q, K, V, *, block_size, top_k, causal, scale=None, kernel="scalar"):
    """Drive a native MSL kernel directly (locks the C ABI). ``kernel`` selects
    the scalar or the tiled (simdgroup-cooperative) f32 kernel."""
    B, Hq, Sq, D = Q.shape
    Hkv, Sk = K.shape[1], K.shape[2]
    scores = ts.ops.msa_index_scores(Q, K, block_size=block_size, scale=scale)
    sel = ts.ops.msa_select_blocks(
        scores, top_k=top_k, block_size=block_size, causal=causal
    )
    native = (R._apple_gpu_msa_block_sparse_tiled_f32() if kernel == "tiled"
              else R._apple_gpu_msa_block_sparse_f32())
    Qc = np.ascontiguousarray(Q, np.float32)
    Kc = np.ascontiguousarray(K, np.float32)
    Vc = np.ascontiguousarray(V, np.float32)
    ids = np.ascontiguousarray(np.asarray(sel, np.int32))
    O = np.zeros((B, Hq, Sq, V.shape[-1]), np.float32)
    fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
    s = (1.0 / np.sqrt(D)) if scale is None else float(scale)
    native(
        Qc.ctypes.data_as(fp), Kc.ctypes.data_as(fp), Vc.ctypes.data_as(fp),
        ids.ctypes.data_as(ip), O.ctypes.data_as(fp),
        ctypes.c_int32(B), ctypes.c_int32(Hq), ctypes.c_int32(Hkv),
        ctypes.c_int32(Sq), ctypes.c_int32(Sk), ctypes.c_int32(D),
        ctypes.c_int32(block_size), ctypes.c_int32(int(ids.shape[-1])),
        ctypes.c_float(s), ctypes.c_int32(1 if causal else 0),
    )
    return O


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_native_kernel_direct_matches_reference(causal, top_k):
    Q, K, V = _qkv(B=1, Hq=4, Hkv=2, Sq=16, Sk=16, D=8, Dv=8, seed=top_k + 10)
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=top_k, causal=causal)
    out = _call_native(Q, K, V, block_size=4, top_k=top_k, causal=causal)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# ── Phase 6 follow-on: GPU-resident selection (huge-batch regime) ────────────

def test_gpu_select_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_msa_select_blocks_f32")
    assert R._apple_gpu_msa_select_blocks_f32() is not None


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_gpu_resident_select_end_to_end_matches_reference(causal, top_k):
    # Force the GPU-resident Index Branch (bmm scoring + GPU select kernel) by
    # dropping the FLOP threshold, then check the full path matches the reference.
    Q, K, V = _qkv(B=2, Hq=4, Hkv=2, Sq=32, Sk=32, D=16, Dv=16, seed=top_k + 50)
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=8, top_k=top_k, causal=causal)
    saved = R._MSA_GPU_SELECT_FLOP_THRESHOLD
    try:
        R._MSA_GPU_SELECT_FLOP_THRESHOLD = 0  # force GPU-resident path
        out = R._apple_gpu_dispatch_sparse_attn(
            "tessera.msa_sparse_attention", [Q, K, V],
            {"block_size": 8, "top_k": top_k, "causal": causal, "force_local_block": True}, np,
        )
    finally:
        R._MSA_GPU_SELECT_FLOP_THRESHOLD = saved
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_gpu_resident_select_is_gated_off_for_small_shapes():
    # The default threshold keeps small shapes on the host path (it should not
    # invoke the GPU select kernel) — both paths still match the reference.
    Q, K, V = _qkv(seed=3)
    out = R._apple_gpu_dispatch_sparse_attn(
        "tessera.msa_sparse_attention", [Q, K, V],
        {"block_size": 4, "top_k": 2, "causal": True, "force_local_block": True}, np,
    )
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2, causal=True)
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# ── Phase 6 follow-on: tiled (simdgroup-cooperative) kernel ──────────────────

def test_tiled_kernel_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_msa_block_sparse_tiled_f32")
    assert R._apple_gpu_msa_block_sparse_tiled_f32() is not None


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_tiled_kernel_direct_matches_reference(causal, top_k):
    Q, K, V = _qkv(B=1, Hq=4, Hkv=2, Sq=16, Sk=16, D=8, Dv=8, seed=top_k + 40)
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=top_k, causal=causal)
    out = _call_native(Q, K, V, block_size=4, top_k=top_k, causal=causal, kernel="tiled")
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("D", [16, 32, 40, 64])
def test_tiled_kernel_strided_owners_across_head_dims(D):
    # D not a multiple of the 32-lane width exercises the per-lane owned-element
    # striding (e.g. D=40 → lanes 0..7 own 2 elements, 8..31 own 1).
    Q, K, V = _qkv(B=1, Hq=4, Hkv=2, Sq=16, Sk=16, D=D, Dv=D, seed=D)
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=2, causal=True)
    out = _call_native(Q, K, V, block_size=4, top_k=2, causal=True, kernel="tiled")
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_native_f16_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_msa_block_sparse_f16")
    assert R._apple_gpu_msa_block_sparse_f16() is not None


def test_tiled_f16_symbol_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_msa_block_sparse_tiled_f16")
    assert R._apple_gpu_msa_block_sparse_tiled_f16() is not None


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("D", [8, 40, 64])
def test_tiled_f16_dispatch_matches_f32_reference(causal, D):
    # The tiled+f16 combo (cooperative dot + half I/O) — validated within f16
    # tolerance, incl. D=40 (non-multiple-of-32 lane striding). The runtime
    # prefers the tiled f16 kernel for f16 inputs.
    Qf, Kf, Vf = _qkv(B=1, Hq=4, Hkv=2, Sq=16, Sk=16, D=D, Dv=D, seed=D + int(causal))
    ref = ts.ops.msa_sparse_attention(Qf, Kf, Vf, block_size=4, top_k=2, causal=causal)
    Q, K, V = Qf.astype(np.float16), Kf.astype(np.float16), Vf.astype(np.float16)
    out = R._apple_gpu_dispatch_sparse_attn(
        "tessera.msa_sparse_attention", [Q, K, V],
        {"block_size": 4, "top_k": 2, "causal": causal, "force_local_block": True}, np,
    )
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_native_f16_dispatch_matches_f32_reference(causal, top_k):
    # f16 inputs route through the native f16 kernel (half I/O, fp32 accum) and
    # match the f32 reference within f16 tolerance; the output dtype is f16.
    Qf, Kf, Vf = _qkv(B=1, Hq=4, Hkv=2, Sq=16, Sk=16, D=8, Dv=8, seed=top_k + 30)
    ref = ts.ops.msa_sparse_attention(Qf, Kf, Vf, block_size=4, top_k=top_k, causal=causal)
    Q, K, V = Qf.astype(np.float16), Kf.astype(np.float16), Vf.astype(np.float16)
    out = R._apple_gpu_dispatch_sparse_attn(
        "tessera.msa_sparse_attention", [Q, K, V],
        {"block_size": 4, "top_k": top_k, "causal": causal, "force_local_block": True}, np,
    )
    assert out.dtype == np.float16
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-2)


def test_native_kernel_gqa_and_dense_equivalence():
    # GQA group=3
    Q, K, V = _qkv(B=2, Hq=6, Hkv=2, Sq=32, Sk=32, D=16, Dv=16, seed=21)
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=8, top_k=2, causal=True)
    np.testing.assert_allclose(
        _call_native(Q, K, V, block_size=8, top_k=2, causal=True), ref,
        rtol=1e-4, atol=1e-4,
    )
    # dense: top_k == num_blocks
    Q, K, V = _qkv(seed=22)
    ref = ts.ops.msa_sparse_attention(Q, K, V, block_size=4, top_k=4, causal=True)
    np.testing.assert_allclose(
        _call_native(Q, K, V, block_size=4, top_k=4, causal=True), ref,
        rtol=1e-4, atol=1e-4,
    )
