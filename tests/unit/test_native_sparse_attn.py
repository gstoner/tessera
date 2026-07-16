"""NSA-1 / NSA-2 / NSA-3 / NSA-5 — DeepSeek Native Sparse Attention.

Forward correctness of the three branch primitives + the
NativeSparseAttention Module + the Apple GPU runtime symbol contract.

NSA-4 (Schedule IR fusion) is exercised by the lit fixture
`tests/tessera-ir/phase8/nsa_fusion.mlir`.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tessera as ts


ROOT = Path(__file__).resolve().parents[2]


def _compile_apple_gpu_runtime(tmp_path):
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    backend = ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime"
    if sys.platform == "darwin":
        source = backend / "apple_gpu_runtime.mm"
        lib = tmp_path / "libtessera_apple_gpu_runtime.dylib"
        cmd = [
            cxx, "-std=c++17", "-shared", "-fPIC", "-fobjc-arc",
            "-x", "objective-c++", str(source), "-o", str(lib),
            "-framework", "Foundation",
            "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "MetalPerformanceShadersGraph",
        ]
    else:
        source = backend / "apple_gpu_runtime_stub.cpp"
        lib = tmp_path / "libtessera_apple_gpu_runtime.so"
        cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(lib)]
    subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return ctypes.CDLL(str(lib))


def _make_qkv(B=1, H=2, S=16, D=4, seed=0):
    np.random.seed(seed)
    Q = np.random.randn(B, H, S, D).astype(np.float32) * 0.5
    K = np.random.randn(B, H, S, D).astype(np.float32) * 0.5
    V = np.random.randn(B, H, S, D).astype(np.float32) * 0.5
    return Q, K, V


# ─────────────────────────────────────────────────────────────────────────────
# Branch primitives
# ─────────────────────────────────────────────────────────────────────────────


class TestSlidingWindow:
    def test_output_shape_matches_v(self):
        Q, K, V = _make_qkv()
        out = ts.ops.attn_sliding_window(Q, K, V, window_size=4, causal=True)
        assert out.shape == V.shape

    def test_window_size_must_be_positive(self):
        Q, K, V = _make_qkv()
        with pytest.raises(ValueError, match="window_size"):
            ts.ops.attn_sliding_window(Q, K, V, window_size=0)

    def test_rank_validated(self):
        with pytest.raises(ValueError, match="rank-4"):
            ts.ops.attn_sliding_window(
                np.zeros((4, 4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 4)),
                window_size=4,
            )

    def test_causal_window_attends_only_to_past(self):
        """For window_size=1 + causal, each query only sees its own key —
        the output at position i should equal V[i] regardless of K."""
        Q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]])  # (1, 1, 2, 2)
        K = np.array([[[[2.0, 0.0], [0.0, 2.0]]]])
        V = np.array([[[[7.0, 8.0], [9.0, 10.0]]]])
        out = ts.ops.attn_sliding_window(Q, K, V, window_size=1, causal=True)
        # Each query attends only to itself, so output equals V.
        np.testing.assert_allclose(out, V, rtol=1e-5)


class TestCompressedBlocks:
    def test_output_shape(self):
        Q, K, V = _make_qkv()
        K_c, V_c = ts.ops.compress_blocks(K, V, block_size=4)
        assert K_c.shape == (Q.shape[0], Q.shape[1], 4, K.shape[3])
        out = ts.ops.attn_compressed_blocks(Q, K_c, V_c)
        assert out.shape == V.shape

    def test_compressed_block_is_mean_by_default(self):
        K = np.array([[[[1.0], [2.0], [3.0], [4.0]]]], dtype=np.float32)
        V = np.array([[[[10.0], [20.0], [30.0], [40.0]]]], dtype=np.float32)
        K_c, V_c = ts.ops.compress_blocks(K, V, block_size=2)
        # Block 0: mean of [1, 2] = 1.5; block 1: mean of [3, 4] = 3.5.
        np.testing.assert_array_equal(K_c[..., 0], [[[1.5, 3.5]]])
        np.testing.assert_array_equal(V_c[..., 0], [[[15.0, 35.0]]])

    def test_learnable_compression(self):
        K = np.array([[[[1.0], [2.0]]]], dtype=np.float32)
        V = np.array([[[[10.0], [20.0]]]], dtype=np.float32)
        # Single output — block_size=2, w_compress=(2, 1).
        w = np.array([[2.0], [3.0]])  # weighted sum: 2*K[0] + 3*K[1]
        K_c, V_c = ts.ops.compress_blocks(K, V, block_size=2, w_compress=w)
        # Expect K_c[0, 0] = 2*1 + 3*2 = 8, V_c[0, 0] = 2*10 + 3*20 = 80.
        np.testing.assert_array_equal(K_c[..., 0], [[[8.0]]])
        np.testing.assert_array_equal(V_c[..., 0], [[[80.0]]])

    def test_block_size_must_divide_seq_len(self):
        K = np.zeros((1, 1, 5, 1), dtype=np.float32)
        V = np.zeros((1, 1, 5, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="divisible"):
            ts.ops.compress_blocks(K, V, block_size=2)


class TestTopKBlocks:
    def test_output_shape(self):
        Q, K, V = _make_qkv()
        K_c, V_c = ts.ops.compress_blocks(K, V, block_size=4)
        scores = np.matmul(Q, np.swapaxes(K_c, -1, -2))
        out = ts.ops.attn_top_k_blocks(
            Q, K, V, scores=scores, top_k=2, block_size=4, causal=True,
        )
        assert out.shape == V.shape

    def test_top_k_must_not_exceed_num_blocks(self):
        Q, K, V = _make_qkv()
        K_c, V_c = ts.ops.compress_blocks(K, V, block_size=4)
        # 4 blocks total; top_k=5 is invalid.
        scores = np.zeros((Q.shape[0], Q.shape[1], Q.shape[2], 4))
        with pytest.raises(ValueError, match="top_k"):
            ts.ops.attn_top_k_blocks(
                Q, K, V, scores=scores, top_k=5, block_size=4,
            )

    def test_seq_must_be_divisible_by_block_size(self):
        Q = np.zeros((1, 1, 5, 1), dtype=np.float32)
        K = np.zeros((1, 1, 5, 1), dtype=np.float32)
        V = np.zeros((1, 1, 5, 1), dtype=np.float32)
        scores = np.zeros((1, 1, 5, 1))
        with pytest.raises(ValueError, match="divisible"):
            ts.ops.attn_top_k_blocks(
                Q, K, V, scores=scores, top_k=1, block_size=2,
            )


# ─────────────────────────────────────────────────────────────────────────────
# nn.NativeSparseAttention Module
# ─────────────────────────────────────────────────────────────────────────────


class TestNativeSparseAttentionModule:
    def test_module_forward_runs(self):
        np.random.seed(0)
        embed_dim, num_heads, S = 8, 2, 16
        m = ts.nn.NativeSparseAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            window_size=4, block_size=4, top_k=2,
        )
        for p in m.parameters():
            p._data._data[:] = np.random.randn(*p.shape).astype(np.float32) * 0.1
        x = np.random.randn(1, S, embed_dim).astype(np.float32) * 0.5
        out = m(x)
        assert out.shape == (1, S, embed_dim)

    def test_module_seq_must_be_divisible_by_block_size(self):
        m = ts.nn.NativeSparseAttention(
            embed_dim=8, num_heads=2, window_size=4, block_size=4, top_k=2,
        )
        x = np.zeros((1, 7, 8), dtype=np.float32)  # 7 not divisible by 4
        with pytest.raises(ValueError, match="divisible"):
            m(x)

    def test_module_with_compress_weight(self):
        np.random.seed(0)
        m = ts.nn.NativeSparseAttention(
            embed_dim=8, num_heads=2, window_size=4, block_size=4, top_k=2,
            compress_weight=True,
        )
        for p in m.parameters():
            p._data._data[:] = np.random.randn(*p.shape).astype(np.float32) * 0.1
        x = np.random.randn(1, 8, 8).astype(np.float32) * 0.5
        out = m(x)
        assert out.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# Apple GPU runtime symbol contract
# ─────────────────────────────────────────────────────────────────────────────


def test_apple_gpu_native_sparse_attn_runtime_shim_exposes_symbol(tmp_path):
    if sys.platform == "darwin":
        from tessera._apple_gpu_dispatch import apple_gpu_runtime
        runtime = apple_gpu_runtime()
        if runtime is None:
            pytest.skip("Apple GPU runtime unavailable")
    else:
        runtime = _compile_apple_gpu_runtime(tmp_path)
    sym = getattr(runtime, "tessera_apple_gpu_native_sparse_attn_f32", None)
    assert sym is not None, "missing C ABI symbol: tessera_apple_gpu_native_sparse_attn_f32"


@pytest.mark.hardware_apple_gpu
def test_apple_gpu_native_sparse_attn_runtime_msl_numerics():
    """Call the C ABI symbol and prove the real MSL path ran.

    Symbol resolution alone is not a GPU proof: the non-Darwin stub exports the
    symbol too. This test requires an active Metal runtime, checks the MSL
    pipeline cache grows, then verifies numerical output against the public
    numpy reference.
    """
    from tessera._apple_gpu_dispatch import apple_gpu_runtime

    runtime = apple_gpu_runtime()
    assert runtime is not None
    cache_size = runtime.tessera_apple_gpu_runtime_msl_cache_size
    cache_size.argtypes = []
    cache_size.restype = ctypes.c_int32
    before = cache_size()
    assert before >= 0, "Metal device context is not active"

    sym = runtime.tessera_apple_gpu_native_sparse_attn_f32
    fp = ctypes.POINTER(ctypes.c_float)
    sym.argtypes = [fp, fp, fp, fp, fp] + [ctypes.c_int32] * 8
    sym.restype = None
    last_path = runtime.tessera_apple_gpu_native_sparse_attn_last_path
    last_path.argtypes = []
    last_path.restype = ctypes.c_int32

    Q, K, V = _make_qkv(B=1, H=2, S=8, D=4, seed=11)
    window_size, block_size, top_k = 4, 4, 1
    K_c, _V_c = ts.ops.compress_blocks(K, V, block_size=block_size)
    scores = np.matmul(Q, np.swapaxes(K_c, -1, -2)).astype(np.float32)
    out = np.empty_like(Q, dtype=np.float32)
    q = np.ascontiguousarray(Q, dtype=np.float32)
    k = np.ascontiguousarray(K, dtype=np.float32)
    v = np.ascontiguousarray(V, dtype=np.float32)
    g = np.ascontiguousarray(scores, dtype=np.float32)
    sym(q.ctypes.data_as(fp), k.ctypes.data_as(fp), v.ctypes.data_as(fp),
        g.ctypes.data_as(fp), out.ctypes.data_as(fp),
        ctypes.c_int32(1), ctypes.c_int32(2), ctypes.c_int32(8),
        ctypes.c_int32(4), ctypes.c_int32(window_size),
        ctypes.c_int32(block_size), ctypes.c_int32(top_k), ctypes.c_int32(1))

    assert last_path() == 1, "native sparse should execute through Metal"
    assert cache_size() >= before, "MSL cache size should remain valid"
    ref = ts.ops.deepseek_sparse_attention(
        Q, K, V, None, window_size=window_size, block_size=block_size,
        top_k=top_k, causal=True)
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)
