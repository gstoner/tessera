"""Focused execute-and-compare proofs for Apple conformance rows.

These tests intentionally exercise correctness through the public Apple target
paths.  Native acceleration is a performance/provenance property; the Apple CPU
reference fallback remains a valid conformance executor on non-Darwin CI.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.cache import KVCacheHandle
from tessera.runtime import apple_gpu_kv_cache_read


def _matmul_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260712)
    return (
        rng.standard_normal((5, 4), dtype=np.float32),
        rng.standard_normal((4, 6), dtype=np.float32),
    )


def test_apple_cpu_matmul_relu_and_softmax_match_numpy() -> None:
    @ts.jit(target="apple_cpu")
    def matmul_relu(a, b):
        return ts.ops.relu(ts.ops.matmul(a, b))

    @ts.jit(target="apple_cpu")
    def matmul_softmax(a, b):
        return ts.ops.softmax(ts.ops.matmul(a, b))

    a, b = _matmul_inputs()
    scores = a @ b
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    np.testing.assert_allclose(matmul_relu(a, b), np.maximum(scores, 0.0),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(matmul_softmax(a, b),
                               exp / exp.sum(axis=-1, keepdims=True),
                               rtol=1e-5, atol=1e-5)


def test_apple_cpu_conv2d_matches_reference() -> None:
    @ts.jit(target="apple_cpu")
    def conv(x, w):
        return ts.ops.conv2d(x, w, stride=1, padding=0, layout="nhwc")

    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, 5, 5, 2), dtype=np.float32)
    w = rng.standard_normal((3, 3, 2, 3), dtype=np.float32)
    expected = np.zeros((1, 3, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            patch = x[0, i:i + 3, j:j + 3]
            for channel in range(3):
                expected[0, i, j, channel] = np.sum(patch * w[..., channel])
    np.testing.assert_allclose(conv(x, w), expected, rtol=1e-5, atol=1e-5)


def test_apple_cpu_flash_attention_matches_reference() -> None:
    @ts.jit(target="apple_cpu")
    def attention(q, k, v):
        return ts.ops.flash_attn(q, k, v)

    rng = np.random.default_rng(11)
    q = rng.standard_normal((1, 2, 4, 8), dtype=np.float32)
    k = rng.standard_normal((1, 2, 4, 8), dtype=np.float32)
    v = rng.standard_normal((1, 2, 4, 8), dtype=np.float32)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(8.0)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = np.matmul(exp / exp.sum(axis=-1, keepdims=True), v)
    np.testing.assert_allclose(attention(q, k, v), expected,
                               rtol=1e-5, atol=1e-5)


def test_apple_cpu_kv_cache_read_matches_source() -> None:
    cache = KVCacheHandle(num_heads=2, head_dim=4, max_seq=16, page_size=4)
    keys = np.arange(8 * 2 * 4, dtype=np.float32).reshape(8, 2, 4)
    values = keys + 100.0
    cache.append(keys, values)
    read_keys, read_values = ts.ops.kv_cache_read(cache, 2, 7)
    np.testing.assert_allclose(read_keys, keys[2:7])
    np.testing.assert_allclose(read_values, values[2:7])


def test_apple_gpu_matmul_relu_matches_numpy() -> None:
    @ts.jit(target="apple_gpu")
    def composed(a, b):
        return ts.ops.relu(ts.ops.matmul(a, b))

    a, b = _matmul_inputs()
    np.testing.assert_allclose(composed(a, b), np.maximum(a @ b, 0.0),
                               rtol=1e-4, atol=1e-5)


def test_apple_gpu_kv_cache_read_matches_source() -> None:
    cache = KVCacheHandle(num_heads=2, head_dim=4, max_seq=16, page_size=4)
    keys = np.arange(8 * 2 * 4, dtype=np.float32).reshape(8, 2, 4)
    values = keys + 50.0
    cache.append(keys, values)
    read_keys, read_values, mode = apple_gpu_kv_cache_read(cache, 1, 6)
    if mode != "metal_runtime":
        pytest.skip("requires the Apple Metal DeviceTensor KV-cache path")
    assert mode == "metal_runtime"
    np.testing.assert_allclose(read_keys, keys[1:6])
    np.testing.assert_allclose(read_values, values[1:6])
