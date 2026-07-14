"""Paged KV-cache reads and decode-attention composition on NVIDIA sm_120."""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import tessera


def _cuda_or_skip():
    if not (shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        pytest.skip("nvcc not installed")
    from tessera import runtime as rt
    if not rt._nvidia_mma_runtime_available():
        pytest.skip("no usable NVIDIA CUDA device")
    return rt


def _artifact(start: int, end: int):
    from tessera import runtime as rt
    return rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120",
        "compiler_path": "nvidia_kv_cache_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["pages", "page_table"],
        "output_name": "slice",
        "ops": [{
            "op_name": "tessera.kv_cache.read",
            "result": "slice",
            "operands": ["pages", "page_table"],
            "kwargs": {"start": start, "end": end},
        }],
    })


def _page(logical: np.ndarray, page_size: int, table: np.ndarray) -> np.ndarray:
    logical_pages = table.size
    padded = np.zeros((logical_pages * page_size, *logical.shape[1:]), np.float32)
    padded[:logical.shape[0]] = logical
    physical = np.empty((logical_pages, page_size, *logical.shape[1:]), np.float32)
    for logical_page, physical_page in enumerate(table):
        physical[physical_page] = padded[
            logical_page * page_size:(logical_page + 1) * page_size]
    return physical


@pytest.mark.slow
@pytest.mark.parametrize("start,end", [(0, 1), (3, 10), (7, 13)])
def test_live_nvidia_paged_kv_read_matches_handle(start, end):
    rt = _cuda_or_skip()
    rng = np.random.default_rng(610 + start + end)
    handle = tessera.cache.KVCacheHandle(num_heads=3, head_dim=8, max_seq=16)
    keys = rng.standard_normal((13, 3, 8)).astype(np.float32)
    handle.append(keys, keys + 100)
    table = np.array([2, 0, 3, 1], np.int32)
    pages = _page(handle.keys.astype(np.float32), 4, table)
    result = rt.launch(_artifact(start, end), (pages, table))
    assert result["ok"] is True, result.get("reason")
    expected, _ = handle.read(start, end)
    np.testing.assert_allclose(result["output"], expected, rtol=0, atol=0)


@pytest.mark.slow
def test_live_nvidia_paged_kv_read_feeds_decode_attention():
    rt = _cuda_or_skip()
    rng = np.random.default_rng(711)
    H, D, S, page_size = 2, 8, 11, 4
    keys = rng.standard_normal((S, H, D)).astype(np.float32) * .2
    values = rng.standard_normal((S, H, D)).astype(np.float32) * .2
    table = np.array([1, 2, 0], np.int32)
    k = rt.launch(_artifact(2, S), (_page(keys, page_size, table), table))["output"]
    v = rt.launch(_artifact(2, S), (_page(values, page_size, table), table))["output"]
    q = (rng.standard_normal((1, H, 1, D)) * .2).astype(np.float32)
    k4 = np.transpose(k, (1, 0, 2))[None]
    v4 = np.transpose(v, (1, 0, 2))[None]
    flash = rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_flash_attn_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["q", "k", "v"], "output_name": "o",
        "ops": [{"op_name": "tessera.flash_attn", "result": "o",
                 "operands": ["q", "k", "v"],
                 "kwargs": {"causal": False, "scale": D ** -0.5}}],
    })
    result = rt.launch(flash, (q, k4, v4))
    assert result["ok"] is True, result.get("reason")
    scores = np.einsum("bhqd,bhkd->bhqk", q, k4) / np.sqrt(D)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = np.einsum("bhqk,bhkd->bhqd", probs, v4)
    np.testing.assert_allclose(result["output"], expected, rtol=2e-5, atol=2e-6)


def test_nvidia_paged_kv_read_rejects_invalid_page_table_before_cuda():
    from tessera.compiler.emit.nvidia_cuda import run_paged_kv_cache_read_f32
    pages = np.zeros((2, 4, 2, 8), np.float32)
    with pytest.raises(ValueError, match="invalid physical page"):
        run_paged_kv_cache_read_f32(pages, np.array([0, 2]), 0, 4)


def test_nvidia_paged_kv_read_rejects_bounds_before_cuda():
    from tessera.compiler.emit.nvidia_cuda import run_paged_kv_cache_read_f32
    pages = np.zeros((2, 4, 2, 8), np.float32)
    with pytest.raises(ValueError, match="bounds"):
        run_paged_kv_cache_read_f32(pages, np.array([0, 1]), 7, 9)
