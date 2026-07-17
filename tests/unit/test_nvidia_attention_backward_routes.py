"""Host-free contracts for NVIDIA backward-attention route policy."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit import nvidia_cuda as nv


def _inputs(dtype=np.float32):
    rng = np.random.default_rng(120)
    q = rng.standard_normal((1, 4, 5, 8), dtype=np.float32).astype(dtype)
    k = rng.standard_normal((1, 2, 7, 8), dtype=np.float32).astype(dtype)
    v = rng.standard_normal((1, 2, 7, 6), dtype=np.float32).astype(dtype)
    do = rng.standard_normal((1, 4, 5, 6), dtype=np.float32).astype(dtype)
    return do, q, k, v


def test_split_reduced_workspace_is_exactly_one_extra_dkdv_footprint():
    _, _, k, v = _inputs()
    expected = (k.size + v.size) * np.dtype(np.float32).itemsize
    assert nv.flash_attention_backward_workspace_bytes(
        k, v, route="split_reduced") == expected
    assert nv.flash_attention_backward_workspace_bytes(k, v, route="atomic") == 0


def test_backward_source_contains_atomic_and_fixed_order_split_candidates():
    source = nv._synthesize_flash_bwd_cuda()
    assert "atomicAdd(&dv" in source
    assert "atomicAdd(&dk" in source
    assert "tsr_flash_bwd_split" in source
    assert "tsr_flash_bwd_reduce" in source
    assert nv._FLASH_BWD_SPLIT_ENTRY in source
    assert "items=B*(long)Hkv*Sk*(D+Dv)" not in source
    assert "items=B*(long)Hkv*Sk" in source


def test_deterministic_request_rejects_atomic_before_cuda_compile(monkeypatch):
    monkeypatch.setattr(
        nv, "_nvidia_cuda_compile_fn",
        lambda *_args, **_kwargs: pytest.fail("must reject before CUDA compile"))
    with pytest.raises(ValueError, match="requires split_reduced"):
        nv.run_flash_attention_backward(
            *_inputs(), scale=0.25, route="atomic", deterministic=True)


def test_split_workspace_limit_rejects_before_cuda_compile(monkeypatch):
    do, q, k, v = _inputs()
    required = nv.flash_attention_backward_workspace_bytes(k, v)
    monkeypatch.setattr(
        nv, "_nvidia_cuda_compile_fn",
        lambda *_args, **_kwargs: pytest.fail("must reject before CUDA compile"))
    with pytest.raises(ValueError, match="exceeding limit"):
        nv.run_flash_attention_backward(
            do, q, k, v, scale=0.25, route="split_reduced",
            workspace_limit_bytes=required - 1)


def test_split_candidate_has_explicit_f32_storage_boundary(monkeypatch):
    monkeypatch.setattr(
        nv, "_nvidia_cuda_compile_fn",
        lambda *_args, **_kwargs: pytest.fail("must reject before CUDA compile"))
    with pytest.raises(ValueError, match="currently requires f32 storage"):
        nv.run_flash_attention_backward(
            *_inputs(np.float16), scale=0.25, route="split_reduced")


def test_deterministic_auto_selects_split_before_cuda_compile(monkeypatch):
    monkeypatch.setattr(
        nv, "_nvidia_cuda_compile_fn",
        lambda *_args, **_kwargs: pytest.fail("must reject before CUDA compile"))
    with pytest.raises(ValueError, match="currently requires f32 storage"):
        nv.run_flash_attention_backward(
            *_inputs(np.float16), scale=0.25, deterministic=True)


@pytest.mark.parametrize("route", ["serial", "rocm_g6c", ""])
def test_unknown_backward_route_rejects_stably(route):
    _, _, k, v = _inputs()
    with pytest.raises(ValueError, match="unknown NVIDIA flash backward route"):
        nv.flash_attention_backward_workspace_bytes(k, v, route=route)
