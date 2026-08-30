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


# ── the deterministic split route does not do asymptotically redundant work ──
#
# The route is forced whenever deterministic=True, and its cost is fed to the
# route arbiter, so redundant work here is charged to determinism.


def _kernel(source, name):
    """The text of one __global__ kernel, so a claim about kernel A is not
    satisfied by a line in kernel B."""
    start = source.index(f"__global__ void {name}(")
    nxt = source.find("__global__ void ", start + 1)
    return source[start:nxt if nxt != -1 else len(source)]


def test_split_route_reads_precomputed_row_stats_instead_of_recomputing_them():
    source = nv._synthesize_flash_bwd_cuda()
    # The row's max/z/delta depend on (b, qh, m) only, but the split kernel is
    # n-parallel: recomputing them per key is an Sk-fold blowup.
    assert "tsr_flash_bwd_stats" in source
    split = _kernel(source, "tsr_flash_bwd_split")
    assert "float mx=sm[r],z=sz[r],delta=sd[r];" in split
    assert "for(long j=0;j<Sk;++j)" not in split


def test_split_route_dq_accumulates_over_keys_once_per_output_row():
    # n outermost into aq[d] -- not the d-outer form that recomputed every score
    # and dp D times per row, which the atomic kernel never did.
    dq = _kernel(nv._synthesize_flash_bwd_cuda(), "tsr_flash_bwd_dq")
    assert "float aq[TSR_FA_CAP]" in dq
    assert "float out=0.f" not in dq


def test_split_route_allocates_and_frees_the_row_stats_workspace():
    source = nv._synthesize_flash_bwd_cuda()
    # Both split entries (plain + timed) own the workspace end to end.
    for buf in ("sm", "sz", "sd"):
        assert source.count(f"cudaMalloc(&{buf},ns)") == 2
        assert source.count(f"if({buf})cudaFree({buf})") == 2        # fail path
        assert source.count(f"cudaFree({buf})") == 4  # + the two success paths
