"""Host-free contracts for NVIDIA backward-attention route policy."""

from __future__ import annotations

import re

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


# ── allocation-failure cleanup (code review 2026-08-29, P3) ─────────────────
def _entry_body(source: str, name: str) -> str:
    """The brace-balanced body of one `extern "C"` entry."""
    start = source.index(f'extern "C" int {name}(')
    open_brace = source.index("{", start)
    depth = 0
    for i in range(open_brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace:i + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _entries_with_device_allocations(source: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'extern "C" int (\w+)\(', source)
            if "cudaMalloc" in _entry_body(source, m.group(1))]


@pytest.mark.parametrize("synth", ["_synthesize_flash_bwd_cuda",
                                   "_synthesize_flash_bwd_f16_cuda"])
def test_flash_backward_entries_free_every_buffer_on_every_exit(synth):
    """A `return` between the first cudaMalloc and the cleanup block strands
    every buffer that already succeeded — nq+nk+no+nv is GB-scale at Sq=Sk=4k,
    and the arbiter process outlives the call, so a retry with a smaller batch
    finds LESS memory than the attempt that failed. The atomic entry and the
    f16 wrapper skipped the `goto fail` cleanup their _timed and SPLIT siblings
    already use."""
    source = getattr(nv, synth)()
    entries = _entries_with_device_allocations(source)
    assert entries, "no allocating entry found — extraction is broken"
    for name in entries:
        body = _entry_body(source, name)
        allocated = set(re.findall(r"cudaMalloc\(&(\w+)", body))
        freed = set(re.findall(r"cudaFree\((\w+)\)", body))
        assert allocated <= freed, f"{name} never frees {sorted(allocated - freed)}"
        assert "fail:" in body, f"{name} has no cleanup label"
        # Every return reached before the cleanup label must already have freed
        # every buffer (that is the entry's own success path); a return that has
        # freed fewer is the leak. The HEAD defect was `return 2;` with zero.
        label = body.index("fail:")
        for hit in re.finditer(r"return \w+;", body[:label]):
            if "cudaMalloc" not in body[:hit.start()]:
                continue
            done = len(re.findall(r"cudaFree\(", body[:hit.start()]))
            assert done >= len(allocated), (
                f"{name} returns at offset {hit.start()} having freed "
                f"{done} of {len(allocated)} buffers")


def test_flash_backward_entries_check_every_transfer():
    """The atomic entry used to fire its H2D copies unchecked and then report
    the sync's status, so a failed upload produced a confidently wrong gradient
    rather than a diagnostic (Decision #21)."""
    body = _entry_body(nv._synthesize_flash_bwd_cuda(),
                       nv._FLASH_BWD_ENTRY)
    copies = re.findall(r"cudaMemcpy\(", body)
    guarded = re.findall(r"(?:if\(|\|\||&&)cudaMemcpy\(", body)
    assert len(copies) == len(guarded), (
        f"{len(copies) - len(guarded)} unchecked cudaMemcpy in the atomic entry")
