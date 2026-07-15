"""Stable paged-KV ABI and ROCm device-gather contract."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.cache import (
    KVGeometry,
    KVKind,
    PageTableEntry,
    PageTier,
    PagedKVBufferABI,
    materialize_paged_kv_abi,
    paged_attention,
)


def _rocm_live() -> bool:
    from tessera import runtime
    return runtime._rocm_compiled_flash_attn_available()


def _permuted_abi() -> tuple[PagedKVBufferABI, np.ndarray, np.ndarray]:
    logical_k = np.arange(12 * 2 * 4, dtype=np.float32).reshape(12, 2, 4) / 100
    logical_v = -logical_k
    table = np.asarray([2, 0, 1], dtype=np.int32)
    kp = np.empty((3, 4, 2, 4), np.float32)
    vp = np.empty_like(kp)
    for logical_page, physical_page in enumerate(table):
        sl = slice(logical_page * 4, (logical_page + 1) * 4)
        kp[physical_page] = logical_k[sl]
        vp[physical_page] = logical_v[sl]
    return PagedKVBufferABI(kp, vp, table, logical_length=12), logical_k, logical_v


class _AbiState:
    kind = KVKind.FULL

    def __init__(self, abi: PagedKVBufferABI) -> None:
        self.abi = abi

    def kv_geometry(self) -> KVGeometry:
        _, page_size, heads, dim = self.abi.k_pages.shape
        return KVGeometry(heads, dim, self.abi.logical_length, page_size)

    def seq_len(self) -> int:
        return self.abi.logical_length

    def quant_bits(self) -> None:
        return None

    def page_table(self) -> list[PageTableEntry]:
        return [PageTableEntry(i, PageTier.RESIDENT)
                for i in range(self.abi.page_table.size)]

    def tier(self, page_id: int) -> PageTier:
        return PageTier.RESIDENT

    def gather(self, token_indices):
        return self.abi.gather(token_indices)

    def paged_kv_abi(self) -> PagedKVBufferABI:
        return self.abi


def test_paged_kv_buffer_abi_follows_permuted_physical_table():
    abi, logical_k, logical_v = _permuted_abi()
    idx = np.asarray([8, 1, 6, 11, 4], dtype=np.int64)
    k, v = abi.gather(idx)
    np.testing.assert_array_equal(k, logical_k[idx])
    np.testing.assert_array_equal(v, logical_v[idx])


def test_materialization_preserves_backend_native_page_placement():
    abi, _, _ = _permuted_abi()
    out = materialize_paged_kv_abi(_AbiState(abi))
    assert out is abi
    np.testing.assert_array_equal(out.page_table, [2, 0, 1])


def test_rocm_attention_receives_paged_abi_not_dense_cache(monkeypatch):
    import tessera.cache.paged_kv as pk
    abi, _, _ = _permuted_abi()
    seen = {}

    def fake_rocm(q, got, token_indices, scale, causal):
        seen["abi"] = got
        k, v = got.gather(token_indices)
        out = pk._reference_attention(
            q, np.transpose(k, (1, 0, 2)), np.transpose(v, (1, 0, 2)),
            scale, causal)
        return out, "native_gpu"

    monkeypatch.setattr(pk, "_paged_attention_rocm", fake_rocm)
    q = np.ones((2, 2, 4), np.float32) * 0.1
    out, execution = paged_attention(
        q, _AbiState(abi), backend="rocm", return_execution=True)
    assert execution == "native_gpu" and seen["abi"] is abi
    assert out.shape == (2, 2, 4)


def test_rocm_hip_gather_source_uses_logical_to_physical_indirection():
    from tessera.compiler.emit.rocm_hip import _synthesize_paged_kv_read_hip
    source = _synthesize_paged_kv_read_hip()
    assert "pp=table[lp]" in source
    assert "pages[(((long long)pp*page_size+off)" in source


def test_rocm_direct_attention_source_consumes_plhd_without_staging():
    from tessera.compiler.emit.rocm_hip import (
        _synthesize_paged_attention_direct_hip,
    )
    source = _synthesize_paged_attention_direct_hip()
    assert "__global__ void paged_attn" in source
    assert "const float*kp,const float*vp,const int*table" in source
    assert "kh=qh/ratio" in source  # grouped/MQA head mapping
    assert "limit=qi+(T>Q?T-Q:0)" in source  # decode causal offset
    assert "pp=table[tok/L]" in source


def test_reference_attention_supports_mqa_and_arbitrary_token_order():
    from tessera.cache.paged_kv import _reference_attention
    rng = np.random.default_rng(91)
    q = rng.standard_normal((4, 3, 8)).astype(np.float32)
    k = rng.standard_normal((1, 7, 8)).astype(np.float32)
    v = rng.standard_normal((1, 7, 8)).astype(np.float32)
    order = np.asarray([6, 1, 5, 0, 3])
    actual = _reference_attention(q, k[:, order], v[:, order], 8 ** -.5, True)
    expected = _reference_attention(
        q, np.repeat(k[:, order], 4, axis=0),
        np.repeat(v[:, order], 4, axis=0), 8 ** -.5, True)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_rocm_route_warm_starts_from_committed_gfx1151_corpus():
    from tessera.cache.paged_kv import _rocm_paged_attention_corpus_winner
    assert _rocm_paged_attention_corpus_winner(4, 4, 1, 512, 32, 16) == "direct"


@pytest.mark.skipif(
    not _rocm_live(),
    reason="requires a live ROCm device and HIP toolchain",
)
def test_live_rocm_paged_gather_handles_permuted_pages():
    from tessera.compiler.emit.rocm_hip import run_paged_kv_cache_read_f32
    abi, logical_k, _ = _permuted_abi()
    idx = np.asarray([9, 0, 7, 4], dtype=np.int64)
    out = run_paged_kv_cache_read_f32(abi.k_pages, abi.page_table, idx)
    np.testing.assert_array_equal(out, logical_k[idx])
