"""Apple GPU — device-resident block-paged KV cache (on-GPU gather).

`tessera.cache.ResidentBlockPagedKVCache` holds the physical block pool in a
resident device buffer; appends write in place (no upload), and a sequence's
(possibly non-contiguous) window is assembled on-GPU via the block-table gather
kernel (`tessera_apple_gpu_gather_blocks_dev_f32`, MPSGraph gather along the
block axis). These tests validate the gather contents, non-contiguous block
tables, concurrent sequences, and pool lifecycle.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R
from tessera.cache import (ResidentBlockPagedKVCache,
                           ResidentBlockPagedKVCacheError)


def _np(win):
    return win.numpy() if hasattr(win, "numpy") else win


def _require():
    if R._apple_gpu_gather_blocks_dev_sym() is None:
        pytest.skip("gather kernel unavailable")


def test_gather_matches_appended_tokens():
    c = ResidentBlockPagedKVCache(latent_dim=8, rope_dim=4, num_blocks=8, block_size=4)
    rng = np.random.RandomState(0)
    c.add_sequence("a")
    toks_c, toks_r = [], []
    for _ in range(6):
        cc = rng.randn(1, 8).astype(np.float32)
        rr = rng.randn(1, 4).astype(np.float32)
        c.append("a", cc, rr)
        toks_c.append(cc)
        toks_r.append(rr)
    np.testing.assert_allclose(_np(c.gather_latent("a")), np.concatenate(toks_c),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(_np(c.gather_rope("a")), np.concatenate(toks_r),
                               rtol=1e-5, atol=1e-5)
    c.free()


def test_non_contiguous_block_table_gather():
    """Interleaving two sequences gives each non-contiguous blocks; the on-GPU
    gather must still reassemble the right window."""
    _require()
    c = ResidentBlockPagedKVCache(latent_dim=6, rope_dim=2, num_blocks=8, block_size=2)
    rng = np.random.RandomState(1)
    c.add_sequence("a"); c.add_sequence("b")
    a_c = []
    for _ in range(3):
        ca = rng.randn(2, 6).astype(np.float32)
        cb = rng.randn(2, 6).astype(np.float32)
        c.append("a", ca, rng.randn(2, 2).astype(np.float32)); a_c.append(ca)
        c.append("b", cb, rng.randn(2, 2).astype(np.float32))
    bt = c.block_table("a")
    assert bt != list(range(bt[0], bt[0] + len(bt)))   # non-contiguous
    np.testing.assert_allclose(_np(c.gather_latent("a")), np.concatenate(a_c),
                               rtol=1e-5, atol=1e-5)
    c.free()


def test_concurrent_sequences():
    c = ResidentBlockPagedKVCache(latent_dim=8, rope_dim=4, num_blocks=32, block_size=4)
    rng = np.random.RandomState(2)
    refs = {}
    for sid, L in (("x", 3), ("y", 9), ("z", 16)):
        c.add_sequence(sid)
        cc = rng.randn(L, 8).astype(np.float32)
        c.append(sid, cc, rng.randn(L, 4).astype(np.float32))
        refs[sid] = cc
    for sid, ref in refs.items():
        np.testing.assert_allclose(_np(c.gather_latent(sid)), ref, rtol=1e-5, atol=1e-5)
    c.free()


def test_free_and_reuse_blocks():
    c = ResidentBlockPagedKVCache(latent_dim=4, rope_dim=2, num_blocks=4, block_size=4)
    rng = np.random.RandomState(3)
    c.add_sequence("a")
    c.append("a", rng.randn(8, 4).astype(np.float32), rng.randn(8, 2).astype(np.float32))
    assert c.num_free_blocks == 2
    used = set(c.block_table("a"))
    c.free_sequence("a")
    assert c.num_free_blocks == 4
    c.add_sequence("b")
    c.append("b", rng.randn(8, 4).astype(np.float32), rng.randn(8, 2).astype(np.float32))
    assert set(c.block_table("b")) == used   # reclaimed pages
    c.free()


def test_pool_exhaustion():
    c = ResidentBlockPagedKVCache(latent_dim=4, rope_dim=2, num_blocks=2, block_size=4)
    rng = np.random.RandomState(4)
    c.add_sequence("a")
    c.append("a", rng.randn(8, 4).astype(np.float32), rng.randn(8, 2).astype(np.float32))
    c.add_sequence("b")
    with pytest.raises(ResidentBlockPagedKVCacheError):
        c.append("b", rng.randn(1, 4).astype(np.float32), rng.randn(1, 2).astype(np.float32))
    c.free()


def test_failed_multiblock_append_is_transactional_and_leak_free():
    c = ResidentBlockPagedKVCache(
        latent_dim=4, rope_dim=2, num_blocks=3, block_size=2)
    rng = np.random.default_rng(41)
    c.add_sequence("a")
    before = c.lifecycle_telemetry()
    with pytest.raises(ResidentBlockPagedKVCacheError, match="exhausted"):
        c.append("a", rng.standard_normal((7, 4)).astype(np.float32),
                 rng.standard_normal((7, 2)).astype(np.float32))
    assert c.sequence_length("a") == 0
    assert c.block_table("a") == []
    after = c.lifecycle_telemetry()
    assert after["free_blocks"] == before["free_blocks"] == 3
    assert after["used_blocks"] == 0
    c.free()
    assert c.lifecycle_telemetry()["live_sequences"] == 0


def test_direct_attention_nonidentity_pages_causal_offsets_and_window():
    c = ResidentBlockPagedKVCache(
        latent_dim=5, rope_dim=3, num_blocks=8, block_size=2)
    rng = np.random.default_rng(42)
    c.add_sequence("a")
    c.add_sequence("interleave")
    values = []
    keys = []
    for _ in range(3):
        v = rng.standard_normal((2, 5)).astype(np.float32)
        k = rng.standard_normal((2, 3)).astype(np.float32)
        values.append(v); keys.append(k)
        c.append("a", v, k)
        c.append("interleave", rng.standard_normal((2, 5)).astype(np.float32),
                 rng.standard_normal((2, 3)).astype(np.float32))
    assert c.block_table("a") == [0, 2, 4]
    q = rng.standard_normal((2, 3)).astype(np.float32)
    got = _np(c.attention(
        "a", q, scale=0.7, causal=True, causal_offset=3, window=3)).copy()
    staged = _np(c.attention(
        "a", q, scale=0.7, causal=True, causal_offset=3, window=3,
        route="staged")).copy()
    dense_v = np.concatenate(values)
    dense_k = np.concatenate(keys)
    want = []
    for qi, limit in enumerate((3, 4)):
        start = limit - 2
        logits = q[qi] @ dense_k[start:limit + 1].T * 0.7
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        want.append(weights @ dense_v[start:limit + 1])
    np.testing.assert_allclose(got, np.asarray(want), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(staged, np.asarray(want), rtol=1e-5, atol=1e-5)
    c.free()


def test_direct_attention_rejects_invalid_window_and_empty_sequence():
    c = ResidentBlockPagedKVCache(
        latent_dim=4, rope_dim=2, num_blocks=2, block_size=2)
    c.add_sequence("a")
    with pytest.raises(ResidentBlockPagedKVCacheError, match="empty"):
        c.attention("a", np.zeros((1, 2), np.float32))
    c.append("a", np.ones((1, 4), np.float32), np.ones((1, 2), np.float32))
    with pytest.raises(ValueError, match="window"):
        c.attention("a", np.zeros((1, 2), np.float32), window=0)
    c.free()


def test_gathered_window_feeds_device_bmm():
    """The on-GPU gathered window (a resident DeviceTensor) feeds a
    device-resident bmm with no host copy."""
    _require()
    c = ResidentBlockPagedKVCache(latent_dim=8, rope_dim=4, num_blocks=8, block_size=4)
    rng = np.random.RandomState(5)
    c.add_sequence("a")
    lat = rng.randn(6, 8).astype(np.float32)
    c.append("a", lat, rng.randn(6, 4).astype(np.float32))
    win = c.gather_latent("a")
    if not hasattr(win, "tensor"):
        c.free(); pytest.skip("not resident")
    W = R.DeviceTensor.from_numpy(rng.randn(1, 8, 5).astype(np.float32))
    out = R._apple_gpu_bmm_device(win.tensor.reshape_view(1, 6, 8), W)
    assert out is not None
    np.testing.assert_allclose(out.numpy()[0],
                               lat.astype(np.float64) @ W.numpy()[0].astype(np.float64),
                               rtol=1e-4, atol=1e-4)
    out.free(); W.free(); win.free(); c.free()


def test_symbols_exported():
    rt = R._load_apple_gpu_runtime()
    assert hasattr(rt, "tessera_apple_gpu_gather_blocks_dev_f32")
    assert hasattr(rt, "tessera_apple_gpu_gather_blocks_dev_f32_enc")
    assert hasattr(rt, "tessera_apple_gpu_paged_latent_attention_dev_f32")
    assert hasattr(rt, "tessera_apple_gpu_dense_latent_attention_dev_f32")


@pytest.mark.hardware_apple_gpu
def test_noncontiguous_gather_retains_native_provenance_and_honest_resources():
    from tessera._apple_gpu_dispatch import (
        clear_dispatch_telemetry, set_dispatch_telemetry_enabled)
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    c = ResidentBlockPagedKVCache(
        latent_dim=6, rope_dim=2, num_blocks=8, block_size=2)
    rng = np.random.default_rng(2007)
    c.add_sequence("a")
    c.add_sequence("b")
    expected = []
    for _ in range(3):
        a = rng.standard_normal((2, 6)).astype(np.float32)
        expected.append(a)
        c.append("a", a, rng.standard_normal((2, 2)).astype(np.float32))
        c.append("b", rng.standard_normal((2, 6)).astype(np.float32),
                 rng.standard_normal((2, 2)).astype(np.float32))
    assert c.block_table("a") == [0, 2, 4]
    try:
        assert set_dispatch_telemetry_enabled(True)
        clear_dispatch_telemetry()
        window = c.gather_latent("a")
        assert c.last_gather_execution == "native_gpu"
        assert c.last_gather_telemetry["capture_enabled"] is True
        resources = c.last_gather_telemetry["resources"]
        assert resources["api"] == "MPSGraph.gatherWithUpdatesTensor"
        assert resources["pipeline_limits"] is None
        np.testing.assert_allclose(
            _np(window), np.concatenate(expected), rtol=1e-5, atol=1e-5)
        window.free()
    finally:
        set_dispatch_telemetry_enabled(False)
        c.free()


@pytest.mark.hardware_apple_gpu
def test_direct_attention_is_native_and_does_not_allocate_staging_window():
    from tessera._apple_gpu_dispatch import (
        clear_dispatch_telemetry, set_dispatch_telemetry_enabled)
    from tests._support.apple import require_apple_metal

    require_apple_metal()
    c = ResidentBlockPagedKVCache(
        latent_dim=16, rope_dim=8, num_blocks=8, block_size=4)
    rng = np.random.default_rng(2042)
    c.add_sequence("a"); c.add_sequence("b")
    values, keys = [], []
    for _ in range(2):
        v = rng.standard_normal((4, 16)).astype(np.float32)
        k = rng.standard_normal((4, 8)).astype(np.float32)
        values.append(v); keys.append(k)
        c.append("a", v, k)
        c.append("b", rng.standard_normal((4, 16)).astype(np.float32),
                 rng.standard_normal((4, 8)).astype(np.float32))
    q = rng.standard_normal((2, 8)).astype(np.float32)
    before = c.lifecycle_telemetry()
    try:
        assert set_dispatch_telemetry_enabled(True)
        clear_dispatch_telemetry()
        out = c.attention("a", q, causal=True, window=5, route="direct")
        assert c.last_attention_execution == "native_gpu"
        got = out.copy_to_host(); out.free()
        staged = c.attention("a", q, causal=True, window=5, route="staged")
        assert c.last_attention_execution == "native_gpu"
        staged_np = staged.copy_to_host(); staged.free()
        # Force the same page-table oracle through the host branch.
        c._resident = False
        want = c.attention("a", q, causal=True, window=5)
        c._resident = True
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(staged_np, want, rtol=2e-4, atol=2e-4)
        after = c.lifecycle_telemetry()
        assert after["used_blocks"] == before["used_blocks"]
        assert after["resident_page_tables"] == before["resident_page_tables"]
    finally:
        set_dispatch_telemetry_enabled(False)
        c.free()
