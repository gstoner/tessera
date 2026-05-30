"""Apple GPU R4 — device-resident latent KV cache.

`tessera.cache.ResidentLatentKVCache` keeps c_kv + k_rope in resident device
buffers: append writes only the new token in place (no upload), and the window
is a zero-copy prefix view. These tests verify the contents, the zero-copy
append semantics, the device-resident window feeding a device-resident bmm
(R1), and the residency invariant (one allocation, no per-append realloc).
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as R
from tessera.cache import ResidentLatentKVCache


def _require():
    if R._apple_gpu_devtensor_api() is None:
        pytest.skip("device-tensor ABI unavailable")


def test_append_and_window_contents():
    c = ResidentLatentKVCache(latent_dim=8, rope_dim=4, max_seq=16)
    rng = np.random.RandomState(0)
    toks_c, toks_r = [], []
    for step in range(5):
        cc = rng.randn(1, 8).astype(np.float32)
        rr = rng.randn(1, 4).astype(np.float32)
        c.append(cc, rr)
        toks_c.append(cc)
        toks_r.append(rr)
    assert c.current_seq == 5
    np.testing.assert_array_equal(c.latent_numpy(), np.concatenate(toks_c))
    np.testing.assert_array_equal(c.rope_numpy(), np.concatenate(toks_r))
    c.free()


def test_prefill_then_append():
    c = ResidentLatentKVCache(latent_dim=6, rope_dim=2, max_seq=32)
    rng = np.random.RandomState(1)
    pre_c, pre_r = rng.randn(5, 6).astype(np.float32), rng.randn(5, 2).astype(np.float32)
    c.append(pre_c, pre_r)
    one_c, one_r = rng.randn(1, 6).astype(np.float32), rng.randn(1, 2).astype(np.float32)
    c.append(one_c, one_r)
    assert c.current_seq == 6
    np.testing.assert_array_equal(c.latent_numpy(), np.concatenate([pre_c, one_c]))
    c.free()


def test_window_is_zero_copy_view_of_resident_buffer():
    """A second append is visible through a window captured earlier — they alias
    the same resident buffer (no copy)."""
    _require()
    c = ResidentLatentKVCache(latent_dim=4, rope_dim=2, max_seq=8)
    if not c.resident:
        c.free()
        pytest.skip("not resident on this host")
    a = np.arange(4, dtype=np.float32).reshape(1, 4)
    c.append(a, np.zeros((1, 2), np.float32))
    win = c.latent_window()                    # DeviceTensor prefix [1,4]
    assert win.numpy()[0, 0] == 0.0
    # mutate the resident storage in place via the cache's view
    c.latent_numpy()[0, 0] = 99.0
    assert win.numpy()[0, 0] == 99.0           # same backing buffer
    c.free()


def test_resident_window_feeds_device_bmm():
    """The resident latent window (R4) feeds a device-resident bmm (R1) with no
    intervening host copy — c_kv [S,Dl] @ W [Dl,N]."""
    _require()
    c = ResidentLatentKVCache(latent_dim=8, rope_dim=4, max_seq=16)
    if not c.resident:
        c.free()
        pytest.skip("not resident")
    rng = np.random.RandomState(2)
    latents = rng.randn(6, 8).astype(np.float32)
    c.append(latents, rng.randn(6, 4).astype(np.float32))
    win = c.latent_window()                              # [6, 8] resident
    W = R.DeviceTensor.from_numpy(rng.randn(1, 8, 5).astype(np.float32))
    # bmm wants [batch,M,K] x [batch|1,K,N]: view window as [1, 6, 8]
    out = R._apple_gpu_bmm_device(win.reshape_view(1, 6, 8), W)
    assert out is not None and out.shape == (1, 6, 5)
    ref = latents.astype(np.float64) @ W.numpy()[0].astype(np.float64)
    np.testing.assert_allclose(out.numpy()[0], ref, rtol=1e-4, atol=1e-4)
    out.free(); W.free(); c.free()


def test_single_allocation_across_appends():
    """Appending does not reallocate — current_seq advances within the same
    resident buffer (latent storage object identity is stable)."""
    c = ResidentLatentKVCache(latent_dim=4, rope_dim=2, max_seq=64)
    base = c._latent_view
    for i in range(20):
        c.append(np.full((1, 4), i, np.float32), np.zeros((1, 2), np.float32))
    assert c._latent_view is base          # same backing view, no realloc
    assert c.current_seq == 20
    c.free()


def test_overflow_raises():
    c = ResidentLatentKVCache(latent_dim=4, rope_dim=2, max_seq=3)
    c.append(np.zeros((3, 4), np.float32), np.zeros((3, 2), np.float32))
    with pytest.raises(ValueError):
        c.append(np.zeros((1, 4), np.float32), np.zeros((1, 2), np.float32))
    c.free()


def test_cache_footprint():
    c = ResidentLatentKVCache(latent_dim=512, rope_dim=64, max_seq=8)
    assert c.cache_bytes_per_token() == (512 + 64) * 4
    c.free()


def test_prefix_view_bounds():
    _require()
    dt = R.DeviceTensor.empty((10, 4), np.float32)
    if dt is None:
        pytest.skip("no device tensor")
    assert dt.prefix_view(3).shape == (3, 4)
    assert dt.prefix_view(0).shape == (0, 4)
    with pytest.raises(ValueError):
        dt.prefix_view(11)
    dt.free()
