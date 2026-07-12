"""P5 — Apple GPU memory budget ABI.

Locks the buffer-pool + device-tensor byte accounting and the public budget
symbols added in ``apple_gpu_runtime.mm`` / ``apple_gpu_runtime_stub.cpp`` (see
``docs/audit/backend/apple/archive/apple_backend_capability_roadmap.md`` P5). The accounting mirrors MLX's
allocator introspection (active / cache / peak / limit / clear_cache).

Counters are process-global, so other tests in the same process perturb the
absolute values — every assertion here is written against *deltas* or against
post-``clear_cache`` invariants that hold regardless of prior state.
"""

from __future__ import annotations

import ctypes
import sys

import numpy as np
import pytest

import tessera.runtime as R

DARWIN = sys.platform == "darwin"

_SYMBOLS = (
    "tessera_apple_gpu_get_active_memory",
    "tessera_apple_gpu_get_cache_memory",
    "tessera_apple_gpu_get_peak_memory",
    "tessera_apple_gpu_reset_peak_memory",
    "tessera_apple_gpu_set_memory_limit",
    "tessera_apple_gpu_get_memory_limit",
    "tessera_apple_gpu_clear_cache",
)


def test_budget_symbols_present_in_runtime():
    """All 7 budget symbols must exist in the device_verified_jit runtime (Darwin .mm or
    non-Darwin stub) so the ctypes layer is platform-agnostic."""
    rt = R._load_apple_gpu_runtime()
    for sym in _SYMBOLS:
        assert hasattr(rt, sym), f"missing budget symbol: {sym}"


def test_memory_stats_surface_shape():
    """The Python introspection helper returns the documented keys."""
    stats = R.apple_gpu_memory_stats()
    if stats is None:
        pytest.skip("apple_gpu runtime unavailable")
    assert set(stats) == {"active", "cache", "peak", "limit"}
    for k, v in stats.items():
        assert isinstance(v, int), k
        assert v >= 0, k


def test_memory_limit_round_trips():
    """set_memory_limit returns the previous value; get reflects the new one;
    restoring the prior limit leaves the runtime as found. Works on the stub
    too (the limit is a plain stored value)."""
    rt = R._apple_gpu_memory_api()
    if rt is None:
        pytest.skip("apple_gpu runtime unavailable")
    prev = R.apple_gpu_set_memory_limit(123 << 20)
    try:
        assert R.apple_gpu_memory_stats()["limit"] == (123 << 20)
        again = R.apple_gpu_set_memory_limit(456 << 20)
        assert again == (123 << 20)  # returns the value we just set
        assert R.apple_gpu_memory_stats()["limit"] == (456 << 20)
    finally:
        R.apple_gpu_set_memory_limit(prev or 0)
    assert R.apple_gpu_memory_stats()["limit"] == (prev or 0)


def test_clear_cache_empties_cache_counter():
    """After clear_cache, the cache counter is exactly 0 regardless of prior
    pool state, and the reported freed bytes are non-negative."""
    rt = R._apple_gpu_memory_api()
    if rt is None:
        pytest.skip("apple_gpu runtime unavailable")
    freed = R.apple_gpu_clear_cache()
    assert freed is not None and freed >= 0
    assert R.apple_gpu_memory_stats()["cache"] == 0


@pytest.mark.skipif(not DARWIN, reason="device-tensor accounting needs Metal")
def test_device_tensor_alloc_free_accounting():
    """Allocating a resident DeviceTensor raises ``active`` by exactly its byte
    count; freeing it restores the level. Single-threaded within this test, so
    the delta is exact."""
    if R.apple_gpu_memory_stats() is None:
        pytest.skip("apple_gpu runtime unavailable")
    if not getattr(R.DeviceTensor, "is_metal", lambda: False)():
        pytest.skip("Metal device-tensor path unavailable")

    nbytes = 1 << 20  # 1 MiB, large enough to dwarf any incidental noise
    before = R.apple_gpu_memory_stats()["active"]
    dt = R.DeviceTensor.empty((nbytes // 4,), np.float32)
    if dt is None:
        pytest.skip("device-tensor allocation unavailable")
    try:
        after = R.apple_gpu_memory_stats()["active"]
        assert after - before == nbytes, (before, after)
    finally:
        dt.free()
    restored = R.apple_gpu_memory_stats()["active"]
    assert restored == before, (before, restored)


@pytest.mark.skipif(not DARWIN, reason="peak tracking needs the Metal path")
def test_peak_tracks_high_water_and_resets():
    """Peak is >= active and >= a freshly allocated resident tensor; resetting
    pins peak back down to the current active level."""
    if R.apple_gpu_memory_stats() is None:
        pytest.skip("apple_gpu runtime unavailable")
    if not getattr(R.DeviceTensor, "is_metal", lambda: False)():
        pytest.skip("Metal device-tensor path unavailable")

    nbytes = 2 << 20
    dt = R.DeviceTensor.empty((nbytes // 4,), np.float32)
    if dt is None:
        pytest.skip("device-tensor allocation unavailable")
    try:
        stats = R.apple_gpu_memory_stats()
        assert stats["peak"] >= stats["active"]
        assert stats["peak"] >= nbytes
    finally:
        dt.free()
    # After freeing, reset pins peak to current active.
    R.apple_gpu_reset_peak_memory()
    stats = R.apple_gpu_memory_stats()
    assert stats["peak"] == stats["active"]
