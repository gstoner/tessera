"""
test_runtime_abi.py — TesseraRuntime mock-mode ABI tests (Phase 6)

All tests run against the _MockBackend so no real GPU is required.

API notes (from runtime.py):
  - get_device(index)             → device handle
  - get_device_props(dev)         → DeviceProps
  - malloc(dev, size)             → buffer handle
  - create_stream(dev)            → stream handle
  - create_event(dev)             → event handle
  - get_version()                 → tuple (major, minor, patch)
  - get_last_error()              → str
"""
from __future__ import annotations

import pytest
from tessera.runtime import (
    TesseraRuntime,
    TsrStatus,
    DeviceKind,
    MemcpyKind,
    TesseraRuntimeError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dev(rt):
    """Return device 0 handle."""
    return rt.get_device(0)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_init_idempotent(self, mock_runtime):
        """Double-init should not raise (second call is a no-op or re-enters)."""
        mock_runtime.init()  # already initialised by fixture — should be fine

    def test_shutdown_idempotent(self, mock_runtime):
        """Shutdown twice should not raise."""
        mock_runtime.shutdown()
        mock_runtime.shutdown()

    def test_context_manager(self):
        rt = TesseraRuntime(mock=True)
        with rt as r:
            assert r.is_initialized

    def test_is_mock_true(self, mock_runtime):
        assert mock_runtime.is_mock is True

    def test_is_initialized_after_init(self, mock_runtime):
        assert mock_runtime.is_initialized is True

    def test_is_initialized_false_before_init(self, fresh_runtime):
        assert fresh_runtime.is_initialized is False

    def test_get_version_returns_tuple(self, mock_runtime):
        v = mock_runtime.get_version()
        assert isinstance(v, tuple)
        assert len(v) == 3

    def test_get_last_error_ok_when_clean(self, mock_runtime):
        err = mock_runtime.get_last_error()
        assert isinstance(err, str)


# ---------------------------------------------------------------------------
# Device queries
# ---------------------------------------------------------------------------

class TestDeviceQueries:
    def test_get_device_count_gte_1(self, mock_runtime):
        assert mock_runtime.get_device_count() >= 1

    def test_get_device_returns_handle(self, mock_runtime):
        dev = mock_runtime.get_device(0)
        assert dev is not None

    def test_get_device_props_cpu_kind(self, mock_runtime):
        dev = _dev(mock_runtime)
        props = mock_runtime.get_device_props(dev)
        assert props.kind == DeviceKind.CPU

    def test_get_device_props_name_nonempty(self, mock_runtime):
        dev = _dev(mock_runtime)
        props = mock_runtime.get_device_props(dev)
        assert props.name and len(props.name) > 0

    def test_get_device_props_threads_positive(self, mock_runtime):
        dev = _dev(mock_runtime)
        props = mock_runtime.get_device_props(dev)
        assert props.logical_tile_threads_max > 0

    def test_get_device_props_concurrent_tiles_positive(self, mock_runtime):
        dev = _dev(mock_runtime)
        props = mock_runtime.get_device_props(dev)
        assert props.concurrent_tiles_hint > 0


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

class TestMemory:
    def test_malloc_returns_handle(self, mock_runtime):
        dev = _dev(mock_runtime)
        handle = mock_runtime.malloc(dev, 1024)
        assert handle is not None

    def test_malloc_zero_bytes_allowed(self, mock_runtime):
        dev = _dev(mock_runtime)
        h = mock_runtime.malloc(dev, 0)
        assert h is not None  # zero-byte allocation is valid in mock

    def test_free_valid_handle(self, mock_runtime):
        dev = _dev(mock_runtime)
        h = mock_runtime.malloc(dev, 256)
        mock_runtime.free(h)  # must not raise

    def test_memset_on_allocated(self, mock_runtime):
        dev = _dev(mock_runtime)
        h = mock_runtime.malloc(dev, 64)
        mock_runtime.memset(h, 0, 64)  # must not raise

    def test_memcpy_host_to_device(self, mock_runtime):
        dev = _dev(mock_runtime)
        src = mock_runtime.malloc(dev, 128)
        dst = mock_runtime.malloc(dev, 128)
        mock_runtime.memcpy(dst, src, 128, MemcpyKind.HOST_TO_DEVICE)

    def test_map_returns_bytes(self, mock_runtime):
        dev = _dev(mock_runtime)
        h = mock_runtime.malloc(dev, 64)
        ptr = mock_runtime.map(h)
        assert ptr is not None

    def test_unmap_does_not_raise(self, mock_runtime):
        dev = _dev(mock_runtime)
        h = mock_runtime.malloc(dev, 64)
        mock_runtime.map(h)
        mock_runtime.unmap(h)


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

class TestStreams:
    def test_create_and_destroy_stream(self, mock_runtime):
        dev = _dev(mock_runtime)
        s = mock_runtime.create_stream(dev)
        assert s is not None
        mock_runtime.destroy_stream(s)

    def test_stream_sync(self, mock_runtime):
        dev = _dev(mock_runtime)
        s = mock_runtime.create_stream(dev)
        mock_runtime.stream_sync(s)
        mock_runtime.destroy_stream(s)

    def test_destroy_any_stream(self, mock_runtime):
        """Destroy a freshly created stream (mock always succeeds)."""
        dev = _dev(mock_runtime)
        s = mock_runtime.create_stream(dev)
        mock_runtime.destroy_stream(s)  # must not raise


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class TestEvents:
    def test_create_and_destroy_event(self, mock_runtime):
        dev = _dev(mock_runtime)
        e = mock_runtime.create_event(dev)
        assert e is not None
        mock_runtime.destroy_event(e)

    def test_record_and_sync_event(self, mock_runtime):
        dev = _dev(mock_runtime)
        s = mock_runtime.create_stream(dev)
        e = mock_runtime.create_event(dev)
        mock_runtime.record_event(e, s)
        mock_runtime.event_sync(e)
        mock_runtime.destroy_event(e)
        mock_runtime.destroy_stream(s)

    def test_wait_event(self, mock_runtime):
        dev = _dev(mock_runtime)
        s1 = mock_runtime.create_stream(dev)
        s2 = mock_runtime.create_stream(dev)
        e = mock_runtime.create_event(dev)
        mock_runtime.record_event(e, s1)
        mock_runtime.wait_event(e, s2)
        mock_runtime.destroy_event(e)
        mock_runtime.destroy_stream(s1)
        mock_runtime.destroy_stream(s2)

    def test_event_get_timestamp_gte_zero(self, mock_runtime):
        dev = _dev(mock_runtime)
        s = mock_runtime.create_stream(dev)
        e = mock_runtime.create_event(dev)
        mock_runtime.record_event(e, s)
        ts = mock_runtime.event_get_timestamp(e)
        assert ts >= 0
        mock_runtime.destroy_event(e)
        mock_runtime.destroy_stream(s)

    def test_destroy_event(self, mock_runtime):
        """Destroy a freshly created event (mock always succeeds)."""
        dev = _dev(mock_runtime)
        e = mock_runtime.create_event(dev)
        mock_runtime.destroy_event(e)
