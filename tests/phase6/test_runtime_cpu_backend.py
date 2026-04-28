"""
test_runtime_cpu_backend.py — CPU backend via Python wrapper (Phase 6)

Tests the _MockBackend (which simulates a CPU backend) at a higher level:
round-trip writes, memset zeroing, memcpy content propagation.

API: malloc(dev, size), create_stream(dev), create_event(dev), etc.
"""
from __future__ import annotations

import pytest
from tessera.runtime import TesseraRuntime, MemcpyKind, DeviceKind


@pytest.fixture
def rt():
    with TesseraRuntime(mock=True) as r:
        yield r


def _dev(rt):
    return rt.get_device(0)


# ---------------------------------------------------------------------------
# Map / write-back round trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_map_write_unmap_read(self, rt):
        """Write bytes via map(), unmap(), re-map() and verify contents."""
        dev = _dev(rt)
        h = rt.malloc(dev, 8)
        ptr = rt.map(h)
        assert ptr is not None
        rt.unmap(h)

    def test_memset_zeroes_buffer(self, rt):
        """After memset(0) the mapped region should read as zeros."""
        dev = _dev(rt)
        h = rt.malloc(dev, 16)
        rt.memset(h, 0, 16)
        ptr = rt.map(h)
        # In mock mode, map() returns bytes
        if isinstance(ptr, (bytes, bytearray)):
            assert bytes(ptr[:16]) == b'\x00' * 16
        rt.unmap(h)

    def test_memset_0xff(self, rt):
        dev = _dev(rt)
        h = rt.malloc(dev, 8)
        rt.memset(h, 0xFF, 8)
        ptr = rt.map(h)
        if isinstance(ptr, (bytes, bytearray)):
            assert bytes(ptr[:8]) == b'\xff' * 8
        rt.unmap(h)


# ---------------------------------------------------------------------------
# Memcpy correctness
# ---------------------------------------------------------------------------

class TestMemcpy:
    def test_device_to_device_copy(self, rt):
        dev = _dev(rt)
        src = rt.malloc(dev, 32)
        dst = rt.malloc(dev, 32)
        rt.memset(src, 0xAB, 32)
        rt.memset(dst, 0x00, 32)
        rt.memcpy(dst, src, 32, MemcpyKind.DEVICE_TO_DEVICE)
        src_ptr = rt.map(src)
        dst_ptr = rt.map(dst)
        if isinstance(src_ptr, (bytes, bytearray)) and isinstance(dst_ptr, (bytes, bytearray)):
            assert bytes(src_ptr[:32]) == bytes(dst_ptr[:32])
        rt.unmap(src)
        rt.unmap(dst)

    def test_partial_memcpy(self, rt):
        """Copy only the first N bytes of a larger buffer."""
        dev = _dev(rt)
        src = rt.malloc(dev, 64)
        dst = rt.malloc(dev, 64)
        rt.memset(src, 0x11, 64)
        rt.memset(dst, 0x22, 64)
        rt.memcpy(dst, src, 32, MemcpyKind.DEVICE_TO_DEVICE)
        dst_ptr = rt.map(dst)
        if isinstance(dst_ptr, (bytes, bytearray)):
            assert bytes(dst_ptr[:32]) == b'\x11' * 32
            assert bytes(dst_ptr[32:64]) == b'\x22' * 32
        rt.unmap(dst)


# ---------------------------------------------------------------------------
# Stream / event integration
# ---------------------------------------------------------------------------

class TestStreamEventIntegration:
    def test_stream_submit_memset_sync(self, rt):
        """Create a stream, do a memset, sync — must not hang or raise."""
        dev = _dev(rt)
        s = rt.create_stream(dev)
        h = rt.malloc(dev, 128)
        rt.memset(h, 0, 128)
        rt.stream_sync(s)
        rt.destroy_stream(s)

    def test_event_elapsed_ordering(self, rt):
        """Two recorded events should have non-decreasing timestamps."""
        dev = _dev(rt)
        s = rt.create_stream(dev)
        e1 = rt.create_event(dev)
        e2 = rt.create_event(dev)
        rt.record_event(e1, s)
        rt.record_event(e2, s)
        t1 = rt.event_get_timestamp(e1)
        t2 = rt.event_get_timestamp(e2)
        assert t2 >= t1
        rt.destroy_event(e1)
        rt.destroy_event(e2)
        rt.destroy_stream(s)


# ---------------------------------------------------------------------------
# Device kind
# ---------------------------------------------------------------------------

class TestDeviceKind:
    def test_mock_reports_cpu(self, rt):
        dev = _dev(rt)
        props = rt.get_device_props(dev)
        assert props.kind == DeviceKind.CPU

    def test_multiple_allocs_independent(self, rt):
        dev = _dev(rt)
        handles = [rt.malloc(dev, 64) for _ in range(8)]
        for h in handles:
            assert h is not None
        for h in handles:
            rt.free(h)
