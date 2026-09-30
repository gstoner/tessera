"""Host-free failure-path cleanup contracts for NVIDIA device sessions."""
from __future__ import annotations

import ctypes

import pytest

from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession


class _Buffer:
    def __init__(self, calls: list[str], name: str) -> None:
        self.calls, self.name = calls, name

    def close(self) -> None:
        self.calls.append(f"free:{self.name}")


class _CleanupLib:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def tessera_nvidia_stream_destroy(self, _stream: ctypes.c_void_p) -> int:
        self.calls.append("destroy-stream")
        return 0


def test_device_session_releases_every_buffer_and_stream_after_sync_failure():
    calls: list[str] = []
    session = NvidiaDeviceSession.__new__(NvidiaDeviceSession)
    session.lib = _CleanupLib(calls)
    session.stream = 17
    session._buffers = [_Buffer(calls, "a"), _Buffer(calls, "b")]
    session.synchronize = lambda: 3

    with pytest.raises(RuntimeError, match="synchronization"):
        session.close()

    assert calls == ["free:b", "free:a", "destroy-stream"]
    assert session.stream == 0
    assert session._buffers == []


class _TimingLib:
    def __init__(self) -> None:
        self.destroyed: list[int] = []
        self.creates = 0

    def tessera_nvidia_event_create(self, output) -> int:
        self.creates += 1
        if self.creates == 2:
            return 3
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(91)
        return 0

    def tessera_nvidia_event_destroy(self, event: ctypes.c_void_p) -> int:
        self.destroyed.append(int(event.value or 0))
        return 0


def test_timing_event_partial_creation_is_destroyed():
    session = NvidiaDeviceSession.__new__(NvidiaDeviceSession)
    session.lib = _TimingLib()
    session.stream = 17
    with pytest.raises(RuntimeError, match="event creation"):
        session.measure(lambda: None, reps=1, warmup=0)
    assert session.lib.destroyed == [91]


@pytest.mark.parametrize("order", ["C", "F"])
def test_strided_upload_rejects_padded_host_views_before_allocating(order):
    import numpy as np

    session = NvidiaDeviceSession.__new__(NvidiaDeviceSession)
    session.stream = 17
    session._buffers = []
    session.lib = object()
    if order == "C":
        backing = np.zeros((5, 9), dtype=np.float16, order="C")
        source = backing[:, :7]
    else:
        backing = np.zeros((9, 7), dtype=np.float16, order="F")
        source = backing[:5, :]
    assert not source.flags.c_contiguous
    assert not source.flags.f_contiguous

    allocated = False
    def unexpected_allocation(*args, **kwargs):
        nonlocal allocated
        allocated = True
        raise AssertionError("invalid strided source reached device allocation")
    session.empty = unexpected_allocation

    with pytest.raises(ValueError, match="compact row-major or column-major"):
        session.upload(source, layout="strided")
    assert allocated is False
