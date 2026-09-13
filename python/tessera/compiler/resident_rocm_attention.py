"""Scoped, reusable HIP attention workspace with queued host submissions.

Each owner uses a nonblocking HIP stream and a host worker. Calls on one owner
remain ordered; separate owners can progress independently. External read-only
pointers are available only through stream-ordered leases. Host results remain
independent snapshots. Saved-LSE admission is not
changed by this owner: the scheduled program's checkpoint policy is preserved.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
from typing import Any, Mapping

import numpy as np


class ResidentROCmAttentionTape:
    """Own one compiled backward program and immutable Q/K/V/bias snapshots."""

    def __init__(self, program: Any, buffers: Mapping[str, Any]):
        from tessera import runtime as rt

        # Copy before dispatch: caller mutation cannot alter queued capture.
        snapshots = {name: np.array(value, copy=True, order="C") for name, value in buffers.items()}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tessera-rocm-attention")
        self._closing = False
        self._failed = False
        self._has_result = False
        self._retirement: Future | None = None
        self._readers = 0
        self._release_scheduled = False
        self._context: Any = None
        self._execute: Any = None
        hip = rt._load_hip_for_launch()
        if hip is None:
            self._executor.shutdown()
            raise RuntimeError("resident attention requires HIP")
        self._hip = hip
        import ctypes as ct
        device = ct.c_int()
        if self._hip.hipGetDevice(ct.byref(device)) != 0:
            self._executor.shutdown()
            raise RuntimeError("resident attention cannot identify its device")
        self._device = device.value
        try:
            self._executor.submit(self._capture, program, snapshots).result()
        except BaseException:
            self._executor.shutdown(wait=True)
            raise

    def _capture(self, program, snapshots):
        self._worker_ident = threading.get_ident()
        from tessera import runtime as rt
        if self._hip.hipSetDevice(self._device) != 0:
            raise RuntimeError("resident attention cannot enter its owning device")
        self._context = rt._bind_rocm_attention_backward_program(program, snapshots)
        self._execute = self._context.__enter__()
        # Locate dO from the validated prepass contract, not a public name.
        forward = {b.name for b in program.descriptors[0].buffers if b.direction == "input"}
        do_names = [b.name for b in program.descriptors[1].buffers
                    if b.layout != "program_workspace" and b.name not in forward]
        if len(do_names) != 1:
            self._context.__exit__(None, None, None)
            raise ValueError("resident attention requires one cotangent input")
        do = snapshots[do_names[0]]
        self._shape, self._dtype = do.shape, do.dtype

    def submit(self, cotangent) -> Future:
        """Queue a backward call, retaining the frame until it completes."""
        with self._lock:
            if self._closing or self._failed or self._readers:
                raise ValueError("resident attention is retiring, failed or has active readers")
            value = np.asarray(cotangent)
            if value.shape != self._shape or value.dtype != self._dtype:
                raise ValueError("resident attention cotangent shape/storage must match capture")
            snapshot = np.array(value, copy=True, order="C")
            return self._executor.submit(self._run, snapshot)

    def _run(self, snapshot):
        if self._failed:
            raise RuntimeError("resident attention previous submission failed")
        try:
            result = self._execute(cotangent=snapshot)
            self._has_result = True
            return result
        except BaseException:
            with self._lock:
                self._failed = True
            raise

    def backward(self, cotangent):
        return self.submit(cotangent).result()["outputs"]

    def reader(self, stream: int):
        """Lease outputs read-only on an explicit same-device HIP stream.

        Consumer work must be enqueued before closing the lease. Pointer use
        after release, writes and transport to another device are forbidden.
        """
        if threading.get_ident() == self._worker_ident:
            raise RuntimeError("blocking reader admission on the owning worker is invalid")
        if type(stream) is not int or stream <= 0:
            raise ValueError("reader requires an explicit non-default HIP stream")
        with self._lock:
            if self._closing or self._failed:
                raise ValueError("resident attention is retiring or failed")
            self._readers += 1
            future = self._executor.submit(_AttentionReader, self, stream)
        try:
            return future.result()
        except BaseException:
            with self._lock:
                self._readers -= 1
                self._schedule_release()
            raise

    def _schedule_release(self):
        if self._retirement is None or self._readers or self._release_scheduled:
            return
        self._release_scheduled = True
        retirement = self._retirement
        work = self._executor.submit(self._release)
        def complete(future):
            try:
                retirement.set_result(future.result())
            except BaseException as exc:
                retirement.set_exception(exc)
        work.add_done_callback(complete)
        self._executor.shutdown(wait=False)

    def retire(self) -> Future:
        """Close admission; teardown waits for submissions and reader releases."""
        with self._lock:
            if self._retirement is None:
                self._closing = True
                self._retirement = Future()
                self._retirement.set_running_or_notify_cancel()
                self._schedule_release()
            return self._retirement

    def _release(self):
        self._execute = None
        self._context.__exit__(None, None, None)
        self._context = None

    def close(self):
        with self._lock:
            if self._readers:
                raise ValueError("release external readers before blocking close; retire is nonblocking")
        if threading.get_ident() == self._worker_ident:
            raise RuntimeError("blocking close on the owning worker is invalid; use retire")
        try:
            self.retire().result()
        finally:
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except BaseException:
            if exc is None:
                raise


class _AttentionReader:
    def __init__(self, owner, stream):
        import ctypes as ct
        from types import MappingProxyType
        if not owner._has_result or owner._failed:
            raise ValueError("reader requires a completed backward result")
        self._owner = owner
        self._stream = ct.c_void_p(stream)
        self._released = False
        self._lock = threading.Lock()
        self._event = ct.c_void_p()
        hip = owner._hip
        get_device = getattr(hip, "hipStreamGetDevice", None)
        if get_device is None:
            raise ValueError("external readers require HIP stream-device validation")
        get_device.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
        device = ct.c_int()
        self._check(get_device(self._stream, ct.byref(device)))
        if device.value != owner._device:
            raise ValueError("external reader stream belongs to another device")
        hip.hipStreamWaitEvent.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint]
        self._check(hip.hipEventCreate(ct.byref(self._event)))
        try:
            self._check(hip.hipEventRecord(self._event, owner._execute.stream))
            self._check(hip.hipStreamWaitEvent(self._stream, self._event, 0))
        except BaseException:
            hip.hipEventDestroy(self._event)
            raise
        self._outputs = MappingProxyType(dict(owner._execute.device_outputs))

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"attention reader HIP status {status}; ownership retained")

    @property
    def outputs(self):
        if self._released:
            raise ValueError("attention reader is released")
        return self._outputs

    def _finish(self):
        # Re-record only after consumer work has been enqueued. If recording or
        # the producer wait fails, retain the lease so release can be retried.
        hip = self._owner._hip
        self._check(hip.hipEventRecord(self._event, self._stream))
        self._check(hip.hipStreamWaitEvent(self._owner._execute.stream, self._event, 0))
        # Event destruction is deferred by HIP until recorded work completes.
        self._check(hip.hipEventDestroy(self._event))
        self._released = True
        with self._owner._lock:
            self._owner._readers -= 1
            self._owner._schedule_release()

    def close(self):
        with self._lock:
            if not self._released:
                if threading.get_ident() == self._owner._worker_ident:
                    self._finish()
                else:
                    self._owner._executor.submit(self._finish).result()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
