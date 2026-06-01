"""Single-command-buffer decode chain — Python ergonomic surface.

Audit Action 6 / Pattern table row #6 (2026-06-01) — scaffold for
"keep prefill / decode / attn / MLP / projection on one command
buffer." See ``docs/audit/single_command_buffer_decode_plan.md``.

Today's encode-session-aware ops:

* :func:`bmm_enc` — encoded device-resident batched matmul
* :func:`layer_norm_enc` — encoded device-resident layer normalization

Both append to a session's command buffer; nothing executes until
the session exits the ``with`` block (which fires
``ts_enc_commit_wait``). Output tensors are ``DeviceTensor`` instances
holding device-resident memory; the caller downloads at the end::

    with batched_session() as s:
        y_dev = layer_norm_enc(s, x_dev, gamma_dev, beta_dev, rows, cols, eps)
        z_dev = bmm_enc(s, y_dev, w_dev, batch, M, N, K)
    # ONE command buffer committed; output downloads back.
    z_host = z_dev.download(np.float32, (batch, M, N))

Stage-2 follow-ons add ``rope_enc`` / ``flash_attn_enc`` /
``softmax_enc`` / ``gelu_enc`` / ``rmsnorm_enc`` — one per decoder-step
op. See the roadmap doc for the full envelope.

Falls back gracefully when the Apple GPU runtime isn't loadable
(``batched_session`` returns ``None`` and the helpers raise so callers
can branch).
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from ._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol


_SYMBOLS_BOUND = False
_ts_enc_begin = None
_ts_enc_commit_wait = None
_bmm_enc = None
_layer_norm_enc = None
_session_commit_count = None
_ts_dev_alloc = None
_ts_dev_upload = None
_ts_dev_download = None
_ts_dev_free = None
_ts_dev_nbytes = None


def _bind_session_symbols() -> bool:
    """Lazy-bind the encode-session C ABI symbols. Returns True iff
    every symbol resolved (runtime loadable + symbols available)."""
    global _SYMBOLS_BOUND
    global _ts_enc_begin, _ts_enc_commit_wait, _bmm_enc, _layer_norm_enc
    global _session_commit_count
    global _ts_dev_alloc, _ts_dev_upload, _ts_dev_download, _ts_dev_free
    global _ts_dev_nbytes
    if _SYMBOLS_BOUND:
        return _ts_enc_begin is not None
    _SYMBOLS_BOUND = True
    if apple_gpu_runtime() is None:
        return False
    _ts_enc_begin = bind_symbol("ts_enc_begin", (), ctypes.c_void_p)
    _ts_enc_commit_wait = bind_symbol(
        "ts_enc_commit_wait", (ctypes.c_void_p,), None)
    _bmm_enc = bind_symbol(
        "tessera_apple_gpu_bmm_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _layer_norm_enc = bind_symbol(
        "tessera_apple_gpu_layer_norm_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_float),
        ctypes.c_int32)
    _session_commit_count = bind_symbol(
        "tessera_apple_gpu_session_commit_count", (), ctypes.c_int64)
    _ts_dev_alloc = bind_symbol(
        "ts_dev_alloc", (ctypes.c_int64,), ctypes.c_void_p)
    _ts_dev_upload = bind_symbol(
        "ts_dev_upload",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64), None)
    _ts_dev_download = bind_symbol(
        "ts_dev_download",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64), None)
    _ts_dev_free = bind_symbol(
        "ts_dev_free", (ctypes.c_void_p,), None)
    _ts_dev_nbytes = bind_symbol(
        "ts_dev_nbytes", (ctypes.c_void_p,), ctypes.c_int64)
    return all(s is not None for s in (
        _ts_enc_begin, _ts_enc_commit_wait, _bmm_enc, _layer_norm_enc,
        _session_commit_count, _ts_dev_alloc, _ts_dev_upload,
        _ts_dev_download, _ts_dev_free, _ts_dev_nbytes))


def session_available() -> bool:
    """Quick capability check — True iff the encode-session C ABI
    is bindable on this host."""
    return _bind_session_symbols()


def session_commit_count() -> int:
    """Monotonic counter of session commits since process start. A
    test that opens a session, runs N ops, commits once, and observes
    ``(count_after - count_before) == 1`` proves the chain stayed on
    one cb. Returns -1 on unavailable runtime."""
    if not _bind_session_symbols():
        return -1
    return int(_session_commit_count())


@dataclass
class DeviceTensor:
    """Device-resident tensor handle. Wraps a ``TsDeviceTensor*``
    and tracks the byte count for safe upload/download. Released
    via :meth:`free` (or context-manager use)."""
    handle: int
    nbytes: int

    def upload(self, arr: np.ndarray) -> None:
        if arr.nbytes != self.nbytes:
            raise ValueError(
                f"upload size {arr.nbytes} != device tensor size "
                f"{self.nbytes}")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        _ts_dev_upload(ctypes.c_void_p(self.handle),
                       arr.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int64(arr.nbytes))

    def download(self, dtype: np.dtype, shape: tuple[int, ...]
                 ) -> np.ndarray:
        nelem = int(np.prod(shape))
        nbytes = nelem * np.dtype(dtype).itemsize
        if nbytes != self.nbytes:
            raise ValueError(
                f"download size mismatch: shape {shape} dtype "
                f"{dtype} = {nbytes} bytes != tensor size "
                f"{self.nbytes}")
        out = np.empty(shape, dtype=dtype)
        _ts_dev_download(ctypes.c_void_p(self.handle),
                         out.ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int64(nbytes))
        return out

    def free(self) -> None:
        if self.handle:
            _ts_dev_free(ctypes.c_void_p(self.handle))
            self.handle = 0
            self.nbytes = 0


def device_tensor(arr: np.ndarray) -> DeviceTensor:
    """Allocate a device-resident tensor and upload ``arr`` into it."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    handle = _ts_dev_alloc(ctypes.c_int64(arr.nbytes))
    if not handle:
        raise RuntimeError("ts_dev_alloc returned NULL")
    dt = DeviceTensor(handle=int(handle), nbytes=int(arr.nbytes))
    dt.upload(arr)
    return dt


def device_empty(nbytes: int) -> DeviceTensor:
    """Allocate a device-resident tensor of ``nbytes`` (no upload)."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    handle = _ts_dev_alloc(ctypes.c_int64(nbytes))
    if not handle:
        raise RuntimeError("ts_dev_alloc returned NULL")
    return DeviceTensor(handle=int(handle), nbytes=int(nbytes))


@contextmanager
def batched_session() -> Iterator[int]:
    """Open an encode-session and yield the opaque session handle
    (as ``int``). The session is automatically committed + waited on
    when the ``with`` block exits — exactly one command-buffer
    submission per ``with``.

    Use the encoded variants of ops (``bmm_enc``, ``layer_norm_enc``,
    ...) inside the block; they take the session as their first arg
    and operate on :class:`DeviceTensor` inputs/outputs. Nothing runs
    on the GPU until the session exits."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    handle = _ts_enc_begin()
    if not handle:
        raise RuntimeError("ts_enc_begin returned NULL")
    try:
        yield int(handle)
    finally:
        _ts_enc_commit_wait(ctypes.c_void_p(int(handle)))


def bmm_enc(session: int, A: DeviceTensor, B: DeviceTensor,
            *, batch: int, M: int, N: int, K: int,
            b_broadcast: bool = False) -> DeviceTensor:
    """Encode a batched matmul into ``session``'s command buffer.
    Allocates a fresh output ``DeviceTensor`` and returns it. The
    actual GPU work runs when the session commits."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(batch * M * N * 4)
    rc = _bmm_enc(ctypes.c_void_p(session),
                  ctypes.c_void_p(A.handle),
                  ctypes.c_void_p(B.handle),
                  ctypes.c_void_p(out.handle),
                  ctypes.c_int32(batch), ctypes.c_int32(M),
                  ctypes.c_int32(N), ctypes.c_int32(K),
                  ctypes.c_int32(1 if b_broadcast else 0))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"bmm_enc returned {rc}")
    return out


def layer_norm_enc(session: int, X: DeviceTensor, gamma: DeviceTensor,
                   beta: DeviceTensor, *, rows: int, cols: int,
                   eps: float = 1e-5) -> DeviceTensor:
    """Encode a layer-normalization into ``session``'s command buffer.
    Operates on (rows, cols) f32 inputs with per-column gamma/beta
    (shape (cols,)). Allocates and returns a fresh output tensor."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(rows * cols * 4)
    rc = _layer_norm_enc(ctypes.c_void_p(session),
                         ctypes.c_void_p(X.handle),
                         ctypes.c_void_p(gamma.handle),
                         ctypes.c_void_p(beta.handle),
                         ctypes.c_void_p(out.handle),
                         ctypes.c_int32(rows), ctypes.c_int32(cols),
                         ctypes.c_float(eps))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"layer_norm_enc returned {rc}")
    return out


__all__ = [
    "DeviceTensor",
    "batched_session",
    "bmm_enc",
    "device_empty",
    "device_tensor",
    "layer_norm_enc",
    "session_available",
    "session_commit_count",
]
