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
# f32 encode-session ABIs
_bmm_enc = None
_layer_norm_enc = None
_rmsnorm_enc = None
_softmax_enc = None
_rope_enc = None
_unary_enc = None
_flash_attn_enc = None
# Project-3 f16 encode-session ABIs (2026-06-01)
_bmm_enc_f16 = None
_layer_norm_enc_f16 = None
_rmsnorm_enc_f16 = None
_softmax_enc_f16 = None
_rope_enc_f16 = None
_unary_enc_f16 = None
_flash_attn_enc_f16 = None
# Project-3 bf16 encode-session ABIs (2026-06-01) — MPSGraph-routed
# ops only. MSL custom kernels (rope, flash_attn) need the on-GPU
# fp32-conversion path which is the Phase-3b follow-on.
_bmm_enc_bf16 = None
_layer_norm_enc_bf16 = None
_rmsnorm_enc_bf16 = None
_softmax_enc_bf16 = None
_unary_enc_bf16 = None
# Phase 3b (2026-06-01) — MSL-kernel bf16 via on-GPU cast.
_rope_enc_bf16 = None
_flash_attn_enc_bf16 = None
_mpsgraph_bf16_supported = None
_session_commit_count = None
_ts_dev_alloc = None
_ts_dev_upload = None
_ts_dev_download = None
_ts_dev_free = None
_ts_dev_nbytes = None
# Project 5 (2026-06-01) — conv2d encode-session ABI.
_conv2d_enc = None


def _bind_session_symbols() -> bool:
    """Lazy-bind the encode-session C ABI symbols. Returns True iff
    every symbol resolved (runtime loadable + symbols available)."""
    global _SYMBOLS_BOUND
    global _ts_enc_begin, _ts_enc_commit_wait, _bmm_enc, _layer_norm_enc
    global _rmsnorm_enc, _softmax_enc, _rope_enc, _unary_enc
    global _flash_attn_enc, _session_commit_count
    global _conv2d_enc
    global _bmm_enc_f16, _layer_norm_enc_f16, _rmsnorm_enc_f16
    global _softmax_enc_f16, _rope_enc_f16, _unary_enc_f16
    global _flash_attn_enc_f16
    global _bmm_enc_bf16, _layer_norm_enc_bf16, _rmsnorm_enc_bf16
    global _softmax_enc_bf16, _unary_enc_bf16
    global _rope_enc_bf16, _flash_attn_enc_bf16
    global _mpsgraph_bf16_supported
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
    _rmsnorm_enc = bind_symbol(
        "tessera_apple_gpu_rmsnorm_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_float),
        ctypes.c_int32)
    _softmax_enc = bind_symbol(
        "tessera_apple_gpu_softmax_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _rope_enc = bind_symbol(
        "tessera_apple_gpu_rope_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _unary_enc = bind_symbol(
        "tessera_apple_gpu_unary_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int64, ctypes.c_int32),
        ctypes.c_int32)
    _flash_attn_enc = bind_symbol(
        "tessera_apple_gpu_flash_attn_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_float, ctypes.c_int32),
        ctypes.c_int32)
    # Project-3 f16 encode-session ABIs (2026-06-01) — same signatures
    # as the f32 variants, just the dtype-bearing buffers carry
    # uint16_t (ml_dtypes.float16) bit patterns.
    _bmm_enc_f16 = bind_symbol(
        "tessera_apple_gpu_bmm_dev_f16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _layer_norm_enc_f16 = bind_symbol(
        "tessera_apple_gpu_layer_norm_dev_f16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_float),
        ctypes.c_int32)
    _rmsnorm_enc_f16 = bind_symbol(
        "tessera_apple_gpu_rmsnorm_dev_f16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_float),
        ctypes.c_int32)
    _softmax_enc_f16 = bind_symbol(
        "tessera_apple_gpu_softmax_dev_f16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _rope_enc_f16 = bind_symbol(
        "tessera_apple_gpu_rope_dev_f16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _unary_enc_f16 = bind_symbol(
        "tessera_apple_gpu_unary_dev_f16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int64, ctypes.c_int32),
        ctypes.c_int32)
    _flash_attn_enc_f16 = bind_symbol(
        "tessera_apple_gpu_flash_attn_dev_f16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_float, ctypes.c_int32),
        ctypes.c_int32)
    # Project-3 bf16 encode-session ABIs (2026-06-01) — MPSGraph-only.
    _bmm_enc_bf16 = bind_symbol(
        "tessera_apple_gpu_bmm_dev_bf16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _layer_norm_enc_bf16 = bind_symbol(
        "tessera_apple_gpu_layer_norm_dev_bf16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_float),
        ctypes.c_int32)
    _rmsnorm_enc_bf16 = bind_symbol(
        "tessera_apple_gpu_rmsnorm_dev_bf16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_float),
        ctypes.c_int32)
    _softmax_enc_bf16 = bind_symbol(
        "tessera_apple_gpu_softmax_dev_bf16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _unary_enc_bf16 = bind_symbol(
        "tessera_apple_gpu_unary_dev_bf16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int64, ctypes.c_int32),
        ctypes.c_int32)
    # Phase 3b (2026-06-01) — MSL bf16 via on-GPU cast.
    _rope_enc_bf16 = bind_symbol(
        "tessera_apple_gpu_rope_dev_bf16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    _flash_attn_enc_bf16 = bind_symbol(
        "tessera_apple_gpu_flash_attn_dev_bf16_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_float, ctypes.c_int32),
        ctypes.c_int32)
    _mpsgraph_bf16_supported = bind_symbol(
        "tessera_apple_gpu_mpsgraph_bf16_supported", (),
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
    # Project 5 (2026-06-01) — conv2d encode-session. Signature:
    # (TsEncodeSession*, X*, W*, bias*, O*, N, H, W, Cin, Cout, kH,
    # kW, strideH, strideW, padH, padW, dilationH, dilationW, groups)
    # → int32. bias may be NULL (pass 0 / None).
    _conv2d_enc = bind_symbol(
        "tessera_apple_gpu_conv2d_dev_f32_enc",
        (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_void_p,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32),
        ctypes.c_int32)
    return all(s is not None for s in (
        _ts_enc_begin, _ts_enc_commit_wait, _bmm_enc, _layer_norm_enc,
        _rmsnorm_enc, _softmax_enc, _rope_enc, _unary_enc,
        _flash_attn_enc,
        _bmm_enc_f16, _layer_norm_enc_f16, _rmsnorm_enc_f16,
        _softmax_enc_f16, _rope_enc_f16, _unary_enc_f16,
        _flash_attn_enc_f16,
        _session_commit_count, _ts_dev_alloc,
        _ts_dev_upload, _ts_dev_download, _ts_dev_free, _ts_dev_nbytes,
        _conv2d_enc))


def session_available() -> bool:
    """Quick capability check — True iff the encode-session C ABI
    is bindable on this host."""
    return _bind_session_symbols()


def bf16_session_available() -> bool:
    """Project-3 (2026-06-01) — True iff MPSGraph accepts bf16 graph
    nodes on this host. macOS 26+ on M2+ supports native bf16
    encode; older hosts return False and bf16 ops fall back to the
    fp32-conversion path (Phase-3b)."""
    if not _bind_session_symbols():
        return False
    if _mpsgraph_bf16_supported is None:
        return False
    return bool(int(_mpsgraph_bf16_supported()))


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


def flash_attn_enc(session: int, Q: DeviceTensor, K: DeviceTensor,
                   V: DeviceTensor, *, B: int, Sq: int, Sk: int, D: int,
                   scale: Optional[float] = None,
                   causal: bool = False) -> DeviceTensor:
    """Encode a flash-attention forward into ``session``'s command
    buffer. Q/K/V are device-resident ``(B, S*, D)`` f32 tensors;
    output ``(B, Sq, D)`` is allocated fresh.

    ``scale`` defaults to ``1/sqrt(D)`` (the standard attention scale).
    Set ``causal=True`` for a lower-triangular mask (decoder
    self-attention)."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if scale is None:
        scale = 1.0 / (float(D) ** 0.5)
    out = device_empty(B * Sq * D * 4)
    rc = _flash_attn_enc(ctypes.c_void_p(session),
                         ctypes.c_void_p(Q.handle),
                         ctypes.c_void_p(K.handle),
                         ctypes.c_void_p(V.handle),
                         ctypes.c_void_p(out.handle),
                         ctypes.c_int32(B), ctypes.c_int32(Sq),
                         ctypes.c_int32(Sk), ctypes.c_int32(D),
                         ctypes.c_float(scale),
                         ctypes.c_int32(1 if causal else 0))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"flash_attn_enc returned {rc}")
    return out


def rmsnorm_enc(session: int, X: DeviceTensor, gamma: DeviceTensor,
                *, rows: int, cols: int, eps: float = 1e-5) -> DeviceTensor:
    """Encode an rmsnorm into ``session``'s command buffer. Operates
    on (rows, cols) f32 with per-column gamma (no beta). Llama-style
    transformers use rmsnorm instead of layer_norm."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(rows * cols * 4)
    rc = _rmsnorm_enc(ctypes.c_void_p(session),
                      ctypes.c_void_p(X.handle),
                      ctypes.c_void_p(gamma.handle),
                      ctypes.c_void_p(out.handle),
                      ctypes.c_int32(rows), ctypes.c_int32(cols),
                      ctypes.c_float(eps))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"rmsnorm_enc returned {rc}")
    return out


def softmax_enc(session: int, X: DeviceTensor, *,
                rows: int, cols: int) -> DeviceTensor:
    """Encode a (free-standing) softmax into ``session``'s command
    buffer. (B*H*Sq, Sk) row-major. Used for classifier heads or
    attention scoring outside the flash_attn fusion."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(rows * cols * 4)
    rc = _softmax_enc(ctypes.c_void_p(session),
                       ctypes.c_void_p(X.handle),
                       ctypes.c_void_p(out.handle),
                       ctypes.c_int32(rows), ctypes.c_int32(cols))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"softmax_enc returned {rc}")
    return out


def rope_enc(session: int, X: DeviceTensor, Theta: DeviceTensor,
             *, M: int, K: int) -> DeviceTensor:
    """Encode a rotary position-embedding apply into ``session``'s
    command buffer. ``X`` is (M, K) f32 (typically flattened
    (B, S, H*D) head-major); ``Theta`` is the same shape with
    per-element phase angle."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(M * K * 4)
    rc = _rope_enc(ctypes.c_void_p(session),
                   ctypes.c_void_p(X.handle),
                   ctypes.c_void_p(Theta.handle),
                   ctypes.c_void_p(out.handle),
                   ctypes.c_int32(M), ctypes.c_int32(K))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"rope_enc returned {rc}")
    return out


# ---------------------------------------------------------------------
# Auto-session decorator — phase 1 of jit auto-detection.
#
# True JIT-level auto-detection (rewriting @jit(target='apple_gpu')
# bodies to spot encode-session-compatible chains and route them
# through batched_session() transparently) is a separate architectural
# change to ``compiler/jit.py``'s metadata-execution path; tracked as
# stage-3 of the single-cb roadmap (see
# ``docs/audit/single_command_buffer_decode_plan.md``).
#
# In the meantime, this decorator gives callers 90% of the ergonomic
# win without touching the JIT: write a function that takes ``s`` as
# its first arg, decorate, and the session lifecycle (open + commit +
# wait) is handled for you.
#
#     @decode_chain
#     def llama_attention(s, x_dev, gamma_dev, w_q, w_k, w_v, w_o,
#                          theta_dev, *, B, S, D):
#         n = rmsnorm_enc(s, x_dev, gamma_dev, rows=B*S, cols=D, eps=1e-5)
#         q = bmm_enc(s, n, w_q, batch=1, M=B*S, N=D, K=D)
#         k = bmm_enc(s, n, w_k, batch=1, M=B*S, N=D, K=D)
#         v = bmm_enc(s, n, w_v, batch=1, M=B*S, N=D, K=D)
#         q_r = rope_enc(s, q, theta_dev, M=B*S, K=D)
#         k_r = rope_enc(s, k, theta_dev, M=B*S, K=D)
#         a = flash_attn_enc(s, q_r, k_r, v, B=B, Sq=S, Sk=S, D=D)
#         return bmm_enc(s, a, w_o, batch=1, M=B*S, N=D, K=D)
#
#     # No explicit `with batched_session():` needed.
#     out_dev = llama_attention(x_dev, gamma_dev, w_q, w_k, w_v, w_o,
#                                theta_dev, B=1, S=8, D=16)
#     # ONE command buffer committed; return value is the output
#     # DeviceTensor, ready to download.

def decode_chain(fn):
    """Decorator — wrap a function whose first positional arg is the
    encode-session handle. The decorator opens a fresh session, calls
    ``fn(session, *args, **kwargs)``, then commits + waits. Inner ops
    encode into the same command buffer; one cb submission per call.

    The wrapped function should call only encode-session helpers
    (``bmm_enc`` / ``layer_norm_enc`` / ``rmsnorm_enc`` / etc.) on the
    session arg. Mixing in non-encode dispatch (which would commit a
    SEPARATE cb under the hood) defeats the single-cb invariant.
    """
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with batched_session() as s:
            return fn(s, *args, **kwargs)
    return wrapped


def _unary_enc_call(session: int, X: DeviceTensor, op_code: int,
                    n: int) -> DeviceTensor:
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(n * 4)
    rc = _unary_enc(ctypes.c_void_p(session),
                    ctypes.c_void_p(X.handle),
                    ctypes.c_void_p(out.handle),
                    ctypes.c_int64(n), ctypes.c_int32(op_code))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"unary_enc(op={op_code}) returned {rc}")
    return out


def silu_enc(session: int, X: DeviceTensor, *, n: int) -> DeviceTensor:
    """Encode SiLU (x * sigmoid(x)) into ``session``'s command buffer.
    Routes through the existing unary encode helper with op-code 4."""
    return _unary_enc_call(session, X, op_code=4, n=n)


def gelu_enc(session: int, X: DeviceTensor, *, n: int) -> DeviceTensor:
    """Encode GELU (Gaussian Error Linear Unit) into ``session``'s
    command buffer. Routes through the existing unary encode helper
    with op-code 5 (the tanh approximation)."""
    return _unary_enc_call(session, X, op_code=5, n=n)


def _conv2d_out_dim(in_dim: int, k: int, stride: int, pad: int,
                     dilation: int) -> int:
    """Mirror the runtime's ``conv2d_out_dim`` so Python can size the
    output DeviceTensor correctly before the encode call."""
    eff = dilation * (k - 1) + 1
    if in_dim + 2 * pad < eff:
        return 0
    return (in_dim + 2 * pad - eff) // stride + 1


def conv2d_enc(session: int, X: DeviceTensor, Wt: DeviceTensor,
                bias: DeviceTensor | None, *,
                N: int, H: int, W: int, Cin: int, Cout: int,
                kH: int, kW: int, strideH: int = 1, strideW: int = 1,
                padH: int = 0, padW: int = 0,
                dilationH: int = 1, dilationW: int = 1,
                groups: int = 1) -> DeviceTensor:
    """Project 5 (2026-06-01) — encode an NHWC f32 conv2d into
    ``session``'s command buffer. Mirrors the runtime's
    ``tessera_apple_gpu_conv2d_dev_f32_enc`` ABI; the dispatch is an
    MPSGraph ``convolution2DWithSourceTensor:`` node appended to the
    cb, so any encode-session chain can include conv2d ops on the
    same command buffer as norm / matmul / softmax / flash_attn.

    Shapes (all NHWC / HWIO row-major, f32):

    * ``X`` — ``(N, H, W, Cin)``, ``N*H*W*Cin*4`` bytes
    * ``Wt`` — ``(kH, kW, Cin/groups, Cout)``, ``kH*kW*(Cin/groups)*Cout*4`` bytes
    * ``bias`` — ``(Cout,)``, ``Cout*4`` bytes, or ``None`` for no bias
    * output — ``(N, outH, outW, Cout)`` where
      ``outH = conv2d_out_dim(H, kH, strideH, padH, dilationH)`` etc.

    Returns the freshly allocated output DeviceTensor.
    """
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if groups <= 0 or Cin % groups or Cout % groups:
        raise ValueError(
            f"conv2d_enc: groups={groups} must evenly divide "
            f"Cin={Cin} and Cout={Cout}")
    outH = _conv2d_out_dim(H, kH, strideH, padH, dilationH)
    outW = _conv2d_out_dim(W, kW, strideW, padW, dilationW)
    out = device_empty(N * outH * outW * Cout * 4)  # f32 = 4 bytes/elem
    bias_handle = ctypes.c_void_p(bias.handle) if bias is not None \
        else ctypes.c_void_p(0)
    rc = _conv2d_enc(ctypes.c_void_p(session),
                      ctypes.c_void_p(X.handle),
                      ctypes.c_void_p(Wt.handle),
                      bias_handle,
                      ctypes.c_void_p(out.handle),
                      ctypes.c_int32(N), ctypes.c_int32(H),
                      ctypes.c_int32(W), ctypes.c_int32(Cin),
                      ctypes.c_int32(Cout), ctypes.c_int32(kH),
                      ctypes.c_int32(kW), ctypes.c_int32(strideH),
                      ctypes.c_int32(strideW), ctypes.c_int32(padH),
                      ctypes.c_int32(padW), ctypes.c_int32(dilationH),
                      ctypes.c_int32(dilationW), ctypes.c_int32(groups))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"conv2d_enc returned {rc}")
    return out


def conv2d_enc_no_bias(session: int, X: DeviceTensor, Wt: DeviceTensor,
                        *, N: int, H: int, W: int, Cin: int, Cout: int,
                        kH: int, kW: int, strideH: int = 1, strideW: int = 1,
                        padH: int = 0, padW: int = 0,
                        dilationH: int = 1, dilationW: int = 1,
                        groups: int = 1) -> DeviceTensor:
    """Registry-friendly conv2d wrapper — same as :func:`conv2d_enc`
    with bias hard-wired to ``None``. The encode-registry tracks
    DeviceTensor args by positional index; an optional ``bias``
    parameter would break that contract. Callers that want bias
    invoke :func:`conv2d_enc` directly (outside the registered
    chain-planning surface)."""
    return conv2d_enc(session, X, Wt, None,
                       N=N, H=H, W=W, Cin=Cin, Cout=Cout, kH=kH, kW=kW,
                       strideH=strideH, strideW=strideW,
                       padH=padH, padW=padW,
                       dilationH=dilationH, dilationW=dilationW,
                       groups=groups)


# ---------------------------------------------------------------------
# Project-3 f16 encode-session Python wrappers (2026-06-01).
#
# Each f32 wrapper above has a sibling ``_f16`` variant here. They
# share the same shape semantics — only the underlying ABI symbol +
# byte size per element differ. The DeviceTensor's byte count tracks
# fp16 = 2 bytes per element automatically.

def bmm_enc_f16(session: int, A: DeviceTensor, B: DeviceTensor,
                *, batch: int, M: int, N: int, K: int,
                b_broadcast: bool = False) -> DeviceTensor:
    """f16 variant of :func:`bmm_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(batch * M * N * 2)  # fp16 = 2 bytes/elem
    rc = _bmm_enc_f16(ctypes.c_void_p(session),
                      ctypes.c_void_p(A.handle),
                      ctypes.c_void_p(B.handle),
                      ctypes.c_void_p(out.handle),
                      ctypes.c_int32(batch), ctypes.c_int32(M),
                      ctypes.c_int32(N), ctypes.c_int32(K),
                      ctypes.c_int32(1 if b_broadcast else 0))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"bmm_enc_f16 returned {rc}")
    return out


def layer_norm_enc_f16(session: int, X: DeviceTensor,
                       gamma: DeviceTensor, beta: DeviceTensor,
                       *, rows: int, cols: int,
                       eps: float = 1e-5) -> DeviceTensor:
    """f16 variant of :func:`layer_norm_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(rows * cols * 2)
    rc = _layer_norm_enc_f16(ctypes.c_void_p(session),
                             ctypes.c_void_p(X.handle),
                             ctypes.c_void_p(gamma.handle),
                             ctypes.c_void_p(beta.handle),
                             ctypes.c_void_p(out.handle),
                             ctypes.c_int32(rows), ctypes.c_int32(cols),
                             ctypes.c_float(eps))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"layer_norm_enc_f16 returned {rc}")
    return out


def rmsnorm_enc_f16(session: int, X: DeviceTensor, gamma: DeviceTensor,
                    *, rows: int, cols: int,
                    eps: float = 1e-5) -> DeviceTensor:
    """f16 variant of :func:`rmsnorm_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(rows * cols * 2)
    rc = _rmsnorm_enc_f16(ctypes.c_void_p(session),
                          ctypes.c_void_p(X.handle),
                          ctypes.c_void_p(gamma.handle),
                          ctypes.c_void_p(out.handle),
                          ctypes.c_int32(rows), ctypes.c_int32(cols),
                          ctypes.c_float(eps))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"rmsnorm_enc_f16 returned {rc}")
    return out


def softmax_enc_f16(session: int, X: DeviceTensor, *,
                    rows: int, cols: int) -> DeviceTensor:
    """f16 variant of :func:`softmax_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(rows * cols * 2)
    rc = _softmax_enc_f16(ctypes.c_void_p(session),
                          ctypes.c_void_p(X.handle),
                          ctypes.c_void_p(out.handle),
                          ctypes.c_int32(rows), ctypes.c_int32(cols))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"softmax_enc_f16 returned {rc}")
    return out


def rope_enc_f16(session: int, X: DeviceTensor, Theta: DeviceTensor,
                 *, M: int, K: int) -> DeviceTensor:
    """f16 variant of :func:`rope_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(M * K * 2)
    rc = _rope_enc_f16(ctypes.c_void_p(session),
                       ctypes.c_void_p(X.handle),
                       ctypes.c_void_p(Theta.handle),
                       ctypes.c_void_p(out.handle),
                       ctypes.c_int32(M), ctypes.c_int32(K))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"rope_enc_f16 returned {rc}")
    return out


def flash_attn_enc_f16(session: int, Q: DeviceTensor, K: DeviceTensor,
                       V: DeviceTensor, *, B: int, Sq: int, Sk: int,
                       D: int, scale: Optional[float] = None,
                       causal: bool = False) -> DeviceTensor:
    """f16 variant of :func:`flash_attn_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if scale is None:
        scale = 1.0 / (float(D) ** 0.5)
    out = device_empty(B * Sq * D * 2)
    rc = _flash_attn_enc_f16(ctypes.c_void_p(session),
                             ctypes.c_void_p(Q.handle),
                             ctypes.c_void_p(K.handle),
                             ctypes.c_void_p(V.handle),
                             ctypes.c_void_p(out.handle),
                             ctypes.c_int32(B), ctypes.c_int32(Sq),
                             ctypes.c_int32(Sk), ctypes.c_int32(D),
                             ctypes.c_float(scale),
                             ctypes.c_int32(1 if causal else 0))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"flash_attn_enc_f16 returned {rc}")
    return out


def _unary_enc_call_f16(session: int, X: DeviceTensor, op_code: int,
                        n: int) -> DeviceTensor:
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    out = device_empty(n * 2)
    rc = _unary_enc_f16(ctypes.c_void_p(session),
                        ctypes.c_void_p(X.handle),
                        ctypes.c_void_p(out.handle),
                        ctypes.c_int64(n), ctypes.c_int32(op_code))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"unary_enc_f16(op={op_code}) returned {rc}")
    return out


def silu_enc_f16(session: int, X: DeviceTensor, *,
                 n: int) -> DeviceTensor:
    """f16 variant of :func:`silu_enc` (op-code 4)."""
    return _unary_enc_call_f16(session, X, op_code=4, n=n)


def gelu_enc_f16(session: int, X: DeviceTensor, *,
                 n: int) -> DeviceTensor:
    """f16 variant of :func:`gelu_enc` (op-code 5)."""
    return _unary_enc_call_f16(session, X, op_code=5, n=n)


# ---------------------------------------------------------------------
# Project-3 bf16 encode-session Python wrappers (2026-06-01).
#
# bf16 uses native MPSGraph MPSDataTypeBFloat16 (macOS 26+ on M2+).
# Each wrapper expects host-side bf16 bit-pattern buffers (uint16
# numpy arrays via ``.view(np.uint16)`` on an ml_dtypes.bfloat16 view
# OR a direct ``view`` on the bf16 high-bytes of an fp32 array — see
# the runtime's bfloat16_to_float_gpu / float_to_bfloat16_gpu
# converters). DeviceTensor stores 2 bytes per element regardless of
# dtype identity at the C ABI boundary.
#
# Sub-byte / non-MPSGraph paths (rope MSL, flash_attn MSL) need on-GPU
# bf16↔fp32 conversion which is the Phase-3b follow-on.

def bmm_enc_bf16(session: int, A: DeviceTensor, B: DeviceTensor,
                 *, batch: int, M: int, N: int, K: int,
                 b_broadcast: bool = False) -> DeviceTensor:
    """bf16 variant of :func:`bmm_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if _bmm_enc_bf16 is None:
        raise RuntimeError("bmm_enc_bf16 symbol not bound")
    out = device_empty(batch * M * N * 2)
    rc = _bmm_enc_bf16(ctypes.c_void_p(session),
                       ctypes.c_void_p(A.handle),
                       ctypes.c_void_p(B.handle),
                       ctypes.c_void_p(out.handle),
                       ctypes.c_int32(batch), ctypes.c_int32(M),
                       ctypes.c_int32(N), ctypes.c_int32(K),
                       ctypes.c_int32(1 if b_broadcast else 0))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"bmm_enc_bf16 returned {rc}")
    return out


def layer_norm_enc_bf16(session: int, X: DeviceTensor,
                        gamma: DeviceTensor, beta: DeviceTensor,
                        *, rows: int, cols: int,
                        eps: float = 1e-5) -> DeviceTensor:
    """bf16 variant of :func:`layer_norm_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if _layer_norm_enc_bf16 is None:
        raise RuntimeError("layer_norm_enc_bf16 symbol not bound")
    out = device_empty(rows * cols * 2)
    rc = _layer_norm_enc_bf16(ctypes.c_void_p(session),
                              ctypes.c_void_p(X.handle),
                              ctypes.c_void_p(gamma.handle),
                              ctypes.c_void_p(beta.handle),
                              ctypes.c_void_p(out.handle),
                              ctypes.c_int32(rows), ctypes.c_int32(cols),
                              ctypes.c_float(eps))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"layer_norm_enc_bf16 returned {rc}")
    return out


def rmsnorm_enc_bf16(session: int, X: DeviceTensor, gamma: DeviceTensor,
                     *, rows: int, cols: int,
                     eps: float = 1e-5) -> DeviceTensor:
    """bf16 variant of :func:`rmsnorm_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if _rmsnorm_enc_bf16 is None:
        raise RuntimeError("rmsnorm_enc_bf16 symbol not bound")
    out = device_empty(rows * cols * 2)
    rc = _rmsnorm_enc_bf16(ctypes.c_void_p(session),
                           ctypes.c_void_p(X.handle),
                           ctypes.c_void_p(gamma.handle),
                           ctypes.c_void_p(out.handle),
                           ctypes.c_int32(rows), ctypes.c_int32(cols),
                           ctypes.c_float(eps))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"rmsnorm_enc_bf16 returned {rc}")
    return out


def softmax_enc_bf16(session: int, X: DeviceTensor, *,
                     rows: int, cols: int) -> DeviceTensor:
    """bf16 variant of :func:`softmax_enc`."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if _softmax_enc_bf16 is None:
        raise RuntimeError("softmax_enc_bf16 symbol not bound")
    out = device_empty(rows * cols * 2)
    rc = _softmax_enc_bf16(ctypes.c_void_p(session),
                           ctypes.c_void_p(X.handle),
                           ctypes.c_void_p(out.handle),
                           ctypes.c_int32(rows), ctypes.c_int32(cols))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"softmax_enc_bf16 returned {rc}")
    return out


def _unary_enc_call_bf16(session: int, X: DeviceTensor, op_code: int,
                         n: int) -> DeviceTensor:
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if _unary_enc_bf16 is None:
        raise RuntimeError("unary_enc_bf16 symbol not bound")
    out = device_empty(n * 2)
    rc = _unary_enc_bf16(ctypes.c_void_p(session),
                         ctypes.c_void_p(X.handle),
                         ctypes.c_void_p(out.handle),
                         ctypes.c_int64(n), ctypes.c_int32(op_code))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"unary_enc_bf16(op={op_code}) returned {rc}")
    return out


def silu_enc_bf16(session: int, X: DeviceTensor, *,
                  n: int) -> DeviceTensor:
    """bf16 variant of :func:`silu_enc` (op-code 4)."""
    return _unary_enc_call_bf16(session, X, op_code=4, n=n)


def gelu_enc_bf16(session: int, X: DeviceTensor, *,
                  n: int) -> DeviceTensor:
    """bf16 variant of :func:`gelu_enc` (op-code 5)."""
    return _unary_enc_call_bf16(session, X, op_code=5, n=n)


def rope_enc_bf16(session: int, X: DeviceTensor, Theta: DeviceTensor,
                  *, M: int, K: int) -> DeviceTensor:
    """bf16 RoPE via on-GPU cast (Phase 3b). The runtime brackets
    the existing f32 MSL kernel with bf16→fp32 → kernel → fp32→bf16
    cast nodes, all in the same command buffer."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if _rope_enc_bf16 is None:
        raise RuntimeError("rope_enc_bf16 symbol not bound")
    out = device_empty(M * K * 2)
    rc = _rope_enc_bf16(ctypes.c_void_p(session),
                        ctypes.c_void_p(X.handle),
                        ctypes.c_void_p(Theta.handle),
                        ctypes.c_void_p(out.handle),
                        ctypes.c_int32(M), ctypes.c_int32(K))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"rope_enc_bf16 returned {rc}")
    return out


def flash_attn_enc_bf16(session: int, Q: DeviceTensor, K: DeviceTensor,
                        V: DeviceTensor, *, B: int, Sq: int, Sk: int,
                        D: int, scale: Optional[float] = None,
                        causal: bool = False) -> DeviceTensor:
    """bf16 flash_attn via on-GPU cast (Phase 3b). Same recipe as
    rope_enc_bf16: f32 MSL kernel sandwiched between cast nodes."""
    if not _bind_session_symbols():
        raise RuntimeError("Apple GPU encode-session runtime unavailable")
    if _flash_attn_enc_bf16 is None:
        raise RuntimeError("flash_attn_enc_bf16 symbol not bound")
    if scale is None:
        scale = 1.0 / (float(D) ** 0.5)
    out = device_empty(B * Sq * D * 2)
    rc = _flash_attn_enc_bf16(ctypes.c_void_p(session),
                              ctypes.c_void_p(Q.handle),
                              ctypes.c_void_p(K.handle),
                              ctypes.c_void_p(V.handle),
                              ctypes.c_void_p(out.handle),
                              ctypes.c_int32(B), ctypes.c_int32(Sq),
                              ctypes.c_int32(Sk), ctypes.c_int32(D),
                              ctypes.c_float(scale),
                              ctypes.c_int32(1 if causal else 0))
    if int(rc) != 1:
        out.free()
        raise RuntimeError(f"flash_attn_enc_bf16 returned {rc}")
    return out


__all__ = [
    "DeviceTensor",
    "batched_session",
    "bf16_session_available",
    "bmm_enc",
    "bmm_enc_bf16",
    "bmm_enc_f16",
    "decode_chain",
    "device_empty",
    "device_tensor",
    "flash_attn_enc",
    "flash_attn_enc_bf16",
    "flash_attn_enc_f16",
    "gelu_enc",
    "gelu_enc_bf16",
    "gelu_enc_f16",
    "layer_norm_enc",
    "layer_norm_enc_bf16",
    "layer_norm_enc_f16",
    "rmsnorm_enc",
    "rmsnorm_enc_bf16",
    "rmsnorm_enc_f16",
    "rope_enc",
    "rope_enc_bf16",
    "rope_enc_f16",
    "session_available",
    "session_commit_count",
    "silu_enc",
    "silu_enc_bf16",
    "silu_enc_f16",
    "softmax_enc",
    "softmax_enc_bf16",
    "softmax_enc_f16",
]
