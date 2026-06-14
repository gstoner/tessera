"""``tessera.stdlib.quant`` — packed low-precision weight layouts + fused
dequantize-into-GEMM (the M1 keystone of the model-class roadmap).

The existing ``tessera.ops.quantize_*`` surface (and ``tessera.quantization``)
is *reference* numerics: per-tensor symmetric, returning already-dequantized
fp32, with no packed storage and no kernel that consumes a separate scale
tensor.  Models like Kimi-K2 (native INT4) and GLM-5 / DeepSeek-V3.2 (FP8)
ship weights that are **packed integer/float codes + a separate per-group
scale tensor**, and the GEMM must dequantize *inside* the contraction with an
fp32 accumulator.  This module makes those two things first-class:

* :class:`PackedQuantTensor` — packed codes (int4 packed two-per-byte, int8,
  or fp8 grid) + the separate ``scales`` tensor + a :class:`QuantScheme` that
  records granularity (per-channel / per-group along the contraction axis).
  The scale is a *load-bearing operand*, not a declared contract.

* :func:`dequant_matmul` / :func:`dequant_grouped_gemm` — the contract is one
  op: packed weights + scales in, fp32-accumulated result out.  The reference
  evaluates it **group-wise** (each K-group dequantized then accumulated in
  fp32), which is the numeric policy real kernels implement; an oracle proves
  it equals dequantize-then-matmul (a DESIL cross-path equivalence).

Honesty note (mirrors the ``moe_swiglu_block`` composed-vs-fused split): on
Apple GPU the heavy matmul runs on the Metal matmul lane after a per-group
host dequant; a single MSL kernel that folds the dequant into the GEMM tile is
a tracked follow-up (M1.1).  What is *real today* and load-bearing: the packed
storage, the separate per-group scales, and the fp32 accumulator policy.

Leaf-ish module: depends only on numpy + ``tessera.compiler.grouped_layout``
(pure contract metadata) + an injected fp8 quantizer, so the audit registry and
runtime can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..compiler import grouped_layout as gl

# Canonical low-precision weight dtypes this module can pack.
QUANT_WEIGHT_DTYPES = ("int8", "int4", "fp8_e4m3", "fp8_e5m2")

_INT_QMAX = {"int8": 127, "int4": 7}


@dataclass(frozen=True)
class QuantScheme:
    """How a weight tensor is quantized.

    ``axis`` is the contraction (K) axis for a ``(K, N)`` weight — scales are
    shared per output channel ``N`` and per group of ``group_size`` rows along
    ``axis``.  ``group_size=None`` means per-channel (one scale per ``N`` over
    the whole ``axis``).  Symmetric (zero-point 0) only — the convention these
    models use for weights.
    """

    dtype: str
    axis: int = 0
    group_size: int | None = None
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in QUANT_WEIGHT_DTYPES:
            raise ValueError(
                f"quant dtype must be one of {QUANT_WEIGHT_DTYPES}; got {self.dtype!r}")
        if not self.symmetric:
            raise ValueError("only symmetric (zero-point 0) weight quant is supported")
        if self.group_size is not None and self.group_size <= 0:
            raise ValueError(f"group_size must be positive or None; got {self.group_size}")

    @property
    def is_int(self) -> bool:
        return self.dtype in _INT_QMAX

    @property
    def scale_layout(self) -> gl.ScaleLayout:
        """The :class:`grouped_layout.ScaleLayout` contract this scheme implies."""
        if self.group_size is None:
            return gl.ScaleLayout(granularity="per_channel", packing="none")
        packing = "none" if self.is_int else "e4m3"
        return gl.ScaleLayout(granularity="block", block=(self.group_size, 1),
                              packing=packing, vector_size=self.group_size,
                              alignment=self.group_size)


@dataclass(frozen=True)
class PackedQuantTensor:
    """Packed weight codes + the separate scale tensor.

    ``codes`` storage depends on the scheme: ``int4`` is packed two nibbles per
    ``uint8`` (so the last axis is halved), ``int8`` is ``int8``, fp8 is the
    fp8-grid value as fp32 (byte-packing to a real fp8 dtype is available via
    ml_dtypes but kept logical here).  ``scales`` is ``(num_groups, N)`` for a
    ``(K, N)`` weight (``num_groups = K // group_size``, or 1 for per-channel).
    """

    codes: np.ndarray
    scales: np.ndarray
    scheme: QuantScheme
    shape: tuple[int, ...]   # logical (unpacked) weight shape, e.g. (K, N)

    @property
    def num_groups(self) -> int:
        return int(self.scales.shape[0])

    def dequantize(self) -> np.ndarray:
        """Reconstruct the fp32 weight ``(K, N)`` from codes + scales."""
        return _dequantize(self)

    def storage_bytes(self) -> int:
        return int(self.codes.nbytes + self.scales.nbytes)


# ── packing helpers (genuine int4 nibble packing) ───────────────────────────
def pack_int4(codes: np.ndarray) -> np.ndarray:
    """Pack signed int4 codes in ``[-8, 7]`` two-per-byte along the last axis.

    Stored as ``uint8`` nibble pairs ``(low, high)``; the last axis must be even
    (callers pad ``K`` to a multiple of 2, which the group alignment guarantees).
    """
    c = np.asarray(codes).astype(np.int64)
    if c.size and (c.min() < -8 or c.max() > 7):
        raise ValueError("int4 codes must be in [-8, 7]")
    if c.shape[-1] % 2 != 0:
        raise ValueError(f"int4 pack needs an even last axis; got {c.shape[-1]}")
    nib = (c & 0xF).astype(np.uint8)                 # two's-complement low nibble
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_int4` → signed int8 codes in ``[-8, 7]``."""
    p = np.asarray(packed).astype(np.uint8)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.int8)
    # sign-extend 4-bit two's complement: values >= 8 are negative.
    out[..., 0::2] = np.where(lo >= 8, lo.astype(np.int16) - 16, lo).astype(np.int8)
    out[..., 1::2] = np.where(hi >= 8, hi.astype(np.int16) - 16, hi).astype(np.int8)
    return out


# ── quantizers ──────────────────────────────────────────────────────────────
def _groups_along_axis(K: int, group_size: int | None) -> int:
    if group_size is None:
        return 1
    if K % group_size != 0:
        raise ValueError(
            f"contraction extent K={K} must be divisible by group_size={group_size}")
    return K // group_size


def quantize_weight(
    w: Any,
    dtype: str,
    *,
    group_size: int | None = None,
    fp8_quantizer: Callable[..., Any] | None = None,
) -> PackedQuantTensor:
    """Quantize a ``(K, N)`` weight to a :class:`PackedQuantTensor`.

    Scales are symmetric, per output channel ``N`` and per group of
    ``group_size`` rows along ``K`` (``None`` → per-channel over all of ``K``).
    ``fp8_quantizer`` (e.g. ``tessera.ops.quantize_fp8``) is injected for the
    fp8 grid rounding so this stays import-light; required only for fp8 dtypes.
    """
    scheme = QuantScheme(dtype=dtype, axis=0, group_size=group_size)
    arr = np.asarray(w._data if hasattr(w, "_data") else w, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"quantize_weight expects a 2-D (K, N) weight; got {arr.shape}")
    K, N = arr.shape
    g = group_size if group_size is not None else K
    ng = _groups_along_axis(K, group_size)
    blocked = arr.reshape(ng, g, N)

    if scheme.is_int:
        qmax = _INT_QMAX[dtype]
        amax = np.max(np.abs(blocked), axis=1)                  # (ng, N)
        scales = np.where(amax > 0, amax / qmax, 1.0).astype(np.float32)
        codes = np.round(blocked / scales[:, None, :]).clip(-qmax, qmax)
        codes = codes.astype(np.int8).reshape(K, N)
        if dtype == "int4":
            # pack along K (axis 0): transpose so K is last, pack, keep (N, K/2)→back.
            codes_t = np.ascontiguousarray(codes.T)             # (N, K)
            packed = pack_int4(codes_t)                          # (N, K/2)
            return PackedQuantTensor(codes=packed, scales=scales,
                                     scheme=scheme, shape=(K, N))
        return PackedQuantTensor(codes=codes, scales=scales, scheme=scheme, shape=(K, N))

    # fp8 block-scaled along K.
    if fp8_quantizer is None:
        raise ValueError("fp8 quantization requires fp8_quantizer= (e.g. ops.quantize_fp8)")
    fmt = "e5m2" if dtype == "fp8_e5m2" else "e4m3"
    codes = np.empty((K, N), dtype=np.float32)
    scales = np.empty((ng, N), dtype=np.float32)
    for gi in range(ng):
        blk = blocked[gi]                                       # (g, N)
        for n in range(N):
            q, s = fp8_quantizer(blk[:, n], format=fmt)
            # q is already (grid_value * s) fp32; store the unit-grid code q/s.
            scales[gi, n] = np.float32(s)
            codes[gi * g:(gi + 1) * g, n] = np.asarray(q, dtype=np.float32) / np.float32(s)
    return PackedQuantTensor(codes=codes, scales=scales, scheme=scheme, shape=(K, N))


def _dequantize(packed: PackedQuantTensor) -> np.ndarray:
    K, N = packed.shape
    scheme = packed.scheme
    g = scheme.group_size if scheme.group_size is not None else K
    ng = packed.num_groups
    if scheme.dtype == "int4":
        codes = unpack_int4(packed.codes).T[:K, :]              # (K, N) signed int8
    else:
        codes = np.asarray(packed.codes, dtype=np.float32)
    codes = np.asarray(codes, dtype=np.float32).reshape(ng, g, N)
    w = codes * packed.scales[:, None, :]
    return w.reshape(K, N).astype(np.float32)


# ── fused dequant-into-GEMM (the contract: one op, fp32 accumulate) ──────────
def unit_codes_and_scales(packed: PackedQuantTensor) -> "tuple[np.ndarray, np.ndarray, int]":
    """Reconstruct ``(codes (K,N) fp32 unit-grid, scales (NG,N) fp32, group_size)``
    so ``dequant(W)[k,n] = codes[k,n] * scales[k//group_size, n]``.  This is the
    operand layout the fused Apple GPU dequant-GEMM kernel consumes (it never
    sees the full fp32 weight — only the unit codes + per-group scales)."""
    K, N = packed.shape
    scheme = packed.scheme
    g = scheme.group_size if scheme.group_size is not None else K
    if scheme.dtype == "int4":
        codes = unpack_int4(packed.codes).T[:K, :].astype(np.float32)
    else:
        codes = np.asarray(packed.codes, dtype=np.float32)
    return (np.ascontiguousarray(codes),
            np.ascontiguousarray(packed.scales.astype(np.float32)), g)


def dequant_matmul(
    x: Any,
    packed_w: PackedQuantTensor,
    *,
    backend: str = "reference",
) -> np.ndarray:
    """``y = x @ dequantize(packed_w)`` with an fp32 accumulator, evaluated
    group-wise so the accumulator policy is real (each K-group is dequantized
    and accumulated, never the full fp32 weight at once in the kernel model).

    ``backend="apple_gpu"`` uses the **fused** Metal dequant-into-GEMM kernel
    (`tessera_apple_gpu_dequant_matmul_f32`) — one dispatch, dequant computed
    in-register from the unit codes + per-group scales (full fp32 weight never
    materialized). Falls back to the composed per-group path on a Metal miss.
    ``"reference"`` uses numpy. Numerically identical to dequantize-then-matmul
    (the oracle).
    """
    xa = np.asarray(x._data if hasattr(x, "_data") else x, dtype=np.float32)
    K, N = packed_w.shape
    if xa.shape[-1] != K:
        raise ValueError(f"dequant_matmul: x last dim {xa.shape[-1]} != weight K {K}")

    if backend == "apple_gpu" and xa.ndim == 2:
        try:
            from .. import _apple_gpu_backend as agb
            codes, scales, g = unit_codes_and_scales(packed_w)
            return np.ascontiguousarray(
                agb.gpu_dequant_matmul(xa, codes, scales, g))
        except Exception:                              # noqa: BLE001 — composed fallback
            pass

    scheme = packed_w.scheme
    g = scheme.group_size if scheme.group_size is not None else K
    ng = packed_w.num_groups
    w_full = packed_w.dequantize()                             # (K, N) fp32
    mm = _matmul_fn(backend)
    acc = np.zeros(xa.shape[:-1] + (N,), dtype=np.float32)
    for gi in range(ng):
        sl = slice(gi * g, (gi + 1) * g)
        acc += mm(np.ascontiguousarray(xa[..., sl]),
                  np.ascontiguousarray(w_full[sl]))
    return acc.astype(np.float32)


def dequant_grouped_gemm(
    x: Any,
    packed_experts: list[PackedQuantTensor],
    group_sizes: Any,
    *,
    backend: str = "reference",
) -> np.ndarray:
    """Quantized M-grouped contiguous grouped-GEMM: token block ``e`` is matmul'd
    by expert ``e``'s packed weight via :func:`dequant_matmul`.

    ``x`` ``(T, K)`` tokens laid out in contiguous per-expert groups; one
    :class:`PackedQuantTensor` ``(K, N)`` per expert; ``group_sizes`` ``(E,)``.
    Returns ``(T, N)``.  This is the M2 MoE-quant building block.
    """
    xa = np.asarray(x._data if hasattr(x, "_data") else x, dtype=np.float32)
    gs = np.asarray(group_sizes).astype(np.int64).reshape(-1)
    E = len(packed_experts)
    if gs.shape[0] != E:
        raise ValueError(f"group_sizes has {gs.shape[0]} groups but {E} experts")
    N = packed_experts[0].shape[1]
    out = np.zeros((xa.shape[0], N), dtype=np.float32)
    off = 0
    for e in range(E):
        n = int(gs[e])
        if n:
            out[off:off + n] = dequant_matmul(
                xa[off:off + n], packed_experts[e], backend=backend)
        off += n
    return out


def reference_dequant_then_matmul(x: Any, packed_w: PackedQuantTensor) -> np.ndarray:
    """Oracle: full-precision ``x @ dequantize(W)`` in fp64 — the value
    :func:`dequant_matmul` must reproduce (the vertical correctness oracle)."""
    xa = np.asarray(x._data if hasattr(x, "_data") else x, dtype=np.float64)
    return (xa @ packed_w.dequantize().astype(np.float64))


# ── backend matmul selection ─────────────────────────────────────────────────
def _matmul_fn(backend: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if backend == "reference":
        return lambda a, b: (a.astype(np.float32) @ b.astype(np.float32))
    if backend == "apple_gpu":
        from .. import _apple_gpu_backend as agb

        def _mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            try:
                return np.asarray(agb.gpu_matmul(np.ascontiguousarray(a),
                                                 np.ascontiguousarray(b)),
                                  dtype=np.float32)
            except Exception:                          # noqa: BLE001 — honest fallback
                return a.astype(np.float32) @ b.astype(np.float32)

        return _mm
    raise ValueError(f"dequant_matmul backend must be 'reference' or 'apple_gpu'; got {backend!r}")


__all__ = [
    "QUANT_WEIGHT_DTYPES",
    "QuantScheme",
    "PackedQuantTensor",
    "pack_int4",
    "unpack_int4",
    "quantize_weight",
    "unit_codes_and_scales",
    "dequant_matmul",
    "dequant_grouped_gemm",
    "reference_dequant_then_matmul",
]
