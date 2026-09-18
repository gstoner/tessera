"""Per-lane operand packing for the compiler-emitted sm_120a NVFP4 warp tile.

The emitted kernel (:func:`tessera.compiler.ptx_emit.emit_nvfp4_block_scale_mma_ptx`)
runs one ``mma.sync.aligned.m16n8k64 ... kind::mxf4nvf4.block_scale.scale_vec::4X``
on one warp: ``D[16x8] f32 = A[16x64] e2m1 * B[64x8] e2m1`` with one ``ue4m3``
scale per 16-wide K block of each A row and each B column. It reads its
operands *already laid out per lane* in the PTX ISA fragment order, so the
layout lives here, in numpy, and nowhere else:

* A: lane ``l`` (``gid = l >> 2``, ``tig = l & 3``) holds four words, each
  eight e2m1 codes with element ``j`` in bits ``4j..4j+3``:
  ``a0 = A[gid, 8tig + j]``, ``a1 = A[gid + 8, 8tig + j]``,
  ``a2 = A[gid, 8tig + 32 + j]``, ``a3 = A[gid + 8, 8tig + 32 + j]``.
* B: two words, ``b0 = B[8tig + j, gid]``, ``b1 = B[8tig + 32 + j, gid]``.
* Scales (``scale_vec::4X``, byte-id 0, thread-id 0): all four K-block bytes
  ride in one word, block ``b`` in byte ``b``. Lanes with ``tig == 0`` supply
  A row ``gid`` and B column ``gid``; lanes with ``tig == 1`` supply A row
  ``gid + 8``; every other lane's scale word is ignored and packed as zero.
* D: ``d0 = D[gid, 2tig]``, ``d1 = D[gid, 2tig + 1]``, ``d2 = D[gid + 8, 2tig]``,
  ``d3 = D[gid + 8, 2tig + 1]``.

This is the layout the on-silicon spike proved on sm_120a with unit,
uniform and mapped non-uniform scales
(``docs/audit/backend/nvidia/spikes/sm120_mma_sync/nvfp4_gemm.cu``). The
exact reference here decodes e2m1 and ue4m3 the way the spike does, so the
device row can be asserted bit-for-bit against it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

M, N, K = 16, 8, 64
LANES = 32
SCALE_BLOCK = 16
SCALE_BLOCKS = K // SCALE_BLOCK
UE4M3_ONE = 0x38  # exponent bias 7, mantissa 0

_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1_decode(codes: Any) -> np.ndarray:
    """fp4 e2m1 codes (``s<<3 | e<<1 | m``, 0..15) to float32."""
    c = np.asarray(codes, dtype=np.uint8)
    if c.size and int(c.max()) > 15:
        raise ValueError("e2m1 codes are 4-bit values")
    magnitude = _E2M1_MAGNITUDES[c & 7]
    return np.where((c >> 3) & 1, -magnitude, magnitude).astype(np.float32)


def e2m1_encode(values: Any) -> np.ndarray:
    """Round-to-nearest e2m1 codes for float inputs (the spike's encoder)."""
    v = np.asarray(values, dtype=np.float32)
    magnitude = np.abs(v)[..., None]
    best = np.argmin(np.abs(magnitude - _E2M1_MAGNITUDES), axis=-1).astype(np.uint8)
    return np.where(v < 0, best | 8, best).astype(np.uint8)


def ue4m3_decode(codes: Any) -> np.ndarray:
    """Unsigned e4m3 scale codes (7 bits, bias 7) to float32."""
    c = np.asarray(codes, dtype=np.uint8)
    if c.size and int(c.max()) > 0x7F:
        raise ValueError("ue4m3 codes are 7-bit values")
    exponent = (c >> 3) & 0xF
    mantissa = (c & 7).astype(np.float32)
    subnormal = np.ldexp(mantissa / 8.0, -6)
    normal = np.ldexp(1.0 + mantissa / 8.0, exponent.astype(np.int32) - 7)
    return np.where(exponent == 0, subnormal, normal).astype(np.float32)


def _check(name: str, array: Any, shape: tuple[int, ...], limit: int) -> np.ndarray:
    a = np.ascontiguousarray(array, dtype=np.uint8)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {a.shape}")
    if a.size and int(a.max()) > limit:
        raise ValueError(f"{name} codes exceed {limit:#x}")
    return a


def nvfp4_tile_reference(a_codes: Any, b_codes: Any, scale_a: Any, scale_b: Any) -> np.ndarray:
    """The exact tile product in float32 (decoded operands, per-block scales)."""
    a = e2m1_decode(_check("A", a_codes, (M, K), 15)).astype(np.float64)
    b = e2m1_decode(_check("B", b_codes, (K, N), 15)).astype(np.float64)
    sa = ue4m3_decode(_check("scale_a", scale_a, (M, SCALE_BLOCKS), 0x7F)).astype(np.float64)
    sb = ue4m3_decode(_check("scale_b", scale_b, (SCALE_BLOCKS, N), 0x7F)).astype(np.float64)
    out = np.zeros((M, N), dtype=np.float64)
    for block in range(SCALE_BLOCKS):
        ks = slice(block * SCALE_BLOCK, (block + 1) * SCALE_BLOCK)
        out += (a[:, ks] * sa[:, block:block + 1]) @ (b[ks, :] * sb[block:block + 1, :])
    return out.astype(np.float32)


def _pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """Eight 4-bit codes along the last axis into one uint32, element j in bits 4j."""
    shifts = (4 * np.arange(8, dtype=np.uint32))
    return (codes.astype(np.uint32) << shifts).sum(axis=-1, dtype=np.uint64).astype(np.uint32)


def pack_nvfp4_mma_fragments(a_codes: Any, b_codes: Any, scale_a: Any, scale_b: Any
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lay the logical tile out per lane: ``(A[32,4], B[32,2], SFa[32], SFb[32])`` uint32."""
    a = _check("A", a_codes, (M, K), 15)
    b = _check("B", b_codes, (K, N), 15)
    sa = _check("scale_a", scale_a, (M, SCALE_BLOCKS), 0x7F)
    sb = _check("scale_b", scale_b, (SCALE_BLOCKS, N), 0x7F)
    lanes = np.arange(LANES)
    gid, tig = lanes >> 2, lanes & 3
    j = np.arange(8)
    cols = 8 * tig[:, None] + j[None, :]                      # [32, 8]
    a_words = np.stack([
        _pack_nibbles(a[gid[:, None], cols]),
        _pack_nibbles(a[gid[:, None] + 8, cols]),
        _pack_nibbles(a[gid[:, None], cols + 32]),
        _pack_nibbles(a[gid[:, None] + 8, cols + 32]),
    ], axis=1).astype(np.uint32)                                # [32, 4]
    b_words = np.stack([
        _pack_nibbles(b[cols, gid[:, None]]),
        _pack_nibbles(b[cols + 32, gid[:, None]]),
    ], axis=1).astype(np.uint32)                                # [32, 2]
    byte_shifts = 8 * np.arange(SCALE_BLOCKS, dtype=np.uint32)
    sfa = np.zeros(LANES, dtype=np.uint32)
    sfb = np.zeros(LANES, dtype=np.uint32)
    lower = tig == 0
    upper = tig == 1
    sfa[lower] = (sa[gid[lower]].astype(np.uint32) << byte_shifts).sum(axis=1)
    sfa[upper] = (sa[gid[upper] + 8].astype(np.uint32) << byte_shifts).sum(axis=1)
    sfb[lower] = (sb[:, gid[lower]].T.astype(np.uint32) << byte_shifts).sum(axis=1)
    return a_words, b_words, sfa, sfb


def unpack_nvfp4_mma_accumulator(d_words: Any) -> np.ndarray:
    """The kernel's ``D[32 lanes, 4]`` f32 back to the logical ``[16, 8]`` tile."""
    d = np.asarray(d_words, dtype=np.float32).reshape(LANES, 4)
    lanes = np.arange(LANES)
    gid, tig = lanes >> 2, lanes & 3
    out = np.empty((M, N), dtype=np.float32)
    out[gid, 2 * tig] = d[:, 0]
    out[gid, 2 * tig + 1] = d[:, 1]
    out[gid + 8, 2 * tig] = d[:, 2]
    out[gid + 8, 2 * tig + 1] = d[:, 3]
    return out


def unpack_nvfp4_mma_fragments(a_words: Any, b_words: Any) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of the A/B half of :func:`pack_nvfp4_mma_fragments` (layout proof)."""
    aw = np.asarray(a_words, dtype=np.uint32).reshape(LANES, 4)
    bw = np.asarray(b_words, dtype=np.uint32).reshape(LANES, 2)
    lanes = np.arange(LANES)
    gid, tig = lanes >> 2, lanes & 3
    j = np.arange(8)
    nib = lambda words: ((words[:, None] >> (4 * j)[None, :]) & 0xF).astype(np.uint8)  # noqa: E731
    a = np.zeros((M, K), dtype=np.uint8)
    b = np.zeros((K, N), dtype=np.uint8)
    cols = 8 * tig[:, None] + j[None, :]
    a[gid[:, None], cols] = nib(aw[:, 0])
    a[gid[:, None] + 8, cols] = nib(aw[:, 1])
    a[gid[:, None], cols + 32] = nib(aw[:, 2])
    a[gid[:, None] + 8, cols + 32] = nib(aw[:, 3])
    b[cols, gid[:, None]] = nib(bw[:, 0])
    b[cols + 32, gid[:, None]] = nib(bw[:, 1])
    return a, b


def pack_e2m1_codes(codes: Any, *, axis: int) -> np.ndarray:
    """Pack 4-bit codes pairwise along ``axis`` (the contraction axis): element
    ``2i`` lands in the low nibble of byte ``i``, ``2i+1`` in the high one; an
    odd extent is zero-padded. This is the general NVFP4 launch ABI's layout
    (``A[M, ceil(K/2)]`` packs along axis 1, ``B[ceil(K/2), N]`` along axis 0)."""
    c = np.asarray(codes, dtype=np.uint8)
    if c.size and int(c.max()) > 15:
        raise ValueError("e2m1 codes are 4-bit values")
    c = np.moveaxis(c, axis, -1)
    if c.shape[-1] % 2:
        c = np.concatenate([c, np.zeros(c.shape[:-1] + (1,), np.uint8)], axis=-1)
    packed = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)
    return np.ascontiguousarray(np.moveaxis(packed, -1, axis))


def nvfp4_gemm_reference(a_codes: Any, b_codes: Any, scale_a: Any, scale_b: Any) -> np.ndarray:
    """The exact general-shape product ``D[M,N]`` for logical (unpacked) e2m1
    code matrices ``A[M,K]``/``B[K,N]`` and raw ue4m3 scales ``SFa[M,ceil(K/16)]``
    / ``SFb[ceil(K/16),N]``: one scale per 16-wide K block, f64 accumulation,
    returned as f32 (every product is an exact binary fraction; the sums the
    device tests use fit an f32 mantissa, so the comparison is bit-exact)."""
    a = e2m1_decode(np.asarray(a_codes, np.uint8)).astype(np.float64)
    b = e2m1_decode(np.asarray(b_codes, np.uint8)).astype(np.float64)
    m, k = a.shape
    n = b.shape[1]
    blocks = (k + 15) // 16
    sa = ue4m3_decode(np.asarray(scale_a, np.uint8)).astype(np.float64)
    sb = ue4m3_decode(np.asarray(scale_b, np.uint8)).astype(np.float64)
    if sa.shape != (m, blocks) or sb.shape != (blocks, n) or b.shape[0] != k:
        raise ValueError("nvfp4 reference shapes disagree")
    out = np.zeros((m, n), np.float64)
    for block in range(blocks):
        ks = slice(block * 16, min((block + 1) * 16, k))
        out += (a[:, ks] * sa[:, block:block + 1]) @ (b[ks, :] * sb[block:block + 1, :])
    return out.astype(np.float32)


__all__ = [
    "M", "N", "K", "LANES", "SCALE_BLOCK", "SCALE_BLOCKS", "UE4M3_ONE",
    "e2m1_decode", "e2m1_encode", "ue4m3_decode", "nvfp4_tile_reference",
    "pack_nvfp4_mma_fragments", "unpack_nvfp4_mma_accumulator",
    "unpack_nvfp4_mma_fragments", "pack_e2m1_codes", "nvfp4_gemm_reference",
]
