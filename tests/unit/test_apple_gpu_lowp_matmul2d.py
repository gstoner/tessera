"""Owning-Mac proof for the SDK27 / MSL 4.1 low-precision MPP ``matmul2d`` lane.

FP8 E4M3 / E5M2 and FP4 E2M1 operands (plus the header's half x low-precision
pairs) bound as strided MTLTensor views: packed layouts, padded row strides and
nonzero tile origins. References start from the exact quantized bytes. This is
a correctness proof only -- not block scaling, not selector admission, not a
performance or fleet claim. Off-Mac and pre-macOS-27 hosts skip honestly; a
Mac whose Metal 4 stack is up but whose loaded runtime lacks the symbol FAILS
(a stale dylib, never a skip -- see the hollow-green pattern).
"""
from __future__ import annotations

import platform

import numpy as np
import pytest

from tessera import runtime as R

FORMATS = ["fp8_e4m3", "fp8_e5m2", "fp4_e2m1"]
#: Apple's MTLTensor row-stride quantum for 8/4-bit data: 128 bytes, i.e. 128
#: FP8 elements or 256 FP4 elements (measured 2026-09-14 on macOS 27.0 / M1 Max:
#: "[tensor.strides extentAtDimensionIndex:1] (64 elements) must be aligned to
#: 128 elements (128 byte aligned) when tensor.dataType is a 8-bit format").
Q = {"fp8_e4m3": 128, "fp8_e5m2": 128, "fp4_e2m1": 256}


def _ld(cols, fmt, extra=0, q=None):
    """Row stride (elements): ``cols`` rounded up to the quantum ``q`` (default
    Apple's 128-byte stride quantum), plus ``extra`` quanta."""
    q = q or Q[fmt]
    return -(-cols // q) * q + extra * q


#: Byte granularity of a reachable view origin: 128 B for FP8 (Apple's
#: data-plane offset rule), 256 B for FP4 (Apple applies 4-bit buffer offsets at
#: 2x the byte value passed -- measured 2026-09-14, see the runtime comment --
#: so the lane halves a 256-B-aligned offset and rejects anything finer).
ORIGIN_BYTES = {"fp8_e4m3": 128, "fp8_e5m2": 128, "fp4_e2m1": 256}


def _origin_elems(fmt, k):
    """Column origin ``k`` origin-quanta into a row, in elements."""
    return k * ORIGIN_BYTES[fmt] * (2 if fmt == "fp4_e2m1" else 1)


def _shapes(fmt):
    q = Q[fmt]
    return [(q, q, q), (2 * q, q, 3 * q), (100, q, 2 * q), (256, 256, 256)]


# The exact-device boundary is the shared `metal4` marker (conftest routes it
# through tests._support.apple.require_apple_metal4), never an inline skip.
pytestmark = pytest.mark.metal4


def _macos_major() -> int:
    try:
        return int(platform.mac_ver()[0].split(".")[0] or 0)
    except ValueError:
        return 0


@pytest.fixture(scope="module")
def lane():
    if _macos_major() < 27:
        pytest.skip("SDK27 low-precision matmul2d needs macOS 27 (shading language 4.1)")
    if R._apple_gpu_mtl4_matmul2d_lowp_sym() is None:
        pytest.fail("Metal 4 is up on macOS 27 but the loaded Apple GPU runtime exports no "
                    "tessera_apple_gpu_mtl4_matmul2d_lowp -- rebuild the dylib against the "
                    "macOS 27 SDK (a stale runtime is a failure, not a skip)")
    return True


# ---------------------------------------------------------------- decoder

@pytest.mark.parametrize("fmt,ml_name", [("fp8_e4m3", "float8_e4m3fn"),
                                         ("fp8_e5m2", "float8_e5m2"),
                                         ("fp4_e2m1", "float4_e2m1fn")])
def test_lowp_decoder_matches_ml_dtypes_for_every_code(fmt, ml_name):
    """Host-free: the pure-numpy decoder the references start from agrees with
    ml_dtypes on every encoding (NaN positions compared as NaN)."""
    ml = pytest.importorskip("ml_dtypes")
    dt = getattr(ml, ml_name)
    if fmt == "fp4_e2m1":
        codes = np.arange(16, dtype=np.uint8)
        packed = (codes[::2] | (codes[1::2] << 4)).reshape(1, -1)
        got = R.apple_lowp_decode(packed, fmt, np).reshape(-1)
    else:
        codes = np.arange(256, dtype=np.uint8)
        got = R.apple_lowp_decode(codes.reshape(1, -1), fmt, np).reshape(-1)
    exp = codes.view(dt).astype(np.float64)
    assert np.array_equal(np.isnan(got), np.isnan(exp))
    m = ~np.isnan(exp)
    np.testing.assert_array_equal(got[m], exp[m])


# ---------------------------------------------------------------- helpers

def _rand_codes(rng, rows, cols, fmt, *, finite=True):
    """Random storage codes. ``finite`` keeps E4M3/E5M2 away from NaN/inf and
    both FP8 formats away from the largest binades so a K-long fp32 dot product
    stays exactly representable (products of two 8-bit values are exact in fp32)."""
    if fmt == "fp4_e2m1":
        n = rng.integers(0, 16, size=(rows, cols * 2), dtype=np.uint8)
        return (n[:, ::2] | (n[:, 1::2] << 4)).astype(np.uint8)
    c = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    if finite:
        if fmt == "fp8_e4m3":
            # exponent 0..11 only (|x| <= 30); avoids NaN code and huge binades
            c = (c & 0x87) | ((c >> 3) & 0xF) % 12 << 3
        else:
            c = (c & 0x83) | ((c >> 2) & 0x1F) % 24 << 2  # exponent 0..23
    return c.astype(np.uint8)


def _storage_cols(cols, fmt):
    return cols // 2 if fmt == "fp4_e2m1" else cols


def _view(codes, fmt, r0, c0, rows, cols):
    dec = R.apple_lowp_decode(codes, fmt, np)
    return dec[r0:r0 + rows, c0:c0 + cols]


def _check(C, ref):
    """fp32 accumulation of exact products vs float64: tolerance from the
    magnitude of the partial sums, not of the result (cancellation-aware)."""
    assert C.dtype == np.float32
    assert np.all(np.isfinite(C))
    scale = np.abs(ref).max() + 1.0
    err = np.abs(C.astype(np.float64) - ref).max()
    assert err <= 4e-6 * scale, (err, scale)


# ---------------------------------------------------------------- packed

@pytest.mark.parametrize("fmt,M,N,K", [(f, *s) for f in FORMATS for s in _shapes(f)])
def test_lowp_matmul2d_packed(lane, fmt, M, N, K):
    """Packed (unpadded) storage whose row strides sit on the quantum; M is
    free (a partial 64-row tile is edge-checked by matmul2d's slice)."""
    rng = np.random.default_rng(M * 7 + N * 3 + K)
    A = _rand_codes(rng, M, _storage_cols(K, fmt), fmt)
    B = _rand_codes(rng, K, _storage_cols(N, fmt), fmt)
    C = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K)
    ref = _view(A, fmt, 0, 0, M, K) @ _view(B, fmt, 0, 0, K, N)
    _check(C, ref)


@pytest.mark.parametrize("fmt,M,N,K", [(f, *s) for f in FORMATS for s in _shapes(f)[:3]])
def test_half_times_lowp_matmul2d(lane, fmt, M, N, K):
    """Header rows `half x metal_fp*_format -> float`: the weight-only shape.
    A is half (no quantum rule of its own here), B must sit on the quantum."""
    rng = np.random.default_rng(M + N + K + 11)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    B = _rand_codes(rng, K, _storage_cols(N, fmt), fmt)
    C = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K, a_dtype="f16")
    ref = A.astype(np.float64) @ _view(B, fmt, 0, 0, K, N)
    _check(C, ref)


# ---------------------------------------------------------------- padded / offset

@pytest.mark.parametrize("fmt", FORMATS)
def test_lowp_matmul2d_padded_strides_and_output_padding(lane, fmt):
    """Row strides wider than the view (one 128-byte quantum of junk per row on
    both operands) and an output buffer wider than N: the junk must not leak
    into the product and the caller's padding columns must survive."""
    M, N, K = 96, 64, 96  # N, K deliberately OFF the quantum: only padding makes them bindable
    rng = np.random.default_rng(5)
    A = _rand_codes(rng, M, _storage_cols(_ld(K, fmt, 1), fmt), fmt, finite=False)  # junk incl. NaN codes
    B = _rand_codes(rng, K, _storage_cols(_ld(N, fmt, 1), fmt), fmt, finite=False)
    # overwrite the view region with finite codes
    A[:, :_storage_cols(K, fmt)] = _rand_codes(rng, M, _storage_cols(K, fmt), fmt)
    B[:, :_storage_cols(N, fmt)] = _rand_codes(rng, K, _storage_cols(N, fmt), fmt)
    out = np.full((M, N + 5), np.float32(-777.0), np.float32)
    got = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K, out=out)
    assert got is out
    ref = _view(A, fmt, 0, 0, M, K) @ _view(B, fmt, 0, 0, K, N)
    _check(out[:, :N], ref)
    assert np.all(out[:, N:] == np.float32(-777.0)), "output padding columns were clobbered"


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("a_origin,b_origin", [((3, 0), (5, 0)), ((0, 1), (0, 2)), ((7, 2), (2, 1))])
def test_lowp_matmul2d_nonzero_tile_origin(lane, fmt, a_origin, b_origin):
    """Views that start inside a bigger padded storage: row offsets, column
    offsets (in origin quanta: 128 B for FP8, 256 B for FP4), and both. Every
    element outside the view is finite-random, so a wrong origin changes the
    answer. For FP4 this is also the canary on Apple's 2x buffer-offset factor
    (see ORIGIN_BYTES): if the OS ever corrects it, these fail."""
    M, N, K = 64, 96, 64
    (ar, ak), (br, bk) = a_origin, b_origin
    ac, bc = _origin_elems(fmt, ak), _origin_elems(fmt, bk)
    # Rows must start on an origin boundary too: round the row stride to it.
    qrow = ORIGIN_BYTES[fmt] * (2 if fmt == "fp4_e2m1" else 1)
    rng = np.random.default_rng(ar + ac + br + bc + 1)
    A = _rand_codes(rng, M + ar + 4, _storage_cols(_ld(K + ac, fmt, 1, qrow), fmt), fmt)
    B = _rand_codes(rng, K + br + 4, _storage_cols(_ld(N + bc, fmt, 1, qrow), fmt), fmt)
    C = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K,
                                       a_origin=(ar, ac), b_origin=(br, bc))
    ref = _view(A, fmt, ar, ac, M, K) @ _view(B, fmt, br, bc, K, N)
    _check(C, ref)


@pytest.mark.parametrize("fmt", FORMATS)
def test_lowp_matmul2d_rejects_sub_128_byte_view_offset(lane, fmt):
    """A column origin of 32 elements (32 B for FP8, 16 B for FP4) is inside
    Apple's 128-byte data-plane offset rule: Apple rejects the descriptor and
    the lane reports it as kind 5 ("offset for plane MTLTensorPlaneTypeData
    (32 bytes) should be aligned to 128 bytes" -- measured 2026-09-14)."""
    M, N, K = 64, Q[fmt], 64
    rng = np.random.default_rng(9)
    A = _rand_codes(rng, M, _storage_cols(_ld(K, fmt, 1), fmt), fmt)
    B = _rand_codes(rng, K, _storage_cols(N, fmt), fmt)
    with pytest.raises(RuntimeError, match=r"(?s)kind 5.*(aligned to 128 bytes|multiple of 256 bytes)"):
        R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K, a_origin=(0, 32))


def test_fp4_rejects_128_byte_origin_and_accepts_256(lane):
    """The FP4 origin granularity is 256 B (Apple's 2x offset factor): a
    128-B column origin is refused with a diagnostic naming the cause, and the
    256-B origin next to it binds the right bytes."""
    fmt = "fp4_e2m1"
    M, N, K = 64, 256, 64
    rng = np.random.default_rng(10)
    A = _rand_codes(rng, M, _storage_cols(1024, fmt), fmt)
    B = _rand_codes(rng, K, _storage_cols(N, fmt), fmt)
    with pytest.raises(RuntimeError, match="multiple of 256 bytes"):
        R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K, a_origin=(0, 256))
    C = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K, a_origin=(0, 512))
    _check(C, _view(A, fmt, 0, 512, M, K) @ _view(B, fmt, 0, 0, K, N))


# ---------------------------------------------------------------- negative

@pytest.mark.parametrize("fmt", FORMATS)
def test_lowp_matmul2d_rejects_packed_sub_quantum_stride(lane, fmt):
    """A packed 64-wide operand is NOT a valid 8/4-bit MTLTensor view: Apple
    rejects the descriptor and the lane surfaces that verbatim as kind 5. This
    pins the measured envelope (128 FP8 / 256 FP4 elements per row stride)."""
    rng = np.random.default_rng(1)
    A = _rand_codes(rng, 64, _storage_cols(64, fmt), fmt)
    B = _rand_codes(rng, 64, _storage_cols(Q[fmt], fmt), fmt)
    with pytest.raises(RuntimeError, match=r"(?s)kind 5.*must be aligned to (128|256) elements \(128 byte aligned\)"):
        R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=64, N=Q[fmt], K=64)


@pytest.mark.parametrize("fmt", FORMATS)
def test_lowp_matmul2d_rejects_unaligned_inner_extent(lane, fmt):
    """Row stride on the quantum but an innermost extent off Apple's 32-element
    rule (K = 48): must be a kind-5 diagnostic, never a wrong answer."""
    rng = np.random.default_rng(2)
    A = _rand_codes(rng, 64, _storage_cols(Q[fmt], fmt), fmt)
    B = _rand_codes(rng, 48, _storage_cols(Q[fmt], fmt), fmt)
    with pytest.raises(RuntimeError, match="kind 5"):
        R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=64, N=Q[fmt], K=48)


def test_lowp_matmul2d_rejects_odd_fp4_origin(lane):
    rng = np.random.default_rng(3)
    A = _rand_codes(rng, 64, _storage_cols(512, "fp4_e2m1"), "fp4_e2m1")
    B = _rand_codes(rng, 64, _storage_cols(256, "fp4_e2m1"), "fp4_e2m1")
    with pytest.raises(RuntimeError, match="byte-aligned"):
        R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt="fp4_e2m1", M=64, N=256, K=64, a_origin=(0, 1))


def test_lowp_matmul2d_rejects_view_past_storage(lane):
    rng = np.random.default_rng(4)
    A = _rand_codes(rng, 64, 128, "fp8_e4m3")
    B = _rand_codes(rng, 64, 128, "fp8_e4m3")
    with pytest.raises(RuntimeError, match="exceeds its storage"):
        R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt="fp8_e4m3", M=64, N=128, K=128, a_origin=(1, 0))


# ---------------------------------------------------------------- fused epilogue (step 4)

def _epi_ref(ref, bias, act):
    v = ref if bias is None else ref + bias[None, :]
    if act == "relu":
        return np.maximum(v, 0.0)
    if act == "gelu":
        t = 0.7978845608028654 * (v + 0.044715 * v ** 3)
        return 0.5 * v * (1.0 + np.tanh(t))
    if act == "silu":
        return v / (1.0 + np.exp(-v))
    return v


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("act,with_bias", [("none", True), ("relu", True), ("gelu", True),
                                           ("silu", False), ("gelu", False)])
def test_lowp_matmul2d_fused_epilogue(lane, fmt, act, with_bias):
    """C = act(A@B + bias) fused on the cooperative tensor, vs the exact-bytes
    reference through the same fp32 formulas (tolerance widened for tanh/exp)."""
    M, N, K = 100, Q[fmt], Q[fmt]
    rng = np.random.default_rng(21)
    A = _rand_codes(rng, M, _storage_cols(K, fmt), fmt)
    B = _rand_codes(rng, K, _storage_cols(N, fmt), fmt)
    bias = (rng.standard_normal(N) * 4).astype(np.float32) if with_bias else None
    C = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K, bias=bias, act=act)
    ref = _epi_ref(_view(A, fmt, 0, 0, M, K) @ _view(B, fmt, 0, 0, K, N),
                   None if bias is None else bias.astype(np.float64), act)
    assert np.all(np.isfinite(C))
    scale = np.abs(ref).max() + 1.0
    assert np.abs(C.astype(np.float64) - ref).max() <= 2e-5 * scale


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("act", ["none", "relu", "gelu", "silu"])
def test_fused_and_decomposed_epilogue_agree(lane, fmt, act):
    """The fused kernel and the two-dispatch route (plain matmul2d, then the
    standalone bias/act pass) evaluate one shared MSL expression; they must
    agree to fp32 rounding, and we record whether they are bit-identical."""
    M, N, K = 64, Q[fmt], Q[fmt]
    rng = np.random.default_rng(22)
    A = _rand_codes(rng, M, _storage_cols(K, fmt), fmt)
    B = _rand_codes(rng, K, _storage_cols(N, fmt), fmt)
    bias = (rng.standard_normal(N) * 4).astype(np.float32)
    fused = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K, bias=bias, act=act)
    plain = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K)
    dec = R.apple_gpu_mtl4_bias_act_f32(plain, np, bias=bias, act=act)
    assert dec is plain
    assert np.all(np.isfinite(fused)) and np.all(np.isfinite(dec))  # NaN == NaN must not pass
    np.testing.assert_allclose(dec, fused, rtol=2e-6, atol=2e-6, equal_nan=False)


@pytest.mark.parametrize("route", ["lowp_fused", "bias_act_pass", "f16_fused"])
def test_gelu_is_finite_for_large_preactivations(lane, route):
    """Regression: under Metal fast math tanh(t) went NaN for |t| > ~45, so
    gelu(v) was NaN for |v| beyond ~12 in every fused epilogue (found 2026-09-14
    via the low-precision lane, whose random operands reach it; the shipped
    f16/bf16 kernels shared the formula). gelu must approach v / 0 instead."""
    if route == "bias_act_pass":
        v = np.linspace(-400.0, 400.0, 257).astype(np.float32)
        C = np.tile(v, (4, 1)).astype(np.float32)
        R.apple_gpu_mtl4_bias_act_f32(C, np, act="gelu")
        got = C[0]
    elif route == "f16_fused":
        v = np.array([-400.0, -60.0, -13.0, -8.0, 0.0, 8.0, 13.0, 60.0, 400.0] * 14 + [0.0] * 2, np.float32)
        A = np.ones((64, 128), np.float16)
        B = np.tile((v / 128.0).astype(np.float16), (128, 1))  # column n sums to v[n] exactly
        C, ran = R.apple_gpu_mtl4_matmul2d_epilogue(A, B, np, bias=None, act="gelu", dtype="f16")
        assert ran
        v = (B.astype(np.float64).sum(axis=0)).astype(np.float32)
        got = C[0]
    else:
        # e4m3 codes: 256 (0x78) x 1.0 over K=128 -> pre-activation 128*256 = 32768 per column
        v = np.full(128, 32768.0, np.float32)
        A = np.full((64, 128), 0x78, np.uint8)
        B = np.full((128, 128), 0x38, np.uint8)
        got = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt="fp8_e4m3", M=64, N=128, K=128, act="gelu")[0]
    assert np.all(np.isfinite(got)), got
    ref = _epi_ref(v.astype(np.float64), None, "gelu")
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_bias_act_pass_respects_view_and_padding(lane):
    M, N, ldc = 37, 96, 130
    rng = np.random.default_rng(23)
    C = rng.standard_normal((M, ldc)).astype(np.float32)
    keep = C[:, N:].copy()
    bias = rng.standard_normal(N).astype(np.float32)
    ref = _epi_ref(C[:, :N].astype(np.float64), bias.astype(np.float64), "relu")
    R.apple_gpu_mtl4_bias_act_f32(C, np, bias=bias, act="relu", N=N)
    np.testing.assert_allclose(C[:, :N], ref, rtol=1e-6, atol=1e-6)
    assert np.array_equal(C[:, N:], keep)


def test_bias_act_pass_guards_ragged_rows(lane):
    """PR #749 review: the dispatch rounds the y grid up to whole threadgroups,
    so a ragged M ran threads at m >= M against C[m*ldc + n] past the buffer
    (masked by pool-bucket rounding for most shapes; M=127, N=ldc=513 sits at a
    bucket edge). Both coordinates are guarded now; the pass must be exact on
    that shape and leave nothing outside the view touched."""
    M, N, ldc = 127, 513, 513
    rng = np.random.default_rng(31)
    C = rng.standard_normal((M, ldc)).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    ref = _epi_ref(C.astype(np.float64), bias.astype(np.float64), "silu")
    R.apple_gpu_mtl4_bias_act_f32(C, np, bias=bias, act="silu", N=N)
    np.testing.assert_allclose(C, ref, rtol=1e-6, atol=1e-6)
    # And a padded output: columns past N untouched with the ragged M.
    C2 = rng.standard_normal((M, ldc + 7)).astype(np.float32)
    keep = C2[:, N:].copy()
    ref2 = _epi_ref(C2[:, :N].astype(np.float64), bias.astype(np.float64), "gelu")
    R.apple_gpu_mtl4_bias_act_f32(C2, np, bias=bias, act="gelu", N=N)
    np.testing.assert_allclose(C2[:, :N], ref2, rtol=1e-5, atol=1e-5)
    assert np.array_equal(C2[:, N:], keep)
