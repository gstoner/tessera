"""Accumulation behaviour of the SDK27 low-precision MPP ``matmul2d`` lane.

Every case is built from exact quantized codes whose products are exact in
fp32, so the only thing under test is the accumulator: small terms added to a
large partial sum, cancellation, long reductions, and one deliberately
order-sensitive case that *characterises* the reduction structure (recorded,
not asserted to a single value). Skips honestly off the owning Mac.
"""
from __future__ import annotations

import platform

import numpy as np
import pytest

from tessera import runtime as R

Q = {"fp8_e4m3": 128, "fp8_e5m2": 128, "fp4_e2m1": 256}  # Apple's 128-byte stride quantum


pytestmark = pytest.mark.metal4  # shared exact-device boundary, not an inline skip


@pytest.fixture(scope="module")
def lane():
    """Same gate as test_apple_gpu_lowp_matmul2d: skip below macOS 27, FAIL on a
    stale dylib where the Metal 4 device is present."""
    try:
        major = int(platform.mac_ver()[0].split(".")[0] or 0)
    except ValueError:
        major = 0
    if major < 27:
        pytest.skip("SDK27 low-precision matmul2d needs macOS 27 (shading language 4.1)")
    if R._apple_gpu_mtl4_matmul2d_lowp_sym() is None:
        pytest.fail("Metal 4 is up on macOS 27 but the loaded runtime exports no "
                    "tessera_apple_gpu_mtl4_matmul2d_lowp -- rebuild the dylib (stale runtime)")
    return True

# exact codes ---------------------------------------------------------------
E4M3 = {448.0: 0x7E, -448.0: 0xFE, 1.0: 0x38, -1.0: 0xB8, 0.0625: 0x18, 0.0: 0x00, 256.0: 0x78}
E5M2 = {448.0: 0x5F, -448.0: 0xDF, 1.0: 0x3C, -1.0: 0xBC, 0.0625: 0x2C, 0.0: 0x00,
        32768.0: 0x78, 2048.0: 0x68}
E2M1 = {6.0: 0x7, -6.0: 0xF, 1.0: 0x2, -1.0: 0xA, 0.5: 0x1, 0.0: 0x0}
CODES = {"fp8_e4m3": E4M3, "fp8_e5m2": E5M2, "fp4_e2m1": E2M1}


def _pack(values, fmt, rows):
    """Storage for ``rows`` identical rows of ``values`` (padded to Q)."""
    table = CODES[fmt]
    for v in values:
        assert v in table, (fmt, v)
    codes = np.array([table[v] for v in values], np.uint8)
    q = Q[fmt]
    ld = -(-codes.size // q) * q
    row = np.zeros(ld, np.uint8)
    row[:codes.size] = codes
    if fmt == "fp4_e2m1":
        row = (row[::2] | (row[1::2] << 4)).astype(np.uint8)
    return np.ascontiguousarray(np.tile(row, (rows, 1)))


def _run(fmt, a_vals, b_vals, *, a_half=False):
    """Every C element = sum_k a_vals[k] * b_vals[k] (A rows / B columns identical)."""
    K = len(a_vals)
    N = Q[fmt]
    M = 64
    if a_half:
        A = np.tile(np.array(a_vals, np.float16), (M, 1))
        assert np.array_equal(A[0].astype(np.float64), np.array(a_vals))  # exact in fp16
    else:
        A = _pack(a_vals, fmt, M)
    # B is K x N with every column equal to b_vals: build row-wise (each row k is b_vals[k] repeated)
    table = CODES[fmt]
    rowcodes = np.array([table[v] for v in b_vals], np.uint8)
    B = np.repeat(rowcodes[:, None], N, axis=1)
    if fmt == "fp4_e2m1":
        B = (B[:, ::2] | (B[:, 1::2] << 4)).astype(np.uint8)
    C = R.apple_gpu_mtl4_matmul2d_lowp(A, B, np, fmt=fmt, M=M, N=N, K=K,
                                       a_dtype="f16" if a_half else "lowp")
    assert np.all(C == C[0, 0]), "identical rows/columns must give one value"
    return float(C[0, 0])


def _k(fmt, n):
    """Pad a case to a K on the stride quantum with exact zeros."""
    return n


# ---------------------------------------------------------------- small into large

@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_small_terms_survive_a_large_partial_sum(lane, fmt):
    """Rigel's hostile cell: 448 + 127 x 0.0625 = 455.9375. An fp16 accumulator
    (ulp 0.25 at 448) loses every small term; fp32 keeps all of them."""
    a = [448.0] + [0.0625] * 127
    b = [1.0] * 128
    assert _run(fmt, a, b) == 455.9375


def test_small_terms_survive_with_half_left_operand(lane):
    a = [448.0] + [0.0625] * 127
    assert _run("fp8_e4m3", a, [1.0] * 128, a_half=True) == 455.9375


def test_small_terms_survive_on_the_fp16_mpp_lane(lane):
    """Cross-lane check: the incumbent fp16 matmul2d accumulates the same cell
    to the same fp32 value."""
    K, M, N = 128, 64, 128
    A = np.tile(np.array([448.0] + [0.0625] * 127, np.float16), (M, 1))
    B = np.ones((K, N), np.float16)
    C, ran = R.apple_gpu_mtl4_matmul2d_f16(A, B, np)
    assert ran
    assert np.all(C == np.float32(455.9375))


def test_fp4_small_terms(lane):
    """FP4 has no value small enough to fall below an fp16 ulp at its max (6),
    so this only proves exact accumulation of 6 + 127 x 0.5 = 69.5."""
    assert _run("fp4_e2m1", [6.0] + [0.5] * 127, [1.0] * 128) == 69.5


# ---------------------------------------------------------------- cancellation

@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_exact_cancellation_of_large_terms(lane, fmt):
    a = [448.0] * 64 + [-448.0] * 64
    assert _run(fmt, a, [1.0] * 128) == 0.0


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_small_residue_after_cancellation(lane, fmt):
    """Pairs that cancel plus two small terms: the residue 0.125 must survive
    whatever order the pairs are summed in (all partial sums are exact in fp32)."""
    a = [448.0, -448.0] * 63 + [0.0625, 0.0625]
    assert _run(fmt, a, [1.0] * 128) == 0.125


def test_fp4_cancellation(lane):
    a = [6.0, -6.0] * 63 + [0.5, 0.5]
    assert _run("fp4_e2m1", a, [1.0] * 128) == 1.0


# ---------------------------------------------------------------- long reductions

@pytest.mark.parametrize("fmt,K", [("fp8_e4m3", 4096), ("fp8_e5m2", 4096), ("fp4_e2m1", 4096),
                                   ("fp8_e4m3", 8192)])
def test_long_reduction_of_ones_is_exact(lane, fmt, K):
    assert _run(fmt, [1.0] * K, [1.0] * K) == float(K)


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2"])
def test_long_reduction_of_small_terms_is_exact(lane, fmt):
    assert _run(fmt, [0.0625] * 8192, [1.0] * 8192) == 512.0


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "fp8_e5m2", "fp4_e2m1"])
def test_long_alternating_reduction_cancels_exactly(lane, fmt):
    K = 8192
    assert _run(fmt, [1.0, -1.0] * (K // 2), [1.0] * K) == 0.0


# ---------------------------------------------------------------- order characterisation

def test_reduction_order_characterisation(lane, record_property):
    """One product of 2^30 followed by 256 products of 1.0 (E5M2). The exact sum
    is 2^30 + 256. A strictly sequential fp32 accumulator returns 2^30 (each +1
    is below half an ulp of 128 and is lost); a tree that sums the ones in
    blocks first lands at 2^30 + 128 or 2^30 + 256. The value is recorded as
    evidence of the reduction structure -- not asserted, because any of these
    is a valid fp32 accumulation."""
    a = [32768.0] + [1.0] * 256 + [0.0] * (512 - 257)
    b = [32768.0] + [1.0] * 256 + [0.0] * (512 - 257)
    got = _run("fp8_e5m2", a, b)
    base = float(2 ** 30)
    assert got in {base, base + 128.0, base + 256.0}, got
    record_property("reduction_order_result", got)
    record_property("reduction_order_delta_from_2pow30", got - base)
