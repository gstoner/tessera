"""Apple stride-alignment enforcement — skills.md Pattern 3 follow-on.

Apple's MTLTensorDescriptor.strides documentation specifies three rules
that Tessera's stride helper must honor when emitting tensor
descriptors for the Metal 4 ML lane:

1. ``strides[0]`` must equal 1 — the innermost dimension is contiguous.
2. For ``MTLTensorUsageMachineLearning`` with byte-sized dtypes
   (≥ 8 bits / element), the second stride's BYTE size
   (``strides[1] * element_bytes``) must be aligned to 64 bytes.
3. For sub-byte dtypes (< 8 bits / element — e.g. int4, fp4), the
   second stride's byte size must be aligned to 128 bytes, regardless
   of usage flag.

The existing ``tessera_apple_gpu_row_major_strides`` helper computes
plain cumulative-product row-major strides (rule #1 only). The new
``_aligned`` variant adds rules #2 and #3. Both helpers coexist;
callers pick based on whether they're describing a HOST buffer
layout (dense — old helper) or a TENSOR DESCRIPTOR's internal layout
(aligned — new helper).

These tests pin:

* **Backward compat** — the legacy helper still produces dense
  cumulative-product strides.
* **Rule 1** — innermost stride is always 1.
* **Rule 2** — ML usage with fp32/fp16/bf16/int8 enforces 64-byte
  alignment on the second stride.
* **Rule 3** — sub-byte dtypes enforce 128-byte alignment regardless
  of usage flag.
* **Generic usage (non-ML, byte+ dtypes)** — no alignment enforced.
* **Cumulative-from-aligned** — third+ strides are products from the
  ALIGNED second stride.
* **Rank-1** — only strides[0] = 1.
* **Invalid inputs** — null pointers / out-of-range rank /
  element_bits return 0.
"""

from __future__ import annotations

import ctypes
import math

import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol


def _bind_aligned():
    if apple_gpu_runtime() is None:
        return None
    return bind_symbol(
        "tessera_apple_gpu_row_major_strides_aligned",
        (ctypes.POINTER(ctypes.c_int64), ctypes.c_int32,
         ctypes.c_int32, ctypes.c_int32,
         ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int32)


def _bind_legacy():
    if apple_gpu_runtime() is None:
        return None
    return bind_symbol(
        "tessera_apple_gpu_row_major_strides",
        (ctypes.POINTER(ctypes.c_int64), ctypes.c_int32,
         ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int32)


def _call_aligned(fn, dims, element_bits, ml_usage):
    rank = len(dims)
    dims_buf = (ctypes.c_int64 * rank)(*dims)
    strides_buf = (ctypes.c_int64 * rank)()
    rc = fn(dims_buf, ctypes.c_int32(rank),
            ctypes.c_int32(element_bits),
            ctypes.c_int32(1 if ml_usage else 0),
            strides_buf)
    return int(rc), [int(s) for s in strides_buf]


def _call_legacy(fn, dims):
    rank = len(dims)
    dims_buf = (ctypes.c_int64 * rank)(*dims)
    strides_buf = (ctypes.c_int64 * rank)()
    rc = fn(dims_buf, ctypes.c_int32(rank), strides_buf)
    return int(rc), [int(s) for s in strides_buf]


# ---- Symbol-availability ------------------------------------------------

def test_aligned_symbol_resolves():
    if apple_gpu_runtime() is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    assert _bind_aligned() is not None


# ---- Rule 1: innermost stride is always 1 ------------------------------

def test_innermost_stride_is_one_for_every_combination():
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    for dims in [(7,), (7, 5), (3, 11, 4), (2, 3, 5, 7)]:
        for bits in (32, 16, 8, 4):
            for ml in (False, True):
                rc, strides = _call_aligned(fn, dims, bits, ml)
                assert rc == len(dims), (dims, bits, ml, rc)
                assert strides[0] == 1, (dims, bits, ml, strides)


# ---- Rule 2: 64-byte alignment for ML usage / byte+ dtypes -------------

def test_ml_usage_fp32_aligns_second_stride_to_16_elements():
    """fp32 = 32 bits. 64 bytes = 16 fp32s. The second stride must
    be a multiple of 16."""
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    # Cases: (innermost_dim, expected_aligned_stride_1).
    cases = [
        (1, 16),     # 1 → round up to 16
        (15, 16),    # 15 → 16
        (16, 16),    # already aligned
        (17, 32),    # 17 → 32
        (33, 48),    # 33 → 48
        (128, 128),  # already aligned
        (129, 144),  # 129 → 144
    ]
    for innermost, expected in cases:
        rc, strides = _call_aligned(fn, [innermost, 7], 32, True)
        assert rc == 2
        assert strides[1] == expected, (innermost, expected, strides)


def test_ml_usage_fp16_aligns_second_stride_to_32_elements():
    """fp16 = 16 bits. 64 bytes = 32 fp16s."""
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    cases = [(1, 32), (31, 32), (32, 32), (33, 64), (65, 96)]
    for innermost, expected in cases:
        rc, strides = _call_aligned(fn, [innermost, 4], 16, True)
        assert rc == 2
        assert strides[1] == expected, (innermost, expected, strides)


def test_ml_usage_int8_aligns_second_stride_to_64_elements():
    """int8 = 8 bits. 64 bytes = 64 int8s."""
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    cases = [(1, 64), (63, 64), (64, 64), (65, 128)]
    for innermost, expected in cases:
        rc, strides = _call_aligned(fn, [innermost, 2], 8, True)
        assert rc == 2
        assert strides[1] == expected, (innermost, expected, strides)


# ---- Rule 3: 128-byte alignment for sub-byte dtypes --------------------

def test_int4_aligns_to_256_elements_regardless_of_usage_flag():
    """int4 = 4 bits. 128 bytes = 1024 bits / 4 = 256 packed-elements.
    The 128-byte rule applies REGARDLESS of ml_usage flag."""
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    cases = [(1, 256), (255, 256), (256, 256), (257, 512)]
    for ml_flag in (False, True):
        for innermost, expected in cases:
            rc, strides = _call_aligned(fn, [innermost, 2], 4, ml_flag)
            assert rc == 2
            assert strides[1] == expected, (innermost, ml_flag, strides)


# ---- Generic (non-ML, byte+) usage: NO alignment enforced --------------

def test_generic_usage_with_byte_dtypes_has_no_alignment_rule():
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    # ml_usage=False with byte+ dtypes — should match legacy cumulative
    # product behavior (no padding).
    for bits in (32, 16, 8):
        rc, strides = _call_aligned(fn, [13, 7], bits, False)
        assert rc == 2
        assert strides == [1, 13], (bits, strides)


# ---- Cumulative-from-aligned: third+ strides use the aligned stride[1] -

def test_third_stride_is_aligned_stride_1_times_second_dim():
    """For rank > 2, strides[2] = strides[1] * dims[1] (cumulative
    product STARTING from the aligned stride[1])."""
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    # rank=3, dims=[13, 7, 5], ml_usage=True, fp32.
    # strides[0] = 1
    # strides[1] = round_up(13, 16) = 16
    # strides[2] = 16 * 7 = 112
    rc, strides = _call_aligned(fn, [13, 7, 5], 32, True)
    assert rc == 3
    assert strides == [1, 16, 112], strides


def test_rank4_strides_cumulate_from_aligned_stride_1():
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    # rank=4, dims=[7, 5, 3, 2], ml_usage=True, fp16.
    # strides[0] = 1
    # strides[1] = round_up(7, 32) = 32   (fp16: 64 bytes = 32 elements)
    # strides[2] = 32 * 5 = 160
    # strides[3] = 160 * 3 = 480
    rc, strides = _call_aligned(fn, [7, 5, 3, 2], 16, True)
    assert rc == 4
    assert strides == [1, 32, 160, 480], strides


# ---- Rank-1: only strides[0] = 1 ---------------------------------------

def test_rank1_has_only_innermost_stride():
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    for bits in (32, 16, 8, 4):
        for ml in (False, True):
            rc, strides = _call_aligned(fn, [100], bits, ml)
            assert rc == 1
            assert strides == [1], (bits, ml, strides)


# ---- Invalid input rejection -------------------------------------------

def test_zero_rank_returns_zero():
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    dims_buf = (ctypes.c_int64 * 0)()
    strides_buf = (ctypes.c_int64 * 0)()
    rc = fn(dims_buf, ctypes.c_int32(0), ctypes.c_int32(32),
            ctypes.c_int32(0), strides_buf)
    assert int(rc) == 0


def test_excessive_rank_returns_zero():
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    dims_buf = (ctypes.c_int64 * 9)(*([1] * 9))
    strides_buf = (ctypes.c_int64 * 9)()
    rc = fn(dims_buf, ctypes.c_int32(9), ctypes.c_int32(32),
            ctypes.c_int32(0), strides_buf)
    assert int(rc) == 0  # rank > 8 rejected


def test_zero_element_bits_returns_zero():
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    dims_buf = (ctypes.c_int64 * 2)(*[7, 3])
    strides_buf = (ctypes.c_int64 * 2)()
    rc = fn(dims_buf, ctypes.c_int32(2), ctypes.c_int32(0),
            ctypes.c_int32(0), strides_buf)
    assert int(rc) == 0


# ---- Backward compat: legacy helper untouched --------------------------

def test_legacy_helper_still_produces_dense_strides():
    """The original ``row_major_strides`` C ABI must keep its existing
    contract — dense cumulative-product strides, no alignment, no new
    args. Used by callers that describe a HOST buffer's layout where
    the dense form is correct."""
    fn = _bind_legacy()
    if fn is None:
        pytest.skip("legacy helper not available")
    rc, strides = _call_legacy(fn, [13, 7, 5])
    assert rc == 3
    assert strides == [1, 13, 91]  # cumulative: 1, 13, 13*7=91


# ---- End-to-end check: byte alignment of the second stride -------------

def test_second_stride_byte_size_meets_alignment_rule():
    """The whole point — for every alignment-eligible case, the second
    stride times the element byte size is a multiple of the right
    alignment (64 for ML+byte+, 128 for sub-byte)."""
    fn = _bind_aligned()
    if fn is None:
        pytest.skip("aligned helper not available")
    cases = [
        # (element_bits, ml_usage, expected_align_bytes)
        (32, True, 64),
        (16, True, 64),
        (8, True, 64),
        (4, False, 128),  # sub-byte rule fires regardless of ml_usage
        (4, True, 128),
    ]
    for bits, ml, align_bytes in cases:
        # Use a tricky innermost dim that forces padding.
        rc, strides = _call_aligned(fn, [13, 5], bits, ml)
        assert rc == 2
        byte_stride_1 = strides[1] * bits // 8  # 4-bit case: 256 * 4 / 8 = 128 bytes
        # For sub-byte (bits < 8), strides are in packed-element units;
        # byte stride is stride * bits / 8 (always an integer for our
        # alignment math).
        assert byte_stride_1 >= 64, (
            f"byte_stride too small: bits={bits} ml={ml} stride={strides[1]} "
            f"byte_stride={byte_stride_1}")
        assert (byte_stride_1 % align_bytes) == 0, (
            f"byte_stride_1={byte_stride_1} not aligned to {align_bytes}: "
            f"bits={bits} ml={ml}")
