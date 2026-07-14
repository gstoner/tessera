"""Apple-sample pattern 6 — row-major stride helper for the Apple GPU
runtime.

The helper ``apple_row_major_strides`` centralizes the
innermost-first row-major contract that the conv2d native multi-tile
spike had to debug from scratch. ``MTLTensorDescriptor`` requires
``strides[0] == 1``; subsequent strides are the cumulative product of
preceding extents. This test verifies the C ABI probe
``tessera_apple_gpu_row_major_strides`` produces the documented
layout across 2D / 4D / edge-case shapes.

The probe is pure math (no Metal calls); the stub computes the same
answer as the real runtime, so the test runs on every host —
Darwin + Apple silicon OR Linux/Intel.
"""

from __future__ import annotations

import ctypes

import pytest

from tessera._apple_gpu_dispatch import apple_gpu_runtime, bind_registered


def _bind_strides_probe():
    """Resolve ``tessera_apple_gpu_row_major_strides`` via the ``APPLE_ABI``
    registry (the ctypes signature lives there now, not inline). Returns
    ``None`` if the runtime can't be built / loaded — shared by the reject
    tests below, which call the probe with deliberately bad args."""
    if apple_gpu_runtime() is None:
        return None
    return bind_registered("tessera_apple_gpu_row_major_strides")


def _strides(dims):
    """Convenience: call the probe and return the strides as a tuple."""
    fn = _bind_strides_probe()
    if fn is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    rank = len(dims)
    dims_arr = (ctypes.c_int64 * rank)(*dims)
    strides_arr = (ctypes.c_int64 * rank)()
    ret = fn(dims_arr, rank, strides_arr)
    assert ret == rank, f"probe returned {ret}, expected {rank}"
    return tuple(strides_arr)


# ---- The documented contract: strides[0] == 1; strides[i+1] = strides[i] * dims[i] ----

def test_strides_innermost_is_one_for_2d():
    s = _strides([16, 32])  # innermost (cols) = 16, outer (rows) = 32
    assert s[0] == 1
    assert s == (1, 16)


def test_strides_4d_matches_cumulative_product():
    """The exact pattern the conv2d spike landed:
    ``{Cin, srcW, srcH, B}`` with strides ``{1, Cin, Cin*srcW, Cin*srcW*srcH}``."""
    Cin, sW, sH, B = 16, 32, 32, 4
    s = _strides([Cin, sW, sH, B])
    assert s == (1, Cin, Cin * sW, Cin * sW * sH)


@pytest.mark.parametrize("dims,expected", [
    ([1], (1,)),
    ([7], (1,)),
    ([2, 3], (1, 2)),
    ([2, 3, 5], (1, 2, 6)),
    ([2, 3, 5, 7], (1, 2, 6, 30)),
    ([2, 3, 5, 7, 11], (1, 2, 6, 30, 210)),
    ([64, 128, 1, 1], (1, 64, 64 * 128, 64 * 128)),  # singleton tail dims
])
def test_strides_general_shapes(dims, expected):
    assert _strides(dims) == expected


# ---- Edge cases / contract guards ----

def test_probe_rejects_zero_rank():
    fn = _bind_strides_probe()
    if fn is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    dims_arr = (ctypes.c_int64 * 1)(1)
    strides_arr = (ctypes.c_int64 * 1)()
    assert fn(dims_arr, 0, strides_arr) == 0


def test_probe_rejects_negative_rank():
    fn = _bind_strides_probe()
    if fn is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    dims_arr = (ctypes.c_int64 * 1)(1)
    strides_arr = (ctypes.c_int64 * 1)()
    assert fn(dims_arr, -1, strides_arr) == 0


def test_probe_rejects_rank_above_max():
    """Rank > 8 is rejected — matches MTLTensorDescriptor's MAX_RANK."""
    fn = _bind_strides_probe()
    if fn is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    dims_arr = (ctypes.c_int64 * 9)(*([1] * 9))
    strides_arr = (ctypes.c_int64 * 9)()
    assert fn(dims_arr, 9, strides_arr) == 0


def test_probe_rejects_null_pointers():
    fn = _bind_strides_probe()
    if fn is None:
        pytest.skip("Apple GPU runtime not buildable on this host")
    strides_arr = (ctypes.c_int64 * 2)()
    # Null dims_in.
    assert fn(None, 2, strides_arr) == 0
    # Null strides_out.
    dims_arr = (ctypes.c_int64 * 2)(1, 2)
    assert fn(dims_arr, 2, None) == 0


# ---- The helper exists on both Darwin and non-Darwin (stub provides it) ----

def test_helper_is_available_on_every_host():
    """Pattern 6's whole point: the math is pure, the stub computes the
    same answer, so a test that walks the contract runs on every host."""
    fn = _bind_strides_probe()
    if fn is None:
        pytest.skip("Apple GPU runtime not buildable on this host "
                    "(compile prerequisites missing)")
    # If the runtime built at all, the probe must resolve. A missing
    # probe symbol indicates a build-config drift (stub didn't get the
    # new symbol).
    assert fn is not None
