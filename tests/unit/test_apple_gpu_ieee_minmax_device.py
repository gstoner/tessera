"""Exact-device IEEE-754-2019 min/max contract on the Apple GPU (M1 Max).

Fleet contract ``IEEE-MINMAX-CONTRACT-2026-08-23`` (rocm plan; owner decision
2026-08-23): ``tessera.maximum``/``minimum`` propagate NaN and ORDER signed
zeros — max tie -> +0.0, min tie -> -0.0 — on EVERY execution route. gfx1151
and the x86 AVX-512 shim were fixed and pinned there; Apple was recorded as an
unaudited follow-up in ``docs/audit/backend/apple/todo.md``
(``JIT-MATH-AUDIT-2026-08-23`` item 3). This module is that audit.

Measured on an M1 Max before the fix, MPSGraph's own
``maximumWithPrimaryTensor``/``minimumWithPrimaryTensor`` are maxNum/minNum:
NaN suppressed (``max(NaN, 1) -> 1``, ``max(NaN, -inf) -> -inf``) and a +/-0
tie resolved to the SECOND operand. 7 of the 12 special-value rows below
disagreed with the contract. The same NaN laundering was measured in the
``scatter_f32`` min/max reduce and in the Cl(3,0) norm kernel's
``sqrt(max(0, s))`` clamp — MSL's ``max``/``min`` on floats are the same
``__metal_fmax``/``__metal_fmin`` intrinsic as ``fmax``/``fmin``
(metal_math header), never a ``>`` comparison.

These assertions are BIT-LEVEL and hit the device symbols directly rather than
``ops.*``: a numpy fallback would satisfy a value-level check while proving
nothing about Metal. numpy is not a valid tie-sign oracle either (it resolves a
+/-0 tie to whatever the host ISA returns), so every expectation here comes
from the shared reference in ``tessera/_ieee_minmax.py``.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera import _ieee_minmax as IM

pytestmark = pytest.mark.hardware_apple_gpu

_NAN = np.float32("nan")
_INF = np.float32("inf")
_PZ = np.float32(0.0)
_NZ = np.float32(-0.0)

#: (a, b) special-value matrix: both tie orders, both tie signs, NaN on each
#: side, NaN vs both infinities, an ordered pair, and NaN vs a zero.
_A = np.array(
    [_PZ, _NZ, _PZ, _NZ, _NAN, 1.0, _NAN, _NAN, -_INF, _INF, 2.0, _NAN], np.float32)
_B = np.array(
    [_NZ, _PZ, _PZ, _NZ, 1.0, _NAN, _NAN, -_INF, _NAN, _NAN, 3.0, _PZ], np.float32)

_CF = ctypes.POINTER(ctypes.c_float)


def _bits(x):
    return np.ascontiguousarray(x, np.float32).view(np.uint32)


def _require(name):
    """Resolve a device symbol, FAILING (not skipping) when the dylib is stale.

    ``hardware_apple_gpu`` already established that Metal is present, so a
    missing symbol here means the built dylib predates the kernel — reporting
    that as a skip would hide a regression behind a green run.
    """
    from tessera import runtime as R

    lib = R._load_apple_gpu_runtime()
    sym = getattr(lib, name, None)
    assert sym is not None, (
        f"{name} missing from {lib._name}; rebuild with "
        "`ninja -C build TesseraAppleRuntimeShared`")
    return sym


# ── binary tessera.maximum / tessera.minimum (MPSGraph lane) ────────────────
@pytest.mark.parametrize("op,name,ref", [
    (4, "maximum", IM.ieee_maximum),
    (5, "minimum", IM.ieee_minimum),
])
def test_mpsgraph_binary_minmax_is_ieee_on_device(op, name, ref):
    from tessera import runtime as R

    sym = R._apple_gpu_mpsgraph_binary_f32()
    assert sym is not None, "MPSGraph binary symbol unavailable; rebuild the dylib"
    a = np.ascontiguousarray(_A)
    b = np.ascontiguousarray(_B)
    out = np.empty(a.size, np.float32)
    sym(ctypes.c_int32(op), a.ctypes.data_as(_CF), b.ctypes.data_as(_CF),
        out.ctypes.data_as(_CF), ctypes.c_int64(a.size))

    want = np.asarray(ref(_A, _B), np.float32)
    bad = [
        f"{name}({_A[i]!s}, {_B[i]!s}) = {out[i]!s} [{_bits(out)[i]:08x}] "
        f"want {want[i]!s} [{_bits(want)[i]:08x}]"
        for i in range(a.size)
        if _bits(out)[i] != _bits(want)[i]
    ]
    assert not bad, "device disagrees with the IEEE-754-2019 contract:\n" + "\n".join(bad)


@pytest.mark.parametrize("op_name,tie_sign", [
    ("tessera.maximum", False),  # max tie -> +0.0
    ("tessera.minimum", True),   # min tie -> -0.0
])
def test_dispatch_route_orders_ties_and_propagates_nan(op_name, tie_sign):
    """The named-op route, not just the raw symbol — ties are explicit, not
    delegated to whatever numpy's host ISA happens to return."""
    from tessera.runtime import _apple_gpu_dispatch_mpsgraph_binary as dispatch

    out = np.asarray(dispatch(op_name, [_A, _B], {}, np), np.float32)
    # rows 0..3 are the four +/-0 tie orders: (+0,-0) (-0,+0) (+0,+0) (-0,-0).
    np.testing.assert_array_equal(
        np.signbit(out[:4]), [tie_sign, tie_sign, False, True])
    # every row with a NaN operand is NaN, and no other row is.
    np.testing.assert_array_equal(
        np.isnan(out), np.isnan(_A) | np.isnan(_B))


# ── C-ABI host recovery path (fires when Metal dispatch fails) ──────────────
@pytest.mark.parametrize("op,name,ref", [
    (4, "maximum", IM.ieee_maximum),
    (5, "minimum", IM.ieee_minimum),
])
def test_c_abi_host_fallback_matches_device_contract(op, name, ref):
    """The f32 binary lane's host recovery path (used when the on-device status
    call fails, and by any direct C-ABI caller that lands there) must carry the
    same IEEE-754-2019 min/max contract as the graph node — a bare ``x > y``
    ternary there would suppress a left-operand NaN and pick the second signed
    zero. Exercised through the exported ``..._binary_f32_host`` symbol, which
    IS that recovery path factored out, so this holds without forcing a Metal
    failure (which a working device cannot do)."""
    sym = _require("tessera_apple_gpu_mpsgraph_binary_f32_host")
    a = np.ascontiguousarray(_A)
    b = np.ascontiguousarray(_B)
    out = np.empty(a.size, np.float32)
    sym(ctypes.c_int32(op), a.ctypes.data_as(_CF), b.ctypes.data_as(_CF),
        out.ctypes.data_as(_CF), ctypes.c_int64(a.size))

    want = np.asarray(ref(_A, _B), np.float32)
    bad = [
        f"host {name}({_A[i]!s}, {_B[i]!s}) = {out[i]!s} [{_bits(out)[i]:08x}] "
        f"want {want[i]!s} [{_bits(want)[i]:08x}]"
        for i in range(a.size)
        if _bits(out)[i] != _bits(want)[i]
    ]
    assert not bad, "host fallback disagrees with the IEEE contract:\n" + "\n".join(bad)


# ── scatter min/max reduce: the reduction result IS the output ──────────────
@pytest.mark.parametrize("mode,name,ref", [
    (2, "min", IM.ieee_minimum),
    (3, "max", IM.ieee_maximum),
])
def test_scatter_reduce_minmax_propagates_nan_and_orders_ties(mode, name, ref):
    sym = _require("tessera_apple_gpu_scatter_f32")
    seed = np.array([[1.0], [_PZ], [_NAN]], np.float32)
    src = np.array([[_NAN], [_NZ], [2.0]], np.float32)
    idx = np.arange(3, dtype=np.int64)

    out = np.ascontiguousarray(seed.copy())
    s = np.ascontiguousarray(src)
    i = np.ascontiguousarray(idx)
    rc = sym(out.ctypes.data_as(_CF), s.ctypes.data_as(_CF),
             i.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
             ctypes.c_int32(3), ctypes.c_int32(3), ctypes.c_int32(1),
             ctypes.c_int32(mode))
    assert rc == 1, f"scatter_{name} did not run on the device (rc={rc})"

    want = np.asarray(ref(seed, src), np.float32)
    np.testing.assert_array_equal(_bits(out), _bits(want))


# ── Cl(3,0) norm: sqrt of a non-negative clamp must not launder NaN ─────────
def test_clifford_norm_clamp_propagates_nan():
    sym = _require("tessera_apple_gpu_clifford_norm_cl30_f32")
    A = np.zeros((2, 8), np.float32)
    A[0, 0] = _NAN          # NaN component -> NaN norm, not 0
    A[1, 0], A[1, 1] = 3.0, 4.0
    a = np.ascontiguousarray(A)
    out = np.zeros(2, np.float32)
    rc = sym(a.ctypes.data_as(_CF), out.ctypes.data_as(_CF), ctypes.c_int32(2))
    assert rc == 1, f"clifford_norm did not run on the device (rc={rc})"

    # ga.ops.norm's reference is sqrt(clip(<a,a>, 0, None)) — np.clip propagates.
    want = np.sqrt(np.clip((A.astype(np.float64) ** 2).sum(1), 0.0, None))
    assert np.isnan(out[0]), f"NaN component laundered to {out[0]!s}"
    np.testing.assert_allclose(out[1], want[1], rtol=1e-6)
