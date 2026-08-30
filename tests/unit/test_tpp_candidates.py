"""TPP stencil arbiter retarget — the tpp-space-time `lower-tpp-to-target-ir`
seam pointed at the D1 candidate arbiter.

Verifies the shipped ``ts_stencil_grad_cpu`` kernel is registered as an F4-gated
candidate for the ``tpp_stencil`` op-kind: the real compiled kernel matches a
numpy central-difference reference through the arbiter, a wrong candidate is
refused, and the arbiter falls back honestly to the reference.
"""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import Candidate, Tier
from tessera.compiler.emit import tpp_candidates as TC
from tessera.compiler.emit.tpp_candidates import (
    OP_TPP_STENCIL,
    StencilGradRegion,
    central_difference_taps,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = {k: list(v) for k, v in C._CANDIDATES.items()}
    yield
    C._CANDIDATES.clear()
    C._CANDIDATES.update(saved)


def _cpu():
    return next(c for c in C.candidates_for("cpu", OP_TPP_STENCIL)
               if c.name == "cpu_stencil_grad")


def _neighbors_periodic_stencil(field, axis, offsets, coeffs):
    """Independent numerical oracle for Neighbors tap/coeff semantics."""
    out = np.zeros_like(field, dtype=np.float32)
    for offset, coeff in zip(offsets, coeffs, strict=True):
        out += np.float32(coeff) * np.roll(field, -offset, axis=axis)
    return out


def test_op_kind_registered():
    assert OP_TPP_STENCIL in C._OP_KIND_VERIFY


@pytest.mark.parametrize(
    "axis,order,spacing",
    [
        (0, 2, (0.5, 2.0)),
        (1, 2, (0.5, 2.0)),
        (0, 4, (0.25, 1.5)),
        (1, 4, (0.25, 1.5)),
    ],
)
def test_cpu_stencil_matches_reference(axis, order, spacing):
    cpu = _cpu()
    if not cpu.available():
        pytest.skip("no C++ toolchain to build the shipped stencil kernel")
    reg = StencilGradRegion(16, 24, axis=axis, order=order, spacing=spacing)
    f = reg.probe_input(0)
    out, tag = cpu.run(reg, f)
    assert tag == "cpu_stencil_grad"
    assert np.allclose(out, reg.reference(f), atol=1e-4)
    assert C.verify_candidate(cpu, reg) is True


@pytest.mark.parametrize("spacing", [(1.0,), (1.0, 0.0), (1.0, float("inf"))])
def test_stencil_region_rejects_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="finite positive"):
        StencilGradRegion(8, 8, spacing=spacing)


@pytest.mark.parametrize("derivative_order", [1, 2])
@pytest.mark.parametrize("accuracy_order", [2, 4])
def test_derivative_taps_scale_by_spacing_power(
    derivative_order, accuracy_order
):
    offsets, unit = central_difference_taps(
        accuracy_order, derivative_order, 1.0
    )
    scaled_offsets, scaled = central_difference_taps(
        accuracy_order, derivative_order, 0.25
    )
    assert scaled_offsets == offsets
    expected_scale = 0.25 ** -derivative_order
    np.testing.assert_allclose(scaled, np.asarray(unit) * expected_scale)


@pytest.mark.parametrize("axis,order", [(0, 2), (1, 2), (0, 4), (1, 4)])
def test_tpp_cpu_matches_neighbors_explicit_coefficient_oracle(axis, order):
    """Declare the two stencil stacks as differential oracles.

    TPP owns the compiled directional-gradient kernel. Neighbors owns the
    general tap/coefficient semantics. Their common central-difference subset
    must agree on a non-unit grid before either physical lane may be promoted.
    """
    cpu = _cpu()
    if not cpu.available():
        pytest.skip("no C++ toolchain")
    spacing = (0.25, 1.5)
    region = StencilGradRegion(
        17, 19, axis=axis, order=order, spacing=spacing
    )
    field = region.probe_input(7)
    offsets, coeffs = central_difference_taps(order, 1, spacing[axis])
    expected = _neighbors_periodic_stencil(field, axis, offsets, coeffs)
    actual, tag = cpu.run(region, field)
    assert tag == "cpu_stencil_grad"
    assert np.allclose(actual, expected, atol=1e-5)


def test_arbiter_picks_and_rejects_wrong():
    if not _cpu().available():
        pytest.skip("no C++ toolchain")
    reg = StencilGradRegion(16, 16, axis=0, order=2)
    win = C.arbitrate(reg, OP_TPP_STENCIL, "cpu")
    assert win is not None and win.name == "cpu_stencil_grad"

    class _Wrong(Candidate):
        name, tier, target, op = "wrong", Tier.HAND_TUNED, "cpu", OP_TPP_STENCIL

        def run(self, region, f, *a, **k):
            return np.full((region.nx, region.ny), 7.0, np.float32), "wrong"

    C.register_candidate(_Wrong())
    assert C.verify_candidate(_Wrong(), reg) is False
    assert C.arbitrate(reg, OP_TPP_STENCIL, "cpu").name == "cpu_stencil_grad"


def test_reference_fallback():
    reg = StencilGradRegion(8, 8, axis=0, order=2)
    f = reg.probe_input(0)
    out, tag = C.run_arbitrated(reg, OP_TPP_STENCIL, "no_such_target", f)
    assert tag == "reference"
    assert np.allclose(out, reg.reference(f))


# --- P2 code review 2026-08-29: declared order must be the order computed -----

@pytest.mark.parametrize("order,rate", [(2, 2.0), (4, 4.0), (6, 6.0)])
def test_declared_order_matches_observed_convergence(order, rate):
    """``order`` selects the operator's truncation error, so the kernel that
    answers must be the one that was asked for.  ``order >= 4`` used to route 6
    and 8 into the fourth-order tap set, returning an O(h^4) answer under an
    order-6 label — an error a convergence study blames on the model."""
    if not _cpu().available():
        pytest.skip("no C++ toolchain")
    lib = TC._cpu_lib()

    def max_error(n: int) -> float:
        h = 2.0 * np.pi / n
        x = np.arange(n) * h
        field = np.ascontiguousarray(np.tile(np.sin(x), (n, 1)).astype(np.float32))
        out = np.empty_like(field)
        status = lib.ts_stencil_grad_cpu(
            TC._cptr(field), TC._cptr(out),
            n, n, 1, order, ctypes.c_float(h))
        assert status == 0
        return float(np.abs(out[0].astype(np.float64) - np.cos(x)).max())

    # Coarse grids only: at n >= 64 the order-6 truncation error is already
    # below float32 round-off, so a finer pair would measure noise, not order.
    observed = np.log2(max_error(8) / max_error(16))
    assert observed == pytest.approx(rate, abs=0.35)


@pytest.mark.parametrize("order", [3, 10, 0, -2])
def test_unimplemented_order_is_refused_not_silently_downgraded(order):
    if not _cpu().available():
        pytest.skip("no C++ toolchain")
    lib = TC._cpu_lib()
    field = np.zeros((8, 8), np.float32)
    out = np.empty_like(field)
    status = lib.ts_stencil_grad_cpu(
        TC._cptr(field), TC._cptr(out),
        8, 8, 1, order, ctypes.c_float(0.1))
    assert status != 0


def test_candidate_raises_rather_than_returning_a_wrong_order():
    if not _cpu().available():
        pytest.skip("no C++ toolchain")
    region = StencilGradRegion(8, 8, axis=0, order=10)
    with pytest.raises(TC.TesseraStencilHookError):
        _cpu().run(region, region.probe_input(0))
