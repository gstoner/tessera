"""TPP stencil arbiter retarget — the tpp-space-time `lower-tpp-to-target-ir`
seam pointed at the D1 candidate arbiter.

Verifies the shipped ``ts_stencil_grad_cpu`` kernel is registered as an F4-gated
candidate for the ``tpp_stencil`` op-kind: the real device_verified_jit kernel matches a
numpy central-difference reference through the arbiter, a wrong candidate is
refused, and the arbiter falls back honestly to the reference.
"""
from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.emit import candidate as C
from tessera.compiler.emit.candidate import Candidate, Tier
from tessera.compiler.emit import tpp_candidates as TC
from tessera.compiler.emit.tpp_candidates import OP_TPP_STENCIL, StencilGradRegion


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = {k: list(v) for k, v in C._CANDIDATES.items()}
    yield
    C._CANDIDATES.clear()
    C._CANDIDATES.update(saved)


def _cpu():
    return next(c for c in C.candidates_for("cpu", OP_TPP_STENCIL)
               if c.name == "cpu_stencil_grad")


def test_op_kind_registered():
    assert OP_TPP_STENCIL in C._OP_KIND_VERIFY


@pytest.mark.parametrize("axis,order", [(0, 2), (1, 2), (0, 4), (1, 4)])
def test_cpu_stencil_matches_reference(axis, order):
    cpu = _cpu()
    if not cpu.available():
        pytest.skip("no C++ toolchain to build the shipped stencil kernel")
    reg = StencilGradRegion(16, 24, axis=axis, order=order)
    f = reg.probe_input(0)
    out, tag = cpu.run(reg, f)
    assert tag == "cpu_stencil_grad"
    assert np.allclose(out, reg.reference(f), atol=1e-4)
    assert C.verify_candidate(cpu, reg) is True


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
