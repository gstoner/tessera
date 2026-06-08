"""Close-out gap #4 — @clifford_jit signature gate (Cl(1,3) front/back mismatch).

The Multivector front-end accepts arbitrary Cl(p,q,r) algebras, but v1 ships
Apple-GPU kernels only for Cl(3,0) (cl30). A @clifford_jit callable invoked with
a non-Cl(3,0) Multivector must refuse with a stable diagnostic
(CLIFFORD_UNSUPPORTED_SIGNATURE) rather than silently routing to the numpy
reference (Decision #21). The plain tessera.ga.* lane stays available for
non-Cl(3,0) algebras.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ga
from tessera.ga.multivector import Multivector
from tessera.ga.signature import Cl
from tessera.compiler.clifford_jit import CliffordJitError, clifford_jit
from tessera.compiler.diagnostics import ConstrainedDiagnosticCode


def _mv(sig, dim, seed):
    return Multivector(
        np.random.default_rng(seed).standard_normal(dim).astype(np.float32), sig)


@clifford_jit(target="apple_gpu")
def _gp(a, b):
    return ga.geometric_product(a, b)


@clifford_jit(target="apple_gpu")
def _sandwich_norm(r, x):
    return ga.norm(ga.rotor_sandwich(r, x))


# The v1 front-end allow-list is exactly {Cl(3,0), Cl(1,3)} (see
# ga/signature.py V1_ALLOWED_SIGNATURES), so Cl(1,3) is the only
# front-end-expressible signature that lacks a GPU kernel — the precise gap #4.


def test_cl30_inputs_run():
    cl30 = Cl(3, 0, 0)
    out = _gp(_mv(cl30, 8, 0), _mv(cl30, 8, 1))
    assert out is not None


def test_cl13_signature_is_gated():
    cl13 = Cl(1, 3, 0)
    a, b = _mv(cl13, 16, 2), _mv(cl13, 16, 3)
    with pytest.raises(CliffordJitError) as exc:
        _gp(a, b)
    assert exc.value.code == ConstrainedDiagnosticCode.CLIFFORD_UNSUPPORTED_SIGNATURE.value
    assert "Cl(1, 3, 0)" in str(exc.value)
    assert "Cl(3,0)" in str(exc.value)


def test_gate_applies_to_multi_op_plan():
    cl13 = Cl(1, 3, 0)
    with pytest.raises(CliffordJitError) as exc:
        _sandwich_norm(_mv(cl13, 16, 4), _mv(cl13, 16, 5))
    assert exc.value.code == ConstrainedDiagnosticCode.CLIFFORD_UNSUPPORTED_SIGNATURE.value


def test_plain_ga_lane_still_runs_cl13():
    # The numpy reference lane is unaffected — only the @clifford_jit path gates.
    cl13 = Cl(1, 3, 0)
    a, b = _mv(cl13, 16, 6), _mv(cl13, 16, 7)
    assert ga.geometric_product(a, b) is not None
