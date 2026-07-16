"""Phase E2 — generator hardening + metamorphic oracle (EVALUATOR_PLAN.md).

  * portable: legal-by-construction safe inputs (DESIL UB-elim / NNSmith) are
    finite + in-domain;
  * Darwin: the metamorphic-equivalence oracle confirms an algebraic invariant
    the compiler must preserve (softmax shift-invariance) and flags a relation
    that is *not* invariant (softmax scale) as divergent — reference-free, on the
    real backend.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.compiler.evaluator import metamorphic_equivalence, safe_input


def _sm(a):
    return ts.ops.softmax(a, axis=-1)


_SM = ts.jit(target="apple_gpu")(_sm)


# ── portable: legal-by-construction inputs ───────────────────────────────────

def test_safe_inputs_are_finite_and_in_domain():
    rng = np.random.default_rng(0)
    shape = (16, 16)
    assert np.all(np.isfinite(safe_input("real", shape, rng)))
    assert np.all(safe_input("positive", shape, rng) > 0.0)        # log/sqrt domain
    assert np.all(np.abs(safe_input("nonzero", shape, rng)) >= 0.1)  # division denom
    assert np.all(np.abs(safe_input("unit", shape, rng)) <= 1.05)
    # bounded — no overflow feeding matmul/exp
    assert np.max(np.abs(safe_input("real", shape, rng))) < 10.0


def test_safe_input_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown safe-input kind"):
        safe_input("bogus", (4, 4), np.random.default_rng(0))


# ── Darwin: metamorphic-equivalence oracle on real Metal ─────────────────────

@pytest.mark.hardware_apple_gpu
def test_softmax_shift_invariance_holds():
    """softmax(x) ≡ softmax(x + c) is an algebraic invariant the compiler must
    preserve — a reference-free numerical-stability check."""
    rng = np.random.default_rng(20260612)
    x = safe_input("real", (16, 16), rng)
    c = np.float32(3.0)
    v = metamorphic_equivalence("apple_gpu", _SM, (x,), (x + c,))
    assert v.relation == "equivalent", v.detail


@pytest.mark.hardware_apple_gpu
def test_softmax_is_not_scale_invariant_oracle_catches_it():
    """softmax(x) ≠ softmax(2x): a relation that is NOT an invariant must be
    flagged divergent — proving the metamorphic oracle has teeth."""
    rng = np.random.default_rng(7)
    x = safe_input("real", (16, 16), rng)
    v = metamorphic_equivalence("apple_gpu", _SM, (x,), (2.0 * x,))
    assert v.is_divergent, v.detail
