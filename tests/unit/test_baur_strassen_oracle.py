"""R1 — Baur–Strassen cost-ratio oracle.

The oracle measures primitive-execution count of a full gradient against a
single forward pass. Baur–Strassen bounds this ratio by a small constant
(≤5 for arithmetic circuits); a value far above it is a structural defect —
canonically an AD implementation that re-runs the forward pass per output
element (review findings B1/B2). This test proves:

  1. the pure classifier behaves (bound, inconclusive-on-zero);
  2. a correct reverse-mode scalar gradient stays within bound;
  3. the oracle *catches* a jacrev-style forward-per-output implementation —
     the exact B1/B2 defect it exists to detect.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import ops
from tessera.autodiff import grad, count_primitive_executions
from tessera.compiler.evaluator import (
    baur_strassen_ratio,
    grad_cost_ratio,
    CostRatioVerdict,
)


# ── 1. Pure classifier ───────────────────────────────────────────────────────

def test_classifier_within_bound():
    v = baur_strassen_ratio(10.0, 25.0, bound=5.0)
    assert isinstance(v, CostRatioVerdict)
    assert v.ratio == pytest.approx(2.5)
    assert v.within_bound and not v.inconclusive


def test_classifier_exceeds_bound():
    v = baur_strassen_ratio(10.0, 90.0, bound=5.0)
    assert v.ratio == pytest.approx(9.0)
    assert not v.within_bound and not v.inconclusive


def test_classifier_inconclusive_on_zero_forward():
    v = baur_strassen_ratio(0.0, 5.0)
    assert v.inconclusive and v.ratio is None and not v.within_bound


# ── 2. Execution counter is live ─────────────────────────────────────────────

def test_execution_counter_counts_ops():
    x = np.array([1.0, 2.0, 3.0])
    with count_primitive_executions() as box:
        y = ops.mul(x, x)
        ops.add(y, x)
    assert box[0] == 2  # exactly two primitive executions


# ── 3. Correct scalar gradient stays within bound ───────────────────────────

def test_correct_grad_is_within_baur_strassen_bound():
    # f(x) = sum((x*x + x)^2)-ish chain; scalar output, reverse mode.
    def f(x):
        y = ops.mul(x, x)
        z = ops.add(y, x)
        return ops.reduce(ops.mul(z, z))

    x = np.array([0.5, -1.0, 2.0, 3.0])
    v = grad_cost_ratio(f, x, bound=5.0)
    assert not v.inconclusive
    assert v.within_bound, v.detail


# ── 4. The oracle catches the B1/B2 defect ──────────────────────────────────

def test_oracle_flags_forward_per_output_jacobian():
    """A Jacobian built by re-running the forward pass once per output element
    (the B1/B2 defect) inflates the primitive-execution ratio to ~M. The oracle
    must flag it — this is the detection capability R1 adds."""
    def f(x):
        # vector output of dimension M; forward has a fixed op count.
        y = ops.mul(x, x)
        return ops.add(y, x)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # M = 5

    # Defective Jacobian: one full forward pass per output element.
    def bad_jacrev(xv):
        rows = []
        M = np.asarray(f(xv)).shape[0]
        for _ in range(M):
            f(xv)  # re-runs the ENTIRE forward pass — the defect
            rows.append(np.zeros_like(xv))
        return np.stack(rows)

    v = grad_cost_ratio(f, x, grad_fn=bad_jacrev, bound=5.0)
    assert not v.inconclusive
    # forward = 2 ops; bad jacrev runs forward M=5 times = 10 ops → ratio 5×...
    # plus the initial shape probe, so strictly above bound.
    assert v.ratio is not None and v.ratio >= 5.0
    assert not v.within_bound, v.detail


def test_real_jacrev_is_within_bound_regression_guard():
    """The in-tree jacrev was fixed in W0.4 to record one forward pass and
    reuse the tape (no forward re-execution per output element). This oracle is
    the regression guard that the fix stays: the ratio must remain ≈1, well
    within bound. If someone reintroduces per-output forward recompute, this
    fails."""
    from tessera.autodiff import jacrev

    def f(x):
        y = ops.mul(x, x)
        return ops.add(y, x)  # vector output, M = 6

    x = np.arange(1.0, 7.0)
    v = grad_cost_ratio(f, x, grad_fn=jacrev(f), bound=5.0)
    assert not v.inconclusive
    assert v.within_bound, v.detail
    assert v.ratio is not None and v.ratio <= 2.0  # ≈1 forward pass, not M
