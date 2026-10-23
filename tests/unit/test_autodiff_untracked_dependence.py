"""A gradient path through raw numpy must refuse, not read as zero.

The tape records ``tessera.ops.*`` calls only. In
``ops.sum(ops.sin(x) * x)`` numpy's ``*`` returns a plain array no recorded op
produced; ``ops.sum`` records it as an operand, the sweep stops there, and
``grad`` returned ``[0, 0, 0]`` for a function whose gradient is
``cos(x) * x + sin(x)``. When only part of the path is laundered the answer is
nonzero and still wrong.

``tessera.autodiff.untracked`` replays the function with the differentiated
arguments perturbed and refuses only when an untracked operand that the sweep
reached actually moves with them. Pinned here:

* ``grad`` and ``jacrev`` refuse the full and the partial laundering, naming
  the op that consumed the untracked value;
* Decision #10a negative fixtures, each of which must still differentiate:
  an all-``ops`` body, a captured constant, a constant built inside the body,
  a piecewise-constant mask derived from ``x`` (its true contribution is
  zero), an unused argument (honest zero), and unseeded randomness (stable
  under perturbation is undecidable there, so the probe stands down).
"""

import numpy as np
import pytest

from tessera import ops
from tessera.autodiff import grad
from tessera.autodiff.tape import TesseraAutodiffError
from tessera.autodiff.transforms import jacrev

X = np.array([0.3, 0.7, 1.1])
W = np.array([2.0, -1.0, 0.5])


def _expected_sin_x_times_x(x):
    return np.cos(x) * x + np.sin(x)


def test_grad_refuses_full_numpy_laundering():
    with pytest.raises(TesseraAutodiffError, match=r"tessera\.ops\.sum"):
        grad(lambda x: ops.sum(ops.sin(x) * x))(X)


def test_grad_refuses_partial_numpy_laundering():
    with pytest.raises(TesseraAutodiffError, match=r"tessera\.ops\.mul"):
        grad(lambda x: ops.sum(ops.mul(ops.sin(x) * x, x)))(X)


def test_jacrev_refuses_numpy_laundering():
    with pytest.raises(TesseraAutodiffError, match=r"jacrev\(fn\)"):
        jacrev(lambda x: ops.mul(ops.sin(x) * x, W))(X)


def test_all_ops_body_is_unchanged():
    g = grad(lambda x: ops.sum(ops.mul(ops.sin(x), x)))(X)
    np.testing.assert_allclose(g, _expected_sin_x_times_x(X), rtol=1e-12)


@pytest.mark.parametrize(
    "fn, expected",
    [
        (lambda x: ops.sum(ops.mul(x, W)), W),
        (lambda x: ops.sum(ops.mul(x, np.array([1.0, 2.0, 3.0]))), np.array([1.0, 2.0, 3.0])),
        (lambda x: ops.sum(ops.mul(x, np.sign(np.asarray(x)))), np.ones(3)),
        (lambda x: ops.sum(ops.mul(W, W)), np.zeros(3)),
    ],
    ids=["captured_constant", "local_constant", "sign_mask", "unused_argument"],
)
def test_constant_operands_still_differentiate(fn, expected):
    np.testing.assert_allclose(grad(fn)(X), expected, rtol=1e-12)


def test_unseeded_randomness_is_not_refused():
    def fn(x):
        noise = np.random.default_rng().standard_normal(x.shape if hasattr(x, "shape") else 3)
        return ops.sum(ops.mul(x, noise))

    assert grad(fn)(X).shape == X.shape


def test_jacrev_constant_operand_still_differentiates():
    jac = jacrev(lambda x: ops.mul(ops.sin(x), W))(X)
    np.testing.assert_allclose(np.diag(jac), np.cos(X) * W, rtol=1e-12)


# ── Review on #1004: a finite step is not a derivative ─────────────────────


def test_threshold_mask_near_its_boundary_is_not_refused():
    """The mask's derivative is zero at x = 1e-5; a coarse step crossed 1e-4."""
    x = np.array([1e-5])
    g = grad(lambda v: ops.sum(ops.mul(v, np.abs(np.asarray(v)) > 1e-4)))(x)
    np.testing.assert_array_equal(g, [0.0])


def test_mask_every_step_crosses_is_a_jump_not_a_slope():
    """At x == 1 any step flips the mask by the same 1: the change does not
    scale with the step, so it is not derivative evidence."""
    x = np.array([1.0])
    g = grad(lambda v: ops.sum(ops.mul(v, np.abs(np.asarray(v) - 1.0) > 0)))(x)
    np.testing.assert_array_equal(g, [0.0])


def test_jacrev_threshold_mask_is_not_refused():
    x = np.array([1e-5, 2.0])
    jac = jacrev(lambda v: ops.mul(v, np.abs(np.asarray(v)) > 1e-4))(x)
    np.testing.assert_array_equal(np.diag(jac), [0.0, 1.0])


# ── Review on #1004: replays must not multiply caller side effects ─────────


def test_local_constant_costs_at_most_one_replay():
    calls = []

    def fn(v):
        calls.append(1)
        return ops.sum(ops.mul(v, np.ones_like(np.asarray(v))))

    np.testing.assert_array_equal(grad(fn)(X), np.ones(3))
    assert len(calls) <= 2, f"{len(calls)} executions for one grad call"


def test_all_ops_program_never_replays():
    calls = []

    def fn(v):
        calls.append(1)
        return ops.sum(ops.mul(v, W))

    grad(fn)(X)
    assert len(calls) == 1


def test_global_random_streams_are_left_as_one_forward_pass_leaves_them():
    import random

    def fn(v):
        noise = np.random.random(3) + random.random()
        return ops.sum(ops.mul(v, noise))

    np.random.seed(1234)
    random.seed(1234)
    fn(X)
    expected = (np.random.random(), random.random())

    np.random.seed(1234)
    random.seed(1234)
    grad(fn)(X)
    assert (np.random.random(), random.random()) == expected
