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
