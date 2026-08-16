"""Oracles for ``tridiagonal_solve`` (P2 tranche / PDE plan §III.1).

Cross-path against the dense solver, adjoint identity through the registered
VJP with finite differences on every operand, fail-closed singular/shape
contracts, and the Decision #15a storage-dtype discipline.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from tessera.solvers_ops import tridiagonal_solve  # noqa: E402
from tessera.solvers_ops import _tridiagonal_solve_vjp  # noqa: E402

N = 9


@pytest.fixture()
def system():
    rng = np.random.default_rng(5)
    dl = rng.standard_normal(N) * 0.3
    d = rng.standard_normal(N) + 4.0        # diagonally dominant
    du = rng.standard_normal(N) * 0.3
    b = rng.standard_normal(N)
    return dl, d, du, b


def _dense(dl, d, du):
    a = np.diag(np.asarray(d, dtype=np.float64))
    a += np.diag(np.asarray(dl, dtype=np.float64)[1:], -1)
    a += np.diag(np.asarray(du, dtype=np.float64)[:-1], +1)
    return a


def test_matches_dense_solve(system):
    dl, d, du, b = system
    x = tridiagonal_solve(dl, d, du, b)
    ref = np.linalg.solve(_dense(dl, d, du), b)
    assert np.allclose(x, ref, atol=1e-12)


def test_batched_rhs(system):
    dl, d, du, _ = system
    rng = np.random.default_rng(6)
    b = rng.standard_normal((4, N))
    x = tridiagonal_solve(dl, d, du, b)
    a = _dense(dl, d, du)
    assert np.allclose(x, np.linalg.solve(a, b.T).T, atol=1e-12)


def test_vjp_matches_finite_differences(system):
    dl, d, du, b = system
    rng = np.random.default_rng(7)
    dout = rng.standard_normal(N)
    gdl, gd, gdu, gb = _tridiagonal_solve_vjp(dout, dl, d, du, b)
    eps = 1e-6
    operands = [dl.copy(), d.copy(), du.copy(), b.copy()]
    grads = [gdl, gd, gdu, gb]
    for oi in range(4):
        for k in (0, N // 2, N - 1):
            for sgn, store in ((+1, {}), (-1, {})):
                ops = [o.copy() for o in operands]
                ops[oi][k] += sgn * eps
                store["v"] = tridiagonal_solve(*ops) @ dout
                if sgn > 0:
                    plus = store["v"]
                else:
                    minus = store["v"]
            fd = (plus - minus) / (2 * eps)
            assert np.isclose(grads[oi][k], fd, atol=1e-4), (
                f"operand {oi} index {k}: vjp {grads[oi][k]} vs fd {fd}")


def test_singular_pivot_fails_closed():
    # d[0] = 0 makes the first Thomas pivot exactly zero.
    with pytest.raises(ValueError, match="singular"):
        tridiagonal_solve(np.zeros(3), np.array([0.0, 1.0, 1.0]),
                          np.zeros(3), np.ones(3))


def test_shape_contracts_fail_closed(system):
    dl, d, du, b = system
    with pytest.raises(ValueError, match="share length"):
        tridiagonal_solve(dl[:-1], d, du, b)
    with pytest.raises(ValueError, match="rhs trailing extent"):
        tridiagonal_solve(dl, d, du, b[:-1])


def test_storage_dtype_preserved(system):
    dl, d, du, b = system
    assert tridiagonal_solve(dl, d, du, b.astype(np.float32)).dtype == np.float32
    assert tridiagonal_solve(dl, d, du, b).dtype == np.float64
