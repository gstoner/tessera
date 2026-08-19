"""Differential algebras — the codomain of the derivative functor (AD-LAW-2).

Groundwork for AD-WEIL-1 (`docs/audit/compiler/AUTODIFF_NEXTGEN_PLAN.md`
§2.2). Every AD mode is the same functor evaluated in a different codomain
algebra ``W = ℝ ⊕ m``; this module supplies the instances so Laws 2 and 4 can
be *executed* rather than argued. It changes no production rule — the
registries are untouched — and exists so the later migration has a proven
substrate to land on.

Two instances today:

``Dual()``
    ``ℝ[ε]/(ε²)`` — today's forward mode, semantics-identical.

``TruncatedJet(k)``
    the Weil algebra ``ℝ[ε]/(ε^{k+1})``. Order-k derivatives at dimension
    ``k+1``, versus the ``2ᵏ`` of k-times-nested forward mode. Nilpotency
    annihilates the Taylor remainder *identically*, so evaluation is exact
    rather than approximate, and the two are related by the diagonal
    embedding ``ε ↦ ε₁+…+ε_k`` (Law 4).

Coefficients follow the **Taylor convention**: ``a[j]`` is
``f⁽ʲ⁾(x)/j!``. That is a semantic key, not a default — the same buffer means
different numbers under the derivative convention, so it is stated here and
asserted in the tests rather than left implicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

__all__ = [
    "DifferentialAlgebra",
    "Dual",
    "TruncatedJet",
    "SCALAR_RECURRENCES",
    "nested_dual_derivative",
]

COEFFICIENT_SCALING = "taylor"   # semantic key (#21a): a[j] = f⁽ʲ⁾/j!


class DifferentialAlgebra(Protocol):
    """The interface every AD mode instantiates."""

    def lift(self, primal: Any, seed: Any) -> Any: ...
    def add(self, a: Any, b: Any) -> Any: ...
    def mul(self, a: Any, b: Any) -> Any: ...
    def scalar_fn(self, name: str, a: Any) -> Any: ...
    def extract(self, a: Any, index: int) -> Any: ...


@dataclass(frozen=True)
class Dual:
    """``ℝ[ε]/(ε²)`` — the first-order instance."""

    order: int = 1

    def lift(self, primal, seed):
        return (np.asarray(primal, dtype=np.float64),
                np.asarray(seed, dtype=np.float64))

    def add(self, a, b):
        return (a[0] + b[0], a[1] + b[1])

    def mul(self, a, b):
        return (a[0] * b[0], a[0] * b[1] + a[1] * b[0])

    def scalar_fn(self, name, a):
        f, df = SCALAR_RECURRENCES[name].pointwise
        return (f(a[0]), df(a[0]) * a[1])

    def extract(self, a, index):
        return a[index]


@dataclass(frozen=True)
class TruncatedJet:
    """``ℝ[ε]/(ε^{k+1})`` — order-k Taylor coefficients, dimension k+1."""

    order: int

    def lift(self, primal, seed):
        c = [np.zeros_like(np.asarray(primal, dtype=np.float64))
             for _ in range(self.order + 1)]
        c[0] = np.asarray(primal, dtype=np.float64)
        if self.order >= 1:
            c[1] = np.asarray(seed, dtype=np.float64)
        return c

    def add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def mul(self, a, b):
        """Truncated Cauchy product — the load-bearing method."""
        k = self.order
        out = [np.zeros_like(np.asarray(a[0])) for _ in range(k + 1)]
        for i in range(k + 1):
            for j in range(k + 1 - i):
                out[i + j] = out[i + j] + a[i] * b[j]
        return out

    def scalar_fn(self, name, a):
        return SCALAR_RECURRENCES[name].jet(self, a)

    def extract(self, a, index):
        return a[index]


# ── Scalar recurrences: one datum per primitive, all orders ─────────────────
# The seed of AD-WEIL-1's holonomic ODE table. Each entry supplies the
# first-order pair (used by `Dual`) AND the order-k coefficient recurrence
# (used by `TruncatedJet`), so both modes derive from the same declaration
# rather than from two hand-written rules.


@dataclass(frozen=True)
class ScalarRecurrence:
    pointwise: tuple[Callable[[Any], Any], Callable[[Any], Any]]
    jet: Callable[["TruncatedJet", Sequence[Any]], list]


def _jet_exp(W: "TruncatedJet", u):
    k = W.order
    w = [np.exp(u[0])] + [np.zeros_like(np.asarray(u[0])) for _ in range(k)]
    for n in range(1, k + 1):
        acc = np.zeros_like(np.asarray(u[0]))
        for j in range(1, n + 1):
            acc = acc + j * u[j] * w[n - j]
        w[n] = acc / n
    return w


def _jet_log(W: "TruncatedJet", u):
    # w = log(u)  ⇒  u·w′ = u′  ⇒  n·w_n·u_0 = n·u_n − Σ_{j=1}^{n-1} j·w_j·u_{n-j}
    k = W.order
    w = [np.log(u[0])] + [np.zeros_like(np.asarray(u[0])) for _ in range(k)]
    for n in range(1, k + 1):
        acc = n * u[n]
        for j in range(1, n):
            acc = acc - j * w[j] * u[n - j]
        w[n] = acc / (n * u[0])
    return w


def _jet_tanh(W: "TruncatedJet", u):
    # w′ = (1 − w²)·u′
    k = W.order
    w = [np.tanh(u[0])] + [np.zeros_like(np.asarray(u[0])) for _ in range(k)]
    for n in range(1, k + 1):
        sq = [np.zeros_like(np.asarray(u[0])) for _ in range(n)]
        for i in range(n):
            for j in range(n - i):
                sq[i + j] = sq[i + j] + w[i] * w[j]
        acc = np.zeros_like(np.asarray(u[0]))
        for j in range(1, n + 1):
            term = (1.0 if n - j == 0 else 0.0) - sq[n - j]
            acc = acc + j * u[j] * term
        w[n] = acc / n
    return w


def _jet_sin(W: "TruncatedJet", u):
    return _jet_sin_cos(W, u)[0]


def _jet_cos(W: "TruncatedJet", u):
    return _jet_sin_cos(W, u)[1]


def _jet_sin_cos(W: "TruncatedJet", u):
    k = W.order
    z = np.zeros_like(np.asarray(u[0]))
    s = [np.sin(u[0])] + [z.copy() for _ in range(k)]
    c = [np.cos(u[0])] + [z.copy() for _ in range(k)]
    for n in range(1, k + 1):
        acc_s = z.copy()
        acc_c = z.copy()
        for j in range(1, n + 1):
            acc_s = acc_s + j * u[j] * c[n - j]
            acc_c = acc_c + j * u[j] * s[n - j]
        s[n] = acc_s / n
        c[n] = -acc_c / n
    return s, c


SCALAR_RECURRENCES: dict[str, ScalarRecurrence] = {
    "exp": ScalarRecurrence((np.exp, np.exp), _jet_exp),
    "log": ScalarRecurrence((np.log, lambda x: 1.0 / x), _jet_log),
    "tanh": ScalarRecurrence((np.tanh, lambda x: 1.0 - np.tanh(x) ** 2), _jet_tanh),
    "sin": ScalarRecurrence((np.sin, np.cos), _jet_sin),
    "cos": ScalarRecurrence((np.cos, lambda x: -np.sin(x)), _jet_cos),
}


# ── The nested-dual reference (Law 4's other side) ───────────────────────────


class _NestedDual:
    """Recursive dual numbers: the 2ᵏ-dimensional tensor-product algebra."""

    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __add__(self, o):
        o = o if isinstance(o, _NestedDual) else _NestedDual(o, 0.0)
        return _NestedDual(_nd_add(self.a, o.a), _nd_add(self.b, o.b))

    __radd__ = __add__

    def __mul__(self, o):
        o = o if isinstance(o, _NestedDual) else _NestedDual(o, 0.0)
        return _NestedDual(_nd_mul(self.a, o.a),
                           _nd_add(_nd_mul(self.a, o.b), _nd_mul(self.b, o.a)))

    __rmul__ = __mul__


def _nd_add(x, y):
    if isinstance(x, _NestedDual) or isinstance(y, _NestedDual):
        x = x if isinstance(x, _NestedDual) else _NestedDual(x, 0.0)
        return x + y
    return x + y


def _nd_mul(x, y):
    if isinstance(x, _NestedDual) or isinstance(y, _NestedDual):
        x = x if isinstance(x, _NestedDual) else _NestedDual(x, 0.0)
        return x * y
    return x * y


def _nd_apply(fn, dfn, x):
    """Lift a scalar function into the nested-dual tower."""
    if isinstance(x, _NestedDual):
        return _NestedDual(_nd_apply(fn, dfn, x.a),
                           _nd_mul(x.b, _nd_apply(dfn, dfn, x.a)))
    return fn(x)


def nested_dual_derivative(program: Callable, x0: float, order: int) -> float:
    """``d^order/dt^order program(x0 + t)`` via `order`-times-nested duals.

    This is the ``2ᵏ``-dimensional path — the thing jet mode replaces. Law 4
    asserts the two agree, which is what licenses retiring the nested route
    (Decision #31: prove the survivor carries what the deleted path carried).
    """
    t: Any = 0.0
    for _ in range(order):
        t = _NestedDual(t, 1.0)
    out = program(_nd_add(x0, t), _NestedScalarOps())
    for _ in range(order):
        out = out.b if isinstance(out, _NestedDual) else 0.0
    while isinstance(out, _NestedDual):
        out = out.a
    return float(out)


class _NestedScalarOps:
    """Scalar functions lifted to the nested tower, keyed like SCALAR_RECURRENCES."""

    def __call__(self, name: str, x):
        f, df = SCALAR_RECURRENCES[name].pointwise
        return self._apply(name, x)

    def _apply(self, name, x):
        f, df = SCALAR_RECURRENCES[name].pointwise
        if isinstance(x, _NestedDual):
            inner = self._apply(name, x.a)
            deriv = self._derivative(name, x.a)
            return _NestedDual(inner, _nd_mul(x.b, deriv))
        return f(x)

    def _derivative(self, name, x):
        """d/dx of `name`, itself lifted (needed for higher orders)."""
        if name == "exp":
            return self._apply("exp", x)
        if name == "log":
            return _nd_reciprocal(x)
        if name == "tanh":
            t = self._apply("tanh", x)
            return _nd_add(1.0, _nd_mul(-1.0, _nd_mul(t, t)))
        if name == "sin":
            return self._apply("cos", x)
        if name == "cos":
            return _nd_mul(-1.0, self._apply("sin", x))
        raise KeyError(name)


def _nd_reciprocal(x):
    if isinstance(x, _NestedDual):
        inv = _nd_reciprocal(x.a)
        return _NestedDual(inv, _nd_mul(-1.0, _nd_mul(x.b, _nd_mul(inv, inv))))
    return 1.0 / x
