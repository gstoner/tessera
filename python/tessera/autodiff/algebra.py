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

**Measured conditioning envelope (AD-WEIL-1).** The plan's §3.8 warned that
monomial jets go conditioning-limited "past ~order 10–15". Measured, that is
**overstated for the coefficient recurrences**: relative error of the order-k
coefficient against the exact derivative stays at ~1e-16 through **k = 30**
in float64 (and at float32's own 1.2e-8 in float32) for exp/sin/log, with no
per-order decay. The reason is the Taylor convention itself — the recurrences
work on the *scaled* coefficients and never form a factorial.

The two real limits, both measured and both elsewhere:

1. **Recovering an unscaled derivative** ``f⁽ᵏ⁾ = k!·w_k`` overflows float64
   at **k ≈ 175**, not because the jet is inaccurate but because ``k!`` is.
   Consume ``w_k`` directly wherever possible.
2. **A small radius of convergence** makes the coefficients themselves grow
   like ``1/R^k`` — ``reciprocal`` at ``x = 0.15`` reaches ``|w₂₀| ≈ 2e17``.
   The relative accuracy stays at 1e-16, so this is dynamic range, not error;
   it is what ``ChebJet`` would address, and it bites long before any
   recurrence conditioning does.

Recorded here because §3.8 gates IR investment on this measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

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


class _Ops(Protocol):
    """The arithmetic a derivative expression is written against."""

    def apply(self, name: str, x: Any) -> Any: ...
    def mul(self, a: Any, b: Any) -> Any: ...
    def add(self, a: Any, b: Any) -> Any: ...
    def neg(self, a: Any) -> Any: ...
    def reciprocal(self, a: Any) -> Any: ...


@dataclass(frozen=True)
class ScalarRecurrence:
    """One primitive, declared once.

    ``value``            the function itself.
    ``derivative_expr``  its derivative, expressed in terms of *registered*
                         functions and ring operations — evaluated with plain
                         numpy for first-order use, and with tower arithmetic
                         for the nested-dual reference. Declaring it once is
                         what keeps the two from drifting; an earlier version
                         restated it as a hardcoded if-chain in
                         ``_NestedScalarOps`` and would have silently gone
                         stale (and KeyError'd) as this table grew.
    ``jet``              the order-k coefficient recurrence.
    """

    value: Callable[[Any], Any]
    derivative_expr: Callable[[_Ops, Any], Any]
    jet: Callable[["TruncatedJet", Sequence[Any]], list]
    guard_expr: Optional[Callable[[_Ops, Any], Any]] = None
    """Maps an arbitrary argument into this primitive's domain.

    Declared per primitive so every consumer applies the SAME guard: the law
    programs evaluate identical text in the jet algebra and the nested-dual
    tower, and a guard that differed between them would silently compare two
    different functions. `None` means the domain is all of ℝ."""

    @property
    def pointwise(self) -> tuple[Callable[[Any], Any], Callable[[Any], Any]]:
        """(f, f') with the derivative DERIVED from `derivative_expr`."""
        return (self.value, lambda x: self.derivative_expr(_SCALAR_OPS, x))


class _ScalarOps:
    """`_Ops` over plain numpy scalars/arrays."""

    def apply(self, name, x):
        return SCALAR_RECURRENCES[name].value(x)

    def mul(self, a, b):
        return a * b

    def add(self, a, b):
        return a + b

    def neg(self, a):
        return -a

    def reciprocal(self, a):
        return 1.0 / a


_SCALAR_OPS = _ScalarOps()


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
    # d/dx exp = exp
    "exp": ScalarRecurrence(np.exp, lambda o, x: o.apply("exp", x), _jet_exp),
    # d/dx log = 1/x
    "log": ScalarRecurrence(np.log, lambda o, x: o.reciprocal(x), _jet_log,
                            guard_expr=lambda o, x: o.add(o.mul(x, x), 1.0)),
    # d/dx tanh = 1 - tanh^2
    "tanh": ScalarRecurrence(
        np.tanh,
        lambda o, x: o.add(1.0, o.neg(o.mul(o.apply("tanh", x),
                                            o.apply("tanh", x)))),
        _jet_tanh),
    # d/dx sin = cos
    "sin": ScalarRecurrence(np.sin, lambda o, x: o.apply("cos", x), _jet_sin),
    # d/dx cos = -sin
    "cos": ScalarRecurrence(np.cos,
                            lambda o, x: o.neg(o.apply("sin", x)), _jet_cos),
}


# ── ODE table expansion (AD-WEIL-1) ─────────────────────────────────────────
# Each entry below is ONE datum from which first order, order-k, and both AD
# modes follow. The recurrences come from the primitive's defining ODE, per
# Griewank & Walther Ch. 13; every one is checked at k=1 against the
# registered JVP and at k>1 against the nested-dual tower.


def _cauchy(a, b, n):
    """Coefficient n of the product of two coefficient lists."""
    acc = None
    for i in range(n + 1):
        if i < len(a) and (n - i) < len(b):
            term = a[i] * b[n - i]
            acc = term if acc is None else acc + term
    return acc


def _jet_from_ode(deriv_coeffs):
    """Build a jet rule from `w' = g(u, w) * u'` given g's coefficients.

    `deriv_coeffs(w, u, n)` returns coefficient n of g. The shared body is
    the chain rule in coefficient space: n*w_n = sum_j j*u_j*g_{n-j}.
    """

    def rule(W, u, _init, _g=deriv_coeffs):
        k = W.order
        w = [_init(u[0])] + [np.zeros_like(np.asarray(u[0])) for _ in range(k)]
        for n in range(1, k + 1):
            acc = np.zeros_like(np.asarray(u[0]))
            for j in range(1, n + 1):
                acc = acc + j * u[j] * _g(w, u, n - j)
            w[n] = acc / n
        return w

    return rule


def _jet_sqrt(W, u):
    # w = sqrt(u): w^2 = u  =>  2*w_0*w_n = u_n - sum_{j=1}^{n-1} w_j*w_{n-j}
    k = W.order
    w = [np.sqrt(u[0])] + [np.zeros_like(np.asarray(u[0])) for _ in range(k)]
    for n in range(1, k + 1):
        acc = u[n]
        for j in range(1, n):
            acc = acc - w[j] * w[n - j]
        w[n] = acc / (2.0 * w[0])
    return w


def _jet_reciprocal(W, u):
    # w = 1/u: u*w = 1  =>  u_0*w_n = -sum_{j=1}^{n} u_j*w_{n-j}
    k = W.order
    w = [1.0 / u[0]] + [np.zeros_like(np.asarray(u[0])) for _ in range(k)]
    for n in range(1, k + 1):
        acc = np.zeros_like(np.asarray(u[0]))
        for j in range(1, n + 1):
            acc = acc - u[j] * w[n - j]
        w[n] = acc / u[0]
    return w


def _jet_sinh_cosh(W, u):
    k = W.order
    z = np.zeros_like(np.asarray(u[0]))
    s = [np.sinh(u[0])] + [z.copy() for _ in range(k)]
    c = [np.cosh(u[0])] + [z.copy() for _ in range(k)]
    for n in range(1, k + 1):
        acc_s = z.copy()
        acc_c = z.copy()
        for j in range(1, n + 1):
            acc_s = acc_s + j * u[j] * c[n - j]
            acc_c = acc_c + j * u[j] * s[n - j]
        s[n] = acc_s / n
        c[n] = acc_c / n
    return s, c


def _jet_sinh(W, u):
    return _jet_sinh_cosh(W, u)[0]


def _jet_cosh(W, u):
    return _jet_sinh_cosh(W, u)[1]


def _jet_atan(W, u):
    # w' = u'/(1+u^2): (1+u^2)*w' = u'
    k = W.order
    denom = [np.zeros_like(np.asarray(u[0])) for _ in range(k + 1)]
    for n in range(k + 1):
        denom[n] = _cauchy(u, u, n)
    denom[0] = denom[0] + 1.0
    w = [np.arctan(u[0])] + [np.zeros_like(np.asarray(u[0])) for _ in range(k)]
    for n in range(1, k + 1):
        acc = n * u[n]
        for j in range(1, n):
            acc = acc - j * w[j] * denom[n - j]
        w[n] = acc / (n * denom[0])
    return w


def _jet_expm1(W, u):
    w = _jet_exp(W, u)
    out = list(w)
    out[0] = np.expm1(u[0])
    return out


def _jet_log1p(W, u):
    shifted = list(u)
    shifted[0] = u[0] + 1.0
    w = _jet_log(W, shifted)
    out = list(w)
    out[0] = np.log1p(u[0])
    return out


def _jet_sigmoid(W, u):
    # s' = s*(1-s)*u'
    k = W.order
    s0 = 1.0 / (1.0 + np.exp(-u[0]))
    w = [s0] + [np.zeros_like(np.asarray(u[0])) for _ in range(k)]
    for n in range(1, k + 1):
        sq = [_cauchy(w, w, m) for m in range(n)]
        acc = np.zeros_like(np.asarray(u[0]))
        for j in range(1, n + 1):
            acc = acc + j * u[j] * (w[n - j] - sq[n - j])
        w[n] = acc / n
    return w


_EXTRA_RECURRENCES = {
    "sqrt": ScalarRecurrence(
        np.sqrt,
        lambda o, x: o.mul(0.5, o.reciprocal(o.apply("sqrt", x))),
        _jet_sqrt,
        guard_expr=lambda o, x: o.add(o.mul(x, x), 1.0)),
    "reciprocal": ScalarRecurrence(
        lambda x: 1.0 / x,
        lambda o, x: o.neg(o.mul(o.reciprocal(x), o.reciprocal(x))),
        _jet_reciprocal,
        guard_expr=lambda o, x: o.add(o.mul(x, x), 1.0)),
    "sinh": ScalarRecurrence(
        np.sinh, lambda o, x: o.apply("cosh", x), _jet_sinh),
    "cosh": ScalarRecurrence(
        np.cosh, lambda o, x: o.apply("sinh", x), _jet_cosh),
    "atan": ScalarRecurrence(
        np.arctan,
        lambda o, x: o.reciprocal(o.add(1.0, o.mul(x, x))),
        _jet_atan),
    "expm1": ScalarRecurrence(
        np.expm1, lambda o, x: o.apply("exp", x), _jet_expm1),
    "log1p": ScalarRecurrence(
        np.log1p, lambda o, x: o.reciprocal(o.add(1.0, x)), _jet_log1p,
        guard_expr=lambda o, x: o.mul(x, x)),
    "sigmoid": ScalarRecurrence(
        lambda x: 1.0 / (1.0 + np.exp(-x)),
        lambda o, x: o.mul(o.apply("sigmoid", x),
                           o.add(1.0, o.neg(o.apply("sigmoid", x)))),
        _jet_sigmoid),
}
SCALAR_RECURRENCES.update(_EXTRA_RECURRENCES)


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
    """`_Ops` over the nested-dual tower — Law 4's reference arithmetic.

    Its `_derivative` evaluates the primitive's declared `derivative_expr`
    with tower arithmetic, so the derivative datum lives in exactly one place
    (the registry) and adding a primitive extends the reference automatically
    instead of raising.
    """

    def apply(self, name, x):
        return self._apply(name, x)

    def mul(self, a, b):
        return _nd_mul(a, b)

    def add(self, a, b):
        return _nd_add(a, b)

    def neg(self, a):
        return _nd_mul(-1.0, a)

    def reciprocal(self, a):
        return _nd_reciprocal(a)

    def _apply(self, name, x):
        if isinstance(x, _NestedDual):
            return _NestedDual(self._apply(name, x.a),
                               _nd_mul(x.b, self._derivative(name, x.a)))
        return SCALAR_RECURRENCES[name].value(x)

    def _derivative(self, name, x):
        """d/dx of `name`, lifted — from the registry's one declaration."""
        return SCALAR_RECURRENCES[name].derivative_expr(self, x)


def _nd_reciprocal(x):
    if isinstance(x, _NestedDual):
        inv = _nd_reciprocal(x.a)
        return _NestedDual(inv, _nd_mul(-1.0, _nd_mul(x.b, _nd_mul(inv, inv))))
    return 1.0 / x


# ─────────────────────────────────────────────────────────────────────────────
# The generic finite-multiplication-table substrate (AD-WEIL-1 / W6.3).
#
# INTEGRATED_COMPILER_PLAN W6.3 asks whether one substrate can carry both
# arbitrary commutative nilpotent Weil algebras AND the Clifford algebras
# `ga/signature.py` implements, noting that the latter is signature-specific
# (blade XOR, metric signs, anti-commutation) and "cannot represent arbitrary
# commutative nilpotent Weil algebras". W6.4's note says to treat the reuse as
# "a design hypothesis to prove, not a sequencing-based cost reduction".
#
# This is the proof. Both families are *monomial* algebras: the product of two
# basis elements is a single basis element times a scalar (possibly zero). So
# one structure-constant table of shape `table[i][j] = (k, coeff)` carries
# both, and `FiniteAlgebra` below is the shared implementation. The Clifford
# instantiation is cross-checked against `ga`'s own product table as oracle in
# `test_autodiff_laws.py`.


@dataclass(frozen=True)
class FiniteAlgebra:
    """A unital algebra given by structure constants over a finite basis.

    ``table[i][j] == (k, c)`` means ``e_i · e_j = c · e_k``; ``c == 0`` means
    the product vanishes (a nilpotent truncation, or a null generator in a
    degenerate Clifford signature). Elements are coefficient lists over the
    basis.
    """

    dim: int
    table: tuple[tuple[tuple[int, float], ...], ...]
    name: str = "finite"

    def zero(self, like=None):
        proto = np.asarray(0.0 if like is None else like, dtype=np.float64)
        return [np.zeros_like(proto) for _ in range(self.dim)]

    def add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def mul(self, a, b):
        out = self.zero(a[0])
        for i in range(self.dim):
            ai = a[i]
            if isinstance(ai, np.ndarray):
                if not ai.any():
                    continue
            elif ai == 0:
                continue
            row = self.table[i]
            for j in range(self.dim):
                k, c = row[j]
                if c:
                    out[k] = out[k] + c * ai * b[j]
        return out


def weil_table(order: int) -> FiniteAlgebra:
    """``ℝ[ε]/(ε^{k+1})`` as structure constants: ``ε^i·ε^j = ε^{i+j}``,
    zero once the exponent passes the truncation."""
    dim = order + 1
    table = tuple(
        tuple(((i + j, 1.0) if i + j <= order else (0, 0.0))
              for j in range(dim))
        for i in range(dim)
    )
    return FiniteAlgebra(dim, table, name=f"Weil(k={order})")


def clifford_table(p: int, q: int, r: int = 0) -> FiniteAlgebra:
    """``Cl(p,q,r)`` as structure constants, sourced from the SAME blade
    product `ga/signature.py` uses — so this is a re-expression of that
    algebra in the shared substrate, not a second implementation of it."""
    from ..ga.signature import _blade_product

    dim = 1 << (p + q + r)
    rows: list[tuple[tuple[int, float], ...]] = []
    for i in range(dim):
        row: list[tuple[int, float]] = []
        for j in range(dim):
            mask, sign = _blade_product(i, j, p, q, r)
            row.append((mask, float(sign)))
        rows.append(tuple(row))
    return FiniteAlgebra(dim, tuple(rows), name=f"Cl({p},{q},{r})")


# ─────────────────────────────────────────────────────────────────────────────
# Law 6 substrate: certified enclosures and randomized operator estimators.
#
# Both change the TYPE of the correctness claim, which is why they need their
# own law (plan §4 row 6). A `TaylorModel` says "the true value lies in this
# interval"; a randomized jet says "this is an unbiased estimator of the
# operator". Neither is an equality, so Laws 1-5 cannot express them.


@dataclass(frozen=True)
class TaylorModel:
    """Order-k jets carried as intervals, with outward-rounded arithmetic.

    Every coefficient is a ``(lo, hi)`` pair that provably brackets the exact
    value. Arithmetic rounds outward — the interval only ever grows — so a
    result interval is a *certificate*: if the enclosure is [a,b] then the
    exact coefficient is in [a,b], full stop. That is strictly stronger than
    an error estimate, and it is what an accuracy-budgeted arbiter
    (Decision #28) can consume as evidence rather than as a hope.

    Outward rounding here uses `np.nextafter`, which is exact for float64 and
    needs no rounding-mode control.
    """

    order: int

    def lift(self, primal, seed):
        lo = [np.float64(0.0)] * (self.order + 1)
        hi = [np.float64(0.0)] * (self.order + 1)
        lo[0] = hi[0] = np.float64(primal)
        if self.order >= 1:
            lo[1] = hi[1] = np.float64(seed)
        return (lo, hi)

    def _widen(self, lo, hi):
        return (np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf))

    def add(self, a, b):
        lo, hi = [], []
        for i in range(self.order + 1):
            l, h = self._widen(a[0][i] + b[0][i], a[1][i] + b[1][i])
            lo.append(l)
            hi.append(h)
        return (lo, hi)

    def mul(self, a, b):
        lo = [np.float64(0.0)] * (self.order + 1)
        hi = [np.float64(0.0)] * (self.order + 1)
        for i in range(self.order + 1):
            for j in range(self.order + 1 - i):
                # Interval product: all four corners, then the hull.
                corners = (a[0][i] * b[0][j], a[0][i] * b[1][j],
                           a[1][i] * b[0][j], a[1][i] * b[1][j])
                l, h = self._widen(min(corners), max(corners))
                lo[i + j] = lo[i + j] + l
                hi[i + j] = hi[i + j] + h
        return ([np.nextafter(v, -np.inf) for v in lo],
                [np.nextafter(v, np.inf) for v in hi])

    def contains(self, model, exact, index: int) -> bool:
        return bool(model[0][index] <= exact <= model[1][index])


def hutchinson_laplacian(fn, x, key, samples: int, *, radius: float = 1.0):
    """Randomized-jet estimator of the Laplacian (the STDE shape, §3.7).

    For ``v`` with i.i.d. zero-mean unit-variance components,
    ``E[vᵀ ∇²f v] = tr(∇²f) = Δf``. The quadratic form is read off a
    **second-order jet** in the direction ``v`` — coefficient 2 of the jet is
    ``½ vᵀ∇²f v`` — so one order-2 jet per sample replaces an ``n``-fold
    loop over coordinates.

    The correctness claim is therefore ``E[estimate] = Δf``, an *unbiased
    estimator*, not an equality. Draws come from the project's Philox stream
    (`tessera.rng`, Decision #18), so a run is deterministic and replayable
    and the claim is testable rather than folklore.
    """
    from ..rng import normal

    n = np.asarray(x).size
    total = np.float64(0.0)
    for s in range(samples):
        v = np.asarray(normal(key.fold_in(s) if hasattr(key, "fold_in") else key,
                              (n,), dtype="fp64"), dtype=np.float64)
        if radius != 1.0:
            v = v * radius
        total = total + 2.0 * _second_order_coefficient(fn, np.asarray(x), v)
    return float(total / samples)


def _second_order_coefficient(fn, x, v):
    """Coefficient 2 of the order-2 jet of ``t -> fn(x + t v)``.

    Equals ``½ vᵀ ∇²f v``; the caller doubles it to recover the quadratic
    form.
    """
    W = TruncatedJet(2)
    lifted = [np.asarray(x, dtype=np.float64),
              np.asarray(v, dtype=np.float64),
              np.zeros_like(np.asarray(x, dtype=np.float64))]
    out = fn(lifted, W)
    return float(np.sum(np.asarray(out[2])))
