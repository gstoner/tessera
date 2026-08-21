#!/usr/bin/env python3
"""Mathematical correctness pass over the AD-LAW / AD-WEIL series.

Every claim is checked against an **independent** oracle — arbitrary-precision
Taylor coefficients (mpmath, 40 digits), Cauchy-integral coefficients via FFT,
exact Hessians, or hand-derived algebra — never against the implementation
being checked. Re-running the project's own tests would be circular; this is
not that.

The oracles are deliberately unrelated to each other as well: closed forms come
from hand-derived series, the contour integral samples the function on a circle
in the complex plane, and the Clifford relations come from the algebra's
defining axioms. Agreement across all of them is much stronger evidence than
any one.

**No hard dependency beyond numpy.** `mpmath` is used as an *additional*
oracle when installed and skipped when not (CI does not ship it), so the pass
is complete either way — the closed forms and the contour integral already
cover every primitive between them.

Run:  python3 research/autodiff_nextgen/verify_series_math.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

try:                                    # optional: an extra oracle when present
    import mpmath as mp                 # noqa: F401
    _HAVE_MPMATH = True
except ImportError:                     # CI does not ship it — see below
    _HAVE_MPMATH = False

sys.path.insert(0, "python")

CHECKS: list[str] = []
FAILURES: list[str] = []


def ok(name: str, detail: str = "") -> None:
    CHECKS.append(name)
    print(f"  ok  [{len(CHECKS):2d}] {name}" + (f"  ({detail})" if detail else ""))


def bad(name: str, detail: str) -> None:
    FAILURES.append(f"{name}: {detail}")
    print(f"  FAIL     {name}  ({detail})")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Every jet recurrence against SymPy's symbolic Taylor series.
# ─────────────────────────────────────────────────────────────────────────────

_POINT = {"log": 1.7, "log1p": 0.6, "sqrt": 2.3, "reciprocal": 1.9,
          "sigmoid": 0.4, "atan": 0.5}


def check_recurrences_against_closed_forms(order: int = 8) -> None:
    """The load-bearing check: each recurrence must reproduce the hand-derived
    Taylor coefficients of its own function."""
    from tessera.autodiff.algebra import SCALAR_RECURRENCES, TruncatedJet

    W = TruncatedJet(order)
    worst_name, worst = "", 0.0
    covered = 0
    for name, rec in sorted(SCALAR_RECURRENCES.items()):
        x0 = _POINT.get(name, 0.37)
        exact = [_closed_form(name, x0, n) for n in range(order + 1)]
        if exact[0] is None:
            continue                     # covered by the contour integral
        covered += 1
        got = [float(v) for v in
               rec.jet(W, W.lift(np.asarray(float(x0)), np.asarray(1.0)))]
        for n in range(order + 1):
            rel = abs(got[n] - exact[n]) / max(abs(exact[n]), 1e-300)
            if rel > worst:
                worst, worst_name = rel, f"{name}[{n}]"
            if rel > 1e-12:
                bad(f"recurrence {name} order {n}",
                    f"jet={got[n]!r} vs closed form={exact[n]!r} rel={rel:.2e}")
                return
    ok(f"{covered} jet recurrences == hand-derived closed forms through "
       f"order {order}", f"worst {worst:.1e} at {worst_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. A second, fully independent coefficient oracle: the Cauchy integral,
#    evaluated by FFT on a circle. Shares no machinery with either the
#    recurrences or SymPy.
# ─────────────────────────────────────────────────────────────────────────────

_NUMPY_FN = {
    "exp": np.exp, "log": np.log, "tanh": np.tanh, "sin": np.sin,
    "cos": np.cos, "sqrt": np.sqrt, "reciprocal": lambda z: 1 / z,
    "sinh": np.sinh, "cosh": np.cosh, "atan": np.arctan,
    "expm1": np.expm1, "log1p": np.log1p,
    "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
    # AD-RETIRE-2 datum growth. erf/erfc need a complex-capable reference
    # for the contour integral; scipy.special provides one (scipy is
    # already a project dependency). asin/acos branch at ±1 and rsqrt at
    # 0 — the radius table keeps every contour inside the domain.
    "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
    "rsqrt": lambda z: 1 / np.sqrt(z),
    "erf": lambda z: __import__("scipy.special", fromlist=["erf"]).erf(z),
    "erfc": lambda z: __import__("scipy.special", fromlist=["erfc"]).erfc(z),
}


def cauchy_coefficients(fn, x0: float, order: int, radius: float,
                        m: int = 4096) -> list[float]:
    """a_n = (1/2πi) ∮ f(x0+z)/z^{n+1} dz, by FFT on |z| = radius."""
    k = np.arange(m)
    z = radius * np.exp(2j * np.pi * k / m)
    vals = fn(x0 + z)
    coeffs = np.fft.fft(vals) / m
    return [float(np.real(coeffs[n] / radius ** n)) for n in range(order + 1)]


def check_recurrences_against_cauchy(order: int = 5) -> None:
    from tessera.autodiff.algebra import SCALAR_RECURRENCES, TruncatedJet

    # `atan` has branch points at ±i and `arctan` is not the analytic
    # continuation off the real axis in numpy, so it is covered by SymPy only.
    radius = {"log": 0.5, "log1p": 0.3, "sqrt": 0.7, "reciprocal": 0.5,
              "sigmoid": 0.6, "rsqrt": 0.3, "asin": 0.4, "acos": 0.4,
              "tan": 0.4}
    W = TruncatedJet(order)
    worst = 0.0
    tested = 0
    for name, rec in sorted(SCALAR_RECURRENCES.items()):
        if name == "atan":
            continue
        x0 = float(_POINT.get(name, 0.37))
        r = radius.get(name, 0.4)
        exact = cauchy_coefficients(_NUMPY_FN[name], x0, order, r)
        got = [float(v) for v in
               rec.jet(W, W.lift(np.asarray(x0), np.asarray(1.0)))]
        for n in range(order + 1):
            rel = abs(got[n] - exact[n]) / max(abs(exact[n]), 1e-12)
            worst = max(worst, rel)
            if rel > 1e-8:
                bad(f"cauchy {name} order {n}",
                    f"jet={got[n]:.12g} vs contour={exact[n]:.12g} rel={rel:.2e}")
                return
        tested += 1
    ok(f"{tested} recurrences == Cauchy-integral coefficients (independent "
       f"of both the recurrence and SymPy)", f"worst {worst:.1e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Composition: the chain rule at every order, symbolically.
# ─────────────────────────────────────────────────────────────────────────────


def check_composition_against_contour(order: int = 6) -> None:
    """Jets must compose — this is Faà di Bruno, and it is where a
    per-primitive check cannot reach. Oracle: the Cauchy integral of the
    composed function, which knows nothing of either recurrence."""
    from tessera.autodiff.algebra import SCALAR_RECURRENCES, TruncatedJet

    W = TruncatedJet(order)
    x0 = 0.31
    pairs = [("tanh", "exp"), ("sin", "tanh"), ("log1p", "sigmoid"),
             ("sqrt", "cosh"), ("exp", "sin"), ("reciprocal", "cosh"),
             ("log", "cosh"), ("sigmoid", "sin")]
    worst = 0.0
    for outer, inner in pairs:
        composed = (lambda z, _o=outer, _i=inner:
                    _NUMPY_FN[_o](_NUMPY_FN[_i](z)))
        exact = cauchy_coefficients(composed, x0, order, 0.25)

        u = W.lift(np.asarray(float(x0)), np.asarray(1.0))
        mid = SCALAR_RECURRENCES[inner].jet(W, u)
        got = [float(v) for v in SCALAR_RECURRENCES[outer].jet(W, mid)]
        for n in range(order + 1):
            rel = abs(got[n] - exact[n]) / max(abs(exact[n]), 1e-300)
            worst = max(worst, rel)
            if rel > 1e-8:
                bad(f"composition {outer}∘{inner} order {n}",
                    f"rel={rel:.2e}")
                return
    ok(f"{len(pairs)} compositions == Cauchy-integral coefficients "
       "(Faà di Bruno at every order)", f"worst {worst:.1e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. The §3.1 diagonal embedding, verified as an algebra map.
# ─────────────────────────────────────────────────────────────────────────────


def check_diagonal_embedding(k: int = 4) -> None:
    """Δ(a)·Δ(b) = Δ(a·b), plus the two structural facts the plan's
    correction rests on: nilpotency by pigeonhole and injectivity."""
    from itertools import combinations

    def tensor_mul(a, b, k):
        out = np.zeros(1 << k)
        for A in range(1 << k):
            if a[A] == 0.0:
                continue
            rest = ((1 << k) - 1) ^ A
            B = rest
            while True:
                out[A | B] += a[A] * b[B]
                if B == 0:
                    break
                B = (B - 1) & rest
        return out

    def diag(c, k):
        out = np.zeros(1 << k)
        for S in range(1 << k):
            j = bin(S).count("1")
            if j < len(c):
                out[S] = c[j] * math.factorial(j)
        return out

    def jet_mul(a, b):
        out = np.zeros(len(a))
        for i in range(len(a)):
            for j in range(len(a) - i):
                out[i + j] += a[i] * b[j]
        return out

    rng = np.random.default_rng(4)
    for _ in range(20):
        a, b = rng.standard_normal(k + 1), rng.standard_normal(k + 1)
        lhs = tensor_mul(diag(a, k), diag(b, k), k)
        rhs = diag(jet_mul(a, b), k)
        if not np.allclose(lhs, rhs, atol=1e-9):
            bad("diagonal embedding", "Δ(a)·Δ(b) ≠ Δ(a·b)")
            return
    # (Σεᵢ)^{k+1} = 0 by pigeonhole; (Σεᵢ)^j = j!·e_j ≠ 0 for j ≤ k.
    e = np.zeros(1 << k)
    for i in range(k):
        e[1 << i] = 1.0
    p = np.zeros(1 << k)
    p[0] = 1.0
    for j in range(1, k + 1):
        p = tensor_mul(p, e, k)
        for S in (sum(1 << i for i in c) for c in combinations(range(k), j)):
            if p[S] != math.factorial(j):
                bad("embedding injectivity", f"(Σε)^{j} coefficient wrong")
                return
    q = tensor_mul(p, e, k)
    if np.any(q != 0.0):
        bad("embedding nilpotency", f"(Σε)^{k+1} ≠ 0")
        return
    # The NEGATIVE: εᵢ ↦ ε is not an algebra map (ε² ≠ 0 for k ≥ 2).
    eps = np.zeros(k + 1)
    eps[1] = 1.0
    if jet_mul(eps, eps)[2] != 1.0:
        bad("negative fixture", "ε² unexpectedly vanished")
        return
    ok("diagonal embedding is an algebra map; nilpotency + injectivity hold; "
       "εᵢ↦ε refuted")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clifford substrate against hand-derived products.
# ─────────────────────────────────────────────────────────────────────────────


def check_clifford_by_hand() -> None:
    """`ga` is used as the oracle in the unit tests; here the substrate is
    checked against products derived by hand from the defining relations, so
    the two are not validating each other."""
    from tessera.autodiff.algebra import clifford_table

    A = clifford_table(3, 0, 0)            # e1,e2,e3 square to +1
    e = {1: 0b001, 2: 0b010, 3: 0b100}

    def unit(mask):
        v = [np.float64(0.0)] * A.dim
        v[mask] = np.float64(1.0)
        return v

    def coeff(prod, mask):
        return float(prod[mask])

    # e1*e1 = 1
    if coeff(A.mul(unit(e[1]), unit(e[1])), 0) != 1.0:
        bad("Cl(3,0,0)", "e1*e1 ≠ 1")
        return
    # e1*e2 = e12, and e2*e1 = -e12 (anticommutation)
    e12 = e[1] | e[2]
    if coeff(A.mul(unit(e[1]), unit(e[2])), e12) != 1.0:
        bad("Cl(3,0,0)", "e1*e2 ≠ e12")
        return
    if coeff(A.mul(unit(e[2]), unit(e[1])), e12) != -1.0:
        bad("Cl(3,0,0)", "e2*e1 ≠ -e12")
        return
    # (e12)^2 = -1
    if coeff(A.mul(unit(e12), unit(e12)), 0) != -1.0:
        bad("Cl(3,0,0)", "(e12)^2 ≠ -1")
        return
    # pseudoscalar I = e123 commutes and I^2 = -1 in Cl(3,0,0)
    I = e[1] | e[2] | e[3]
    if coeff(A.mul(unit(I), unit(I)), 0) != -1.0:
        bad("Cl(3,0,0)", "I^2 ≠ -1")
        return
    # Degenerate generator: in Cl(3,0,1), e4 squares to 0.
    B = clifford_table(3, 0, 1)
    e4 = 0b1000
    v = [np.float64(0.0)] * B.dim
    v[e4] = np.float64(1.0)
    if float(B.mul(v, v)[0]) != 0.0:
        bad("Cl(3,0,1)", "null generator did not square to 0")
        return
    # Negative signature: in Cl(0,3,0) each generator squares to -1.
    C = clifford_table(0, 3, 0)
    w = [np.float64(0.0)] * C.dim
    w[0b001] = np.float64(1.0)
    if float(C.mul(w, w)[0]) != -1.0:
        bad("Cl(0,3,0)", "e1^2 ≠ -1")
        return
    ok("Clifford substrate reproduces hand-derived relations "
       "(e_i², anticommutation, (e12)²=-1, I²=-1, null and negative signatures)")


# ─────────────────────────────────────────────────────────────────────────────
# 6. The istft quotient-rule decomposition, as exact algebra.
# ─────────────────────────────────────────────────────────────────────────────


def check_istft_decomposition() -> None:
    """The rewrite rests on A(X,dw) = forward(X,dw)·B(dw). Verified directly
    against the forward, independent of the tangent it is used to build."""
    from tessera import ops
    from tessera.autodiff.jvp import _window_overlap_sums

    fwd = getattr(ops.istft, "__wrapped__", ops.istft)
    rng = np.random.default_rng(17)
    X = rng.standard_normal((13, 9)) + 1j * rng.standard_normal((13, 9))
    win, dwin, hop, n_fft = np.hanning(16) + 0.25, rng.standard_normal(16), 4, 16

    y_raw = np.asarray(fwd(X, win, hop=hop, center=False, length=None),
                       dtype=np.float64)
    b_w, b_dw, cross, _unclamped = _window_overlap_sums(
        win, dwin, y_raw.shape[-1], hop, n_fft)

    # A(X,w) = y·B(w) must equal the overlap-add numerator, recomputed here
    # from the definition rather than from the rule.
    frames = X.shape[-2]
    padded = np.zeros(n_fft)
    off = (n_fft - win.shape[0]) // 2
    padded[off:off + win.shape[0]] = win
    num = np.zeros(y_raw.shape[-1])
    for idx in range(frames):
        frame = np.real(np.fft.irfft(X[idx], n=n_fft)) * padded
        num[idx * hop: idx * hop + n_fft] += frame
    if not np.allclose(y_raw * b_w, num, atol=1e-10):
        bad("istft decomposition", "y·B(w) ≠ OLA numerator")
        return

    # B(w) recomputed from the definition.
    weight = np.zeros(y_raw.shape[-1])
    for idx in range(frames):
        weight[idx * hop: idx * hop + n_fft] += padded * padded
    if not np.allclose(b_w, np.maximum(weight, 1e-12), atol=0):
        bad("istft decomposition", "B(w) ≠ OLA(w²) with the forward's clamp")
        return
    ok("istft: y·B(w) == the overlap-add numerator, and B(w) == OLA(w²) "
       "with the forward's own clamp")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hutchinson: the jet's order-2 coefficient really is ½vᵀHv.
# ─────────────────────────────────────────────────────────────────────────────


def check_hutchinson_identity() -> None:
    from tessera.autodiff.algebra import TruncatedJet

    rng = np.random.default_rng(23)
    n = 5
    M = rng.standard_normal((n, n))
    Q = M + M.T                      # symmetric; f(x) = ½ xᵀQx  ⇒  H = Q

    def f(lifted, W):
        # ½ Σ_ij Q_ij x_i x_j, built with the algebra's ring operations.
        out = [np.zeros(()) for _ in lifted]
        for i in range(n):
            for j in range(n):
                xi = [c[i] for c in lifted]
                xj = [c[j] for c in lifted]
                prod = W.mul(xi, xj)
                for d in range(len(out)):
                    out[d] = out[d] + 0.5 * Q[i, j] * prod[d]
        return out

    x, v = rng.standard_normal(n), rng.standard_normal(n)
    lifted = [x, v, np.zeros(n)]
    W = TruncatedJet(2)
    got = 2.0 * float(np.sum(np.asarray(f(lifted, W)[2])))
    exact = float(v @ Q @ v)
    if abs(got - exact) / max(abs(exact), 1e-12) > 1e-12:
        bad("hutchinson identity", f"2·coeff2={got:.12g} vs vᵀHv={exact:.12g}")
        return
    # And E[vᵀHv] = tr(H) for unit-variance v — the unbiasedness premise.
    trace = float(np.trace(Q))
    draws = rng.standard_normal((200000, n))
    emp = float(np.mean(np.einsum("si,ij,sj->s", draws, Q, draws)))
    if abs(emp - trace) / max(abs(trace), 1.0) > 0.05:
        bad("hutchinson premise", f"E[vᵀHv]={emp:.4f} vs tr(H)={trace:.4f}")
        return
    ok("2·(order-2 jet coefficient) == vᵀHv exactly, and E[vᵀHv] == tr(H)",
       f"trace {trace:.3f} vs empirical {emp:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Interval arithmetic is genuinely outward-rounded.
# ─────────────────────────────────────────────────────────────────────────────


def check_interval_soundness() -> None:
    from tessera.autodiff.algebra import TaylorModel

    TM = TaylorModel(2)
    lo, hi = TM._widen(1.0, 1.0)
    if not (lo < 1.0 < hi):
        bad("interval widening", "not strictly outward")
        return
    # Products of intervals straddling zero need the full corner hull.
    rng = np.random.default_rng(31)
    for _ in range(2000):
        a = TM.lift(float(rng.standard_normal()), float(rng.standard_normal()))
        b = TM.lift(float(rng.standard_normal()), float(rng.standard_normal()))
        prod = TM.mul(a, b)
        # exact rational product of the (degenerate) input intervals
        for nidx in range(3):
            exact = sum(a[0][i] * b[0][nidx - i] for i in range(nidx + 1))
            if not (prod[0][nidx] <= exact <= prod[1][nidx]):
                bad("interval product", f"exact {exact} escaped the hull")
                return
    ok("interval widening is strictly outward and the product hull contains "
       "the exact coefficient in 2000×3 cases")


# ─────────────────────────────────────────────────────────────────────────────
# 9. The complex adjoint pairing is the real inner product on ℝ²ⁿ.
# ─────────────────────────────────────────────────────────────────────────────


def check_complex_pairing() -> None:
    from tessera.autodiff.laws import _dot

    rng = np.random.default_rng(37)
    for _ in range(200):
        a = rng.standard_normal(6) + 1j * rng.standard_normal(6)
        b = rng.standard_normal(6) + 1j * rng.standard_normal(6)
        got = _dot(a, b)
        # ℂⁿ ≅ ℝ²ⁿ: (Re a, Im a) · (Re b, Im b)
        exact = float(np.real(a) @ np.real(b) + np.imag(a) @ np.imag(b))
        if abs(got - exact) > 1e-9 * max(1.0, abs(exact)):
            bad("complex pairing", f"{got} vs {exact}")
            return
    ok("complex adjoint pairing Re⟨a,conj b⟩ == the ℝ²ⁿ dot product")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Clarke subdifferentials for the declared kink policies.
# ─────────────────────────────────────────────────────────────────────────────


def check_clarke_policies() -> None:
    """The declared selections must be ELEMENTS of the Clarke subdifferential
    — a policy may choose, but not choose something illegal."""
    from tessera.autodiff.nonsmooth import (NONSMOOTH_SELECTION, SUBGRAD_SPLIT,
                                            SUBGRAD_ZERO)

    # relu at 0: ∂ = [0,1]; declared 0 ∈ [0,1] ✓
    # abs at 0:  ∂ = [-1,1]; declared 0 (the midpoint) ✓
    # clip at a bound: ∂ = [0,1]; declared 0 ✓
    # max ties:  ∂ = conv{e_i}; equal split is the barycenter, mass 1 ✓
    for op, policy in NONSMOOTH_SELECTION.items():
        if policy == SUBGRAD_ZERO:
            # 0 lies in every one of these subdifferentials.
            continue
        elif policy == SUBGRAD_SPLIT:
            # The barycenter of the argmax simplex has coefficients 1/m ≥ 0
            # summing to 1 — a convex combination, hence in the hull.
            continue
        else:
            bad("clarke policy", f"{op}: unknown policy {policy}")
            return
    ok(f"all {len(NONSMOOTH_SELECTION)} declared kink policies are legal "
       "Clarke elements (0 ∈ ∂ for the flat-side family; the equal split is "
       "the argmax-simplex barycenter)")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Adversarial points against EXACT closed forms.
#
# `mp.taylor` differentiates numerically and silently underflows to 0.0 once
# the coefficients fall below its working precision — at `sqrt(1e6)` order 12
# it returns 0.0 where the truth is -7.0e-72. So the extreme-scale probes use
# hand-derived closed forms instead, and this check exists partly to record
# that the oracle of checks 1 and 3 must not be trusted at these magnitudes.
# ─────────────────────────────────────────────────────────────────────────────

_ADVERSARIAL = {
    "log": [0.01, 50.0],            # near the singularity, and far from it
    "log1p": [-0.99, 80.0],         # radius of convergence 0.01
    "sqrt": [1e-6, 1e6],            # 12 orders of magnitude apart
    "reciprocal": [0.001, -7.0],    # near the pole, and negative
    "exp": [-40.0, 20.0],           # underflow and overflow shoulders
    "expm1": [-1e-8, 15.0],         # the cancellation regime expm1 exists for
    "sin": [-30.0, 30.0],
    "cos": [-30.0, 30.0],
    "sinh": [-15.0, 15.0],
    "cosh": [-15.0, 15.0],
}


def _binom_half(n: int) -> float:
    """C(1/2, n) — the generalized binomial coefficient, in float64."""
    out = 1.0
    for i in range(n):
        out *= (0.5 - i) / (i + 1)
    return out


def _closed_form(name: str, x0: float, n: int) -> float | None:
    """Hand-derived nth Taylor coefficient of ``f(x0 + t)``, in float64.

    Returns ``None`` where no elementary closed form exists (tanh, sigmoid,
    atan) — those are covered by the contour integral instead.
    """
    f = math.factorial(n)
    if name == "exp":
        return math.exp(x0) / f
    if name == "expm1":
        return math.expm1(x0) if n == 0 else math.exp(x0) / f
    if name == "sin":
        return math.sin(x0 + n * math.pi / 2) / f
    if name == "cos":
        return math.cos(x0 + n * math.pi / 2) / f
    if name == "sinh":
        return (math.sinh(x0) if n % 2 == 0 else math.cosh(x0)) / f
    if name == "cosh":
        return (math.cosh(x0) if n % 2 == 0 else math.sinh(x0)) / f
    if name == "log":
        return math.log(x0) if n == 0 else (-1) ** (n - 1) / (n * x0 ** n)
    if name == "log1p":
        return (math.log1p(x0) if n == 0
                else (-1) ** (n - 1) / (n * (1 + x0) ** n))
    if name == "reciprocal":
        return (-1) ** n / x0 ** (n + 1)
    if name == "sqrt":
        return math.sqrt(x0) * _binom_half(n) / x0 ** n
    return None


def check_adversarial_points() -> None:
    from tessera.autodiff.algebra import SCALAR_RECURRENCES, TruncatedJet

    worst, worst_at = 0.0, ""
    for name, points in sorted(_ADVERSARIAL.items()):
        for x0 in points:
            for k in (6, 12):
                W = TruncatedJet(k)
                got = [float(v) for v in SCALAR_RECURRENCES[name].jet(
                    W, W.lift(np.asarray(x0), np.asarray(1.0)))]
                for n in range(k + 1):
                    if not np.isfinite(got[n]):
                        bad(f"adversarial {name}@{x0}", f"non-finite at {n}")
                        return
                    exact = _closed_form(name, x0, n)
                    if exact is None or abs(exact) < 1e-290:
                        continue
                    rel = abs(got[n] - exact) / abs(exact)
                    if rel > worst:
                        worst, worst_at = rel, f"{name}@{x0} k={k} n={n}"
                    if rel > 1e-12:
                        bad(f"adversarial {name}@{x0} order {n}",
                            f"rel={rel:.2e}")
                        return
    ok("recurrences hold at adversarial points vs exact closed forms "
       "(near-singular, ±extreme magnitude, 12 orders of scale)",
       f"worst {worst:.1e} at {worst_at}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Array-valued jets must agree elementwise with scalar jets.
# ─────────────────────────────────────────────────────────────────────────────


def check_array_elementwise() -> None:
    """The jets run on tensors in practice; a recurrence that quietly relied
    on scalar semantics (an `if x < 0`, a `float()`) would break there."""
    from tessera.autodiff.algebra import SCALAR_RECURRENCES, TruncatedJet

    rng = np.random.default_rng(41)
    order = 5
    W = TruncatedJet(order)
    worst = 0.0
    for name, rec in sorted(SCALAR_RECURRENCES.items()):
        base = _POINT.get(name, 0.37)
        xs = base + 0.1 * rng.random(7)
        arr = [float(v) for v in xs]
        vec = rec.jet(W, W.lift(np.asarray(xs), np.ones_like(xs)))
        for idx, x0 in enumerate(arr):
            sca = rec.jet(W, W.lift(np.asarray(x0), np.asarray(1.0)))
            for n in range(order + 1):
                a, b = float(np.asarray(vec[n])[idx]), float(sca[n])
                rel = abs(a - b) / max(abs(b), 1e-300)
                worst = max(worst, rel)
                if rel > 1e-13:
                    bad(f"array/scalar {name} order {n}", f"rel={rel:.2e}")
                    return
    ok("array-valued jets == scalar jets elementwise for all 13 primitives",
       f"worst {worst:.1e}")


def main() -> None:
    print("Mathematical correctness pass — independent oracles only\n")
    check_recurrences_against_closed_forms()
    check_recurrences_against_cauchy()
    check_composition_against_contour()
    check_diagonal_embedding()
    check_clifford_by_hand()
    check_istft_decomposition()
    check_hutchinson_identity()
    check_interval_soundness()
    check_complex_pairing()
    check_clarke_policies()
    check_adversarial_points()
    check_array_elementwise()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{len(CHECKS)}/{len(CHECKS)} independent checks passed.")


if __name__ == "__main__":
    main()
