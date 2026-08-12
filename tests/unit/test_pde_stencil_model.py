"""Executable mathematical contract for the PDE / stencil capability.

Verifies the derived results in docs/audit/compiler/PDE_STENCIL_CAPABILITY_PLAN.md:
the exact von Neumann stability decider (Part II A2-A3) and its goldens, the
principal-symbol reduction and type classification (Part I.2, Part II A1), and
the graph-theoretic scheduling oracles (Part IV).

Everything here is decided in EXACT rational arithmetic (`fractions.Fraction`)
because a stability verdict flips on the boundary: FTCS heat at s = 1/2 is
stable and at s = 51/100 is not, and no float tolerance separates those
honestly. The Sturm/root-condition helpers below are the reference the C++
`lib/Analysis/VonNeumann.cpp` must agree with (plan Part III.6); keep them
dependency-free (numpy + stdlib only -- no sympy, no torch on the fleet).

The load-bearing negative result is `test_endpoint_sampling_cannot_prove_stable`:
sampling only ever refutes. A PROVEN_STABLE verdict that traces to a sample grid
is the Decision #30 fail-open pattern, and that test is what makes it a build
failure rather than a code-review opinion.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Sequence

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Exact rational polynomial arithmetic (plan Part II, A3)
# Polynomials are coefficient lists, lowest degree first.
# ---------------------------------------------------------------------------

Poly = list[Fraction]


def _trim(p: Poly) -> Poly:
    q = list(p)
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    return q


def poly(*coeffs: object) -> Poly:
    """Build a polynomial from ints/Fractions, lowest degree first."""
    return _trim([Fraction(c) for c in coeffs])  # type: ignore[arg-type]


def poly_eval(p: Poly, x: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * x + c
    return acc


def poly_add(a: Poly, b: Poly) -> Poly:
    n = max(len(a), len(b))
    return _trim(
        [
            (a[i] if i < len(a) else Fraction(0))
            + (b[i] if i < len(b) else Fraction(0))
            for i in range(n)
        ]
    )


def poly_neg(a: Poly) -> Poly:
    return _trim([-c for c in a])


def poly_sub(a: Poly, b: Poly) -> Poly:
    return poly_add(a, poly_neg(b))


def poly_mul(a: Poly, b: Poly) -> Poly:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        if ca == 0:
            continue
        for j, cb in enumerate(b):
            out[i + j] += ca * cb
    return _trim(out)


def poly_scale(a: Poly, k: Fraction) -> Poly:
    return _trim([c * k for c in a])


def poly_deriv(a: Poly) -> Poly:
    if len(a) <= 1:
        return [Fraction(0)]
    return _trim([a[i] * i for i in range(1, len(a))])


def poly_divmod(a: Poly, b: Poly) -> tuple[Poly, Poly]:
    b = _trim(b)
    if b == [Fraction(0)]:
        raise ZeroDivisionError("polynomial division by zero")
    a = _trim(a)
    q = [Fraction(0)] * max(1, len(a) - len(b) + 1)
    r = list(a)
    while len(_trim(r)) >= len(b) and _trim(r) != [Fraction(0)]:
        r = _trim(r)
        d = len(r) - len(b)
        if d < 0:
            break
        c = r[-1] / b[-1]
        q[d] += c
        for i, cb in enumerate(b):
            r[i + d] -= c * cb
        r = _trim(r)
    return _trim(q), _trim(r)


def poly_gcd(a: Poly, b: Poly) -> Poly:
    a, b = _trim(a), _trim(b)
    while b != [Fraction(0)]:
        _, r = poly_divmod(a, b)
        a, b = b, _trim(r)
    if a[-1] != 0:  # normalize to monic so the chain is canonical
        a = poly_scale(a, Fraction(1) / a[-1])
    return a


def poly_squarefree(p: Poly) -> Poly:
    """p / gcd(p, p') -- same real roots, all simple. Sturm needs this."""
    p = _trim(p)
    if p == [Fraction(0)]:
        return p
    d = poly_deriv(p)
    if d == [Fraction(0)]:
        return p
    g = poly_gcd(p, d)
    if len(g) == 1:
        return p
    q, _ = poly_divmod(p, g)
    return q


def sturm_chain(p: Poly) -> list[Poly]:
    """Standard Sturm chain of the squarefree part (plan Part II A3 step 3)."""
    p0 = poly_squarefree(p)
    chain = [p0, poly_deriv(p0)]
    while _trim(chain[-1]) != [Fraction(0)] and len(_trim(chain[-1])) > 1:
        _, r = poly_divmod(chain[-2], chain[-1])
        nxt = poly_neg(r)
        if _trim(nxt) == [Fraction(0)]:
            break
        chain.append(nxt)
    return chain


def _sign_changes(chain: Sequence[Poly], x: Fraction) -> int:
    signs = []
    for q in chain:
        v = poly_eval(q, x)
        if v != 0:
            signs.append(1 if v > 0 else -1)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def count_distinct_roots(p: Poly, a: Fraction, b: Fraction) -> int:
    """Number of distinct real roots of p in the half-open interval (a, b]."""
    chain = sturm_chain(p)
    return _sign_changes(chain, a) - _sign_changes(chain, b)


def nonneg_on(
    p: Poly, a: Fraction, b: Fraction, depth: int = 24
) -> tuple[bool, Fraction | None]:
    """Decide p(x) >= 0 for all x in [a, b], exactly.

    Returns (True, None) when p is nonnegative on the interval, else
    (False, witness) with an exact rational witness where p is negative.
    Sound in BOTH directions -- this is what a PROVEN_STABLE verdict may cite,
    unlike sampling (see `test_endpoint_sampling_cannot_prove_stable`).
    """

    def rec(lo: Fraction, hi: Fraction, d: int) -> Fraction | None:
        for x in (lo, hi):
            if poly_eval(p, x) < 0:
                return x
        # No roots in (lo, hi] => constant sign there; one interior probe decides.
        if count_distinct_roots(p, lo, hi) == 0:
            m = (lo + hi) / 2
            return m if poly_eval(p, m) < 0 else None
        if d == 0:
            for i in range(1, 32):
                x = lo + (hi - lo) * Fraction(i, 32)
                if poly_eval(p, x) < 0:
                    return x
            return None
        m = (lo + hi) / 2
        return rec(lo, m, d - 1) or rec(m, hi, d - 1)

    w = rec(a, b, depth)
    return (w is None, w)


MINUS_ONE, ONE = Fraction(-1), Fraction(1)


def stable_on_cos(r: Poly) -> tuple[bool, Fraction | None]:
    """r(c) = 1 - |xi(c)|^2 must be >= 0 for c = cos(theta) in [-1, 1]."""
    return nonneg_on(r, MINUS_ONE, ONE)


# ---------------------------------------------------------------------------
# Amplification factors (plan Part I.3). Each returns r(c) = 1 - |xi|^2.
# The cosine reduction (Part II A2): for REAL tap coefficients |xi|^2 is a
# polynomial in c = cos(theta) via the tap autocorrelation -- symmetry of the
# stencil is NOT required, which is why upwind advection below is decidable.
# ---------------------------------------------------------------------------


def ftcs_heat_r(s: Fraction) -> Poly:
    """A2: FTCS heat, xi = 1 - 2s(1 - c).  Stable iff s <= 1/2."""
    xi = poly(1 - 2 * s, 2 * s)  # (1 - 2s) + 2s*c
    return poly_sub(poly(1), poly_mul(xi, xi))


def theta_scheme_num(s: Fraction, q: Fraction) -> Poly:
    """A2: theta/Q-scheme, D^2 - N^2 = 4w(1 + w(2Q - 1)) with w = s(1 - c).

    Unconditionally stable for Q >= 1/2 (Crank-Nicolson is Q = 1/2).
    D > 0 on the interval, so the sign of D^2 - N^2 decides |xi| <= 1.
    """
    w = poly(s, -s)  # s(1 - c)
    inner = poly_add(poly(1), poly_scale(w, 2 * q - 1))
    return poly_scale(poly_mul(w, inner), Fraction(4))


def centered_advection_r(a: Fraction) -> Poly:
    """A2: forward Euler + centered advection, |xi|^2 = 1 + a^2(1 - c^2).

    UNCONDITIONALLY UNSTABLE for every a != 0. This is the negative golden that
    makes `scheme` a semantic key (Decision #21a).
    """
    return poly(-a * a, 0, a * a)  # a^2 c^2 - a^2 = -(a^2)(1 - c^2)


def upwind_advection_r(a: Fraction) -> Poly:
    """A2: upwind advection, 1 - |xi|^2 = 2a(1 - a)(1 - c). Stable iff 0<=a<=1."""
    k = 2 * a * (1 - a)
    return poly(k, -k)


def advection_diffusion_r(s: Fraction, a: Fraction) -> Poly:
    """A2: centered advection + centered diffusion.

    xi = 1 - 2s(1 - c) - i*a*sin(theta), so
        |xi|^2 = (1 - 2s(1-c))^2 + a^2 (1 - c^2).
    """
    diff = poly(1 - 2 * s, 2 * s)
    return poly_sub(poly(1), poly_add(poly_mul(diff, diff), poly(a * a, 0, -a * a)))


def rk4_stability_excess() -> Poly:
    """A2/A5: |R_RK4(i y)|^2 - 1 = y^6 (y^2 - 8) / 576, as a polynomial in y^2."""
    # In the variable t = y^2: t^3 (t - 8) / 576.
    return poly_scale(poly(0, 0, 0, -8, 1), Fraction(1, 576))


# ---------------------------------------------------------------------------
# Multi-level root condition (plan Part II A3, case B.2)
# ---------------------------------------------------------------------------


def leapfrog_roots(s: Fraction, c: Fraction) -> tuple[complex, complex]:
    """Roots of xi^2 - 2(1+p) xi + 1 = 0 with p = s^2 (c - 1) (leapfrog wave)."""
    p = float(s) ** 2 * (float(c) - 1.0)
    b = 2.0 * (1.0 + p)
    disc = complex(b * b - 4.0, 0.0) ** 0.5
    return ((b + disc) / 2.0, (b - disc) / 2.0)


def leapfrog_root_condition(s: Fraction, samples: int = 4001) -> tuple[bool, bool]:
    """(root_condition_holds, has_defective_unimodular_root) over theta.

    Root condition for a monic degree-2 with constant term 1: both roots lie on
    the closed unit disk iff |1 + p| <= 1, i.e. -2 <= p <= 0. A REPEATED
    unimodular root (p = -2 exactly) still satisfies |xi| <= 1 but grows
    linearly in n -- which is why the simple-root clause is not decoration.
    """
    ok, defective = True, False
    for i in range(samples):
        theta = np.pi * i / (samples - 1)
        c = Fraction(np.cos(theta)).limit_denominator(10**9)
        p = s * s * (c - 1)
        if not (Fraction(-2) <= p <= Fraction(0)):
            ok = False
        if p == Fraction(-2):
            defective = True
    return ok, defective


def leapfrog_peak_amplitude(s: float, n_modes: int, steps: int) -> float:
    """Directly integrate the leapfrog recurrence for the theta = pi mode."""
    theta = np.pi  # the mode that goes defective at s = 1
    p = s * s * (np.cos(theta) - 1.0)
    prev, cur = 1.0, 1.0
    peak = 1.0
    for _ in range(steps):
        nxt = 2.0 * (1.0 + p) * cur - prev
        prev, cur = cur, nxt
        peak = max(peak, abs(cur))
    return peak


# ---------------------------------------------------------------------------
# Principal symbol and classification (plan Part I.2, Part II A1)
# ---------------------------------------------------------------------------


def exact_inertia(a: list[list[Fraction]]) -> tuple[int, int, int]:
    """(n_pos, n_neg, n_zero) of a symmetric rational matrix, exactly.

    Symmetric Gaussian elimination (congruence), which preserves inertia by
    Sylvester's law. Uses a symmetric row/col swap when a diagonal pivot is zero
    but the block is not, and handles the 2x2 hyperbolic case explicitly.
    """
    m = [row[:] for row in a]
    n = len(m)
    pos = neg = zero = 0
    idx = list(range(n))
    while idx:
        # Prefer a nonzero diagonal pivot.
        k = next((i for i in idx if m[i][i] != 0), None)
        if k is None:
            # All diagonals zero. If the whole remaining block is zero we are done.
            off = [(i, j) for i in idx for j in idx if i < j and m[i][j] != 0]
            if not off:
                zero += len(idx)
                break
            # A 2x2 block [[0, b], [b, 0]] has inertia (1, 1, 0). Realize it by
            # the congruence x_i -> x_i + x_j, which makes a diagonal nonzero.
            i, j = off[0]
            for t in idx:
                m[i][t] += m[j][t]
            for t in idx:
                m[t][i] += m[t][j]
            continue
        piv = m[k][k]
        if piv > 0:
            pos += 1
        else:
            neg += 1
        rest = [i for i in idx if i != k]
        for i in rest:
            f = m[i][k] / piv
            if f == 0:
                continue
            for j in rest:
                m[i][j] -= f * m[k][j]
        for i in rest:
            m[i][k] = m[k][i] = Fraction(0)
        idx = rest
    return pos, neg, zero


def classify_second_order(a: list[list[Fraction]], time_axis: int | None = None) -> str:
    """Classify a scalar second-order operator from its coefficient matrix.

    Complete and exact for m = 2 (plan Part I.2 verdict table). Returns one of
    elliptic / hyperbolic / ultrahyperbolic / degenerate / unknown. Parabolicity
    is NOT decidable from the principal part alone -- see
    `classify_parabolic_direction`.
    """
    pos, neg, zero = exact_inertia(a)
    n = len(a)
    if zero == 0 and (pos == n or neg == n):
        return "elliptic"
    if zero == 0 and min(pos, neg) == 1:
        if time_axis is None:
            return "unknown"  # hyperbolicity is relative to a covector
        return "hyperbolic"
    if zero == 0 and min(pos, neg) >= 2:
        return "ultrahyperbolic"
    return "degenerate"


def classify_parabolic_direction(
    spatial: list[list[Fraction]], dt_coeff: Fraction
) -> str:
    """Forward vs backward parabolic, from the SUB-principal D_t coefficient.

    The heat operator's principal part is degenerate along the time axis, so the
    principal symbol alone cannot decide well-posedness. With D = d/dx,
    d_t u_hat = (1/c_t) q_x(xi) u_hat, so the forward problem is well posed iff
    the definite spatial part carries the sign OPPOSITE to c_t.
    """
    pos, neg, zero = exact_inertia(spatial)
    n = len(spatial)
    if zero != 0 or not (pos == n or neg == n):
        return "unknown"
    spatial_sign = 1 if pos == n else -1
    ct_sign = 1 if dt_coeff > 0 else -1
    return "parabolic_forward" if spatial_sign != ct_sign else "parabolic_backward"


def principal_symbol_real_part(
    terms: dict[tuple[int, ...], Fraction], xi: Sequence[float]
) -> complex:
    """p_m(xi) = sum_alpha a_alpha (i xi)^alpha, evaluated directly."""
    total = 0j
    for alpha, coeff in terms.items():
        term = complex(float(coeff), 0.0)
        for axis, e in enumerate(alpha):
            term *= (1j * xi[axis]) ** e
        total += term
    return total


def real_form(terms: dict[tuple[int, ...], Fraction], xi: Sequence[float]) -> float:
    """q_m(xi) = sum_alpha a_alpha xi^alpha -- the real form p_m = i^m q_m."""
    total = 0.0
    for alpha, coeff in terms.items():
        term = float(coeff)
        for axis, e in enumerate(alpha):
            term *= xi[axis] ** e
        total += term
    return total


# ---------------------------------------------------------------------------
# Graph oracles (plan Part IV)
# ---------------------------------------------------------------------------


def _augment(u: int, adj: list[set[int]], match_r: list[int], seen: list[bool]) -> bool:
    for v in adj[u]:
        if not seen[v]:
            seen[v] = True
            if match_r[v] == -1 or _augment(match_r[v], adj, match_r, seen):
                match_r[v] = u
                return True
    return False


def max_bipartite_matching(adj: list[set[int]], n_right: int) -> list[int]:
    """Kuhn's algorithm. Returns match_r[v] = u, or -1."""
    match_r = [-1] * n_right
    for u in range(len(adj)):
        _augment(u, adj, match_r, [False] * n_right)
    return match_r


def peel_rounds(demand: np.ndarray) -> list[tuple[int, ...]]:
    """Decompose a k-regular bipartite demand into k perfect matchings.

    Frobenius: a k-regular bipartite graph always has a perfect matching; peel it
    and the residual is (k-1)-regular. Konig's line-coloring theorem makes the
    round count exactly Delta -- NOT Vizing's Delta+1, which is the general-graph
    bound and would over-provision every schedule by a round.
    """
    residual = demand.copy()
    n = residual.shape[0]
    k = int(residual.sum(axis=1).max())
    rounds: list[tuple[int, ...]] = []
    for _ in range(k):
        adj = [set(np.nonzero(residual[u])[0].tolist()) for u in range(n)]
        match_r = max_bipartite_matching(adj, n)
        perm = [-1] * n
        for v, u in enumerate(match_r):
            if u >= 0:
                perm[u] = v
        if any(p < 0 for p in perm):
            raise AssertionError("no perfect matching in a regular residual")
        for u, v in enumerate(perm):
            residual[u][v] -= 1
        rounds.append(tuple(perm))
    return rounds


def hall_defect(adj: list[set[int]], n_right: int, capacity: int = 1) -> int:
    """max_U (|U| - |N(U)| * capacity) over subsets U of the left vertices.

    Brute force over subsets -- this is the ORACLE, so exhaustive is correct.
    """
    n_left = len(adj)
    best = 0
    for mask in range(1 << n_left):
        u_set = [i for i in range(n_left) if mask & (1 << i)]
        if not u_set:
            continue
        nbr: set[int] = set()
        for i in u_set:
            nbr |= adj[i]
        best = max(best, len(u_set) - len(nbr) * capacity)
    return best


def spanning_tree_count(adjacency: np.ndarray) -> int:
    """Kirchhoff matrix-tree count via a principal Laplacian minor."""
    deg = np.diag(adjacency.sum(axis=1))
    lap = deg - adjacency
    return int(round(float(np.linalg.det(lap[1:, 1:]))))


def spanning_tree_count_eigen(adjacency: np.ndarray) -> int:
    """Same count via the eigenvalue form: product of nonzero eigenvalues / n."""
    deg = np.diag(adjacency.sum(axis=1))
    lap = deg - adjacency
    n = adjacency.shape[0]
    ev = np.sort(np.linalg.eigvalsh(lap))[1:]  # drop the zero eigenvalue
    return int(round(float(np.prod(ev)) / n))


def de_bruijn_sequence(k: int, n: int) -> list[int]:
    """Cyclic sequence of length k^n covering every length-n window once."""
    a = [0] * (k * n)
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
            return
        a[t] = a[t - p]
        db(t + 1, p)
        for j in range(a[t - p] + 1, k):
            a[t] = j
            db(t + 1, t)

    db(1, 1)
    return seq


# ===========================================================================
# Stability goldens (plan Part I.3 / III.6, rows E.1-E.9)
# ===========================================================================


@pytest.mark.parametrize(
    ("s", "expected"),
    [
        (Fraction(1, 4), True),
        (Fraction(3, 8), True),
        (Fraction(1, 2), True),  # exact boundary: stable
        (Fraction(51, 100), False),  # 0.01 past it: not
        (Fraction(1), False),
    ],
)
def test_ftcs_heat_threshold(s: Fraction, expected: bool) -> None:
    """E.1: FTCS heat is stable iff s = k/h^2 <= 1/2, decided exactly."""
    ok, witness = stable_on_cos(ftcs_heat_r(s))
    assert ok is expected, f"s={s}: got {ok}, witness={witness}"
    if not ok:
        assert witness is not None
        assert poly_eval(ftcs_heat_r(s), witness) < 0  # the witness is checkable


def test_ftcs_boundary_is_exactly_one_half() -> None:
    """The threshold is a sharp rational, not a tolerance."""
    assert stable_on_cos(ftcs_heat_r(Fraction(1, 2)))[0] is True
    assert stable_on_cos(ftcs_heat_r(Fraction(1, 2) + Fraction(1, 10**6)))[0] is False


@pytest.mark.parametrize("q", [Fraction(1, 2), Fraction(3, 4), Fraction(1)])
@pytest.mark.parametrize(
    "s", [Fraction(1, 4), Fraction(1), Fraction(10), Fraction(1000)]
)
def test_theta_scheme_unconditional_for_q_at_least_half(
    q: Fraction, s: Fraction
) -> None:
    """E.2: Q >= 1/2 (incl. Crank-Nicolson Q = 1/2) is unconditionally stable."""
    ok, witness = stable_on_cos(theta_scheme_num(s, q))
    assert ok, f"Q={q}, s={s} should be unconditionally stable; witness={witness}"


@pytest.mark.parametrize(
    ("q", "s", "expected"),
    [
        (Fraction(0), Fraction(1, 2), True),  # Q=0 is FTCS: s <= 1/2
        (Fraction(0), Fraction(51, 100), False),
        (Fraction(1, 4), Fraction(1), True),  # Q=1/4: s <= 1/(2(1-2Q)) = 1
        (Fraction(1, 4), Fraction(101, 100), False),
    ],
)
def test_theta_scheme_conditional_below_half(
    q: Fraction, s: Fraction, expected: bool
) -> None:
    """E.2: for Q < 1/2 the bound is s <= 1/(2(1 - 2Q)), derived not asserted."""
    assert stable_on_cos(theta_scheme_num(s, q))[0] is expected


@pytest.mark.parametrize("a", [Fraction(1, 100), Fraction(1, 2), Fraction(1)])
def test_centered_explicit_advection_is_unconditionally_unstable(a: Fraction) -> None:
    """E.6: the negative golden. |xi|^2 = 1 + a^2(1 - c^2) > 1 for any a != 0.

    This is why `scheme` is a semantic key: LegalizeSpaceTime currently defaults
    a missing scheme to "central", which silently turns an unannotated advection
    kernel into a divergent one.
    """
    ok, witness = stable_on_cos(centered_advection_r(a))
    assert not ok, f"a={a} must be refuted"
    assert witness is not None
    assert poly_eval(centered_advection_r(a), witness) < 0


@pytest.mark.parametrize(
    ("a", "expected"),
    [
        (Fraction(1, 2), True),
        (Fraction(1), True),  # exact CFL boundary
        (Fraction(11, 10), False),
        (Fraction(-1, 10), False),  # wrong-sign wind, caught for free
    ],
)
def test_upwind_advection_cfl(a: Fraction, expected: bool) -> None:
    """E.6: upwind is stable iff 0 <= a <= 1. Note the stencil is ASYMMETRIC --
    decidable anyway, because the cosine reduction needs only real coefficients.
    """
    assert stable_on_cos(upwind_advection_r(a))[0] is expected


def test_endpoint_sampling_cannot_prove_stable() -> None:
    """THE load-bearing negative: sampling only ever refutes.

    Centered advection attains |xi| = 1 at BOTH theta = 0 and theta = pi, so an
    endpoint sampler reports "stable" for a scheme that is unconditionally
    unstable. Its true maximum is interior (theta = pi/2). A PROVEN_STABLE
    verdict must therefore never trace to a sample grid (Decision #30).
    """
    a = Fraction(1, 100)
    r = centered_advection_r(a)

    # An endpoint-only sampler sees no violation ...
    assert poly_eval(r, MINUS_ONE) == 0
    assert poly_eval(r, ONE) == 0

    # ... and even a coarse interior grid can miss it if it straddles the peak,
    # while the exact decider refutes with a checkable witness.
    ok, witness = stable_on_cos(r)
    assert not ok and witness is not None
    assert poly_eval(r, witness) < 0

    # The true maximum of |xi| is at theta = pi/2 (c = 0), not at an endpoint.
    worst = max(
        (float(np.cos(t)) for t in np.linspace(0.0, np.pi, 2001)),
        key=lambda c: -float(poly_eval(r, Fraction(c).limit_denominator(10**9))),
    )
    assert abs(worst) < 1e-2, "peak should be at c ~ 0, i.e. theta ~ pi/2"


def test_rk4_imaginary_axis_bound() -> None:
    """E.7: |R_RK4(iy)|^2 - 1 = y^6(y^2 - 8)/576, so RK4 advection needs a <= 2*sqrt(2).

    This derives the hand-written `dt = 0.25*dx/c; // CFL-safe` comment in
    src/solvers/tpp/benchmarks/correctness_microbench.cpp -- and supplies the
    headroom that comment does not state.
    """
    excess = rk4_stability_excess()  # in t = y^2
    # Closed form check against a direct evaluation of the RK4 polynomial.
    for y in (0.5, 1.0, 2.0, 2.8284271, 3.0):
        z = 1j * y
        r = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
        lhs = abs(r) ** 2 - 1.0
        rhs = float(poly_eval(excess, Fraction(y * y).limit_denominator(10**12)))
        assert abs(lhs - rhs) < 1e-9, f"y={y}: {lhs} vs {rhs}"

    # Stable region is t = y^2 <= 8, i.e. |y| <= 2*sqrt(2).
    assert poly_eval(excess, Fraction(8)) == 0
    assert poly_eval(excess, Fraction(79, 10)) < 0  # inside
    assert poly_eval(excess, Fraction(81, 10)) > 0  # outside
    assert abs(math.sqrt(8) - 2 * math.sqrt(2)) < 1e-15


@pytest.mark.parametrize(
    ("s", "a", "expected"),
    [
        (Fraction(1, 2), Fraction(1, 2), True),
        (Fraction(1, 2), Fraction(11, 10), False),  # violates a^2 <= 2s
        (Fraction(1, 4), Fraction(1, 2), True),
        (Fraction(1, 4), Fraction(4, 5), False),
        (Fraction(3, 5), Fraction(1, 2), False),  # violates s <= 1/2
        (Fraction(1, 8), Fraction(1, 2), True),
    ],
)
def test_advection_diffusion_admissible_region(
    s: Fraction, a: Fraction, expected: bool
) -> None:
    """E.8: the emitted BOUND is a region {s <= 1/2 and a^2 <= 2s}, not a verdict.

    An autotuner given the region can maximize dt inside it; given only a
    verdict it can merely probe and be refused.
    """
    exact = stable_on_cos(advection_diffusion_r(s, a))[0]
    region = (s <= Fraction(1, 2)) and (a * a <= 2 * s)
    assert exact is expected
    assert region is exact, f"closed-form region disagrees with Sturm at s={s}, a={a}"


@pytest.mark.parametrize(
    ("s", "expected_ok"),
    [(Fraction(9, 10), True), (Fraction(1), True), (Fraction(101, 100), False)],
)
def test_leapfrog_root_condition_is_cfl(s: Fraction, expected_ok: bool) -> None:
    """E.3: leapfrog satisfies the closed-disk root condition iff s <= 1."""
    ok, _ = leapfrog_root_condition(s)
    assert ok is expected_ok


def test_leapfrog_is_defective_at_the_cfl_boundary() -> None:
    """E.3: at s = 1 the root condition HOLDS but a unimodular root is repeated,
    giving linear-in-n growth. |xi| <= 1 alone is the wrong test.
    """
    ok, defective = leapfrog_root_condition(Fraction(1))
    assert ok and defective, "s=1 satisfies |xi|<=1 yet has a double unimodular root"

    r1, r2 = leapfrog_roots(Fraction(1), Fraction(-1))  # theta = pi
    assert abs(abs(r1) - 1.0) < 1e-12 and abs(abs(r2) - 1.0) < 1e-12
    assert abs(r1 - r2) < 1e-9, "the two roots coincide at s=1, theta=pi"

    # Growth is linear in the step count, not bounded and not exponential.
    peaks = [leapfrog_peak_amplitude(1.0, 64, n) for n in (1000, 10_000, 100_000)]
    ratios = [peaks[1] / peaks[0], peaks[2] / peaks[1]]
    for ratio in ratios:
        assert 9.0 < ratio < 11.0, f"expected ~10x per decade (linear), got {ratios}"

    # And strictly below the boundary it is genuinely bounded.
    bounded = [leapfrog_peak_amplitude(0.9, 64, n) for n in (1000, 100_000)]
    assert bounded[1] / bounded[0] < 1.5


def test_spectral_radius_is_not_sufficient_for_systems() -> None:
    """E.9: rho(G) <= 1 is NECESSARY but NOT SUFFICIENT for power boundedness.

    The leapfrog companion at p = -2 has rho = 1 exactly and ||G^n|| = 2n. A
    pass that concludes PROVEN_STABLE from the spectral radius of a non-normal
    amplification matrix is wrong, so the design returns UNKNOWN there.
    """
    g = np.array([[2.0 * (1.0 + -2.0), -1.0], [1.0, 0.0]])
    rho = max(abs(np.linalg.eigvals(g)))
    assert abs(rho - 1.0) < 1e-12, "spectral radius is exactly 1"

    assert not np.allclose(g @ g.T, g.T @ g), "G is non-normal (that is the point)"

    norms = []
    acc = np.eye(2)
    for _ in range(1000):
        acc = acc @ g
        norms.append(np.linalg.norm(acc, 2))
    assert norms[-1] > 1000.0, "unbounded despite rho == 1"
    # Growth is linear: ||G^n|| ~ 2n.
    assert abs(norms[999] / 2000.0 - 1.0) < 0.05


# ===========================================================================
# Principal symbol and classification (plan Part I.2)
# ===========================================================================


def test_principal_symbol_factors_as_i_to_the_m_times_a_real_form() -> None:
    """I.2: p_m(xi) = i^m q_m(xi) with q_m REAL.

    This is why no complex arithmetic appears in the classifier: nonvanishing of
    p_m and of q_m are the same question.
    """
    rng = np.random.default_rng(0)
    laplace = {(2, 0): Fraction(1), (0, 2): Fraction(1)}
    biharm = {(4, 0): Fraction(1), (2, 2): Fraction(2), (0, 4): Fraction(1)}
    for terms, m in ((laplace, 2), (biharm, 4)):
        for _ in range(20):
            xi = rng.normal(size=2)
            p = principal_symbol_real_part(terms, xi)
            q = real_form(terms, xi)
            assert abs(p - (1j**m) * q) < 1e-9


def test_odd_degree_scalar_operators_are_never_elliptic() -> None:
    """I.2: a homogeneous form of ODD degree in n >= 2 variables always has a
    nontrivial real zero (q(-xi) = -q(xi) plus IVT on the connected sphere).

    Consequence the higher tiers rely on: every elliptic scalar operator of
    order >= 3 has even order.
    """
    terms = {(3, 0): Fraction(1), (0, 3): Fraction(1)}
    thetas = np.linspace(0.0, 2 * np.pi, 4001)
    vals = [real_form(terms, (np.cos(t), np.sin(t))) for t in thetas]
    assert max(vals) > 0 and min(vals) < 0, "must change sign on the unit circle"
    assert min(abs(v) for v in vals) < 1e-3, "hence a zero on the sphere"


@pytest.mark.parametrize(
    ("matrix", "time_axis", "expected"),
    [
        ([[1, 0], [0, 1]], None, "elliptic"),  # Laplace
        ([[-1, 0], [0, -1]], None, "elliptic"),  # -Laplace: sign-flip invariant
        ([[1, 0], [0, -1]], 0, "hyperbolic"),  # wave
        ([[1, 0], [0, -1]], None, "unknown"),  # hyperbolicity needs a covector
        ([[0, 0], [0, -1]], None, "degenerate"),  # heat principal part
        (
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]],
            0,
            "ultrahyperbolic",
        ),
    ],
)
def test_second_order_classification_is_exact(
    matrix: list[list[int]], time_axis: int | None, expected: str
) -> None:
    """I.2: for m = 2 the inertia of the coefficient matrix decides completely."""
    a = [[Fraction(v) for v in row] for row in matrix]
    assert classify_second_order(a, time_axis) == expected


def test_inertia_handles_the_zero_diagonal_case() -> None:
    """[[0,1],[1,0]] is the xy-form: inertia (1,1,0), i.e. hyperbolic, not degenerate."""
    a = [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]]
    assert exact_inertia(a) == (1, 1, 0)


def test_forward_and_backward_heat_are_distinguished() -> None:
    """I.2: parabolicity is NOT decidable from the principal part alone.

    Both u_t - u_xx and u_t + u_xx have the SAME degenerate principal symbol;
    only the sub-principal D_t coefficient separates the well-posed forward
    problem from the ill-posed backward one.
    """
    forward_spatial = [[Fraction(-1)]]  # u_t - u_xx = 0  ->  a_xx = -1
    backward_spatial = [[Fraction(1)]]  # u_t + u_xx = 0  ->  a_xx = +1
    ct = Fraction(1)

    assert classify_parabolic_direction(forward_spatial, ct) == "parabolic_forward"
    assert classify_parabolic_direction(backward_spatial, ct) == "parabolic_backward"

    # The principal parts are indistinguishable, which is the point.
    fwd = classify_second_order(
        [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(-1)]]
    )
    bwd = classify_second_order(
        [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(1)]]
    )
    assert fwd == bwd == "degenerate"


def test_backward_heat_amplification_is_self_incriminating() -> None:
    """A solver that does NOT refuse backward heat is detectable numerically:
    mode k is amplified by exp(k^2 t), which no plausible output can hide.
    """
    k, t = 20.0, 0.05
    assert np.exp(k * k * t) > 4.8e8


def test_tricomi_is_classified_mixed_not_guessed() -> None:
    """I.2: y*u_xx + u_yy changes type across y = 0.

    The classifier must return the DEFINITE verdict `mixed` by enumerating the
    realizable sign patterns -- not `elliptic` (which is what sampling y at one
    point would give) and not `unknown`.
    """
    reached = set()
    for y in (Fraction(1), Fraction(-1), Fraction(0)):
        a = [[y, Fraction(0)], [Fraction(0), Fraction(1)]]
        reached.add(classify_second_order(a, time_axis=0))
    assert reached == {"elliptic", "hyperbolic", "degenerate"}
    assert len(reached) > 1, "more than one type reached => mixed"
    # Specifically: sampling only y > 0 would wrongly report a uniform type.
    assert (
        classify_second_order([[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]])
        == "elliptic"
    )


# ===========================================================================
# Graph-theoretic scheduling oracles (plan Part IV)
# ===========================================================================


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_regular_demand_peels_into_exactly_k_permutation_rounds(n: int, k: int) -> None:
    """IV.1: Frobenius + Konig give EXACTLY Delta rounds, and the bound is tight.

    Not Vizing's Delta+1 -- the exchange graph is bipartite, so the chromatic
    index is exactly Delta and citing Vizing would over-provision every schedule.
    """
    rng = np.random.default_rng(n * 100 + k)
    demand = np.zeros((n, n), dtype=int)
    for _ in range(k):
        demand[np.arange(n), rng.permutation(n)] += 1

    rounds = peel_rounds(demand)

    assert len(rounds) == k, "round count must equal Delta exactly"
    for perm in rounds:  # each round is a permutation: no port conflict
        assert sorted(perm) == list(range(n))
    acc = np.zeros_like(demand)  # union of rounds reproduces the demand
    for perm in rounds:
        for u, v in enumerate(perm):
            acc[u][v] += 1
    assert (acc == demand).all()


def test_hall_defect_equals_the_exact_drop_count() -> None:
    """IV.2: max matching = |X| - max_U(|U| - |N(U)|), and the maximizing U is
    the DIAGNOSTIC -- which token group starved which expert group.
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        n_left, n_right = 5, 4
        adj = [
            {v for v in range(n_right) if rng.random() < 0.35} for _ in range(n_left)
        ]
        matched = sum(1 for u in max_bipartite_matching(adj, n_right) if u >= 0)
        defect = hall_defect(adj, n_right)
        assert matched == n_left - max(
            0, defect
        ), f"Hall-Konig defect mismatch: matched={matched}, defect={defect}"


def test_matrix_tree_three_forms_agree_and_give_cayley() -> None:
    """IV.4: the minor form and the eigenvalue form are independent computations
    of the same count -- a self-checking oracle. tau(K_n) = n^(n-2).
    """
    for n in range(2, 9):
        k_n = np.ones((n, n), dtype=float) - np.eye(n)
        assert spanning_tree_count(k_n) == n ** (n - 2)
        assert spanning_tree_count_eigen(k_n) == n ** (n - 2)

    # A tree has exactly one spanning tree; a disconnected graph has none.
    path = np.zeros((4, 4))
    for i in range(3):
        path[i][i + 1] = path[i + 1][i] = 1.0
    assert spanning_tree_count(path) == 1
    split = np.zeros((4, 4))
    split[0][1] = split[1][0] = split[2][3] = split[3][2] = 1.0
    assert spanning_tree_count(split) == 0


@pytest.mark.parametrize(("k", "n"), [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2)])
def test_de_bruijn_covers_every_window_exactly_once(k: int, n: int) -> None:
    """IV.4: length k^n, every length-n window exactly once.

    The count of CYCLIC classes is (k!)^(k^(n-1)) / k^n. The unquotiented
    (k!)^(k^(n-1)) counts labeled sequences and is k^n times too large for this
    assertion -- an easy and silent off-by-a-factor.
    """
    seq = de_bruijn_sequence(k, n)
    assert len(seq) == k**n
    windows = {
        tuple(seq[(i + j) % len(seq)] for j in range(n)) for i in range(len(seq))
    }
    assert len(windows) == k**n

    expected_classes = math.factorial(k) ** (k ** (n - 1)) // k**n
    assert expected_classes >= 1
    if (k, n) == (2, 4):
        assert expected_classes == 16
    if (k, n) == (3, 2):
        assert expected_classes == 24


# ===========================================================================
# Reference-function preconditions (negative tests)
# ===========================================================================


def test_reference_helpers_reject_malformed_input() -> None:
    with pytest.raises(ZeroDivisionError):
        poly_divmod(poly(1, 2), poly(0))
    with pytest.raises(AssertionError):
        # A non-regular demand has no perfect matching at full multiplicity.
        peel_rounds(np.array([[1, 1], [0, 0]]))


def test_sturm_counts_known_root_multiplicities() -> None:
    """Sanity-check the decider itself against hand-computed root counts."""
    # (c - 1/2)(c + 1/2) has two distinct roots in (-1, 1].
    p = poly_mul(poly(Fraction(-1, 2), 1), poly(Fraction(1, 2), 1))
    assert count_distinct_roots(p, MINUS_ONE, ONE) == 2
    # A perfect square has ONE distinct root (Sturm counts distinct roots).
    sq = poly_mul(poly(0, 1), poly(0, 1))
    assert count_distinct_roots(sq, MINUS_ONE, ONE) == 1
    # ... and it is nonnegative, which the decider must not confuse with a crossing.
    assert nonneg_on(sq, MINUS_ONE, ONE)[0] is True
