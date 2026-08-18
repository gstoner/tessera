#!/usr/bin/env python3
"""AD-LAW-1 — machine-check of AUTODIFF_NEXTGEN_PLAN.md §3's mathematics.

Discipline copied from ``research/core_substrate/verify_substrate_math.py``
(CORE_SUBSTRATE_VIEW.md §0.1): every prose-only mathematical claim the plan
reasons from is executed here, so no slice builds on unverified prose. Pure
numpy + stdlib; host-free; deterministic (fixed seeds, no wall clock).

Checks (plan section in parens):

  1.  (§3.2) exp jet recurrence  w_k = (1/k) Σ j·u_j·w_{k−j}          vs nested duals
  2.  (§3.2) tanh jet recurrence w_k = (1/k) Σ j·u_j·(δ − s)_{k−j}    vs nested duals
  3.  (§3.2) sin/cos joint pair recurrence                            vs nested duals
  4.  (§3.2) the k=1 rows reproduce the hand-written first-order JVP forms
  5.  (§3.3) jet-matmul == truncated polynomial matrix product (Cauchy)
  6.  (§3.3) coefficient-product count through order k == (k+1)(k+2)/2
  7.  (§3.1) pigeonhole nilpotency: (ε₁+…+ε_k)^{k+1} = 0 in ⊗ᵢ ℝ[εᵢ]/(εᵢ²)
  8.  (§3.1) injectivity fuel: (Σεᵢ)^j = j!·e_j ≠ 0 for j ≤ k
  9.  (§3.1) the diagonal embedding Δ(Σ c_j ε^j)[S] = c_{|S|}·|S|! is an
      algebra homomorphism: Δ(a)·Δ(b) = Δ(a·b) with truncation matching
      the quotient
  10. (§3.1, NEGATIVE — review correction 2026-08-18, #10a) the previously
      claimed surjection εᵢ ↦ ε is NOT an algebra map: the ideal generator
      εᵢ² maps to ε², which is nonzero in ℝ[ε]/(ε^{k+1}) for k ≥ 2
  11. (§3.1 / Law 4 seed) jet extraction == k-nested duals on the diagonal
      seed with factorial bookkeeping, end-to-end through exp/tanh

Run:  python3 research/autodiff_nextgen/verify_autodiff_math.py
Exit 0 with "N/N" on success; any failure raises AssertionError.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

CHECKS: list[str] = []


def check(name: str) -> None:
    CHECKS.append(name)
    print(f"  ok [{len(CHECKS):2d}] {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Reference implementation 1: recursive dual numbers (nested forward mode).
# Dual(a, b) represents a + b·ε with ε² = 0; components may themselves be
# Duals, giving the 2^k-dimensional tensor-product algebra by construction.
# ─────────────────────────────────────────────────────────────────────────────


class Dual:
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = a, b

    # ring ops — generic over scalar or Dual components
    def __add__(self, o):
        o = _lift(o)
        return Dual(_add(self.a, o.a), _add(self.b, o.b))

    __radd__ = __add__

    def __mul__(self, o):
        o = _lift(o)
        return Dual(_mul(self.a, o.a), _add(_mul(self.a, o.b), _mul(self.b, o.a)))

    __rmul__ = __mul__


def _lift(x):
    return x if isinstance(x, Dual) else Dual(x, 0.0)


def _add(x, y):
    if isinstance(x, Dual) or isinstance(y, Dual):
        return _lift(x) + _lift(y)
    return x + y


def _mul(x, y):
    if isinstance(x, Dual) or isinstance(y, Dual):
        return _lift(x) * _lift(y)
    return x * y


def d_exp(x):
    if isinstance(x, Dual):
        e = d_exp(x.a)
        return Dual(e, _mul(x.b, e))
    return math.exp(x)


def d_tanh(x):
    if isinstance(x, Dual):
        t = d_tanh(x.a)
        one_minus_t2 = _add(1.0, _mul(-1.0, _mul(t, t)))
        return Dual(t, _mul(x.b, one_minus_t2))
    return math.tanh(x)


def d_sin(x):
    if isinstance(x, Dual):
        return Dual(d_sin(x.a), _mul(x.b, d_cos(x.a)))
    return math.sin(x)


def d_cos(x):
    if isinstance(x, Dual):
        return Dual(d_cos(x.a), _mul(-1.0, _mul(x.b, d_sin(x.a))))
    return math.cos(x)


def top_coefficient(x, depth: int) -> float:
    """The coefficient of ε₁⋯ε_depth: follow .b at every level."""
    for _ in range(depth):
        x = x.b if isinstance(x, Dual) else 0.0
    while isinstance(x, Dual):
        x = x.a
    return float(x)


def nested_derivative(phi, u_coeffs: np.ndarray, order: int) -> float:
    """d^order/dt^order  φ(u(t)) at t=0 via `order`-times-nested duals.

    u(t) = Σ u_j t^j is evaluated by Horner in the nested-dual algebra with
    t = the fully mixed nilpotent Σ-style seed built by nesting Dual(t, 1)
    `order` times; the top mixed coefficient of the result is exactly the
    order-th derivative (each εᵢ contributes one d/dt).
    """
    t = 0.0
    for _ in range(order):
        t = Dual(t, 1.0)
    acc = 0.0
    for c in u_coeffs[::-1]:
        acc = _add(_mul(acc, t), float(c))
    return top_coefficient(phi(acc), order)


# ─────────────────────────────────────────────────────────────────────────────
# The §3.2 recurrences (Taylor-coefficient convention: w_j = φ(u)⁽ʲ⁾/j!).
# ─────────────────────────────────────────────────────────────────────────────


def jet_exp(u: np.ndarray) -> np.ndarray:
    k = len(u) - 1
    w = np.zeros(k + 1)
    w[0] = math.exp(u[0])
    for n in range(1, k + 1):
        w[n] = sum(j * u[j] * w[n - j] for j in range(1, n + 1)) / n
    return w


def jet_tanh(u: np.ndarray) -> np.ndarray:
    k = len(u) - 1
    w = np.zeros(k + 1)
    w[0] = math.tanh(u[0])
    for n in range(1, k + 1):
        s = np.convolve(w[:n], w[:n])[: n]  # (w²) coefficients 0..n-1
        w[n] = sum(
            j * u[j] * ((1.0 if n - j == 0 else 0.0) - s[n - j])
            for j in range(1, n + 1)
        ) / n
    return w


def jet_sin_cos(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = len(u) - 1
    s, c = np.zeros(k + 1), np.zeros(k + 1)
    s[0], c[0] = math.sin(u[0]), math.cos(u[0])
    for n in range(1, k + 1):
        s[n] = sum(j * u[j] * c[n - j] for j in range(1, n + 1)) / n
        c[n] = -sum(j * u[j] * s[n - j] for j in range(1, n + 1)) / n
    return s, c


def _check_recurrence(name, jet_fn, phi, rng, k=5):
    u = rng.standard_normal(k + 1) * 0.5
    w = jet_fn(u)
    for order in range(k + 1):
        ref = nested_derivative(phi, u, order) if order else float(
            np.atleast_1d(np.asarray(phi(float(u[0]))))[0]
        )
        got = w[order] * math.factorial(order)
        assert abs(got - ref) < 1e-8 * max(1.0, abs(ref)), (
            f"{name}: order {order}: recurrence {got} vs nested {ref}"
        )
    check(f"§3.2 {name} recurrence == nested duals through order {k}")


# ─────────────────────────────────────────────────────────────────────────────
# §3.1 — the explicit 2^k algebra as subset-indexed arrays, the diagonal
# embedding, and the negative fixture.
# ─────────────────────────────────────────────────────────────────────────────


def tensor_mul(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Product in ⊗ᵢ ℝ[εᵢ]/(εᵢ²), components indexed by subset bitmask."""
    out = np.zeros(1 << k)
    for A in range(1 << k):
        if a[A] == 0.0:
            continue
        rest = ((1 << k) - 1) ^ A
        # enumerate B disjoint from A
        B = rest
        while True:
            out[A | B] += a[A] * b[B]
            if B == 0:
                break
            B = (B - 1) & rest
    return out


def epsilon_sum(k: int) -> np.ndarray:
    v = np.zeros(1 << k)
    for i in range(k):
        v[1 << i] = 1.0
    return v


def diag_embed(c: np.ndarray, k: int) -> np.ndarray:
    """Δ(Σ c_j ε^j)[S] = c_{|S|} · |S|!  — the §3.1 embedding."""
    out = np.zeros(1 << k)
    for S in range(1 << k):
        j = bin(S).count("1")
        if j < len(c):
            out[S] = c[j] * math.factorial(j)
    return out


def jet_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Truncated Cauchy product in ℝ[ε]/(ε^{k+1}), k+1 = len(a)."""
    k1 = len(a)
    out = np.zeros(k1)
    for i in range(k1):
        for j in range(k1 - i):
            out[i + j] += a[i] * b[j]
    return out


def main() -> None:
    rng = np.random.default_rng(20260818)

    # 1–3: the scalar recurrences vs nested duals
    _check_recurrence("exp", jet_exp, d_exp, rng)
    _check_recurrence("tanh", jet_tanh, d_tanh, rng)

    u = rng.standard_normal(6) * 0.5
    s, c = jet_sin_cos(u)
    for order in range(6):
        ref_s = nested_derivative(d_sin, u, order)
        ref_c = nested_derivative(d_cos, u, order)
        assert abs(s[order] * math.factorial(order) - ref_s) < 1e-8 * max(1, abs(ref_s))
        assert abs(c[order] * math.factorial(order) - ref_c) < 1e-8 * max(1, abs(ref_c))
    check("§3.2 sin/cos joint pair recurrence == nested duals through order 5")

    # 4: the k=1 rows are the familiar first-order JVPs
    x, v = 0.37, 1.9
    u1 = np.array([x, v])
    assert abs(jet_exp(u1)[1] - v * math.exp(x)) < 1e-12
    assert abs(jet_tanh(u1)[1] - v * (1 - math.tanh(x) ** 2)) < 1e-12
    s1, c1 = jet_sin_cos(u1)
    assert abs(s1[1] - v * math.cos(x)) < 1e-12
    assert abs(c1[1] + v * math.sin(x)) < 1e-12
    check("§3.2 k=1 rows reproduce the hand-written first-order JVP forms")

    # 5: jet-matmul == truncated polynomial matrix product
    k = 4
    A = [rng.standard_normal((3, 4)) for _ in range(k + 1)]
    B = [rng.standard_normal((4, 2)) for _ in range(k + 1)]
    cauchy = [
        sum(A[i] @ B[n - i] for i in range(n + 1)) for n in range(k + 1)
    ]
    # independent reference: full polynomial product entrywise, then truncate
    for n in range(k + 1):
        ref = np.zeros((3, 2))
        for i in range(k + 1):
            for j in range(k + 1):
                if i + j == n:
                    ref += A[i] @ B[j]
        np.testing.assert_allclose(cauchy[n], ref, atol=1e-10)
    check("§3.3 jet-matmul coefficients == truncated polynomial matrix product")

    # 6: product count
    n_products = sum(1 for i in range(k + 1) for j in range(k + 1) if i + j <= k)
    assert n_products == (k + 1) * (k + 2) // 2, n_products
    check("§3.3 coefficient-product count through order k == (k+1)(k+2)/2")

    # 7: pigeonhole nilpotency
    for kk in (2, 3, 4):
        e = epsilon_sum(kk)
        p = e.copy()
        for _ in range(kk):  # p = e^{kk+1}
            p = tensor_mul(p, e, kk)
        assert np.all(p == 0.0), f"(Σε)^{kk+1} ≠ 0 at k={kk}"
    check("§3.1 pigeonhole: (ε₁+…+ε_k)^{k+1} = 0 in the nested algebra")

    # 8: injectivity fuel — (Σεᵢ)^j = j!·e_j ≠ 0 for j ≤ k
    kk = 4
    e = epsilon_sum(kk)
    p = np.zeros(1 << kk)
    p[0] = 1.0
    for j in range(1, kk + 1):
        p = tensor_mul(p, e, kk)
        for S in map(lambda comb: sum(1 << i for i in comb),
                     combinations(range(kk), j)):
            assert p[S] == math.factorial(j), (j, S, p[S])
        assert np.any(p != 0.0)
    check("§3.1 (Σεᵢ)^j = j!·e_j ≠ 0 for j ≤ k (injectivity of Δ)")

    # 9: Δ is an algebra homomorphism
    kk = 4
    for trial in range(5):
        a = rng.standard_normal(kk + 1)
        b = rng.standard_normal(kk + 1)
        lhs = tensor_mul(diag_embed(a, kk), diag_embed(b, kk), kk)
        rhs = diag_embed(jet_mul(a, b), kk)
        np.testing.assert_allclose(lhs, rhs, atol=1e-9)
    check("§3.1 Δ(a)·Δ(b) = Δ(a·b): the diagonal embedding is a homomorphism")

    # 10: NEGATIVE — the claimed surjection εᵢ ↦ ε is not an algebra map.
    # An algebra map must send the ideal generator εᵢ² to 0; its image is ε²,
    # which is NONZERO in ℝ[ε]/(ε^{k+1}) for every k ≥ 2.
    for kk in (2, 3, 5):
        eps = np.zeros(kk + 1)
        eps[1] = 1.0
        eps_sq = jet_mul(eps, eps)
        assert np.any(eps_sq != 0.0), f"ε² unexpectedly 0 at k={kk}"
        assert eps_sq[2] == 1.0
    check("§3.1 NEGATIVE (#10a): εᵢ ↦ ε is not an algebra map — ε² ≠ 0 for k ≥ 2")

    # 11: end-to-end Law-4 seed — jet == nested duals on the diagonal, with
    # factorial bookkeeping, through a two-op composition tanh(exp(·)).
    kk = 4
    u = rng.standard_normal(kk + 1) * 0.4
    w = jet_tanh(jet_exp(u))
    for order in range(kk + 1):
        ref = nested_derivative(lambda t: d_tanh(d_exp(t)), u, order)
        got = w[order] * math.factorial(order)
        assert abs(got - ref) < 1e-7 * max(1.0, abs(ref)), (order, got, ref)
    check("§3.1/Law 4 seed: composed jets == k-nested duals (diagonal, factorial-scaled)")

    print(f"\n{len(CHECKS)}/{len(CHECKS)} checks passed.")


if __name__ == "__main__":
    main()
