#!/usr/bin/env python3
"""Matrix calculus in Tessera — a runnable tour of the linear-operator view.

Companion to Bright, Edelman & Johnson, *Matrix Calculus (for Machine Learning
and Beyond)* (arXiv:2501.14787), the MIT 18.063 notes. Each section maps a
result from the notes onto the Tessera surface that already implements it, and
checks the result numerically rather than asserting it.

The one idea underneath every section:

    a derivative is the LINEAR OPERATOR f'(x) that maps a small change dx in
    the input to the first-order change df = f'(x)[dx] in the output.

Everything else — gradients, Jacobians, forward vs reverse mode, adjoint
methods, Hessians — is that one statement specialized to a vector space, an
inner product, or an association order.

Run:  python3 examples/matrix_calculus/matrix_calculus_tutorial.py
Host: any (pure Python/numpy reference lane; no device required).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Allow running straight from a source checkout.
_repo_python = Path(__file__).resolve().parents[2] / "python"
if _repo_python.is_dir() and str(_repo_python) not in sys.path:
    sys.path.insert(0, str(_repo_python))

import tessera  # noqa: E402
from tessera import ops  # noqa: E402
from tessera.autodiff import grad, jacfwd, jacrev, jvp  # noqa: E402


def banner(n: int, title: str, ref: str) -> None:
    print()
    print("=" * 78)
    print(f"  {n}. {title}")
    print(f"     notes: {ref}")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# 1. The derivative is a linear operator, not a number and not "2A"
#    §2.6:  f(A) = A²  ⇒  df = dA·A + A·dA.  This is NOT 2A·dA, because dA
#    and A do not commute. Tessera's JVP rule for `matmul` carries exactly
#    the product rule, so `jvp` gives f'(A)[dA] with no hand derivation.
# ─────────────────────────────────────────────────────────────────────────────
def section_1_linear_operator(rng: np.random.Generator) -> None:
    banner(1, "The derivative is a linear operator", "§2.2, §2.6")

    A = rng.standard_normal((4, 4))
    dA = rng.standard_normal((4, 4))

    square = lambda X: ops.matmul(X, X)  # noqa: E731
    value, tangent = jvp(square, A, dA)

    print(f"  f(A) = A@A                      matches A@A : {np.allclose(value, A @ A)}")
    print(f"  f'(A)[dA] == dA@A + A@dA        : {np.allclose(tangent, dA @ A + A @ dA)}")
    print(f"  f'(A)[dA] == 2*A@dA (the trap)  : {np.allclose(tangent, 2 * A @ dA)}")
    print()
    print("  The operator f'(A) has a 16x16 Jacobian if you insist on one")
    print("  (§3.3, via vec/Kronecker), but you never need to build it: the")
    print("  rule `dA·A + A·dA` IS the operator, and it costs two GEMMs.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Directional derivative == f'(x)[v], and finite differences check it
#    §2.2.1: f'(x)[v] = d/dα f(x + αv) at α=0. So ONE pair of function
#    evaluations along a random direction v tests the whole operator —
#    no per-coordinate sweep, and it works for matrix-valued inputs where
#    "one coordinate at a time" is the wrong unit anyway.
# ─────────────────────────────────────────────────────────────────────────────
def section_2_directional(rng: np.random.Generator) -> None:
    banner(2, "Directional derivative = one basis-free finite difference",
           "§2.2.1, §4")

    A = rng.standard_normal((4, 4))
    V = rng.standard_normal((4, 4))
    f = lambda X: ops.matmul(ops.tanh(X), X)  # noqa: E731
    _, exact = jvp(f, A, V)

    s = 1e-6
    central = (f(A + s * V) - f(A - s * V)) / (2 * s)
    rel = np.linalg.norm(central - exact) / np.linalg.norm(exact)
    print(f"  ||FD - f'(A)[V]|| / ||f'(A)[V]||  = {rel:.3e}   (2 evaluations)")
    print(f"  a coordinate-wise check of the same operator would need "
          f"{A.size} pairs")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Order of accuracy: truncation vs roundoff
#    §4.4–4.6. Forward differences are 1st-order (error ∝ s); central are
#    2nd-order (error ∝ s²) — until roundoff takes over near s ≈ sqrt(eps).
#    Reading the SLOPE is a far stronger test than a fixed tolerance: it is
#    scale-free, dtype-honest, and it fails loudly when a rule is subtly wrong.
# ─────────────────────────────────────────────────────────────────────────────
def section_3_order_of_accuracy(rng: np.random.Generator) -> None:
    banner(3, "Order of accuracy: truncation error vs roundoff", "§4.4–4.6")

    A = rng.standard_normal((4, 4))
    V = rng.standard_normal((4, 4))
    f = lambda X: ops.matmul(ops.sin(X), X)  # noqa: E731
    _, exact = jvp(f, A, V)
    den = np.linalg.norm(exact)

    print(f"  {'step s':>10}  {'forward rel err':>16}  {'central rel err':>16}")
    for e in range(0, 15, 2):
        s = 10.0 ** (-e)
        fwd = (f(A + s * V) - f(A)) / s
        ctr = (f(A + s * V) - f(A - s * V)) / (2 * s)
        print(f"  {s:10.0e}  {np.linalg.norm(fwd - exact) / den:16.3e}"
              f"  {np.linalg.norm(ctr - exact) / den:16.3e}")

    eps = np.finfo(np.float64).eps
    print()
    print("  forward error falls like s, central like s^2, then BOTH rise:")
    print(f"  catastrophic cancellation below s ~ sqrt(eps) = {np.sqrt(eps):.2e}.")
    print("  Rule of thumb from the notes: use ||dx|| ~ sqrt(eps)*||x||,")
    print(f"  i.e. s ~ {np.sqrt(eps) * np.linalg.norm(A):.2e} here — NOT a fixed 1e-4.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. The gradient lives in an inner product
#    §5.1. For scalar-valued f, f'(x) is a linear FORM; the gradient is the
#    thing you take an inner product with to recover df. On matrices the
#    default is Frobenius, <A,B> = tr(AᵀB), which is why ∇f has the same
#    shape as A. Change the inner product and you change the gradient.
# ─────────────────────────────────────────────────────────────────────────────
def section_4_inner_product(rng: np.random.Generator) -> None:
    banner(4, "The gradient is defined by an inner product", "§5.1")

    A = rng.standard_normal((5, 3))

    def frobenius_norm(X):
        return ops.sqrt(ops.reduce(ops.mul(X, X), op="sum"))

    g = grad(frobenius_norm)(A)
    closed_form = A / np.linalg.norm(A)
    print(f"  grad ||A||_F  ==  A / ||A||_F   : {np.allclose(g, closed_form)}")

    dA = rng.standard_normal((5, 3)) * 1e-7
    df_pred = float(np.sum(g * dA))                     # <grad f, dA>_F
    df_true = float(frobenius_norm(A + dA) - frobenius_norm(A))
    print(f"  <grad f, dA>_F = {df_pred: .12e}")
    print(f"  f(A+dA)-f(A)   = {df_true: .12e}")
    print()
    print("  Under a weighted inner product <x,y>_W = x^T W y the SAME")
    print("  differential gives grad_W f = W^-1 grad f (§5.1). That is the")
    print("  whole content of preconditioning, natural gradient, and Muon:")
    print("  a different metric, not a different derivative.")
    W = np.diag(rng.uniform(0.5, 2.0, size=A.size))
    g_w = (np.linalg.inv(W) @ g.reshape(-1)).reshape(A.shape)
    print(f"  ||grad_W f - grad f|| = {np.linalg.norm(g_w - g):.3e} "
          f"(same df, different vector)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Forward vs reverse is an ASSOCIATION ORDER, and the cost is not symmetric
#    §2.5.1, §8.4. (a'b')c' vs a'(b'c'). Reverse costs one forward pass plus
#    one backward sweep per OUTPUT; forward costs one pass per INPUT.
# ─────────────────────────────────────────────────────────────────────────────
def section_5_forward_vs_reverse(rng: np.random.Generator) -> None:
    banner(5, "Forward vs reverse mode = which end you associate from",
           "§2.5.1, §8.4")

    evals = [0]

    def many_in_one_out(x):        # R^32 -> R    (the ML/optimization regime)
        evals[0] += 1
        return ops.reduce(ops.mul(ops.tanh(x), ops.tanh(x)), op="sum")

    x = rng.standard_normal(32)

    evals[0] = 0
    t0 = time.perf_counter()
    Jr = jacrev(many_in_one_out)(x)
    t_rev = time.perf_counter() - t0
    n_rev = evals[0]

    evals[0] = 0
    t0 = time.perf_counter()
    Jf = jacfwd(many_in_one_out)(x)
    t_fwd = time.perf_counter() - t0
    n_fwd = evals[0]

    print("  f: R^32 -> R   (32 inputs, 1 output)")
    print(f"    jacrev : {n_rev:3d} forward evaluations + 1 backward sweep"
          f"   [{1e3 * t_rev:6.2f} ms]")
    print(f"    jacfwd : {n_fwd:3d} forward evaluations"
          f"                     [{1e3 * t_fwd:6.2f} ms]")
    print(f"    same answer: {np.allclose(Jr, Jf)}")
    print()
    print("  This asymmetry is the entire reason backpropagation exists")
    print("  (§2.5.1): with n inputs and 1 output, left-to-right is Theta(n^2)")
    print("  and right-to-left is Theta(n^3). Flip the shape and the ranking")
    print("  flips with it.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Adjoint differentiation — the whole point, on a real Tessera solver
#    §6.3.3 is literally this problem: A(p) tridiagonal symmetric, and
#         g(p) = (c^T A(p)^-1 b)^2 .
#    The notes derive ∂g/∂p_k = v_k x_{k+1} + v_{k+1} x_k from TWO solves
#    (one forward, one transposed). `tessera.ops.tridiagonal_solve` ships a
#    VJP that is exactly that transposed solve, so Tessera's reverse mode
#    reproduces the hand derivation — and stays O(n), not O(n) solves.
# ─────────────────────────────────────────────────────────────────────────────
def section_6_adjoint_method(rng: np.random.Generator) -> None:
    banner(6, "Adjoint differentiation: two solves, not n solves", "§6.3")

    n = 8
    a = rng.uniform(2.0, 3.0, size=n)      # main diagonal (diagonally dominant)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)

    def g(p):
        # A(p) = tridiag(p, a, p). Storage convention: du[i] = A[i,i+1],
        # dl[i] = A[i,i-1]; du[n-1] and dl[0] are unused by the recurrence.
        du = p
        dl = ops.roll(p, shift=1)
        x = ops.tridiagonal_solve(dl, a, du, b)
        s = ops.reduce(ops.mul(c, x), op="sum")
        return ops.mul(s, s)

    p_free = rng.uniform(-0.3, 0.3, size=n - 1)
    p = np.concatenate([p_free, [0.0]])    # last slot is structurally unused

    g_ad = grad(g)(p)

    # The notes' hand-derived adjoint, using dense linear algebra as an oracle.
    A = np.diag(a) + np.diag(p_free, 1) + np.diag(p_free, -1)
    x = np.linalg.solve(A, b)                       # forward solve
    v = np.linalg.solve(A.T, -2.0 * (c @ x) * c)    # adjoint solve
    hand = np.array([v[k] * x[k + 1] + v[k + 1] * x[k] for k in range(n - 1)])

    print(f"  g(p) = {float(g(p)):.10f}")
    print(f"  max |AD - hand-derived adjoint| = "
          f"{np.max(np.abs(g_ad[:n - 1] - hand)):.3e}")
    print(f"  gradient of the structurally unused slot p[{n-1}] = {g_ad[-1]:+.1e}")

    dp = rng.standard_normal(n)
    dp[-1] = 0.0
    s = 1e-7
    fd = float((g(p + s * dp) - g(p - s * dp)) / (2 * s))
    print(f"  directional check:  FD = {fd: .10e}")
    print(f"                      AD = {float(g_ad @ dp): .10e}")
    print()
    print("  Forward mode / finite differences would need one solve PER")
    print("  parameter (§6.3). The adjoint costs one extra solve, total,")
    print("  because A^T of a tridiagonal matrix is still tridiagonal.")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Second derivatives are symmetric BILINEAR maps
#    §12.2. f''(x)[u,v] = f''(x)[v,u], and the notes give a second-difference
#    formula that needs no AD at all:
#         f''(x)[u,v] ~ f(x+u+v) + f(x) - f(x+u) - f(x+v)
#    That is an independent oracle for any Hessian-vector product you compute.
# ─────────────────────────────────────────────────────────────────────────────
def section_7_second_derivatives(rng: np.random.Generator) -> None:
    banner(7, "Second derivatives: a symmetric bilinear map", "§12.2")

    n = 6
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)

    def f(z):
        # Deliberately NOT separable: the cross terms are what make the
        # Hessian genuinely non-diagonal, so symmetry is a real check.
        return ops.mul(
            ops.reduce(ops.sin(z), op="sum"),
            ops.reduce(ops.mul(z, z), op="sum"),
        )

    def second_difference(fn, x0, du, dv, s):
        """The boxed formula of §12.2 — pure function evaluations."""
        return float(
            fn(x0 + s * du + s * dv) + fn(x0) - fn(x0 + s * du) - fn(x0 + s * dv)
        ) / (s * s)

    s = 1e-4
    f_uv = second_difference(f, x, u, v, s)
    f_vu = second_difference(f, x, v, u, s)
    print(f"  f''(x)[u,v] = {f_uv: .8f}")
    print(f"  f''(x)[v,u] = {f_vu: .8f}   symmetry gap {abs(f_uv - f_vu):.2e}")

    # Cross-check against an explicit Hessian, built by differencing the
    # reverse-mode gradient. (Tessera's eager `hvp` does the same thing; an
    # exact forward-over-reverse HVP lives on the compiled path.)
    eps = 1e-5
    gradf = grad(f)
    hess = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        hess[:, i] = (gradf(x + e) - gradf(x - e)) / (2 * eps)
    print(f"  Hessian symmetry ||H - H^T||    = "
          f"{np.linalg.norm(hess - hess.T):.3e}")
    print(f"  u^T H v vs the difference formula: "
          f"{float(u @ hess @ v): .8f} vs {f_uv: .8f}")
    print()
    print("  Symmetry is free (§12.2: it follows from '+' being commutative),")
    print("  which makes it a law any Hessian path can be tested against —")
    print("  including a compiled forward-over-reverse HVP.")


def main() -> int:
    rng = np.random.default_rng(20260820)
    print(f"Tessera matrix-calculus tutorial  (tessera {getattr(tessera, '__version__', 'dev')})")
    section_1_linear_operator(rng)
    section_2_directional(rng)
    section_3_order_of_accuracy(rng)
    section_4_inner_product(rng)
    section_5_forward_vs_reverse(rng)
    section_6_adjoint_method(rng)
    section_7_second_derivatives(rng)
    print()
    print("Done. Every number above was computed, not asserted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
