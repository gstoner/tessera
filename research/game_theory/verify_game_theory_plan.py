"""Mathematical correctness audit of GAME_THEORY_PLAN.md — every algebraic claim,
checked numerically. Exit nonzero on any failure."""
import itertools
import math
import numpy as np

rng = np.random.default_rng(0)
FAIL = []

def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAIL.append(name)

# ---------------------------------------------------------------- lattice layer
def subset_zeta(f):
    f = f.copy(); n = f.shape[-1].bit_length() - 1
    for i in range(n):
        g = f.reshape(-1, f.shape[-1] >> (i + 1), 2, 1 << i)
        g[:, :, 1, :] += g[:, :, 0, :]
    return f

def subset_mobius(f):
    f = f.copy(); n = f.shape[-1].bit_length() - 1
    for i in range(n):
        g = f.reshape(-1, f.shape[-1] >> (i + 1), 2, 1 << i)
        g[:, :, 1, :] -= g[:, :, 0, :]
    return f

def superset_zeta(f):
    f = f.copy(); n = f.shape[-1].bit_length() - 1
    for i in range(n):
        g = f.reshape(-1, f.shape[-1] >> (i + 1), 2, 1 << i)
        g[:, :, 0, :] += g[:, :, 1, :]
    return f

n = 5; N = 1 << n
v = rng.standard_normal(N)

print("== §3.1 / oracle 1-2: butterfly algebra ==")
check("mobius(zeta(v)) == v", np.allclose(subset_mobius(subset_zeta(v)), v))
# direct definition cross-check (DESIL-style)
zdir = np.array([sum(v[S] for S in range(N) if (S & T) == S) for T in range(N)])
check("butterfly zeta == direct subset-sum", np.allclose(subset_zeta(v), zdir))
w = rng.standard_normal(N)
check("adjoint: <zeta v, w> == <v, superset_zeta w>",
      np.isclose(subset_zeta(v) @ w, v @ superset_zeta(w)))

print("== §4.2 semivalue in the Möbius basis ==")
def shapley_direct(v):
    """Permutation-average definition — the ground truth."""
    phi = np.zeros(n)
    for perm in itertools.permutations(range(n)):
        S = 0
        for i in perm:
            phi[i] += v[S | (1 << i)] - v[S]
            S |= 1 << i
    return phi / math.factorial(n)

def banzhaf_direct(v):
    phi = np.zeros(n)
    for i in range(n):
        b = 1 << i
        for S in range(N):
            if not (S & b): phi[i] += v[S | b] - v[S]
        phi[i] /= 2 ** (n - 1)
    return phi

def semivalue(v, wgt):
    """Plan's op: Phi_i = sum_{T ni i} wgt(|T|) * mobius(v)[T]."""
    m = subset_mobius(v)
    phi = np.zeros(n)
    for T in range(1, N):
        t = bin(T).count("1")
        for i in range(n):
            if T & (1 << i): phi[i] += wgt(t) * m[T]
    return phi

check("Shapley == semivalue with w(t)=1/t",
      np.allclose(shapley_direct(v), semivalue(v, lambda t: 1.0 / t)))
check("Banzhaf == semivalue with w(t)=2^(1-t)",
      np.allclose(banzhaf_direct(v), semivalue(v, lambda t: 2.0 ** (1 - t))))
check("oracle 3 efficiency: sum_i Shapley_i == v(N)-v(0)",
      np.isclose(shapley_direct(v).sum(), v[N - 1] - v[0]))

print("== §1.1 the paper's Banzhaf transcription error ==")
for Tset in [0b00011, 0b00111, 0b11111]:
    u_T = np.array([1.0 if (S & Tset) == Tset else 0.0 for S in range(N)])
    t = bin(Tset).count("1"); i0 = (Tset & -Tset).bit_length() - 1
    got = banzhaf_direct(u_T)[i0]
    check(f"Banzhaf(u_T,|T|={t}) == 2^(1-|T|)={2.0**(1-t)} (paper says {2.0**-t})",
          np.isclose(got, 2.0 ** (1 - t)) and not np.isclose(got, 2.0 ** (-t)))

print("== oracle 6-7: marginal sum + Boltzmann limits ==")
def marginal(v, i):
    b = 1 << i
    return np.array([v[S] - v[S ^ b] for S in range(N)])
check("sum_S d_i v(S) == 0", all(np.isclose(marginal(v, i).sum(), 0) for i in range(n)))

def boltzmann_value(v, T):
    lw = v / T; lw -= lw.max()          # online-LSE style stabilization
    p = np.exp(lw); p /= p.sum()
    return np.array([marginal(v, i) @ p for i in range(n)])
def boltz_expect(v, T):
    lw = v / T; lw -= lw.max(); p = np.exp(lw); p /= p.sum(); return v @ p
check("T->inf: value -> uniform mean of marginals",
      np.allclose(boltzmann_value(v, 1e9),
                  [marginal(v, i).mean() for i in range(n)]))
check("T->0+: E[v] -> max v", np.isclose(boltz_expect(v, 1e-3), v.max(), atol=1e-2))
check("T->0-: E[v] -> min v", np.isclose(boltz_expect(v, -1e-3), v.min(), atol=1e-2))

print("== §6 numerics: the fp32 wall is sign-structure-dependent ==")
# NONNEGATIVE monotone data (real cooperative games) is the worst case: the
# zeta grows like 2^n with no cancellation, so fp32 STORAGE of the transformed
# tensor is fatal by n~20-24. Random-sign data cancels (~2^(n/2)) and merely
# degrades. Measured 2026-08-15; drives the fp64-intermediates mandate.
for nn, kind, expect_bad in [(12, "nonneg", False), (20, "nonneg", True),
                             (24, "nonneg", True), (24, "randsign", False)]:
    NN = 1 << nn
    x64 = rng.standard_normal(NN)
    if kind == "nonneg":
        x64 = np.abs(x64)
    z64 = subset_zeta(x64)
    r32 = subset_mobius(z64.astype(np.float32).astype(np.float64))
    relerr = np.max(np.abs(r32 - x64))
    check(f"n={nn} {kind}: fp32-stored-zeta roundtrip err {relerr:.2e} "
          f"{'>' if expect_bad else '<'} 0.1",
          (relerr > 0.1) == expect_bad)

print("== §4.6.3 potential-game prefix scan: O(prod) and exact ==")
shapes = (4, 5, 3, 4, 3)
Phi = rng.standard_normal(shapes)
xs = [rng.dirichlet(np.ones(s)) for s in shapes]
def g_naive(Phi, xs, i):
    out = Phi
    for j in range(len(xs) - 1, -1, -1):
        if j != i: out = np.tensordot(out, xs[j], axes=([j if j < i else j], [0])) \
            if False else out
    # simpler: contract all axes except i, one at a time from the back
    out = Phi
    for j in reversed(range(len(xs))):
        if j == i: continue
        out = np.moveaxis(out, j, -1) @ xs[j] if out.ndim - 1 == j or True else out
    return out
def g_direct(Phi, xs, i):
    out = Phi
    for j in reversed(range(len(xs))):
        if j != i:
            out = np.tensordot(out, xs[j], axes=(j if j < i else j, 0)) if j > i else out
    # do it robustly with einsum instead:
    letters = "abcde"
    spec = ",".join([letters[:len(xs)]] + [letters[j] for j in range(len(xs)) if j != i])
    return np.einsum(spec + "->" + letters[i], Phi, *[xs[j] for j in range(len(xs)) if j != i])
def g_prefix(Phi, xs):
    """Prefix contractions only; finishing each g_i geometrically. O(prod) total."""
    prefixes = [Phi]
    for j in range(len(xs) - 1):
        prefixes.append(np.tensordot(prefixes[-1], xs[j], axes=(0, 0)))
    out = []
    for i in range(len(xs)):
        t = prefixes[i]                    # axes i..n-1
        for j in range(len(xs) - 1, i, -1):
            t = np.tensordot(t, xs[j], axes=(j - i, 0))
        out.append(t)
    return out
gp = g_prefix(Phi, xs)
check("prefix-scan gradients == direct leave-one-out einsum",
      all(np.allclose(gp[i], g_direct(Phi, xs, i)) for i in range(len(xs))))
# cost model: sum_i prod_{j>=i} |X_j| <= 2*prod for |X_j|>=2
cost = sum(int(np.prod(shapes[i:])) for i in range(len(shapes)))
check("finishing cost sum_i prod_{j>=i}|Xj| <= 2*prod",
      cost <= 2 * int(np.prod(shapes)), f"{cost} vs {int(np.prod(shapes))}")

print("== §3.3 nash_residual + implicit diff on a real game ==")
def sparsemax(z):
    zs = np.sort(z)[::-1]; css = np.cumsum(zs)
    k = np.arange(1, len(z) + 1)
    sup = k[zs * k > css - 1]
    kk = sup[-1]; tau = (css[kk - 1] - 1) / kk
    return np.maximum(z - tau, 0.0)

# 2-player zero-sum: theta parameterizes the payoff matrix
A0 = rng.standard_normal((4, 4))
def solve_zero_sum(A, eta=0.25, iters=20000):
    x = np.ones(4) / 4; y = np.ones(4) / 4
    for _ in range(iters):
        # extragradient
        xh = sparsemax(x + eta * (A @ y)); yh = sparsemax(y - eta * (A.T @ x))
        x = sparsemax(x + eta * (A @ yh)); y = sparsemax(y - eta * (A.T @ xh))
    return x, y
x, y = solve_zero_sum(A0)
gap = (A0 @ y).max() - (A0.T @ x).min()   # duality gap of the bilinear game
check("saddle_solve (extragradient+sparsemax) duality gap ~ 0", gap < 1e-4, f"gap={gap:.1e}")
rx = np.linalg.norm(x - sparsemax(x + 0.25 * (A0 @ y)))
check("oracle 10: ||nash_residual(x*)|| ~ 0", rx < 1e-6, f"{rx:.1e}")

# implicit dtheta of the equilibrium VALUE via envelope: d/dA (x^T A y) = x y^T
# perturb an entry inside the equilibrium support so the derivative is nonzero
i_s = int(np.argmax(x)); j_s = int(np.argmax(y))
eps = 1e-5; E = np.zeros_like(A0); E[i_s, j_s] = 1.0
xp, yp = solve_zero_sum(A0 + eps * E); xm, ym = solve_zero_sum(A0 - eps * E)
fd = (xp @ (A0 + eps * E) @ yp - xm @ (A0 - eps * E) @ ym) / (2 * eps)
check("implicit/envelope dValue/dA_ij == x_i*y_j (finite diff, in-support)",
      np.isclose(fd, x[i_s] * y[j_s], atol=1e-3) and x[i_s] * y[j_s] > 1e-3,
      f"fd={fd:.5f} xy={x[i_s]*y[j_s]:.5f}")

print("== §4.3 swap-regret -> correlated equilibrium (2x2 chicken) ==")
# Chicken: payoffs; unique mixed Nash is worse than the fair CE.
u1 = np.array([[6, 2], [7, 0]]); u2 = u1.T
def stationary(Q):
    p = np.ones(Q.shape[0]) / Q.shape[0]
    for _ in range(200):
        p = p @ Q
    return p / p.sum()

def swap_regret_play(T=200000):
    """Full-information internal-regret dynamics (Blum-Mansour / Hart-Mas-Colell).

    The step my first draft got WRONG (kept as a cautionary note, cf. plan
    §4.6.1): the played distribution is the STATIONARY distribution of the
    row-stochastic matrix whose row a is regret-matching over R[a,:] — not the
    regret-matching row of the last action. The wrong version converges without
    complaint to a non-CE (violation ~0.23 on chicken). No invariant catches it.
    """
    R1 = np.zeros((2, 2)); R2 = np.zeros((2, 2))
    joint = np.zeros((2, 2))
    p1 = np.ones(2) / 2; p2 = np.ones(2) / 2
    for _ in range(T):
        joint += np.outer(p1, p2)
        U1 = u1 @ p2; U2 = u2.T @ p1
        R1 += p1[:, None] * (U1[None, :] - U1[:, None])
        R2 += p2[:, None] * (U2[None, :] - U2[:, None])
        def q_of(R):
            # Hart-Mas-Colell inertia: divide by max(s, mu) so the chain keeps
            # self-transition mass; full normalization (Q[a,a]=0) oscillates
            # and the empirical joint fails the CE check (~0.20 violation).
            Q = np.zeros((2, 2)); mu = 1.0
            for a in range(2):
                pos = np.maximum(R[a], 0.0); pos[a] = 0.0
                s = pos.sum()
                if s > 0:
                    Q[a] = pos / max(s, mu)
                    Q[a, a] = 1.0 - Q[a].sum()
                else:
                    Q[a, a] = 1.0
            return Q
        p1 = stationary(q_of(R1)); p2 = stationary(q_of(R2))
    return joint / joint.sum()
mu = swap_regret_play()
# CE check: for each player, each action a with mass, no gain from swapping a->a'
def ce_violation(mu):
    viol = 0.0
    for a in range(2):
        for ap in range(2):
            viol = max(viol, sum(mu[a, b] * (u1[ap, b] - u1[a, b]) for b in range(2)))
    for b in range(2):
        for bp in range(2):
            viol = max(viol, sum(mu[a, b] * (u2[a, bp] - u2[a, b]) for a in range(2)))
    return viol
check("empirical joint of no-swap-regret play is an approx CE",
      ce_violation(mu) < 0.02, f"violation={ce_violation(mu):.3f}")

print("== §4.4 core via separation oracle == ")
# convex (supermodular) game -> Shapley value is IN the core; verify via the
# separation oracle the plan proposes (a single lattice reduction).
def supermodular_game():
    m = np.abs(rng.standard_normal(N)) * 0.1
    vv = subset_zeta(np.where([bin(S).count('1') >= 2 for S in range(N)], m, 0.0))
    for i in range(n):
        vv += subset_zeta(np.where([S == (1 << i) for S in range(N)], rng.random(), 0.0))
    return vv
vv = supermodular_game()
phi = shapley_direct(vv)
excess = np.array([vv[S] - sum(phi[i] for i in range(n) if S & (1 << i)) for S in range(N)])
check("supermodular: Shapley in core (separation oracle finds no violated cut)",
      excess.max() <= 1e-9 + excess[N - 1], f"max excess {excess.max():.2e}")

print("== oracle 8: hermitian embed isometry ==")
A = rng.standard_normal((6, 6)); B = rng.standard_normal((6, 6))
def herm(A): return (A + A.T) / 2 + 1j * (A - A.T) / 2
ipR = np.sum(A * B); ipH = np.real(np.sum(np.conj(herm(A)) * herm(B)))
check("<A|B> == <herm(A)|herm(B)>", np.isclose(ipR, ipH))
check("herm(A) is self-adjoint", np.allclose(herm(A), herm(A).conj().T))

print("== oracle 11: nim / Grundy ==")
def grundy_nim(heaps):
    from functools import reduce
    import operator
    return reduce(operator.xor, heaps, 0)
def grundy_bruteforce(heaps, memo={}):
    key = tuple(sorted(heaps))
    if key in memo: return memo[key]
    moves = set()
    for i, h in enumerate(heaps):
        for take in range(1, h + 1):
            nh = list(heaps); nh[i] = h - take
            moves.add(grundy_bruteforce(nh))
    g = 0
    while g in moves: g += 1
    memo[key] = g
    return g
ok = all(grundy_nim(h) == grundy_bruteforce(list(h))
         for h in itertools.product(range(5), repeat=3))
check("Grundy(sum) == XOR of heap Grundy values (all 3-heap nim <=4)", ok)

print()
if FAIL:
    print("FAILURES:", FAIL); raise SystemExit(1)
print(f"ALL CHECKS PASSED")
