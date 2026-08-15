"""Diagnose the two failures from the main verification run."""
import numpy as np
rng = np.random.default_rng(0)

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

print("== Failure 1 diagnosis: sign structure decides the fp32 wall ==")
print("   (random-sign data cancels; NONNEGATIVE data — i.e. real cooperative")
print("    games — is the worst case: zeta grows like 2^n, no cancellation)")
for nn in (12, 16, 20, 24):
    NN = 1 << nn
    for kind in ("random-sign", "nonneg"):
        x = rng.standard_normal(NN)
        if kind == "nonneg":
            x = np.abs(x)
        z = subset_zeta(x)
        r = subset_mobius(z.astype(np.float32).astype(np.float64))
        err = np.max(np.abs(r - x))
        print(f"   n={nn:2d} {kind:12s} zeta_max={np.max(np.abs(z)):9.3e}  "
              f"roundtrip_err={err:9.3e}")

print()
print("== Failure 2 diagnosis: my first swap-regret loop skipped the ==")
print("   stationary-distribution step of Blum-Mansour. Correct algorithm:")
print("   p = stationary(Q) where Q's row a = regret-matching over R[a,:]")
u1 = np.array([[6., 2.], [7., 0.]]); u2 = u1.T   # chicken

def stationary(Q):
    # power iteration on the row-stochastic Q
    p = np.ones(Q.shape[0]) / Q.shape[0]
    for _ in range(200):
        p = p @ Q
    return p / p.sum()

def internal_regret_play(T=200000):
    R1 = np.zeros((2, 2)); R2 = np.zeros((2, 2))
    joint = np.zeros((2, 2))
    p1 = np.ones(2) / 2; p2 = np.ones(2) / 2
    for t in range(T):
        joint += np.outer(p1, p2)          # expected-play accumulation
        # full-information utility vectors given opponent mixture
        U1 = u1 @ p2                        # utility of each of my actions
        U2 = u2.T @ p1
        # internal regret update: R[a,a'] += p[a] * (U[a'] - U[a])
        R1 += np.outer(p1, np.ones(2)) * (U1[None, :] - U1[:, None] > -1e18) * (U1[None, :] - U1[:, None])
        R2 += np.outer(p2, np.ones(2)) * (U2[None, :] - U2[:, None])
        def q_of(R):
            Q = np.zeros((2, 2))
            for a in range(2):
                pos = np.maximum(R[a], 0.0); pos[a] = 0.0
                s = pos.sum()
                if s > 0:
                    mu = 1.0  # normalization mass
                    Q[a] = pos / max(s, mu)
                    Q[a, a] = 1.0 - Q[a].sum()
                else:
                    Q[a, a] = 1.0
            return Q
        p1 = stationary(q_of(R1)); p2 = stationary(q_of(R2))
    return joint / joint.sum()

mu = internal_regret_play()

def ce_violation(mu):
    viol = 0.0
    for a in range(2):
        for ap in range(2):
            viol = max(viol, sum(mu[a, b] * (u1[ap, b] - u1[a, b]) for b in range(2)))
    for b in range(2):
        for bp in range(2):
            viol = max(viol, sum(mu[a, b] * (u2[a, bp] - u2[a, b]) for a in range(2)))
    return viol

v = ce_violation(mu)
print(f"   corrected Blum-Mansour: joint=\n{np.round(mu,3)}\n   CE violation = {v:.4f}")
print(f"   {'PASS' if v < 0.02 else 'FAIL'}  corrected reduction converges to a CE")
