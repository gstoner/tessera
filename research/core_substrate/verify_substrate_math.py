#!/usr/bin/env python3
"""Machine-check every load-bearing mathematical claim CORE_SUBSTRATE_VIEW.md
inherits from its six sources that was previously PROSE-ONLY.

Scope discipline: game theory (research/game_theory/, 27 checks), PDE/stencil
(tests/unit/test_pde_stencil_model.py, 78 assertions), and SparDA
(tests/unit/test_sparda_contracts.py) already have executable harnesses and are
NOT re-verified here. What had none: the CAKE statistics (compiler_enhancement
§2, §7 Phase 4 gate) and the TileRT models (TILERT_ASSESSMENT §1 M1-M5), plus
two arithmetic spot-checks the view repeats from the game-theory plan.

Pure numpy + stdlib. Run: python3 research/core_substrate/verify_substrate_math.py
"""

import itertools
import math
import random

import numpy as np

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# CAKE — compiler_enhancement.md §2 (exact small-sample statistics)
# ---------------------------------------------------------------------------

CAKE_SPEEDUP_TREATMENT = (1.041, 1.144, 1.205)   # CAKE IR arm
CAKE_SPEEDUP_CONTROL = (0.852, 0.928, 1.151)     # direct CUDA/PTX arm
CAKE_EVOLVE_TREATMENT = (1.02, 1.89, 2.33)       # hours (lower better)
CAKE_EVOLVE_CONTROL = (3.59, 3.73, 4.34)


def exact_one_sided_p_greater(treatment, control):
    """Exact permutation p-value for H1: treatment stochastically greater.

    n=m=3, so the reported triples ARE the complete samples; enumerate all
    C(6,3)=20 assignments of the pooled values to the treatment arm and count
    those whose rank-sum >= the observed treatment rank-sum.
    """
    pooled = sorted(treatment + control)
    ranks = {v: i + 1 for i, v in enumerate(pooled)}  # all values distinct
    observed = sum(ranks[v] for v in treatment)
    count = 0
    total = 0
    for combo in itertools.combinations(pooled, 3):
        total += 1
        if sum(ranks[v] for v in combo) >= observed:
            count += 1
    return count, total


@check("CAKE §2.1: best-speedup exact one-sided p = 4/20 = 0.20 (arms overlap)")
def cake_speedup_not_significant():
    count, total = exact_one_sided_p_greater(
        CAKE_SPEEDUP_TREATMENT, CAKE_SPEEDUP_CONTROL)
    assert total == math.comb(6, 3) == 20
    assert count == 4, f"expected 4 assignments >= observed rank-sum, got {count}"
    # And the overlap fact: control's best beats treatment's median + 2 of 3 runs.
    assert max(CAKE_SPEEDUP_CONTROL) > sorted(CAKE_SPEEDUP_TREATMENT)[1]
    assert sum(1 for t in CAKE_SPEEDUP_TREATMENT
               if max(CAKE_SPEEDUP_CONTROL) > t) == 2


@check("CAKE §2.1: evolve-time completely separated, p = 1/20 = 0.05")
def cake_evolve_time_separated():
    # complete separation (every treatment run faster than every control run)
    assert max(CAKE_EVOLVE_TREATMENT) < min(CAKE_EVOLVE_CONTROL)
    # one-sided p for 'treatment lower': by symmetry use negated values
    count, total = exact_one_sided_p_greater(
        tuple(-v for v in CAKE_EVOLVE_TREATMENT),
        tuple(-v for v in CAKE_EVOLVE_CONTROL))
    assert (count, total) == (1, 20)


@check("CAKE §2.1: plateau 3/3 vs 0/3 Fisher exact one-sided p = 1/20 = 0.05")
def cake_fisher_plateau():
    # Hypergeometric: P(all 3 plateauers land in the treatment arm | margins)
    p = (math.comb(3, 3) * math.comb(3, 0)) / math.comb(6, 3)
    assert abs(p - 0.05) < 1e-12


@check("CAKE §2.2: family-wise floor — min p = 0.05, Bonferroni 0.15, Holm rejects nothing")
def cake_family_wise():
    min_p = 1 / math.comb(6, 3)
    assert abs(min_p - 0.05) < 1e-12
    bonferroni_floor = 3 * min_p
    assert abs(bonferroni_floor - 0.15) < 1e-12 and bonferroni_floor > 0.05
    # Holm on sorted (0.05, 0.05, 0.20): compare smallest against alpha/3.
    ps = sorted([0.05, 0.05, 0.20])
    alpha = 0.05
    rejections = 0
    for i, p in enumerate(ps):
        if p <= alpha / (len(ps) - i):
            rejections += 1
        else:
            break  # Holm stops at the first failure
    assert rejections == 0, "Holm should stop at step 1 (0.05 > 0.0167)"


@check("CAKE §7 Phase 4: gain = 1/((c_s/c_g) + (1-p)); net-positive iff c_s/c_g < p")
def cake_filter_gate_formula():
    rng = random.Random(7)
    for _ in range(2000):
        n = 1000
        p = rng.uniform(0.01, 0.99)     # rejection fraction
        c_g = rng.uniform(0.5, 50.0)    # GPU measurement cost
        c_s = rng.uniform(0.001, 2 * p * c_g)  # static filter cost (both sides)
        # Direct simulation: filtered pipeline pays c_s for every candidate and
        # c_g for the (1-p) fraction that survives; unfiltered pays c_g for all.
        cost_unfiltered = n * c_g
        cost_filtered = n * c_s + n * (1 - p) * c_g
        gain_sim = cost_unfiltered / cost_filtered
        gain_formula = 1.0 / ((c_s / c_g) + (1.0 - p))
        assert abs(gain_sim - gain_formula) < 1e-9 * max(1.0, gain_formula)
        assert (gain_formula > 1.0) == (c_s / c_g < p)


# ---------------------------------------------------------------------------
# TileRT — TILERT_ASSESSMENT.md §1 (M1-M5)
# ---------------------------------------------------------------------------

@check("TileRT M1: bubble B >= 0 always; B = 0 iff one resource dominates every op")
def tilert_bubble():
    rng = np.random.default_rng(11)
    for _ in range(500):
        w = rng.uniform(0, 10, size=(rng.integers(1, 30), 3))
        bubble = w.max(axis=1).sum() - w.sum(axis=0).max()
        assert bubble >= -1e-12
    # One resource dominates every op -> B == 0.
    w = rng.uniform(0, 5, size=(20, 3))
    w[:, 0] = w.max(axis=1) + rng.uniform(0.1, 1, size=20)
    assert abs(w.max(axis=1).sum() - w.sum(axis=0).max()) < 1e-12
    # Alternating dominant resources (the MoE shape) -> strictly positive B.
    w = np.zeros((6, 3))
    for i in range(6):
        w[i, i % 3] = 10.0
    assert w.max(axis=1).sum() - w.sum(axis=0).max() == 40.0 > 0


@check("TileRT M1: overlap ceiling T_barrier/T_tile <= |R| = 3, approached by resource rotation; 2 resources -> 2")
def tilert_ceiling():
    rng = np.random.default_rng(13)
    for n_res in (2, 3):
        for _ in range(300):
            w = rng.uniform(0, 10, size=(rng.integers(1, 40), n_res))
            t_barrier = w.max(axis=1).sum()
            t_tile_lb = w.sum(axis=0).max()  # T_tile >= max_r W_r
            assert t_barrier / t_tile_lb <= n_res + 1e-9
    # Tightness: n ops rotating one dominant resource, no cross-op deps.
    for n_res in (2, 3):
        n = 3000
        w = np.zeros((n, n_res))
        for i in range(n):
            w[i, i % n_res] = 1.0
        ratio = w.max(axis=1).sum() / w.sum(axis=0).max()
        assert ratio > n_res - 0.01, f"rotation should approach {n_res}, got {ratio}"


@check("TileRT M2: roofline arithmetic — 0.58 ms floor, ~1730 tok/s, ~2.9x headroom")
def tilert_roofline():
    floor_ms = 37 / 64e3 * 1e3          # 37 GB / 64 TB/s
    assert abs(floor_ms - 0.578) < 0.005
    ceiling_toks = 1e3 / floor_ms
    assert abs(ceiling_toks - 1730) < 5
    observed_ms = 1e3 / 600
    assert abs(observed_ms - 1.67) < 0.005
    assert abs(observed_ms / floor_ms - 2.88) < 0.05  # "~2.9x on the table"


@check("TileRT M3: composition/selection non-commutation counterexample is exact")
def tilert_m3():
    # kernels (compute, net); step carries 8 units of network work.
    k_a, k_b, step_net = (10.0, 0.0), (4.0, 4.0), 8.0
    standalone = lambda k: max(k)
    assert standalone(k_b) < standalone(k_a)          # arbiter picks k_B
    makespan = lambda k: max(k[0], step_net + k[1])   # steady-state, resource sums
    assert makespan(k_b) == 12.0 and makespan(k_a) == 10.0
    assert makespan(k_a) < makespan(k_b)              # composition inverts the pick
    assert abs(makespan(k_b) / makespan(k_a) - 1.2) < 1e-12  # "20% faster"


@check("TileRT M4: Graham — greedy list scheduling <= (2 - 1/m) x optimal; bound is tight")
def tilert_graham():
    def greedy_makespan(tasks, m, order):
        loads = [0.0] * m
        for i in order:
            loads[loads.index(min(loads))] += tasks[i]
        return max(loads)

    rng = random.Random(17)
    for _ in range(400):
        m = rng.randint(2, 6)
        tasks = [rng.uniform(0.1, 10) for _ in range(rng.randint(1, 25))]
        lower = max(sum(tasks) / m, max(tasks))  # LB on optimal makespan
        order = list(range(len(tasks)))
        rng.shuffle(order)
        ratio = greedy_makespan(tasks, m, order) / lower
        assert ratio <= 2 - 1 / m + 1e-9
    # Tight family: m(m-1) unit tasks then one task of length m, worst order.
    for m in (2, 3, 5):
        tasks = [1.0] * (m * (m - 1)) + [float(m)]
        worst = greedy_makespan(tasks, m, list(range(len(tasks))))
        assert abs(worst / m - (2 - 1 / m)) < 1e-12   # optimal = m


@check("TileRT M5: MTP alpha = (1-p^(k+1))/(1-p); alpha=2.77 at k=3 -> p ~= 0.76")
def tilert_mtp():
    alpha = lambda p, k: (1 - p ** (k + 1)) / (1 - p)
    # alpha is sum_{i=0..k} p^i (expected accepted tokens incl. the base token)
    for p in (0.3, 0.76, 0.9):
        assert abs(alpha(p, 3) - sum(p ** i for i in range(4))) < 1e-12
    # Solve alpha(p, 3) = 2.77 by bisection; assert the implied p ~= 0.76.
    lo, hi = 0.01, 0.999
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if alpha(mid, 3) < 2.77 else (lo, mid)
    assert abs(lo - 0.76) < 0.005, f"implied p = {lo}"


# ---------------------------------------------------------------------------
# Arithmetic the view repeats from GAME_THEORY_PLAN §6 (its own harness covers
# the numerics; these two closed-form claims are re-derived independently).
# ---------------------------------------------------------------------------

@check("Game theory §6: memory wall 2^n * 4B — n=28 -> 1 GiB, 30 -> 4 GiB, 32 -> 16 GiB")
def gt_memory_wall():
    for n, gib in ((28, 1), (30, 4), (32, 16)):
        assert (2 ** n) * 4 == gib * 2 ** 30


@check("Game theory §6: fp64 digits-gone wall n ~= 53 (half-ulp model, nonneg worst case)")
def gt_fp64_wall():
    # Nonneg game: zeta magnitude ~ 2^n, consumer needs O(1) differences.
    # Absolute half-ulp error of storing 2^n in fp64 is 2^(n-53); differences
    # die when that reaches O(1) -> smallest such n is 53.
    n_wall = min(n for n in range(20, 80) if 2.0 ** (n - 53) >= 1.0)
    assert n_wall == 53
    # Direct float check: at n=53 adjacent fp64 values around 2^53 differ by 2.
    assert np.spacing(np.float64(2.0 ** 53)) == 2.0
    assert np.spacing(np.float64(2.0 ** 52)) == 1.0


# ---------------------------------------------------------------------------

def main():
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks pass")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
