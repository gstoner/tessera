"""G1 oracles for the coalition-lattice family (GAME_THEORY_PLAN.md §7).

Axiom-level metamorphic identities, not transcribed closed forms (§1.1): the
plan's oracle rows 1–6, the §1.1 factor-of-two pin, finite-difference checks of
the declared transposes (the VJPs of the linear primitives), and the fail-closed
shape contracts. Seeded from research/game_theory/verify_game_theory_plan.py
(27/27) — these are the registry-facing versions of those checks.
"""

from __future__ import annotations

import itertools
import math

import pytest

np = pytest.importorskip("numpy")

from tessera.game import (  # noqa: E402
    banzhaf_value,
    banzhaf_weights,
    coalition_marginal,
    lattice_players,
    semivalue,
    shapley_value,
    shapley_weights,
    subset_mobius,
    subset_zeta,
    superset_mobius,
    superset_zeta,
)

N_PLAYERS = 5
SIZE = 1 << N_PLAYERS


@pytest.fixture()
def v():
    return np.random.default_rng(7).standard_normal(SIZE)


# ── oracle 1: the butterflies invert each other, both directions ──────────────

def test_mobius_inverts_zeta(v):
    assert np.allclose(subset_mobius(subset_zeta(v)), v)
    assert np.allclose(subset_zeta(subset_mobius(v)), v)
    assert np.allclose(superset_mobius(superset_zeta(v)), v)


def test_zeta_matches_direct_subset_sum(v):
    direct = np.array([sum(v[S] for S in range(SIZE) if (S & T) == S)
                       for T in range(SIZE)])
    assert np.allclose(subset_zeta(v), direct)


# ── oracle 2: superset zeta is the adjoint (the transpose_rule consumer) ─────

def test_adjoint_identity(v):
    w = np.random.default_rng(11).standard_normal(SIZE)
    assert np.isclose(subset_zeta(v) @ w, v @ superset_zeta(w))
    assert np.isclose(subset_mobius(v) @ w, v @ superset_mobius(w))


# ── oracles 3–4: semivalue axioms, independent of any closed form ────────────

def test_shapley_efficiency(v):
    phi = shapley_value(v)
    assert np.isclose(phi.sum(), v[SIZE - 1] - v[0])


def test_semivalue_symmetry_dummy_additivity(v):
    rng = np.random.default_rng(13)
    # Symmetry: swapping two symmetric players swaps their values. Build a
    # game symmetric in players 0 and 1 by symmetrizing.
    perm = np.arange(SIZE)
    b0, b1 = 1, 2
    swapped = ((perm & ~(b0 | b1))
               | ((perm & b0) << 1) | ((perm & b1) >> 1))
    sym = v + v[swapped]
    phi = shapley_value(sym)
    assert np.isclose(phi[0], phi[1])
    # Dummy: a player contributing nothing gets exactly zero.
    no3 = np.array([sym[S & ~8] for S in range(SIZE)])  # player 3 never matters
    assert np.isclose(shapley_value(no3)[3], 0.0)
    # Additivity: Φ(u + w) = Φ(u) + Φ(w).
    u, w = rng.standard_normal(SIZE), rng.standard_normal(SIZE)
    assert np.allclose(shapley_value(u + w), shapley_value(u) + shapley_value(w))


def test_semivalue_matches_direct_definitions(v):
    # Shapley: permutation-average ground truth.
    phi = np.zeros(N_PLAYERS)
    for perm in itertools.permutations(range(N_PLAYERS)):
        S = 0
        for i in perm:
            phi[i] += v[S | (1 << i)] - v[S]
            S |= 1 << i
    phi /= math.factorial(N_PLAYERS)
    assert np.allclose(shapley_value(v), phi)
    # Banzhaf: subset-average ground truth.
    bz = np.zeros(N_PLAYERS)
    for i in range(N_PLAYERS):
        b = 1 << i
        for S in range(SIZE):
            if not (S & b):
                bz[i] += v[S | b] - v[S]
        bz[i] /= 2 ** (N_PLAYERS - 1)
    assert np.allclose(banzhaf_value(v), bz)


# ── oracle 5: the §1.1 factor-of-two pin ─────────────────────────────────────

def test_banzhaf_unanimity_pins_the_paper_erratum():
    for tset in (0b00011, 0b00111, 0b11111):
        u_t = np.array([1.0 if (S & tset) == tset else 0.0
                        for S in range(SIZE)])
        t = bin(tset).count("1")
        i0 = (tset & -tset).bit_length() - 1
        got = banzhaf_value(u_t)[i0]
        assert np.isclose(got, 2.0 ** (1 - t))       # the correct constant
        assert not np.isclose(got, 2.0 ** (-t))      # the text's transcription


# ── oracle 6: marginal sums vanish ⟹ uniform-π value is zero ────────────────

def test_marginal_sum_is_zero(v):
    marg = coalition_marginal(v)
    assert marg.shape == (N_PLAYERS, SIZE)
    assert np.allclose(marg.sum(axis=-1), 0.0)


# ── transposes are the true adjoints (finite-difference VJP checks) ──────────

@pytest.mark.parametrize("prim", [subset_zeta, subset_mobius,
                                  superset_zeta, superset_mobius])
def test_butterfly_transpose_is_adjoint(prim, v):
    dout = np.random.default_rng(17).standard_normal(SIZE)
    # ⟨f(v), dout⟩ == ⟨v, fᵀ(dout)⟩ through the REGISTERED rule.
    grad = prim.transpose_rule(dout, v)
    assert np.isclose(prim(v) @ dout, v @ grad)


def test_marginal_transpose_is_adjoint(v):
    rng = np.random.default_rng(19)
    dout = rng.standard_normal((N_PLAYERS, SIZE))
    grad = coalition_marginal.transpose_rule(dout, v)
    assert np.isclose(np.sum(coalition_marginal(v) * dout), v @ grad)


def test_semivalue_transpose_is_adjoint(v):
    rng = np.random.default_rng(23)
    vhat = subset_mobius(v)
    w = shapley_weights(N_PLAYERS)
    dout = rng.standard_normal(N_PLAYERS)
    grad, wgrad = semivalue.transpose_rule(dout, vhat, w)
    assert wgrad is None
    assert np.isclose(semivalue(vhat, w) @ dout, vhat @ grad)


# ── fail-closed contracts (G0): n derived, never defaulted ───────────────────

def test_non_power_of_two_lattice_fails_closed():
    with pytest.raises(ValueError, match="power-of-two"):
        subset_zeta(np.zeros(24))
    with pytest.raises(ValueError, match="power-of-two"):
        lattice_players(np.zeros(0))


def test_wrong_weight_length_fails_closed(v):
    with pytest.raises(ValueError, match=r"\(n\+1,\)"):
        semivalue(subset_mobius(v), np.ones(N_PLAYERS))  # needs n+1


# ── §6 numerics: fp64 ACCUMULATION with storage dtype preserved (#15a) ───────

def test_storage_dtype_preserved_with_fp64_accumulation(v):
    # Storage follows the tensor (Decision #15a); fp64 is the accumulator
    # contract, not a silent storage promotion.
    assert subset_zeta(v.astype(np.float32)).dtype == np.float32
    assert subset_zeta(v).dtype == np.float64
    assert shapley_value(v.astype(np.float32)).dtype == np.float32
    # fp64 accumulation is real: an fp32-storage input still gets exact
    # fp64-computed values (n=5 values are exactly representable deltas).
    got32 = subset_zeta(v.astype(np.float32)).astype(np.float64)
    ref64 = subset_zeta(v.astype(np.float32).astype(np.float64))
    assert np.allclose(got32, ref64, atol=1e-6)


# ── oracle 7 + H5: Boltzmann value limits on an ASYMMETRIC fixture ───────────

def test_boltzmann_uniform_limit(v):
    from tessera.game import boltzmann_value
    expect = coalition_marginal(v).mean(axis=-1)
    assert np.allclose(boltzmann_value(v, temperature=1e9), expect, atol=1e-6)


def test_boltzmann_zero_limits_are_argmax_argmin_selections():
    from tessera.game import boltzmann_value
    # Asymmetric by construction (H5: a sign slip is undetectable on
    # symmetric data) — distinct max and min coalitions.
    rng = np.random.default_rng(29)
    v = rng.standard_normal(SIZE)
    v[13] += 10.0   # unique argmax
    v[6] -= 10.0    # unique argmin
    marg = coalition_marginal(v)
    hot = boltzmann_value(v, temperature=1e-2)
    cold = boltzmann_value(v, temperature=-1e-2)
    assert np.allclose(hot, marg[..., :, 13], atol=1e-3)   # T→0⁺ → ∂ at argmax
    assert np.allclose(cold, marg[..., :, 6], atol=1e-3)   # T→0⁻ → ∂ at argmin
    assert not np.allclose(hot, cold)


def test_boltzmann_zero_temperature_fails_closed(v):
    from tessera.game import boltzmann_value
    with pytest.raises(ValueError, match="finite and nonzero"):
        boltzmann_value(v, temperature=0.0)


def test_boltzmann_vjp_matches_finite_differences(v):
    from tessera.game import boltzmann_value
    from tessera.game.values import _boltzmann_value_vjp
    rng = np.random.default_rng(31)
    dout = rng.standard_normal(N_PLAYERS)
    grad = _boltzmann_value_vjp(dout, v, temperature=0.7)
    eps = 1e-6
    for r in rng.integers(0, SIZE, size=8):
        vp, vm = v.copy(), v.copy()
        vp[r] += eps
        vm[r] -= eps
        fd = (boltzmann_value(vp, temperature=0.7) - boltzmann_value(vm, temperature=0.7)) @ dout
        assert np.isclose(grad[r], fd / (2 * eps), atol=1e-5)


# ── coalition excess: the subset_zeta-of-an-additive-game identity ───────────

def test_excess_is_v_minus_zeta_of_additive_game(v):
    from tessera.game import coalition_excess
    rng = np.random.default_rng(37)
    x = rng.standard_normal(N_PLAYERS)
    additive = np.zeros(SIZE)
    for i in range(N_PLAYERS):
        additive[1 << i] = x[i]
    assert np.allclose(coalition_excess(v, x), v - subset_zeta(additive))


def test_excess_transpose_is_adjoint(v):
    from tessera.game import coalition_excess
    rng = np.random.default_rng(41)
    x = rng.standard_normal(N_PLAYERS)
    dout = rng.standard_normal(SIZE)
    gv, gx = coalition_excess.transpose_rule(dout, v, x)
    lhs = coalition_excess(v, x) @ dout
    assert np.isclose(lhs, v @ gv + x @ gx)


# ── oracle 11: mex + Grundy/nim ──────────────────────────────────────────────

def test_mex_basics():
    from tessera.game import mex
    vals = np.array([0, 1, 3, 0, 2, 5, 7], dtype=np.int64)
    segs = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    out = mex(vals, segs, num_segments=4)
    assert out.tolist() == [2, 1, 0, 0]  # incl. empty segment -> 0


def test_grundy_of_nim_sum_is_xor():
    from tessera.game import mex
    # Grundy of a single nim pile of size k is k (computed via the mex
    # recursion over reachable piles), and the Grundy number of a two-pile
    # position is the XOR of the pile Grundy values (Sprague–Grundy).
    grundy = {0: 0}
    for k in range(1, 8):
        reach = np.array([grundy[j] for j in range(k)], dtype=np.int64)
        grundy[k] = int(mex(reach, np.zeros(k, dtype=np.int64),
                            num_segments=1)[0])
    assert all(grundy[k] == k for k in range(8))
    for a in range(6):
        for b in range(6):
            reach = ([grundy[j] ^ b for j in range(a)]
                     + [a ^ grundy[j] for j in range(b)])
            g2 = int(mex(np.array(reach, dtype=np.int64),
                         np.zeros(len(reach), dtype=np.int64),
                         num_segments=1)[0])
            assert g2 == (a ^ b)


def test_mex_fails_closed():
    from tessera.game import mex
    with pytest.raises(ValueError, match="integer"):
        mex(np.zeros(3), np.zeros(3, dtype=np.int64), num_segments=1)
    with pytest.raises(ValueError, match="out of range"):
        mex(np.zeros(3, dtype=np.int64),
            np.array([0, 1, 5], dtype=np.int64), num_segments=2)


def test_excess_jvp_is_dv_minus_M_dx(v):
    from tessera.game.values import _coalition_excess_jvp, _membership
    rng = np.random.default_rng(43)
    x = rng.standard_normal(N_PLAYERS)
    dv = rng.standard_normal(SIZE)
    dx = rng.standard_normal(N_PLAYERS)
    m = _membership(N_PLAYERS, SIZE)
    _, tangent = _coalition_excess_jvp((v, x), (dv, dx))
    assert np.allclose(tangent, dv - dx @ m)
    # single-tangent cases: the other primal must NOT leak into the tangent
    _, tv = _coalition_excess_jvp((v, x), (dv, None))
    assert np.allclose(tv, dv)
    _, tx = _coalition_excess_jvp((v, x), (None, dx))
    assert np.allclose(tx, -dx @ m)


def test_boltzmann_reverse_through_tape(v):
    # The end-to-end path the review flagged: keyword-only temperature must
    # survive tape recording into the registered VJP.
    from tessera.autodiff import tape as tape_mod
    from tessera.game import boltzmann_value
    if not hasattr(tape_mod, "tape"):
        pytest.skip("tape() entry point not exposed here")
    with tape_mod.tape() as t:
        out = boltzmann_value(v, temperature=0.7)
    assert out.shape == (N_PLAYERS,)
