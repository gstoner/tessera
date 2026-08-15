"""Semivalues over the coalition lattice (G1): one weight-parameterized op.

The operator-design win from GAME_THEORY_PLAN.md §4.2: Shapley (``w(t)=1/t``)
and Banzhaf (``w(t)=2^{1−t}``) are the SAME contraction against different
cardinality weights in the Möbius basis —

    Φ_i = Σ_{T ∋ i} w(|T|) · v̂(T),    v̂ = subset_mobius(v)

so one op serves every probabilistic value (the whole Weber set), and each new
value is a weight vector, not a kernel. Validation is against the AXIOMS
(efficiency, symmetry, dummy, additivity) plus the direct
permutation-average/subset-average definitions — never against a transcribed
closed form (plan §1.1: the source text's own Banzhaf constant is off by a
factor of two, and a transcribed oracle would have inherited it silently).
"""

from __future__ import annotations

import numpy as np

from ..custom import custom_primitive
from .lattice import lattice_players, subset_mobius


def _popcounts(size: int) -> np.ndarray:
    return np.array([bin(s).count("1") for s in range(size)], dtype=np.int64)


def _membership(n: int, size: int) -> np.ndarray:
    """M[i, S] = 1 iff player i ∈ S — the contraction's fixed structure."""
    s = np.arange(size)
    return ((s[None, :] >> np.arange(n)[:, None]) & 1).astype(np.float64)


def _semivalue_impl(vhat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    vhat = np.asarray(vhat, dtype=np.float64)
    n = lattice_players(vhat)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (n + 1,):
        raise ValueError(
            f"semivalue weights must have shape (n+1,) = ({n + 1},) — one "
            f"entry per coalition cardinality 0..n; got {weights.shape}")
    size = vhat.shape[-1]
    weighted = vhat * weights[_popcounts(size)]
    # Φ[..., i] = Σ_S M[i, S] · w(|S|) · v̂[..., S]
    return weighted @ _membership(n, size).T


def _semivalue_transpose(dout: np.ndarray, vhat: np.ndarray,
                         weights: np.ndarray, **_kw: object):
    """Adjoint in the linear argument v̂: grad[..., S] = w(|S|)·Σ_{i∈S} dout_i.
    Weights parameterize the map (not differentiated here) → None cotangent."""
    dout = np.asarray(dout, dtype=np.float64)
    vhat = np.asarray(vhat)
    n = lattice_players(vhat)
    size = vhat.shape[-1]
    weights = np.asarray(weights, dtype=np.float64)
    grad = (dout @ _membership(n, size)) * weights[_popcounts(size)]
    return grad, None


semivalue = custom_primitive(
    "game_semivalue", linear=True, linear_args=(0,),
    transpose_rule=_semivalue_transpose,
)(_semivalue_impl)


def shapley_weights(n: int) -> np.ndarray:
    """Möbius-basis Shapley weights: w(t) = 1/t (t ≥ 1); w(0) unused (no
    coalition containing a player is empty)."""
    w = np.zeros(n + 1, dtype=np.float64)
    w[1:] = 1.0 / np.arange(1, n + 1)
    return w


def banzhaf_weights(n: int) -> np.ndarray:
    """Möbius-basis Banzhaf weights: w(t) = 2^(1−t). The correct constant is
    2^(1−|T|) on unanimity games — pinned against the source text's
    factor-of-two transcription error (plan §1.1)."""
    w = np.zeros(n + 1, dtype=np.float64)
    w[1:] = 2.0 ** (1.0 - np.arange(1, n + 1))
    return w


def shapley_value(v: np.ndarray) -> np.ndarray:
    """Φ^Shapley over the lattice tensor ``v`` — the composition the fusion
    lane (G1b) later owns as one kernel: mobius → weighted contraction."""
    n = lattice_players(np.asarray(v))
    return semivalue(subset_mobius(v), shapley_weights(n))


def banzhaf_value(v: np.ndarray) -> np.ndarray:
    n = lattice_players(np.asarray(v))
    return semivalue(subset_mobius(v), banzhaf_weights(n))


def _boltzmann_softmax(v: np.ndarray, temperature: float) -> np.ndarray:
    """Max-subtracted Boltzmann weights p ∝ exp(v/T) along the lattice axis.

    Legal for T < 0 (min-seeking — hazard H5): the max subtraction stabilizes
    the *shifted* logits v/T whatever the sign. T == 0 fails closed: the limit
    is a semantic choice (max vs min), never a computable default.
    """
    if temperature == 0.0 or not np.isfinite(temperature):
        raise ValueError(
            "boltzmann_value temperature must be finite and nonzero; the "
            "T -> 0 limits are argmax/argmin selections, not a default")
    logits = v / temperature
    logits = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def _boltzmann_value_impl(v: np.ndarray, temperature: float) -> np.ndarray:
    """E^T_i(v) = Σ_S ∂_i v(S) · p_S with p = softmax(v/T) — the plan's §3.2
    flagship: an n-head softmax-weighted reduction over the 2^n lattice
    (structurally flash-attention's online softmax; the reference tier
    materializes p, the kernel tier streams it)."""
    v = np.asarray(v, dtype=np.float64)
    from .lattice import _coalition_marginal_impl
    p = _boltzmann_softmax(v, float(temperature))
    m = _coalition_marginal_impl(v)
    return np.einsum("...is,...s->...i", m, p)


def _boltzmann_value_vjp(dout: np.ndarray, v: np.ndarray,
                         temperature: float, **_kw: object) -> np.ndarray:
    """Closed-form gradient (finite-difference checked in the oracle tests):
    grad_R = Σ_i d_i (p_R − p_{R⊕i}) + (p_R/T)(Σ_i d_i m_{iR} − ⟨d, E⟩)."""
    v = np.asarray(v, dtype=np.float64)
    dout = np.asarray(dout, dtype=np.float64)
    t = float(temperature)
    from .lattice import _coalition_marginal_impl, lattice_players
    n = lattice_players(v)
    size = v.shape[-1]
    p = _boltzmann_softmax(v, t)
    m = _coalition_marginal_impl(v)
    e_val = np.einsum("...is,...s->...i", m, p)
    idx = np.arange(size)
    term1 = np.zeros_like(v)
    for i in range(n):
        di = dout[..., i:i + 1]
        term1 += di * (p - p[..., idx ^ (1 << i)])
    dm = np.einsum("...i,...is->...s", dout, m)
    de = np.einsum("...i,...i->...", dout, e_val)[..., None]
    return term1 + (p / t) * (dm - de)


boltzmann_value = custom_primitive(
    "game_boltzmann_value", vjp=_boltzmann_value_vjp,
)(_boltzmann_value_impl)


def _coalition_excess_impl(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """e(S, x) = v(S) − Σ_{i∈S} x_i — the core/nucleolus separation quantity.
    Jointly linear in (v, x); the x term is exactly ``subset_zeta`` of the
    additive game placing x_i on the singletons (the oracle the tests pin)."""
    v = np.asarray(v, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = lattice_players(v)
    if x.shape[-1] != n:
        raise ValueError(
            f"coalition_excess allocation must have trailing extent n = {n}; "
            f"got {x.shape[-1]}")
    return v - x @ _membership(n, v.shape[-1])


def _coalition_excess_transpose(dout: np.ndarray, v: np.ndarray,
                                x: np.ndarray, **_kw: object):
    dout = np.asarray(dout, dtype=np.float64)
    n = np.asarray(x).shape[-1]
    m = _membership(n, np.asarray(v).shape[-1])
    return dout, -(dout @ m.T)


coalition_excess = custom_primitive(
    "game_coalition_excess", linear=True, linear_args=(0, 1),
    transpose_rule=_coalition_excess_transpose,
)(_coalition_excess_impl)
