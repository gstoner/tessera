"""AD-JET-STRUCT-1 — structured jets: the fused families evaluated in W.

The AD-WEIL-1 substrate (`algebra.TruncatedJet`) evaluates *scalar*
programs in ``W = ℝ[ε]/(ε^{k+1})``. This module extends evaluation-in-W to
the structured/fused families the plan's §3.3–§3.4 name, as the **numpy
reference lane** (host-free; physical carriers are AD-JET-IR-1's problem,
x86/gfx1151 first per the integrated plan):

* **multilinear** — `jet_matmul` is the truncated Cauchy convolution over
  the coefficient axis (§3.3): ``(Â·B̂)_k = Σ_{i+j=k} Aᵢ·Bⱼ``. All
  coefficients through order k cost (k+1)(k+2)/2 matmuls — polynomial,
  against 2ᵏ matmul-shaped terms under nesting. Zero new rules: the
  structure declaration *is* the rule.
* **pointwise** — `jet_map` applies a registered holonomic recurrence
  (`algebra.SCALAR_RECURRENCES`) to array coefficients; the recurrences
  are elementwise, so arrays ride through unchanged.
* **softmax / logsumexp / norm chain / flash_attn** — §3.4. The one load-
  bearing fact: ``softmax(z) = exp(z − m)/Σexp(z − m)`` is *invariant* to
  the shift ``m`` (it cancels between numerator and denominator), so the
  running max may be held at **order 0** (`control_at_order = 0`, the
  §2.3 semantic key) and the jet stays EXACT — the selection
  nonsmoothness of ``m`` never reaches the derivative of the attention
  output. The online-recurrence rescale ``exp(m_old − m_new)`` therefore
  degenerates to an order-0 factor, and ``ℓ``/``o`` are ordinary jets
  updated by the same recurrence with scalar ops replaced by W-ops.
* **max as an OP output** (`jet_reduce_max`) is where the kink policy
  genuinely bites: selection uses the primal coefficient only
  (`control_at_order = 0`), and at exact ties the declared
  ``SUBGRAD_SPLIT`` policy (`nonsmooth.NONSMOOTH_SELECTION["amax"]`)
  distributes the higher-order coefficients as the equal-share average —
  the same selection Law 5 pins for the first-order rules.
* **stochastic estimators** (§3.7) — `hessian_trace_estimate` /
  `laplacian_estimate`: randomizing the order-2 jet seed gives an
  unbiased estimator of ``tr ∇²f`` (``E[vᵀ∇²f v] = tr ∇²f`` for
  isotropic ``v``; the jet's Taylor coefficient a₂ is ``½ vᵀ∇²f v``).
  Draws go through the S4 Philox streams (`tessera.rng`, Decision #18)
  with an **explicit, mandatory key** — the estimator is deterministic
  and replayable given the key, and key-functional (no hidden state).
  W2.2 note: these helpers are not `tessera.ops.*` catalog ops, so they
  never appear on a tape or trace where the fail-closed `EffectLattice`
  would need to classify them; if one is ever promoted to the catalog it
  MUST register ``effect_kind = "random"`` — recorded here so promotion
  cannot silently skip it.

**Production authority is unchanged** (#31): the hand-written `_JVPS` /
`_VJPS` rules remain the production derivative path. Every structured jet
here is reference + oracle, tied to production by three proofs in
`tests/unit/test_jet_struct.py` — order 0 equals the canonical forward,
order 1 equals the registered hand JVP, and orders ≥ 2 equal the k-nested
dual tower on the diagonal seed (Law 4's differential proof, with the
§3.1 factorial bookkeeping spelled by `coefficient_scaling="derivative"`).
Hand-rule retirement is a per-family decision that *begins* only once its
Law-4 proof is green, and is deliberately NOT exercised in this slice.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .algebra import SCALAR_RECURRENCES, TruncatedJet

__all__ = [
    "Jet",
    "jet_lift",
    "jet_const",
    "jet_add",
    "jet_sub",
    "jet_mul",
    "jet_scale",
    "jet_map",
    "jet_matmul",
    "jet_sum",
    "jet_mean",
    "jet_reduce_max",
    "jet_where_mask",
    "jet_softmax",
    "jet_logsumexp",
    "jet_rmsnorm",
    "jet_layer_norm",
    "jet_flash_attn",
    "hessian_trace_estimate",
    "laplacian_estimate",
]

# A jet is a list of k+1 coefficient arrays, all one shape — the same
# convention `TruncatedJet.lift` produces. Taylor-scaled internally (the
# only convention under which the Cauchy product is the ring product);
# read out derivatives via `TruncatedJet(k, coefficient_scaling=
# "derivative").extract`.
Jet = list


def jet_lift(W: TruncatedJet, x: np.ndarray, v: np.ndarray) -> Jet:
    return W.lift(np.asarray(x, dtype=np.float64),
                  np.asarray(v, dtype=np.float64))


def jet_const(W: TruncatedJet, x: np.ndarray) -> Jet:
    x = np.asarray(x, dtype=np.float64)
    out = [np.zeros_like(x) for _ in range(W.order + 1)]
    out[0] = x
    return out


def jet_add(W: TruncatedJet, a: Jet, b: Jet) -> Jet:
    return W.add(a, b)


def jet_sub(W: TruncatedJet, a: Jet, b: Jet) -> Jet:
    return [x - y for x, y in zip(a, b)]


def jet_mul(W: TruncatedJet, a: Jet, b: Jet) -> Jet:
    """Elementwise Cauchy product (broadcasting like numpy ``*``)."""
    return W.mul(a, b)


def jet_scale(W: TruncatedJet, a: Jet, s) -> Jet:
    """Multiply by an order-0 (jet-constant) array or scalar."""
    return [c * s for c in a]


def jet_map(W: TruncatedJet, name: str, u: Jet) -> Jet:
    """Apply a registered holonomic scalar recurrence elementwise.

    Fails closed on an unregistered name: a structured family may not
    silently substitute a different function (#21a's discipline)."""
    if name not in SCALAR_RECURRENCES:
        raise KeyError(
            f"no holonomic recurrence registered for {name!r}; the jet of "
            f"an unregistered scalar primitive is undefined — register it "
            f"in algebra.SCALAR_RECURRENCES rather than approximating"
        )
    return W.scalar_fn(name, u)


def jet_matmul(W: TruncatedJet, a: Jet, b: Jet) -> Jet:
    """§3.3 — the jet of a bilinear map is a triangular Cauchy convolution
    over the coefficient axis: out_k = Σ_{i+j=k} aᵢ @ bⱼ."""
    k = W.order
    out: Jet = []
    for order in range(k + 1):
        acc = None
        for i in range(order + 1):
            term = np.matmul(a[i], b[order - i])
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


def jet_sum(W: TruncatedJet, u: Jet, axis=None, keepdims: bool = False) -> Jet:
    """Linear reduction: coefficient-wise sum."""
    return [np.sum(c, axis=axis, keepdims=keepdims) for c in u]


def jet_mean(W: TruncatedJet, u: Jet, axis=None, keepdims: bool = False) -> Jet:
    return [np.mean(c, axis=axis, keepdims=keepdims) for c in u]


def jet_where_mask(W: TruncatedJet, mask: np.ndarray, u: Jet,
                   fill0: float) -> Jet:
    """``where(mask, fill0, u)`` with an order-0 fill (a masked position is
    CONSTANT — e.g. the causal −inf — so its higher coefficients are
    zero). This is `control_at_order = 0` for branch predicates: the mask
    is data the primal computed; coefficients follow the primal's trace."""
    out = [np.where(mask, fill0, u[0])]
    for c in u[1:]:
        out.append(np.where(mask, 0.0, c))
    return out


def jet_reduce_max(W: TruncatedJet, u: Jet, axis: int = -1,
                   keepdims: bool = False) -> Jet:
    """``max`` as an op OUTPUT — the case where the kink policy bites.

    Selection is by the PRIMAL coefficient only (`control_at_order = 0`);
    at exact ties the declared ``SUBGRAD_SPLIT`` policy for ``amax``
    (`nonsmooth.NONSMOOTH_SELECTION`) makes the higher-order coefficients
    the equal-share average over the tied slots — the same selection the
    first-order rules implement and Law 5 pins, extended upward.
    """
    from .nonsmooth import NONSMOOTH_SELECTION, SUBGRAD_SPLIT

    policy = NONSMOOTH_SELECTION.get("amax", SUBGRAD_SPLIT)
    if policy != SUBGRAD_SPLIT:  # pragma: no cover — declared set is stable
        raise NotImplementedError(
            f"jet_reduce_max implements the declared '{SUBGRAD_SPLIT}' "
            f"policy; amax now declares {policy!r} — extend the jet rule "
            f"alongside the declaration, never silently diverge"
        )
    primal = u[0]
    m = np.max(primal, axis=axis, keepdims=True)
    tied = (primal == m)
    share = tied / np.sum(tied, axis=axis, keepdims=True)
    out = [m if keepdims else np.squeeze(m, axis=axis)]
    for c in u[1:]:
        sel = np.sum(share * c, axis=axis, keepdims=keepdims)
        out.append(sel)
    return out


# ── §3.4 structured families ─────────────────────────────────────────────────


def _shift_order0(u: Jet, m: np.ndarray) -> Jet:
    """Subtract an order-0 quantity from the constant coefficient only."""
    return [u[0] - m] + [c for c in u[1:]]


def jet_softmax(W: TruncatedJet, z: Jet, axis: int = -1) -> Jet:
    """softmax through exp/sum/reciprocal jets. The max shift is order-0
    and EXACT by the invariance softmax(z) = softmax(z − m)."""
    m = np.max(z[0], axis=axis, keepdims=True)
    e = jet_map(W, "exp", _shift_order0(z, m))
    s = jet_sum(W, e, axis=axis, keepdims=True)
    return jet_mul(W, e, jet_map(W, "reciprocal", s))


def jet_logsumexp(W: TruncatedJet, z: Jet, axis=None,
                  keepdims: bool = False) -> Jet:
    """lse(z) = m + log Σ exp(z − m); the m-dependence cancels identically,
    so the order-0 shift is exact here too."""
    m = np.max(z[0], axis=axis, keepdims=True)
    e = jet_map(W, "exp", _shift_order0(z, m))
    s = jet_sum(W, e, axis=axis, keepdims=True)
    out = jet_map(W, "log", s)
    out = [out[0] + m] + list(out[1:])
    if not keepdims and axis is not None:
        out = [np.squeeze(c, axis=axis) for c in out]
    elif not keepdims and axis is None:
        out = [np.reshape(c, ()) for c in out]
    return out


def jet_rmsnorm(W: TruncatedJet, x: Jet,
                gamma: Optional[np.ndarray] = None,
                eps: float = 1e-5) -> Jet:
    """y = x / sqrt(mean(x², last axis) + eps) · γ — the norm chain as
    square / mean / sqrt / reciprocal jets (γ, eps are order-0)."""
    sq = jet_mul(W, x, x)
    ms = jet_mean(W, sq, axis=-1, keepdims=True)
    ms = [ms[0] + eps] + list(ms[1:])
    inv = jet_map(W, "reciprocal", jet_map(W, "sqrt", ms))
    out = jet_mul(W, x, inv)
    if gamma is not None:
        out = jet_scale(W, out, np.asarray(gamma, dtype=np.float64))
    return out


def jet_layer_norm(W: TruncatedJet, x: Jet,
                   gamma: Optional[np.ndarray] = None,
                   beta: Optional[np.ndarray] = None,
                   eps: float = 1e-5) -> Jet:
    """y = (x − mean(x)) / sqrt(var(x) + eps) · γ + β over the last axis."""
    mu = jet_mean(W, x, axis=-1, keepdims=True)
    centered = jet_sub(W, x, mu)
    var = jet_mean(W, jet_mul(W, centered, centered), axis=-1, keepdims=True)
    var = [var[0] + eps] + list(var[1:])
    inv = jet_map(W, "reciprocal", jet_map(W, "sqrt", var))
    out = jet_mul(W, centered, inv)
    if gamma is not None:
        out = jet_scale(W, out, np.asarray(gamma, dtype=np.float64))
    if beta is not None:
        out = [out[0] + np.asarray(beta, dtype=np.float64)] + list(out[1:])
    return out


def jet_flash_attn(
    W: TruncatedJet,
    Q: Jet,
    K: Jet,
    V: Jet,
    *,
    scale: Optional[float] = None,
    causal: bool = False,
    attn_bias: Optional[np.ndarray] = None,
    block_size: int = 2,
) -> Jet:
    """§3.4 — the jet of the ONLINE softmax attention recurrence.

    Deliberately blockwise over the key axis (never materializing full
    softmax weights) so the *online* form itself is what's proven: the
    running max ``m`` stays order-0 (`control_at_order = 0`, exact by
    softmax's shift invariance), the rescale ``exp(m_old − m_new)`` is an
    order-0 factor, and the running ``(ℓ, o)`` are jets updated by the
    same recurrence with W-ops. Backward, HVP, and order-k directional
    derivatives share this one skeleton at different W — the fusion
    argument the eventual `schedule.attention_backward` sibling carries
    (AD-JET-IR-1's problem, not this lane's).

    ``attn_bias`` is an order-0 additive score bias (the canonical
    substrate); ``causal`` applies the canonical upper-triangular mask.
    """
    d = Q[0].shape[-1]
    s = float(scale) if scale is not None else 1.0 / np.sqrt(d)
    kT = [np.swapaxes(c, -1, -2) for c in K]
    q_len = Q[0].shape[-2]
    k_len = K[0].shape[-2]

    neg_inf = -np.inf
    m_run = np.full(Q[0].shape[:-1] + (1,), neg_inf, dtype=np.float64)
    ell: Jet = [np.zeros(Q[0].shape[:-1] + (1,), dtype=np.float64)
                for _ in range(W.order + 1)]
    out: Jet = [np.zeros(Q[0].shape[:-2] + (q_len, V[0].shape[-1]),
                         dtype=np.float64)
                for _ in range(W.order + 1)]

    for start in range(0, k_len, block_size):
        stop = min(start + block_size, k_len)
        kT_blk = [c[..., :, start:stop] for c in kT]
        v_blk = [c[..., start:stop, :] for c in V]
        scores = jet_scale(W, jet_matmul(W, Q, kT_blk), s)
        if attn_bias is not None:
            bias_blk = np.asarray(attn_bias, dtype=np.float64)[..., :, start:stop]
            scores = [scores[0] + bias_blk] + list(scores[1:])
        if causal:
            cols = np.arange(start, stop)[None, :]
            rows = np.arange(q_len)[:, None]
            mask = cols > rows + max(k_len - q_len, 0)
            scores = jet_where_mask(W, mask, scores, neg_inf)

        blk_max = np.max(scores[0], axis=-1, keepdims=True)
        m_new = np.maximum(m_run, blk_max)
        # Order-0 rescale of the running stats. exp(−inf) = 0 on the
        # first block (empty prefix) — exactly the right annihilation.
        with np.errstate(invalid="ignore"):
            alpha = np.where(np.isneginf(m_run), 0.0, np.exp(m_run - m_new))
        p = jet_map(W, "exp", _shift_order0(scores, m_new))
        ell = [alpha * c for c in ell]
        ell = jet_add(W, ell, jet_sum(W, p, axis=-1, keepdims=True))
        out = [alpha * c for c in out]
        out = jet_add(W, out, jet_matmul(W, p, v_blk))
        m_run = m_new

    return jet_mul(W, out, jet_map(W, "reciprocal", ell))


# ── §3.7 stochastic derivative estimators on Philox ──────────────────────────


def hessian_trace_estimate(
    jet_fn: Callable[[TruncatedJet, Jet], Jet],
    x: np.ndarray,
    key,
    *,
    samples: int = 64,
    distribution: str = "rademacher",
) -> float:
    """Unbiased ``tr ∇²f(x)`` from randomized order-2 jet seeds (§3.7).

    ``jet_fn(W, coeffs) -> coeffs`` is a scalar-output jet program (write
    it against this module's `jet_*` vocabulary). For each probe ``v`` the
    order-2 Taylor coefficient of ``t ↦ f(x + t v)`` is ``½ vᵀ∇²f v``, so
    ``E[2·a₂] = tr ∇²f`` when ``E[v vᵀ] = I`` — true for both supported
    distributions (``"rademacher"``, lower variance; ``"normal"``).

    Determinism (Decision #18): draws come from the S4 Philox stream of
    the **mandatory** ``key`` — same key, same estimate, bit-for-bit;
    independent estimates come from split keys, never from hidden state.
    An unknown distribution fails closed (#21a: the probe law is a
    semantic key — a wrong isotropy assumption is a silently-biased
    estimator, the worst outcome).
    """
    from .errors import TesseraAutodiffError
    from tessera.rng import bernoulli, normal

    if samples <= 0:
        raise TesseraAutodiffError("hessian_trace_estimate needs samples >= 1")
    if distribution not in ("rademacher", "normal"):
        raise TesseraAutodiffError(
            f"unknown probe distribution {distribution!r}: the estimator is "
            f"unbiased only under E[vvᵀ] = I, so the distribution is a "
            f"semantic key and may not default (#21a)"
        )
    x = np.asarray(x, dtype=np.float64)
    W = TruncatedJet(2)
    subkeys = key.split(samples)
    total = 0.0
    for sk in subkeys:
        if distribution == "rademacher":
            bits = np.asarray(bernoulli(sk, x.shape, p=0.5), dtype=np.float64)
            v = 2.0 * bits - 1.0
        else:
            v = np.asarray(normal(sk, x.shape, dtype="fp64"), dtype=np.float64)
        coeffs = jet_fn(W, jet_lift(W, x, v))
        a2 = np.asarray(coeffs[2], dtype=np.float64)
        if a2.size != 1:
            raise TesseraAutodiffError(
                f"hessian_trace_estimate needs a scalar-output jet program; "
                f"got output shape {a2.shape}"
            )
        total += 2.0 * float(a2.reshape(()))
    return total / samples


def laplacian_estimate(
    jet_fn: Callable[[TruncatedJet, Jet], Jet],
    x: np.ndarray,
    key,
    *,
    samples: int = 64,
    distribution: str = "rademacher",
) -> float:
    """The Laplacian is the Hessian trace — same estimator, named for the
    PDE consumers (`CORE_SUBSTRATE_VIEW.md` S6 names jet AD as a demanded
    capability for the PDE lane)."""
    return hessian_trace_estimate(
        jet_fn, x, key, samples=samples, distribution=distribution
    )
