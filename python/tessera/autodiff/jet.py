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
        # A row whose scores are all −inf in this block (an attn_bias padding
        # mask) has m_new = −inf, and −inf − (−inf) is NaN, which then
        # contaminates ℓ and o for every later block through the alpha
        # rescale. Shifting by 0 instead leaves the score at −inf, so
        # exp gives the 0 weight the mask asks for and a row masked only in
        # its leading blocks stays finite — matching `ops.flash_attn`.
        shift = np.where(np.isneginf(m_new), 0.0, m_new)
        p = jet_map(W, "exp", _shift_order0(scores, shift))
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


def _jet_div(W: TruncatedJet, a: Jet, b: Jet) -> Jet:
    """`a / b` in W, as `a * reciprocal(b)`.

    `reciprocal` is a registered holonomic recurrence, so division needs no
    rule of its own -- which also means it inherits that recurrence's
    behaviour at `b = 0` rather than inventing a second convention.
    """
    return jet_mul(W, a, jet_map(W, "reciprocal", b))


#: `ops.<name>` -> how that op evaluates in W. Absence is meaningful: an op
#: with no entry here FAILS CLOSED in `jet_trace` rather than being treated as
#: a constant, because silently dropping an op to order 0 returns a derivative
#: that is wrong without being obviously wrong.
_JET_REPLAY: "dict[str, Callable]" = {
    "add": lambda W, a, kw: jet_add(W, a[0], a[1]),
    "sub": lambda W, a, kw: jet_sub(W, a[0], a[1]),
    "mul": lambda W, a, kw: jet_mul(W, a[0], a[1]),
    "div": lambda W, a, kw: _jet_div(W, a[0], a[1]),
    "neg": lambda W, a, kw: jet_scale(W, a[0], -1.0),
    "matmul": lambda W, a, kw: jet_matmul(W, a[0], a[1]),
    "gemm": lambda W, a, kw: jet_matmul(W, a[0], a[1]),
    "sum": lambda W, a, kw: jet_sum(W, a[0], axis=kw.get("axis"),
                                    keepdims=bool(kw.get("keepdims", False))),
    "mean": lambda W, a, kw: jet_mean(W, a[0], axis=kw.get("axis"),
                                      keepdims=bool(kw.get("keepdims", False))),
    "softmax": lambda W, a, kw: jet_softmax(W, a[0], axis=int(kw.get("axis", -1))),
    "logsumexp": lambda W, a, kw: jet_logsumexp(
        W, a[0], axis=kw.get("axis"), keepdims=bool(kw.get("keepdims", False))),
}
# The 21 registered pointwise recurrences ride in unchanged: `jet_map` already
# applies them coefficient-wise, so each is a rule only in the sense that its
# NAME must be listed -- which is what keeps an unregistered scalar failing
# closed instead of silently resolving.
_JET_REPLAY.update({
    _name: (lambda _n: lambda W, a, kw: jet_map(W, _n, a[0]))(_name)
    for _name in SCALAR_RECURRENCES
})


def jet_trace(fn: "Callable[[np.ndarray], np.ndarray]") -> "Callable[[TruncatedJet, Jet], Jet]":
    """Lift a plain `tessera.ops.*` program into jet arithmetic (MSW-2).

    Returns a `jet_fn(W, coeffs)` of the shape `laplacian_exact` and
    `hessian_trace_estimate` consume, so a caller writes ordinary
    `ops.*` code instead of hand-translating it into this module's `jet_*`
    vocabulary -- which was the practical barrier to using either.

    **It reuses the tape rather than adding a second tracer** (#31). `fn` is
    run once under `tape()`, which already records every `ops.*` call in
    order with its operands and kwargs; this replays that linear record with
    each buffer bound to a jet. There is one interception mechanism in the
    codebase and this is not a new one.

    **It re-traces on every call, deliberately.** The tape captures the
    control flow taken at ONE point, so a record reused at another point
    would silently evaluate the wrong branch. Re-tracing costs one numpy
    forward per evaluation and removes the entire class; the straight-line
    restriction that remains is the tape's own, inherited rather than
    invented.

    Values `fn` closes over are constants in W (order 0 only), which is
    correct -- they do not vary with the seed direction -- and so are Python
    scalar literals.
    """
    from .errors import TesseraAutodiffError
    from .tape import tape

    # One-entry trace cache, keyed on the PRIMAL POINT. The record depends on
    # `coeffs[0]` alone -- control flow, captured constants and literal
    # operands are all fixed once the point is -- and NOT on the seed
    # direction, so `laplacian_exact`'s d evaluations at one point can share a
    # single trace. Measured on a d=128 field: tracing was 40% of
    # `laplacian_exact`, all of it redundant.
    #
    # One entry, not an unbounded dict: d consecutive calls at the same point
    # is exactly the access pattern, and a growing cache of traces (each
    # holding every intermediate array) is a memory leak wearing an
    # optimisation's clothes. A different point re-traces, which is what
    # `test_jet_trace_retraces_so_a_second_point_is_not_stale` pins.
    #
    # The cache holds the probe array itself, not just its id: `env` is keyed
    # on `id(probe)`, and a collected probe could see its id reused by another
    # object, silently binding the wrong buffer to the input jet.
    cache: dict = {"key": None, "probe": None, "entries": None, "out_id": None}

    def _trace_at(point: np.ndarray):
        key = (point.shape, point.dtype.str, point.tobytes())
        if cache["key"] == key:
            return cache["probe"], cache["entries"], cache["out_id"]
        probe = np.array(point, copy=True)
        with tape() as recorded:
            out = fn(probe)
        if not recorded.entries:
            raise TesseraAutodiffError(
                "jet_trace recorded no tessera.ops.* calls. The function must "
                "build its result through ops.*; raw numpy is invisible to the "
                "tape, so the jet would be a constant and every derivative "
                "would come back zero."
            )
        cache.update(key=key, probe=probe, entries=tuple(recorded.entries),
                     out_id=id(out))
        return probe, cache["entries"], cache["out_id"]

    def jet_fn(W: TruncatedJet, coeffs: Jet) -> Jet:
        probe, entries, out_id = _trace_at(np.asarray(coeffs[0], dtype=np.float64))
        env: dict[int, Jet] = {id(probe): coeffs}

        def resolve(desc) -> Jet:
            if desc.array_id in env:
                return env[desc.array_id]
            # A literal or a captured constant: order 0 only.
            return jet_const(W, np.asarray(desc.array, dtype=np.float64))

        for entry in entries:
            rule = _JET_REPLAY.get(entry.op)
            if rule is None:
                raise TesseraAutodiffError(
                    f"jet_trace has no jet rule for tessera.ops.{entry.op}. "
                    f"Refusing rather than treating it as a constant: an op "
                    f"dropped to order 0 yields a derivative that is wrong "
                    f"without looking wrong. Known ops: "
                    f"{', '.join(sorted(_JET_REPLAY))}."
                )
            env[entry.output_id] = rule(W, [resolve(i) for i in entry.inputs],
                                        entry.kwargs)

        if out_id not in env:
            raise TesseraAutodiffError(
                "jet_trace could not tie the returned value back to a recorded "
                "op: the function returned something the tape did not produce "
                "(a raw numpy result, or a value built outside ops.*)."
            )
        return env[out_id]

    return jet_fn


def laplacian_exact(
    jet_fn: "Callable[[TruncatedJet, Jet], Jet]",
    x: np.ndarray,
) -> float:
    """Exact ``tr ∇²f(x)`` from `d` deterministic jet evaluations (MSW-2).

    The estimator above randomizes the order-2 seed and averages; this walks
    the coordinate directions instead. Seeding ``v = e_i`` makes the order-2
    Taylor coefficient of ``t ↦ f(x + t e_i)`` exactly ``½ ∂²f/∂x_i²``, so

        Σ_i 2·a₂(e_i) = Σ_i ∂²f/∂x_i²  =  tr ∇²f  =  Δf

    with no variance and no key. It is the same quantity
    `laplacian_estimate` approximates, which is why they are checked against
    each other rather than only against closed forms.

    **Cost is the reason both exist.** This is exactly ``d`` jet evaluations
    for a ``d``-element input -- cheaper than the estimator only when ``d``
    is smaller than the sample count, and unusable when ``d`` is large. The
    estimator's error falls as ``1/sqrt(samples)`` independently of ``d``.
    Choose by dimension, not by preference: exact below a few hundred
    elements, sampled above.

    No key, deliberately. A signature that accepted one would suggest the
    result varies with it, and an exact method that quietly ignored a key
    would be the more confusing of the two failures.
    """
    from .errors import TesseraAutodiffError

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise TesseraAutodiffError(
            "laplacian_exact needs at least one input element; the Laplacian "
            "of a zero-element field is not 0, it is undefined"
        )
    W = TruncatedJet(2)
    total = 0.0
    flat_basis = np.zeros(x.size, dtype=np.float64)
    for i in range(x.size):
        flat_basis[i] = 1.0
        v = flat_basis.reshape(x.shape)
        coeffs = jet_fn(W, jet_lift(W, x, v))
        a2 = np.asarray(coeffs[2], dtype=np.float64)
        if a2.size != 1:
            raise TesseraAutodiffError(
                f"laplacian_exact needs a scalar-output jet program; got "
                f"output shape {a2.shape}"
            )
        total += 2.0 * float(a2.reshape(()))
        flat_basis[i] = 0.0
    return total


# ── AD-RETIRE-2: the structured family's production rules derive here ────────
#
# Second retirement wave (after the ODE family in `derivative_contract`).
# The production JVP/VJP pairs for softmax / logsumexp / rmsnorm-core are
# first-order specializations of the structured jets above; the displaced
# hand rules become declared oracles (#31) in
# `derivative_contract.RETIRED_HAND_RULES`, same ledger as the ODE family.
#
# Envelope audit (the §8 bar, PR-recorded): softmax's pair speaks `axis`;
# logsumexp's speaks `axis` (incl. None) + `keepdims`; rmsnorm's is the
# last-axis core with `eps` inside the sqrt PLUS the optional broadcast
# `gamma` operand (dx via the symmetric core kernel on the scaled
# cotangent, dγ broadcast-reduced) — the hand pair was x-only, so the γ
# half is a gap CLOSED by retirement, not carried from the oracle; its
# proof is adjoint + finite-difference + tape-reverse, not differential.
#
# Dtype (the PR #600 lesson): production rules must preserve the input
# dtype, so the derivation feeds NATIVE-dtype coefficient pairs `[x, dx]`
# straight into the structured jets (the fp64 cast lives only in
# `jet_lift`, which these rules bypass; every coefficient op below is
# dtype-preserving). The jet API's `numeric_policy="fp64"` contract is
# about jet-order work — a first-order production specialization follows
# the canonical dtype flow instead.
#
# VJP derivations, stated: softmax and the rmsnorm core have SYMMETRIC
# Jacobians (diag(p) − p pᵀ on the softmax simplex; the I/n − x xᵀ/(d n³)
# projection kernel), so the pullback IS the pushforward applied to the
# cotangent — the AD-LAW-1n delegation pattern, now in production.
# logsumexp's linearization is the softmax row-functional J = pᵀ, so its
# transpose is broadcast-multiply: Jᵀu = p · u.


def _order1(name: str):
    W1 = TruncatedJet(1)
    # The branches deliberately bind different signatures (each op's kwarg
    # envelope, and rmsnorm's optional γ operand); declare the names as
    # general callables so the variants type-check.
    jvp: Callable[..., object]
    vjp: Callable[..., object]

    if name == "softmax":
        def jvp(primals, tangents, *, axis=-1, **_):
            x = np.asarray(primals[0])
            dx = np.asarray(tangents[0])
            c = jet_softmax(W1, [x, dx], axis=axis)
            return c[0], c[1]

        def vjp(dout, x, *, axis=-1, **_):
            _, g = jvp((x,), (np.asarray(dout),), axis=axis)
            return (g,)

    elif name == "logsumexp":
        def jvp(primals, tangents, *, axis=None, keepdims=False, **_):
            x = np.asarray(primals[0])
            dx = np.asarray(tangents[0])
            c = jet_logsumexp(W1, [x, dx], axis=axis, keepdims=keepdims)
            return c[0], c[1]

        def vjp(dout, x, *, axis=None, keepdims=False, **_):
            a = np.asarray(x)
            p = jet_softmax(W1, [a, np.zeros_like(a)],
                            axis=-1 if axis is None else axis)[0] \
                if axis is not None else None
            if axis is None:
                m = np.max(a)
                e = np.exp(a - m)
                p = e / np.sum(e)
                return (p * np.asarray(dout),)
            do = np.asarray(dout)
            if not keepdims:
                do = np.expand_dims(do, axis=axis)
            return (p * do,)

    elif name == "rmsnorm":
        # Envelope: the canonical forward is `core(x) · γ` with γ optional
        # (broadcast, typically last-dim). The displaced hand pair was
        # x-only, so tape-reverse through `ops.rmsnorm(x, gamma)` was
        # ALREADY broken before retirement (the hand VJP swallowed γ via
        # `**_` and returned one cotangent for two operands); the derived
        # pair closes that gap rather than reproducing it. With γ:
        #   JVP   dy = J_core(dx)·γ + core(x)·dγ            (product rule)
        #   VJP   dx = J_core(γ⊙dout)   (core kernel is symmetric — the
        #             AD-LAW-1n delegation, applied to the scaled cotangent)
        #         dγ = Σ_broadcast dout⊙core(x), reduced to γ's shape.
        def _core_pair(x, dx, eps):
            c = jet_rmsnorm(W1, [x, dx], gamma=None, eps=eps)
            return c[0], c[1]

        def jvp(primals, tangents, *, eps=1e-5, **_):
            x = np.asarray(primals[0])
            dx = np.asarray(tangents[0])
            gamma = primals[1] if len(primals) > 1 else None
            core, dcore = _core_pair(x, dx, eps)
            if gamma is None:
                return core, dcore
            gam = np.asarray(gamma)
            y = core * gam
            t = dcore * gam
            dgamma = tangents[1] if len(tangents) > 1 else None
            if dgamma is not None:
                t = t + core * np.asarray(dgamma)
            return y, t

        def vjp(dout, x, gamma=None, *, eps=1e-5, **_):
            a = np.asarray(x)
            do = np.asarray(dout)
            if gamma is None:
                _, g = _core_pair(a, do, eps)
                return (g,)
            from .vjp import _sum_to_shape
            gam = np.asarray(gamma)
            core, g = _core_pair(a, do * gam, eps)
            dgamma = _sum_to_shape(do * core, gam.shape)
            return (g, dgamma)

    else:  # pragma: no cover — the registration loop is the only caller
        raise KeyError(name)

    jvp._derived_from_jet = name  # type: ignore[attr-defined]
    vjp._derived_from_jet = name  # type: ignore[attr-defined]
    jvp.__name__ = f"jvp_{name}__jet"
    vjp.__name__ = f"vjp_{name}__jet"
    return jvp, vjp


STRUCTURED_RETIREES = ("softmax", "logsumexp", "rmsnorm")


def register_jet_derived_structured_rules() -> list[str]:
    """Switch the structured family's production pairs to the jet-derived
    first-order specializations; displaced hand rules join the #31 oracle
    ledger. Idempotent; fails closed on an incomplete hand pair."""
    from .derivative_contract import RETIRED_HAND_RULES
    from .jvp import get_jvp, register_jvp
    from .vjp import get_vjp, register_vjp

    switched: list[str] = []
    for name in STRUCTURED_RETIREES:
        current_jvp = get_jvp(name)
        if current_jvp is not None and getattr(
                current_jvp, "_derived_from_jet", None) == name:
            switched.append(name)
            continue
        current_vjp = get_vjp(name)
        if current_jvp is not None and current_vjp is not None:
            RETIRED_HAND_RULES[name] = (current_jvp, current_vjp)
        elif current_jvp is not None or current_vjp is not None:
            raise ValueError(
                f"refusing to retire {name!r}: exactly one hand rule "
                f"exists — a half-displaced pair would leave one mode "
                f"anchored and one not (#31)"
            )
        # else: fill mode (post-prune; see derivative_contract).
        derived_jvp, derived_vjp = _order1(name)
        register_jvp(name, derived_jvp)
        register_vjp(name, derived_vjp)
        switched.append(name)
    return switched
