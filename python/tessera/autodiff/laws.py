"""AD-LAW-1 — executable algebraic laws over the derivative registries.

The design authority is ``docs/audit/compiler/AUTODIFF_NEXTGEN_PLAN.md`` §4:
correctness of the derivative rules is checked as *algebraic law*, not only as
pointwise finite differences. This module implements the first two laws over
the existing ``_VJPS``/``_JVPS`` registries, without changing any production
rule (the registries are read, never written):

Law 3 — **adjoint**:      ⟨J v, u⟩ = ⟨v, Jᵀ u⟩ for every probe pair (v, u).
    Complete for the *transpose relationship* between a JVP and its paired
    VJP. NOT a derivative-correctness test on its own: a matched-wrong pair
    (e.g. both identically zero) passes — see the plan's §3.5 caveat. That
    is what Law 1 is for.

Law 1 — **chain** (functoriality, the derivative-correctness complement):
    the chained JVP tangent of ``g∘f`` must match a central finite
    difference of the composed primal. Catches matched-wrong JVP/VJP pairs
    that Law 3 structurally cannot.

Inputs come from the declarative per-op registry in ``law_inputs.py``; ops
without a spec are *reported* as unswept, never silently skipped (claim
integrity: the dashboard distinguishes "checked" from "not yet checkable").

Everything here is deterministic: probe RNG is seeded from the op name, so
the generated dashboard is byte-stable across runs and hosts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "LawResult",
    "adjoint_check",
    "chain_check",
    "enclosure_check",
    "homomorphism_check",
    "kink_check",
    "quotient_check",
    "unbiasedness_check",
    "op_rng",
    "run_law_sweep",
]

# Tolerances. Both sides of the adjoint identity are computed in float64 on
# identical data, so the residual is pure rounding noise plus reduction
# reordering; 1e-8 relative is generous. The chain law compares against a
# central finite difference (eps 1e-5 ⇒ O(eps²) truncation), so its
# tolerance is necessarily looser.
ADJOINT_RTOL = 1e-8
CHAIN_RTOL = 5e-4
_CHAIN_EPS = 1e-5


@dataclass(frozen=True)
class LawResult:
    """Outcome of one (op, law) evaluation."""

    op: str
    registry: str            # "tensor" | "geometric"
    law: str                 # "adjoint" | "chain"
    status: str              # pass | fail | rule_error | no_spec | jvp_only |
                             # vjp_only | not_applicable
    probes: int
    max_rel_residual: Optional[float]   # informational; NOT in gated artifacts
    detail: str = ""


def op_rng(op: str, law: str) -> np.random.Generator:
    """Deterministic, platform-stable RNG per (op, law)."""
    digest = hashlib.sha256(f"ad-law-1:{law}:{op}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


# ── tree helpers ─────────────────────────────────────────────────────────────


def _is_diff_dtype(arr: np.ndarray) -> bool:
    return (np.issubdtype(arr.dtype, np.floating)
            or np.issubdtype(arr.dtype, np.complexfloating))


def _leaves(x: Any) -> list[np.ndarray]:
    """Flatten an op output (array / tuple / list / dict) to float/complex
    arrays. Complex arrays are first-class leaves — the pairing treats ℂ as
    ℝ² via Re⟨a, conj b⟩ (see `_dot`), which is the inner product under
    which the fft-family adjoint conventions are stated."""
    if x is None:
        return []
    if isinstance(x, (tuple, list)):
        out: list[np.ndarray] = []
        for e in x:
            out.extend(_leaves(e))
        return out
    if isinstance(x, dict):
        out = []
        for k in sorted(x):
            out.extend(_leaves(x[k]))
        return out
    arr = np.asarray(x)
    return [arr] if _is_diff_dtype(arr) else []


def _like_leaves(x: Any, make: Callable[[np.ndarray], np.ndarray]) -> Any:
    """Rebuild `x`'s structure with `make(leaf)` at each float leaf."""
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(_like_leaves(e, make) for e in x)
    if isinstance(x, list):
        return [_like_leaves(e, make) for e in x]
    if isinstance(x, dict):
        return {k: _like_leaves(v, make) for k, v in x.items()}
    arr = np.asarray(x)
    if _is_diff_dtype(arr):
        return make(arr)
    return np.zeros_like(arr)


def _dot(a: Any, b: Any) -> float:
    """Real inner product; ℂ is paired as ℝ² via Re⟨a, conj b⟩."""
    la, lb = _leaves(a), _leaves(b)
    if len(la) != len(lb):
        raise ValueError(f"structure mismatch: {len(la)} vs {len(lb)} leaves")
    total = 0.0
    for x, y in zip(la, lb):
        xa, ya = np.asarray(x), np.asarray(y)
        if (np.issubdtype(xa.dtype, np.complexfloating)
                or np.issubdtype(ya.dtype, np.complexfloating)):
            total += float(np.real(np.sum(xa.astype(np.complex128)
                                          * np.conj(ya.astype(np.complex128)))))
        else:
            total += float(np.sum(xa.astype(np.float64) * ya.astype(np.float64)))
    return total


def _rand_like(rng: np.random.Generator, arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(np.asarray(arr).dtype, np.complexfloating):
        return (rng.standard_normal(np.shape(arr))
                + 1j * rng.standard_normal(np.shape(arr)))
    return rng.standard_normal(np.shape(arr))


def _scale(*xs: Any) -> float:
    total = 0.0
    for x in xs:
        for leaf in _leaves(x):
            total += float(np.sum(np.abs(leaf)))
    return max(total, 1e-30)


# ── Law 3: adjoint ───────────────────────────────────────────────────────────


def adjoint_check(op: str, spec, jvp_fn: Callable, vjp_fn: Callable,
                  registry: str = "tensor") -> LawResult:
    """⟨J v, u⟩ = ⟨v, Jᵀ u⟩ over `spec.probes` deterministic probe pairs."""
    rng = op_rng(op, "adjoint")
    try:
        primals, kwargs = spec.make(rng)
        diff = spec.diff_args if spec.diff_args is not None else tuple(range(len(primals)))

        tol = spec.rtol if spec.rtol is not None else ADJOINT_RTOL
        max_res = 0.0
        for _ in range(spec.probes):
            tangents = tuple(
                _rand_like(rng, p) if i in diff and _is_float(p)
                else _zero_like(p)
                for i, p in enumerate(primals)
            )
            _, t_out = jvp_fn(primals, tangents, **kwargs)
            u = _like_leaves(t_out, lambda leaf: _rand_like(rng, leaf))
            lhs = _dot(t_out, u)

            grads = vjp_fn(u if not _single_leaf(t_out) else _leaves(u)[0],
                           *primals, **kwargs)
            if not isinstance(grads, tuple):
                grads = (grads,)
            rhs = 0.0
            for i in diff:
                if i < len(grads) and grads[i] is not None:
                    rhs += _dot(tangents[i], grads[i])

            res = abs(lhs - rhs) / max(abs(lhs) + abs(rhs), tol * _scale(t_out, u))
            max_res = max(max_res, res)

        status = "pass" if max_res <= tol * 10 else "fail"
        # A degenerate all-zero tangent stream would vacuously pass; flag it —
        # unless the spec declares the derivative genuinely 0 a.e. (`sign`).
        if status == "pass" and not spec.zero_tangent_ok and all(
            not np.any(np.asarray(l)) for l in _leaves(t_out)
        ):
            status = "fail"
            return LawResult(op, registry, "adjoint", status, spec.probes, max_res,
                             "tangent output identically zero — vacuous pairing "
                             "(matched-zero shape; see Law 1)")
        return LawResult(op, registry, "adjoint", status, spec.probes, max_res)
    except Exception as e:  # noqa: BLE001 — a sweep must survive any one rule
        return LawResult(op, registry, "adjoint", "rule_error", 0, None,
                         f"{type(e).__name__}: {e}")


def _is_float(p: Any) -> bool:
    return _is_diff_dtype(np.asarray(p))


def _zero_like(p: Any) -> np.ndarray:
    return np.zeros_like(np.asarray(p), dtype=np.float64) if _is_float(p) \
        else np.zeros(np.shape(p))


def _single_leaf(x: Any) -> bool:
    return not isinstance(x, (tuple, list, dict))


# ── Law 1: chain (functoriality vs finite differences) ───────────────────────


def chain_check(op: str, spec, jvp_fn: Callable,
                registry: str = "tensor") -> LawResult:
    """Chained-JVP tangent of tanh∘f vs central FD of the **canonical** primal.

    This is the derivative-correctness complement to the adjoint law: a
    matched-zero JVP/VJP pair passes Law 3 and fails here.

    The finite difference is anchored on the *registered forward op*
    (resolved via ``linear._resolve_forward``), never on the JVP's own
    primal output — a JVP that self-consistently implements the derivative
    of the wrong function (say ``exp(2x)`` with tangent ``2·dx·exp(2x)``)
    agrees with an FD of its own primal on every probe, so a
    self-referential check would certify exactly the matched-wrong class
    this law exists to catch. When no canonical forward resolves for the
    op name, the check reports ``not_applicable`` rather than silently
    degrading to self-consistency. The JVP's primal output is additionally
    required to match the canonical forward at the base point.
    """
    from .jvp import get_jvp
    from .linear import _resolve_forward

    g_jvp = get_jvp("tanh")
    if g_jvp is None:  # cannot happen in-tree; keep the sweep honest anyway
        return LawResult(op, registry, "chain", "not_applicable", 0, None,
                         "no tanh JVP to chain with")
    forward = _resolve_forward(op)
    if forward is None:
        return LawResult(op, registry, "chain", "not_applicable", 0, None,
                         "no canonical forward resolved — a self-consistency "
                         "FD would not anchor the primal")
    rng = op_rng(op, "chain")
    try:
        primals, kwargs = spec.make(rng)
        diff = spec.diff_args if spec.diff_args is not None else tuple(range(len(primals)))

        def primal_out(ps: tuple) -> Any:
            return forward(*ps, **kwargs)

        # Scale the output into tanh's active region before composing —
        # a large primal (e.g. an unreduced loss ≈ ±15) saturates tanh,
        # collapsing both the chained tangent and the finite difference into
        # cancellation noise and turning the residual into garbage. The scale
        # is a constant computed once at the base point, so it is transparent
        # to differentiation.
        y0_raw = primal_out(primals)
        if not _single_leaf(y0_raw):
            return LawResult(op, registry, "chain", "not_applicable", 0, None,
                             "multi-output op; chain check needs one output")
        if np.issubdtype(np.asarray(y0_raw).dtype, np.complexfloating):
            return LawResult(op, registry, "chain", "not_applicable", 0, None,
                             "complex output; the chain anchor composes a real "
                             "tanh — adjoint law still applies via Re⟨·,conj·⟩")
        y0 = np.asarray(y0_raw, dtype=np.float64)
        s = 1.0 / (1.0 + float(np.max(np.abs(y0))))

        # Primal-consistency gate: the JVP's own primal output must be the
        # canonical forward. A JVP whose primal half already disagrees is
        # differentiating some other function, whatever its tangent says.
        zeros = tuple(_zero_like(p) for p in primals)
        y_jvp = np.asarray(jvp_fn(primals, zeros, **kwargs)[0], dtype=np.float64)
        primal_rtol = spec.rtol if spec.rtol is not None else 1e-9
        if y_jvp.shape != y0.shape or not np.allclose(y_jvp, y0, rtol=primal_rtol,
                                                      atol=primal_rtol * 1e-3):
            return LawResult(op, registry, "chain", "fail", 1, None,
                             "JVP primal output disagrees with the canonical "
                             "forward — the rule differentiates a different "
                             "function")

        max_res = 0.0
        for _ in range(max(2, spec.probes // 2)):
            tangents = tuple(
                rng.standard_normal(np.shape(p)) if i in diff and _is_float(p)
                else _zero_like(p)
                for i, p in enumerate(primals)
            )
            y, dy = jvp_fn(primals, tangents, **kwargs)
            if not _single_leaf(y):
                return LawResult(op, registry, "chain", "not_applicable", 0, None,
                                 "multi-output op; chain check needs one output")
            _, dz = g_jvp((s * np.asarray(y, dtype=np.float64),),
                          (s * np.asarray(dy, dtype=np.float64),))

            def shifted(sign: float) -> np.ndarray:
                ps = tuple(
                    np.asarray(p, dtype=np.float64) + sign * _CHAIN_EPS * t
                    if i in diff and _is_float(p) else p
                    for i, (p, t) in enumerate(zip(primals, tangents))
                )
                return np.tanh(s * np.asarray(primal_out(ps), dtype=np.float64))

            fd = (shifted(+1.0) - shifted(-1.0)) / (2.0 * _CHAIN_EPS)
            num = float(np.max(np.abs(np.asarray(dz) - fd)))
            den = float(np.max(np.abs(fd))) + float(np.max(np.abs(np.asarray(dz)))) + 1e-12
            max_res = max(max_res, num / den)

        chain_tol = max(CHAIN_RTOL, spec.rtol or 0.0)
        status = "pass" if max_res <= chain_tol else "fail"
        return LawResult(op, registry, "chain", status,
                         max(2, spec.probes // 2), max_res)
    except Exception as e:  # noqa: BLE001
        return LawResult(op, registry, "chain", "rule_error", 0, None,
                         f"{type(e).__name__}: {e}")


# ── The sweep ────────────────────────────────────────────────────────────────


def run_law_sweep() -> list[LawResult]:
    """Evaluate Laws 1 and 3 over every op in the derivative registries.

    Every op appears in the result exactly once per applicable law, with an
    explicit status — including the ops that cannot be checked yet
    (``no_spec``) and the ops where the adjoint law is inapplicable because
    only one mode exists (``jvp_only`` / ``vjp_only``). The geometric
    registry is enumerated the same way so its sweep debt is visible.
    """
    from .jvp import _JVPS
    from .vjp import _VJPS
    from .law_inputs import KINK_SPECS, LAW_INPUT_SPECS

    results: list[LawResult] = []

    tensor_ops = sorted(set(_VJPS) | set(_JVPS))
    for op in tensor_ops:
        jvp_fn, vjp_fn = _JVPS.get(op), _VJPS.get(op)
        if jvp_fn is None or vjp_fn is None:
            status = "vjp_only" if jvp_fn is None else "jvp_only"
            results.append(LawResult(op, "tensor", "adjoint", status, 0, None))
            continue
        spec = LAW_INPUT_SPECS.get(op)
        if spec is None:
            results.append(LawResult(op, "tensor", "adjoint", "no_spec", 0, None))
            continue
        results.append(adjoint_check(op, spec, jvp_fn, vjp_fn))
        if spec.chain:
            results.append(chain_check(op, spec, jvp_fn))

    # Law 5 — at-the-kink policy for every declared-nonsmooth op. Ops with a
    # declared policy but no probe are reported, not skipped.
    from .nonsmooth import NONSMOOTH_SELECTION
    for op in sorted(NONSMOOTH_SELECTION):
        vjp_fn = _VJPS.get(op)
        if vjp_fn is None:
            results.append(LawResult(op, "tensor", "kink", "vjp_only", 0, None,
                                     "declared nonsmooth but no VJP registered"))
            continue
        kspec = KINK_SPECS.get(op)
        if kspec is None:
            results.append(LawResult(op, "tensor", "kink", "no_spec", 0, None))
            continue
        results.append(kink_check(op, kspec, vjp_fn, _JVPS.get(op)))

    # Laws 2 and 4 — algebra-level, the AD-WEIL-1 gate. These are keyed by
    # the codomain (W1..W4), not by op: they assert that the *substrate* a
    # future jet mode would run on is a faithful algebra, and that order-k
    # jets agree with the k-nested route they would replace.
    for _order in (1, 2, 3, 4):
        results.append(homomorphism_check(_order))
        results.append(quotient_check(_order))
        results.append(enclosure_check(_order))
    results.append(unbiasedness_check())

    try:
        from .geometric.registry import _JVPS_GEO, _VJPS_GEO
        for op in sorted(set(_VJPS_GEO) | set(_JVPS_GEO)):
            if op not in _JVPS_GEO or op not in _VJPS_GEO:
                status = "vjp_only" if op not in _JVPS_GEO else "jvp_only"
            else:
                status = "no_spec"  # multivector input specs are follow-on work
            results.append(LawResult(op, "geometric", "adjoint", status, 0, None))
    except Exception as e:  # noqa: BLE001 — geometric import must not sink the sweep
        results.append(LawResult("<geometric-registry>", "geometric", "adjoint",
                                 "rule_error", 0, None, f"{type(e).__name__}: {e}"))

    return results


# ── Law 5: kink policy ───────────────────────────────────────────────────────


def kink_check(op: str, spec, vjp_fn: Callable, jvp_fn: Optional[Callable],
               registry: str = "tensor") -> LawResult:
    """Evaluate a nonsmooth rule **exactly at** its kink against the declared
    policy in ``nonsmooth.py`` (Decision #21a).

    Laws 1/3 deliberately sample inside a smooth piece — a finite difference
    straddling a kink is meaningless. This law covers the one input where a
    legal-but-different Clarke selection is observable, which is exactly where
    a backend kernel and the numpy reference can silently disagree. It asserts
    the *declared* selection, not "a" subgradient: which element is returned
    is a semantic choice, so an undeclared op fails closed rather than
    defaulting.

    Both modes are checked when the op has both: a JVP that picks a different
    kink element than its VJP is the same defect class as the paired-default
    drift, and neither the adjoint law (both sides agree on a wrong pick) nor
    the chain law (which avoids kinks) can see it.
    """
    from .nonsmooth import SUBGRAD_SPLIT, SUBGRAD_ZERO, selection_policy

    try:
        policy = selection_policy(op)          # fail-closed on absence (#21a)
    except KeyError as e:
        return LawResult(op, registry, "kink", "rule_error", 0, None,
                         f"undeclared nonsmooth policy: {e}")
    try:
        primals, kwargs = spec.make()
        masks = spec.kink_mask(primals)

        # Reverse mode: seed a unit cotangent so the returned gradient IS the
        # selected subgradient (SUBGRAD_ZERO) or the tie share (SUBGRAD_SPLIT).
        out = vjp_fn(*(_unit_cotangent(jvp_fn, primals, kwargs),), *primals,
                     **kwargs)
        grads = out if isinstance(out, tuple) else (out,)

        detail = []
        if spec.expected == "split":
            if policy != SUBGRAD_SPLIT:
                return LawResult(op, registry, "kink", "fail", 1, None,
                                 f"spec expects a tie split but policy is {policy}")
            # Mass conservation: the tied elements' shares sum to 1.
            shares = []
            for g, m in zip(grads, masks):
                if g is None:
                    continue
                sel = np.asarray(g)[np.asarray(m)]
                if sel.size:
                    shares.append(sel)
            if not shares:
                return LawResult(op, registry, "kink", "fail", 1, None,
                                 "no tied elements found in the probe")
            stacked = np.concatenate([s.ravel() for s in shares])
            total = float(np.sum(stacked))
            n_tied = stacked.size
            equal = np.allclose(stacked, stacked[0], atol=1e-12)
            # Mass conservation is the whole point of SUBGRAD_SPLIT: a unit
            # cotangent arriving at one output must leave as shares summing to
            # 1 across its tie group. The probe declares how many independent
            # groups it contains, so the target is that count — hardcoding 1
            # would fail a correct rule on any multi-group probe.
            # (An earlier form also accepted `total == n_tied * share`, which
            # is vacuously true whenever the shares are equal — it let a rule
            # giving every tied element the FULL mass pass. Caught by the
            # mutation test in test_autodiff_laws.py.)
            groups = getattr(spec, "tie_groups", 1)
            conserved = abs(total - float(groups)) < 1e-9
            if not equal:
                return LawResult(op, registry, "kink", "fail", 1, None,
                                 f"tie shares not equal: {stacked.tolist()}")
            if not conserved:
                return LawResult(op, registry, "kink", "fail", 1, None,
                                 f"tie mass not conserved: sum={total} "
                                 f"over {groups} declared group(s)")
            detail.append(f"{n_tied} tied elements share {float(stacked[0]):.4g}")
        else:
            if policy != SUBGRAD_ZERO:
                return LawResult(op, registry, "kink", "fail", 1, None,
                                 f"spec expects a fixed value but policy is {policy}")
            for g, m in zip(grads, masks):
                if g is None:
                    continue
                sel = np.asarray(g)[np.asarray(m)]
                if sel.size and not np.allclose(sel, spec.expected, atol=1e-12):
                    return LawResult(
                        op, registry, "kink", "fail", 1, None,
                        f"at-kink value {sel.tolist()} != declared "
                        f"{spec.expected} ({policy})")

        # Forward mode must select the same element as reverse mode.
        if jvp_fn is not None:
            tangents = tuple(np.ones_like(np.asarray(p), dtype=np.float64)
                             if _is_float(p) else _zero_like(p) for p in primals)
            try:
                _, t_out = jvp_fn(primals, tangents, **kwargs)
            except Exception as e:  # noqa: BLE001
                return LawResult(op, registry, "kink", "rule_error", 1, None,
                                 f"jvp at kink: {type(e).__name__}: {e}")
            if spec.expected != "split":
                # With a unit tangent the JVP tangent equals the selection.
                for m in masks[:1]:
                    sel = np.asarray(t_out)[np.asarray(m)] \
                        if np.shape(t_out) == np.shape(m) else np.array([])
                    if sel.size and not np.allclose(sel, spec.expected, atol=1e-12):
                        return LawResult(
                            op, registry, "kink", "fail", 1, None,
                            f"forward mode selects {sel.tolist()} at the kink, "
                            f"reverse mode selects {spec.expected} — the modes "
                            f"disagree on the declared policy")
            detail.append("both modes agree")

        return LawResult(op, registry, "kink", "pass", 1, None, "; ".join(detail))
    except Exception as e:  # noqa: BLE001
        return LawResult(op, registry, "kink", "rule_error", 0, None,
                         f"{type(e).__name__}: {e}")


def _unit_cotangent(jvp_fn: Optional[Callable], primals: tuple, kwargs: dict):
    """A ones-cotangent shaped like the op's output."""
    if jvp_fn is not None:
        zeros = tuple(_zero_like(p) for p in primals)
        out = jvp_fn(primals, zeros, **kwargs)[0]
        return np.ones_like(np.asarray(out), dtype=np.float64)
    return np.ones_like(np.asarray(primals[0]), dtype=np.float64)


# ── Laws 2 and 4: the algebra laws that gate AD-WEIL-1 ───────────────────────


def _random_program(rng: np.random.Generator, depth: int):
    """A random straight-line program over +, x and (optionally) scalar fns.

    Returned as a closure taking (value, algebra-ops) so the SAME program text
    can be evaluated in any codomain — which is the whole point being tested.
    """
    from .algebra import SCALAR_RECURRENCES

    names = list(SCALAR_RECURRENCES)
    ops: list[tuple[str, Any]] = []
    for _ in range(depth):
        roll = rng.random()
        if roll < 0.4:
            ops.append(("add_const", float(rng.standard_normal())))
        elif roll < 0.8:
            ops.append(("square", None))
        else:
            ops.append(("fn", names[int(rng.integers(len(names)))]))
    return ops


def _eval_program(ops, x, W):
    """Evaluate a program in algebra `W` starting from lifted value `x`."""
    cur = x
    for kind, arg in ops:
        if kind == "add_const":
            const = W.lift(np.asarray(arg, dtype=np.float64),
                           np.zeros_like(np.asarray(arg, dtype=np.float64)))
            cur = W.add(cur, const)
        elif kind == "square":
            cur = W.mul(cur, cur)
        else:
            cur = _guard_jet(W, arg, cur)
            cur = W.scalar_fn(arg, cur)
    return cur


class _JetOps:
    """`_Ops` over a jet algebra, so a declared guard evaluates here too."""

    def __init__(self, W):
        self.W = W

    def apply(self, name, x):
        return self.W.scalar_fn(name, x)

    def mul(self, a, b):
        return self.W.mul(self._lift(a), self._lift(b))

    def add(self, a, b):
        return self.W.add(self._lift(a), self._lift(b))

    def neg(self, a):
        return self.mul(-1.0, a)

    def reciprocal(self, a):
        return self.W.scalar_fn("reciprocal", self._lift(a))

    def _lift(self, v):
        if isinstance(v, (list, tuple)):
            return v
        return self.W.lift(np.asarray(float(v)), np.asarray(0.0))


def _guard_jet(W, name, cur):
    from .algebra import SCALAR_RECURRENCES

    guard = SCALAR_RECURRENCES[name].guard_expr
    return cur if guard is None else guard(_JetOps(W), cur)


def homomorphism_check(order: int, trials: int = 8,
                       registry: str = "algebra") -> LawResult:
    """Law 2 — evaluation in ``W`` is an ℝ-algebra homomorphism.

    Tested against **analytically known** Taylor coefficients: a random
    polynomial is evaluated using only the algebra's `add`/`mul`, and its jet
    is compared with the exact derivatives obtained by differentiating the
    polynomial's coefficient vector. That is the substantive claim — the
    algebra's ring operations reproduce the true derivative tower — and it is
    where nilpotency does the work: for a polynomial of degree ≤ k there is no
    truncation error at all, so the only residual is float rounding.

    (An earlier draft compared ``W.mul(f, g)`` against ``W.mul(f, g)`` and was
    vacuously true, and used an absolute threshold on values that a random
    program can make large. Both are pinned by the tests.)
    """
    import math

    from .algebra import TruncatedJet

    W = TruncatedJet(order)
    rng = op_rng(f"W{order}", "homomorphism")
    try:
        worst = 0.0
        for _ in range(trials):
            deg = int(rng.integers(2, 6))
            coeffs = rng.standard_normal(deg + 1)      # c[0] + c[1]x + ...
            x0 = float(rng.standard_normal())

            # Evaluate by Horner using ONLY the algebra's ring operations.
            x = W.lift(np.asarray(x0), np.asarray(1.0))
            acc = W.lift(np.asarray(coeffs[-1]), np.asarray(0.0))
            for c in coeffs[-2::-1]:
                acc = W.add(W.mul(acc, x),
                            W.lift(np.asarray(c), np.asarray(0.0)))

            # Exact reference: differentiate the coefficient vector.
            for j in range(order + 1):
                d = coeffs.copy()
                for _ in range(j):
                    d = np.array([i * d[i] for i in range(1, len(d))]) \
                        if len(d) > 1 else np.array([0.0])
                exact = float(np.polyval(d[::-1], x0)) / math.factorial(j)
                got = float(np.asarray(acc[j]))
                worst = max(worst, abs(got - exact) / max(abs(exact), 1.0))

        status = "pass" if worst <= 1e-12 else "fail"
        return LawResult(f"W{order}", registry, "homomorphism", status,
                         trials, worst,
                         "polynomial jets == exact Taylor coefficients "
                         "(no truncation error by nilpotency)")
    except Exception as e:  # noqa: BLE001
        return LawResult(f"W{order}", registry, "homomorphism", "rule_error",
                         0, None, f"{type(e).__name__}: {e}")


def quotient_check(order: int, trials: int = 6,
                   registry: str = "algebra") -> LawResult:
    """Law 4 — order-k jet ≡ k-nested duals on the diagonal seed.

    This is the differential proof that licenses retiring the nested route
    (Decision #31: the survivor must carry what the deleted path carried).
    The two algebras are related by the diagonal embedding
    ``ε ↦ ε₁+…+ε_k``; reading the jet's order-j coefficient back requires the
    factorial scaling ``j!`` (the coefficient-convention semantic key).
    """
    import math

    from .algebra import TruncatedJet, nested_dual_derivative

    W = TruncatedJet(order)
    rng = op_rng(f"W{order}", "quotient")
    try:
        worst = 0.0
        for _ in range(trials):
            x0 = float(rng.standard_normal()) * 0.4
            ops = _random_program(rng, 3)
            jet = _eval_program(ops, W.lift(np.asarray(x0), np.asarray(1.0)), W)

            def program(t, scalar_ops, _ops=ops):
                from .algebra import Dual  # noqa: F401  (documents the pair)
                cur = t
                for kind, arg in _ops:
                    if kind == "add_const":
                        cur = cur + arg
                    elif kind == "square":
                        cur = cur * cur
                    else:
                        from .algebra import SCALAR_RECURRENCES
                        guard = SCALAR_RECURRENCES[arg].guard_expr
                        if guard is not None:
                            cur = guard(scalar_ops, cur)
                        cur = scalar_ops._apply(arg, cur)
                return cur

            for j in range(1, order + 1):
                ref = nested_dual_derivative(program, x0, j)
                got = float(np.asarray(jet[j])) * math.factorial(j)
                if not (np.isfinite(got) and np.isfinite(ref)):
                    return LawResult(f"W{order}", registry, "quotient", "fail",
                                     trials, None,
                                     "non-finite value in the comparison — a "
                                     "domain guard is missing or wrong")
                denom = max(abs(ref), 1.0)
                worst = max(worst, abs(got - ref) / denom)

        status = "pass" if worst <= 1e-9 else "fail"
        return LawResult(f"W{order}", registry, "quotient", status, trials,
                         worst, f"jet(k={order}) == {order}-nested duals")
    except Exception as e:  # noqa: BLE001
        return LawResult(f"W{order}", registry, "quotient", "rule_error", 0,
                         None, f"{type(e).__name__}: {e}")


# ── Law 6: enclosures and unbiasedness ───────────────────────────────────────
# These change the TYPE of the correctness claim. Laws 1-5 assert equalities;
# here the claims are "the exact value lies in this interval" and "this is an
# unbiased estimator of the operator". Both are testable, and neither is
# expressible as an equality — which is why the plan gives them their own row.


def enclosure_check(order: int, trials: int = 60,
                    registry: str = "algebra") -> LawResult:
    """Law 6a — a `TaylorModel` interval provably contains the exact jet.

    Runs the same program through the exact jet algebra and the outward-
    rounded interval algebra and asserts containment coefficient by
    coefficient. A certificate that did not contain the truth would be worse
    than no certificate, so a single escape is a failure.
    """
    from .algebra import TaylorModel, TruncatedJet

    TM, W = TaylorModel(order), TruncatedJet(order)
    rng = op_rng(f"W{order}", "enclosure")
    try:
        escapes = 0
        widest = 0.0
        for _ in range(trials):
            x = float(rng.standard_normal())
            c = float(rng.standard_normal())
            m = TM.mul(TM.add(TM.lift(x, 1.0), TM.lift(c, 0.0)), TM.lift(x, 1.0))
            j = W.mul(W.add(W.lift(np.asarray(x), np.asarray(1.0)),
                            W.lift(np.asarray(c), np.asarray(0.0))),
                      W.lift(np.asarray(x), np.asarray(1.0)))
            for i in range(order + 1):
                if not TM.contains(m, float(j[i]), i):
                    escapes += 1
                widest = max(widest, float(m[1][i] - m[0][i]))
        status = "pass" if escapes == 0 else "fail"
        return LawResult(f"W{order}", registry, "enclosure", status, trials,
                         widest,
                         f"{escapes} escape(s); widest interval {widest:.2e}")
    except Exception as e:  # noqa: BLE001
        return LawResult(f"W{order}", registry, "enclosure", "rule_error", 0,
                         None, f"{type(e).__name__}: {e}")


def unbiasedness_check(registry: str = "algebra") -> LawResult:
    """Law 6b — the randomized-jet Laplacian estimator is unbiased.

    Asserts the property that actually defines the estimator, not a
    tolerance: the error must fall like ``1/sqrt(N)``. A *biased* estimator
    would plateau instead, which a single-N tolerance check would happily
    accept. Draws come from the Philox stream, so this is deterministic and
    replayable (Decision #18) rather than a flaky statistical test.
    """
    from ..rng import RNGKey
    from .algebra import hutchinson_laplacian

    try:
        n = 6
        x = np.zeros(n)
        exact = 2.0 * n           # f = sum(x^2) + 3 sum(x)  =>  Laplacian 2n

        def quad(lifted, W):
            sq = W.mul(lifted, lifted)
            return [sq[i] + 3.0 * lifted[i] for i in range(len(lifted))]

        errs = []
        for s in (64, 1024):
            est = hutchinson_laplacian(quad, x, RNGKey(0), s)
            errs.append(abs(est - exact))
        # 16x the samples should cut the error ~4x; allow a wide factor so the
        # test checks the SHAPE of convergence, not a lucky draw.
        ratio = errs[0] / max(errs[1], 1e-12)
        status = "pass" if ratio > 2.0 else "fail"
        return LawResult("hutchinson", registry, "unbiasedness", status, 2,
                         ratio,
                         f"error {errs[0]:.3f} -> {errs[1]:.3f} over 16x "
                         f"samples (ratio {ratio:.2f}, 1/sqrt(N) predicts 4)")
    except Exception as e:  # noqa: BLE001
        return LawResult("hutchinson", registry, "unbiasedness", "rule_error",
                         0, None, f"{type(e).__name__}: {e}")
