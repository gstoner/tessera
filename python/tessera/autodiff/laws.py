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


def _leaves(x: Any) -> list[np.ndarray]:
    """Flatten an op output (array / tuple / list / dict) to float arrays."""
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
    return [arr] if np.issubdtype(arr.dtype, np.floating) else []


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
    if np.issubdtype(arr.dtype, np.floating):
        return make(arr)
    return np.zeros_like(arr)


def _dot(a: Any, b: Any) -> float:
    la, lb = _leaves(a), _leaves(b)
    if len(la) != len(lb):
        raise ValueError(f"structure mismatch: {len(la)} vs {len(lb)} leaves")
    return float(sum(np.sum(np.asarray(x, dtype=np.float64) * np.asarray(y, dtype=np.float64))
                     for x, y in zip(la, lb)))


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

        max_res = 0.0
        for _ in range(spec.probes):
            tangents = tuple(
                rng.standard_normal(np.shape(p)) if i in diff and _is_float(p)
                else _zero_like(p)
                for i, p in enumerate(primals)
            )
            _, t_out = jvp_fn(primals, tangents, **kwargs)
            u = _like_leaves(t_out, lambda leaf: rng.standard_normal(leaf.shape))
            lhs = _dot(t_out, u)

            grads = vjp_fn(u if not _single_leaf(t_out) else _leaves(u)[0],
                           *primals, **kwargs)
            if not isinstance(grads, tuple):
                grads = (grads,)
            rhs = 0.0
            for i in diff:
                if i < len(grads) and grads[i] is not None:
                    rhs += _dot(tangents[i], grads[i])

            res = abs(lhs - rhs) / max(abs(lhs) + abs(rhs), ADJOINT_RTOL * _scale(t_out, u))
            max_res = max(max_res, res)

        status = "pass" if max_res <= ADJOINT_RTOL * 10 else "fail"
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
    return np.issubdtype(np.asarray(p).dtype, np.floating)


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
        y0 = np.asarray(primal_out(primals), dtype=np.float64)
        s = 1.0 / (1.0 + float(np.max(np.abs(y0))))

        # Primal-consistency gate: the JVP's own primal output must be the
        # canonical forward. A JVP whose primal half already disagrees is
        # differentiating some other function, whatever its tangent says.
        zeros = tuple(_zero_like(p) for p in primals)
        y_jvp = np.asarray(jvp_fn(primals, zeros, **kwargs)[0], dtype=np.float64)
        if y_jvp.shape != y0.shape or not np.allclose(y_jvp, y0, rtol=1e-9, atol=1e-12):
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

        status = "pass" if max_res <= CHAIN_RTOL else "fail"
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
    from .law_inputs import LAW_INPUT_SPECS

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
