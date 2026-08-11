"""Stochastic computation-graph typing (C3).

Motivation (from the *Elements of Differentiable Programming* review). A program
with randomness is a DAG of two node kinds — **function nodes** and
**distribution nodes** — and a node's output is a random variable iff its set of
random parents is non-empty (Blondel & Roulet §11.5). That is a *derivable*
forward-dataflow property. W2.2 retired the former source-name walker; this
module supplies the concrete trace certificate that additionally resolves an
RNG reached through an alias, local, helper, ``getattr``, or dict dispatch.

This module derives the same fact from the **traced IR** instead. Every op is
present in the trace by its canonical ``op_name`` regardless of how the Python
reached it, so randomness cannot hide behind an alias (Decision #30 — derive,
don't ask). Given a straight-line IR body (a list of ``IROp``-like nodes with
``result`` / ``op_name`` / ``operands``) it computes:

  * which SSA values are random (forward propagation from distribution nodes);
  * whether the function is deterministic (its outputs' dependency cone contains
    no random node) — the fail-closed certificate ``@jit(deterministic=True)``
    should consult;
  * which gradient estimator is legal per output (§11.5): a purely-function-node
    path admits the **pathwise / reparameterization** estimator; any distribution
    node on the path forces the **score-function** estimator unless every such
    node is reparameterizable.

It is intentionally standalone (analysis + certificate), the substrate the W2
dataflow-effects work can wire into the real ``deterministic=True`` gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from .effects import Effect, _OP_EFFECTS


def _random_op_names() -> frozenset[str]:
    """Distribution-node op names, derived from the existing effect table plus
    the rng sampler surface — not a hand-copied list (Decision #29: reuse the
    declared source). Names are matched on the last dotted component so
    ``tessera.normal`` and ``normal`` both resolve."""
    names = {
        name for name, eff in _OP_EFFECTS.items() if eff == Effect.random
    }
    # The rng sampler surface (rng.__all__ minus the key type and the iterative
    # MCMC samplers, which are function nodes over an explicit noise input).
    names |= {
        "uniform", "normal", "truncated_normal", "bernoulli", "categorical",
        "gumbel", "multinomial", "randint", "permutation", "gamma", "beta",
        "dirichlet", "poisson", "randn", "rand", "dropout",
        "rng_philox_uniform", "rng_philox_normal",
    }
    return frozenset(names)


DISTRIBUTION_OPS: frozenset[str] = _random_op_names()

# Continuous distributions with a reparameterization (location-scale or
# inverse-CDF) → the pathwise estimator applies (lower variance). Discrete
# distributions have no pathwise gradient → score-function estimator.
REPARAMETERIZABLE_OPS: frozenset[str] = frozenset({
    "uniform", "normal", "truncated_normal", "gumbel", "gamma", "beta",
    "dirichlet", "rng_philox_uniform", "rng_philox_normal",
})


class EstimatorProvenance(str, Enum):
    """Compiler-visible meaning of a stochastic derivative request."""

    CONSTANT_NOISE = "constant_noise"
    PATHWISE = "pathwise"
    SCORE_FUNCTION = "score_function"
    REJECT = "reject"


@dataclass(frozen=True)
class EstimatorDecision:
    provenance: EstimatorProvenance
    legal: bool
    reason: str


def resolve_estimator_provenance(
    op_name: str, *, requested: str | None = None,
    explicit_key: bool = False, replayable: bool = False,
) -> EstimatorDecision:
    """Fail-closed estimator selection for one stochastic Graph operation.

    Selection is never inferred from a generic ``random`` effect alone. A
    discrete distribution needs an explicit score-function request, continuous
    pathwise differentiation needs explicit counter state, and constant-noise
    differentiation needs an exact replay identity.
    """
    base = _base_name(op_name)
    try:
        choice = EstimatorProvenance(requested or "reject")
    except ValueError:
        return EstimatorDecision(
            EstimatorProvenance.REJECT, False,
            f"unknown estimator provenance {requested!r}")
    if choice == EstimatorProvenance.CONSTANT_NOISE:
        if replayable and explicit_key:
            return EstimatorDecision(choice, True, "explicit key/counter replay")
        return EstimatorDecision(
            EstimatorProvenance.REJECT, False,
            "constant_noise requires an explicit replayable key/counter identity")
    if choice == EstimatorProvenance.PATHWISE:
        if base in REPARAMETERIZABLE_OPS and explicit_key:
            return EstimatorDecision(choice, True, "explicit reparameterized noise")
        return EstimatorDecision(
            EstimatorProvenance.REJECT, False,
            "pathwise requires a reparameterizable operation and explicit key/counter")
    if choice == EstimatorProvenance.SCORE_FUNCTION:
        if is_distribution_op(op_name):
            return EstimatorDecision(choice, True, "explicit score-function estimator")
        return EstimatorDecision(
            EstimatorProvenance.REJECT, False,
            "score_function applies only to distribution operations")
    return EstimatorDecision(
        EstimatorProvenance.REJECT, False,
        "stochastic differentiation requires explicit estimator provenance")


def _base_name(op_name: str) -> str:
    base = op_name.rsplit(".", 1)[-1]
    if base.startswith("rng_") and base not in {
        "rng_philox_uniform", "rng_philox_normal"
    }:
        return base.removeprefix("rng_")
    return base


def _ssa_name(value: str) -> str:
    """Canonicalize Graph IR's optional SSA ``%`` prefix."""
    return value[1:] if value.startswith("%") else value


def is_distribution_op(op_name: str) -> bool:
    return _base_name(op_name) in DISTRIBUTION_OPS


def is_reparameterizable_op(op_name: str) -> bool:
    return _base_name(op_name) in REPARAMETERIZABLE_OPS


@dataclass(frozen=True)
class StochasticAnalysis:
    """Result of typing a straight-line IR body as a stochastic graph."""

    random_values: frozenset[str]        # SSA names that are random variables
    distribution_nodes: tuple[str, ...]  # SSA names produced by a distribution op
    is_deterministic: bool               # outputs' cone has no random node
    estimator: str                       # 'none' | 'pathwise' | 'score_function'
    reason: str

    def certificate(self) -> tuple[bool, str]:
        """Fail-closed determinism certificate for @jit(deterministic=True)."""
        return self.is_deterministic, self.reason


def _iter_nodes(body: Iterable[Any]) -> Iterable[tuple[str | None, str, tuple[str, ...]]]:
    """Normalize IROp-like nodes to ``(result, op_name, operands)`` triples.

    Accepts real ``IROp`` objects (duck-typed) or plain ``(result, op_name,
    operands)`` tuples, so callers can analyze a live trace or a synthetic
    graph without an adapter.
    """
    for node in body:
        if isinstance(node, tuple):
            result, op_name, operands = node
            yield result, op_name, tuple(operands)
        else:
            yield (
                getattr(node, "result", None),
                getattr(node, "op_name"),
                tuple(getattr(node, "operands", ()) or ()),
            )


def analyze_stochastic_graph(
    body: Sequence[Any],
    outputs: Sequence[str],
    *,
    random_inputs: Iterable[str] = (),
) -> StochasticAnalysis:
    """Type a straight-line IR body as a stochastic computation graph.

    ``body`` is a sequence of ``IROp``-like nodes (or ``(result, op_name,
    operands)`` tuples) in topological order; ``outputs`` are the result SSA
    names of interest; ``random_inputs`` are argument SSA names already known to
    be random (e.g. an externally-sampled noise input).

    Randomness propagates forward: a value is random if it is produced by a
    distribution op or if any operand is random. Determinism is decided over the
    outputs' dependency cone only — a discarded random draw does not make the
    function non-deterministic in its outputs.
    """
    random_values: set[str] = {_ssa_name(v) for v in random_inputs}
    distribution_nodes: list[str] = []
    producer: dict[str, tuple[str, tuple[str, ...]]] = {}

    for result, op_name, operands in _iter_nodes(body):
        result = _ssa_name(result) if result is not None else None
        operands = tuple(_ssa_name(op) for op in operands)
        if result is not None:
            producer[result] = (op_name, operands)
        is_dist = is_distribution_op(op_name)
        if is_dist and result is not None:
            distribution_nodes.append(result)
        random_here = is_dist or any(op in random_values for op in operands)
        if random_here and result is not None:
            random_values.add(result)

    # Dependency cone of the outputs (backward reachability).
    cone: set[str] = set()
    stack = [_ssa_name(output) for output in outputs]
    while stack:
        v = stack.pop()
        if v in cone:
            continue
        cone.add(v)
        if v in producer:
            _op, ops = producer[v]
            stack.extend(ops)

    cone_random = [v for v in cone if v in random_values]
    is_deterministic = not any(_ssa_name(o) in random_values for o in outputs) and not cone_random

    # Estimator classification over the distribution nodes inside the cone.
    cone_dist_ops = [
        producer[v][0] for v in distribution_nodes if v in cone and v in producer
    ]
    if not cone_dist_ops and is_deterministic:
        estimator = "none"
        reason = "no distribution node in the outputs' dependency cone"
    elif cone_dist_ops and all(is_reparameterizable_op(op) for op in cone_dist_ops):
        estimator = "pathwise"
        reason = "all distribution nodes on the path are reparameterizable"
    else:
        estimator = "score_function"
        reason = (
            "a non-reparameterizable (discrete) distribution node is on the path"
            if cone_dist_ops
            else "outputs depend on a random input with no reparameterization"
        )

    return StochasticAnalysis(
        random_values=frozenset(random_values),
        distribution_nodes=tuple(distribution_nodes),
        is_deterministic=is_deterministic,
        estimator=estimator,
        reason=reason,
    )


def certify_deterministic(
    body: Sequence[Any],
    outputs: Sequence[str],
    *,
    random_inputs: Iterable[str] = (),
) -> tuple[bool, str]:
    """Fail-closed determinism certificate over traced IR.

    Returns ``(True, reason)`` only when no random node reaches the outputs.
    Unlike the AST effect walker, this cannot be fooled by aliasing an RNG op —
    the op is in the trace by its canonical name. This is the derivation
    ``@jit(deterministic=True)`` should trust once wired (Decision #30).
    """
    return analyze_stochastic_graph(
        body, outputs, random_inputs=random_inputs
    ).certificate()
