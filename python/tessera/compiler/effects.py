"""
tessera.compiler.effects — EffectLattice and Effect type system.

Effects flow upward through the call graph. A function that calls an RNG op
is tagged `random`. A function that calls a `write` collective is tagged `io`.
A @jit(deterministic=True) block FORBIDS `random` unless the function is
also decorated with seed=N.

Lattice order (least → most permissive):
    pure < random < movement < state < collective < memory < io < top

The EffectLattice consumes canonical Graph-IR operation records. Python call
spellings are deliberately irrelevant: aliases and dispatch must be resolved
by Graph emission or concrete tracing before effects are trusted.

Reference: CLAUDE.md §Key Design Contracts — Effect Lattice
           src/programming_model/docs/Tessera_Programming_Model_v1_1_Plan_20250917_212640.md §1.2
"""

from __future__ import annotations
import enum
import inspect
import weakref
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .op_catalog import OP_SPECS, get_op_spec


# ─────────────────────────────────────────────────────────────────────────────
# Error type
# ─────────────────────────────────────────────────────────────────────────────

class TesseraEffectError(Exception):
    """
    Raised when an effect contract is violated.

    Most common case: a @jit(deterministic=True) function contains or calls
    an op with `random` effect without a seed.

    Attributes:
        fn_name       : name of the function with the violation
        declared      : the effect declared by the @jit contract
        inferred      : the effect inferred from the function body
        offending_ops : list of op names that caused the violation
    """

    def __init__(
        self,
        fn_name: str,
        declared: "Effect",
        inferred: "Effect",
        offending_ops: Optional[List[str]] = None,
        message: Optional[str] = None,
    ) -> None:
        self.fn_name = fn_name
        self.declared = declared
        self.inferred = inferred
        self.offending_ops = offending_ops or []
        self._message = message
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        if self._message:
            return self._message
        msg = (
            f"Effect contract violation in {self.fn_name!r}: "
            f"declared {self.declared.name!r} but inferred {self.inferred.name!r}"
        )
        if self.offending_ops:
            msg += f". Offending ops: {self.offending_ops}"
        return msg


# ─────────────────────────────────────────────────────────────────────────────
# Effect enum — the lattice
# ─────────────────────────────────────────────────────────────────────────────

class Effect(enum.IntEnum):
    """
    Effect lattice for Tessera functions. Values are ordered from least to
    most permissive (pure=0 is the strictest).

    Lattice:
        pure < random < movement < state < collective < memory < io < top

    Semantics:
        pure   — no side effects; output depends only on inputs; recompute safe
        random — may call RNG; result varies across identical inputs
        movement — explicit prefetch/copy/wait movement effects
        state  — reads or writes compiler-visible state (e.g., KV cache)
        collective — performs async device/rank communication
        memory — writes mutable tensors or aliases host-visible memory
        io     — performs host I/O or unknown external calls
        top    — unknown / unconstrained (conservative fallback)
    """
    pure       = 0
    random     = 1
    movement   = 2
    state      = 3
    collective = 4
    memory     = 5
    io         = 6
    top        = 7

    def join(self, other: "Effect") -> "Effect":
        """
        Lattice join (least upper bound). Used to propagate effects upward
        through a call graph: the caller inherits the max effect of all callees.
        """
        return Effect(max(self.value, other.value))

    def __le__(self, other: "Effect") -> bool:  # type: ignore[override]
        return self.value <= other.value

    def __lt__(self, other: "Effect") -> bool:  # type: ignore[override]
        return self.value < other.value

    def __ge__(self, other: "Effect") -> bool:  # type: ignore[override]
        return self.value >= other.value

    def __gt__(self, other: "Effect") -> bool:  # type: ignore[override]
        return self.value > other.value


# ─────────────────────────────────────────────────────────────────────────────
# Known op → effect mappings
# ─────────────────────────────────────────────────────────────────────────────

# Compatibility view of the canonical catalog for public-name lookup.
# Inference itself consumes emitted/traced Graph operation records.
_OP_EFFECTS: Dict[str, Effect] = {
    name: Effect[spec.effect]
    for name, spec in OP_SPECS.items()
}
_OP_EFFECTS.update({
    "randn": Effect.random,
    "rand": Effect.random,
    "bernoulli": Effect.random,
    "normal": Effect.random,
    "prefetch": Effect.movement,
    "async_copy": Effect.movement,
    "await_movement": Effect.movement,
    "kv_cache_create": Effect.state,
    "kv_cache_read": Effect.state,
    "kv_cache_write": Effect.state,
    # SD1-3 speculative-decode cache cursor ops — typed state effect (the
    # IR-visible form of advance_kv / advance_ssm; no hidden Python mutation).
    "cache_commit": Effect.state,
    "cache_rollback": Effect.state,
    "all_to_all": Effect.collective,
    "await": Effect.collective,
    "send": Effect.collective,
    "recv": Effect.collective,
    "barrier": Effect.collective,
})

def registered_op_effect(
    op_name: str, attrs: Mapping[str, Any] | None = None,
) -> Effect:
    """Return the effect declared by a canonical Graph operation record.

    Unknown operations fail closed as ``top``. An explicit traced
    ``tessera.effect_kind`` may refine a custom operation; otherwise the
    canonical operation catalog is authoritative.
    """
    attrs = attrs or {}
    explicit = attrs.get("tessera.effect_kind", attrs.get("effect"))
    if isinstance(explicit, str) and explicit in Effect.__members__:
        return Effect[explicit]
    spec = get_op_spec(op_name)
    if spec is not None:
        return Effect[spec.effect]
    bare = op_name.rsplit(".", 1)[-1]
    return _OP_EFFECTS.get(bare, Effect.top)


def infer_graph_effects(body: Iterable[Any]) -> tuple[Effect, List[str]]:
    """Join registered effects across IROp-like traced Graph records."""
    inferred = Effect.pure
    offending: List[str] = []
    for op in body:
        name = str(getattr(op, "op_name", ""))
        effect = registered_op_effect(name, getattr(op, "kwargs", None))
        inferred = inferred.join(effect)
        if effect > Effect.pure:
            offending.append(name or "<unregistered-op>")
    return inferred, offending


# ─────────────────────────────────────────────────────────────────────────────
# EffectLattice
# ─────────────────────────────────────────────────────────────────────────────

class EffectLattice:
    """
    Infers and validates the effect level of a Tessera function.

    Effects are inferred from emitted/traced Graph operations. The former AST
    call-name visitor was retired because aliases could make it fail open.

    Usage:
        lattice = EffectLattice()
        inferred = lattice.infer(fn)           # infer from source

        # Validate deterministic contract:
        lattice.check_deterministic(fn, seed=42)   # raises if fn has random effect

    The @jit decorator calls this automatically. Users rarely interact with
    EffectLattice directly.
    """

    def __init__(self) -> None:
        # Cache keyed by the function object via a weak reference, so a GC'd
        # function cannot have its id() reused by a different function and return
        # a stale inferred effect (and the cache can't grow unbounded).
        self._cache: "weakref.WeakKeyDictionary[Callable, Effect]" = weakref.WeakKeyDictionary()

    def infer(self, fn: Callable, source_text: Optional[str] = None) -> Effect:
        """
        Infer the effect level from the function's emitted Graph IR.

        Returns:
            Effect — the inferred effect level

        Note: Functions whose source cannot be retrieved (built-ins, C
        extensions) are conservatively assigned Effect.top.
        """
        use_cache = source_text is None
        if use_cache:
            try:
                cached = self._cache.get(fn)
            except TypeError:
                use_cache = False  # fn not weak-referenceable (e.g. a builtin)
            else:
                if cached is not None:
                    return cached

        try:
            from .graph_ir import GraphIRBuilder

            source = source_text if source_text is not None else inspect.getsource(fn)
            builder = GraphIRBuilder()
            builder.lower(fn, source_text=source)
            if any(d.severity in {"warning", "error"} for d in builder.diagnostics):
                result = Effect.top
                if use_cache:
                    self._cache[fn] = result
                return result
            body = [op for graph_fn in builder.module().functions
                    for op in graph_fn.body]
            result, _ = infer_graph_effects(body)
        except Exception:
            if use_cache:
                self._cache[fn] = Effect.top
            return Effect.top
        if use_cache:
            self._cache[fn] = result
        return result

    def infer_with_ops(self, fn: Callable, source_text: Optional[str] = None):
        """
        Like infer(), but also returns the list of offending op names.

        Returns:
            (Effect, List[str]) — effect level and offending ops
        """
        try:
            from .graph_ir import GraphIRBuilder

            source = source_text if source_text is not None else inspect.getsource(fn)
            builder = GraphIRBuilder()
            builder.lower(fn, source_text=source)
            if any(d.severity in {"warning", "error"} for d in builder.diagnostics):
                return Effect.top, ["<untraceable-graph>"]
            body = [op for graph_fn in builder.module().functions
                    for op in graph_fn.body]
            return infer_graph_effects(body)
        except Exception:
            return Effect.top, ["<untraceable-graph>"]

    def check_deterministic(
        self,
        fn: Callable,
        seed: Optional[int] = None,
        source_text: Optional[str] = None,
    ) -> None:
        """
        Validate that fn satisfies the @jit(deterministic=True) contract.

        A deterministic function may contain movement, state, and collective
        effects only when they are represented in Tessera IR, where the runtime
        can impose stream/order contracts. RNG requires a seed. Host I/O and
        unknown calls remain forbidden.

        Args:
            fn   : the function to validate
            seed : if provided, random ops are allowed (seeded RNG is deterministic)

        Raises:
            TesseraEffectError: if fn has unseeded RNG or host I/O/unknown effect
        """
        inferred, offending_ops = self.infer_with_ops(fn, source_text=source_text)

        if inferred >= Effect.io:
            raise TesseraEffectError(
                fn_name=fn.__name__,
                declared=Effect.pure,
                inferred=inferred,
                offending_ops=offending_ops,
                message=(
                    f"@jit(deterministic=True) function {fn.__name__!r} performs "
                    f"host I/O or unknown external work ({inferred.name}), which "
                    f"cannot be made deterministic. Remove deterministic=True or "
                    f"eliminate the offending ops: {offending_ops}"
                ),
            )

        random_ops = [
            op for op in offending_ops
            if registered_op_effect(op) == Effect.random
        ]
        if random_ops and seed is None:
            raise TesseraEffectError(
                fn_name=fn.__name__,
                declared=Effect.pure,
                inferred=inferred,
                offending_ops=random_ops,
                message=(
                    f"@jit(deterministic=True) function {fn.__name__!r} calls RNG ops "
                    f"({random_ops}) without a seed. Either add seed=<int> to "
                    f"@jit(deterministic=True, seed=42) or remove the RNG calls."
                ),
            )

    def join(self, effects: List[Effect]) -> Effect:
        """Compute the join (least upper bound) of a list of effects."""
        result = Effect.pure
        for e in effects:
            result = result.join(e)
        return result

    def invalidate(self, fn: Callable) -> None:
        """Remove a cached inference result (e.g., after function mutation)."""
        try:
            self._cache.pop(fn, None)
        except TypeError:
            pass  # not weak-referenceable → was never cached

    def __repr__(self) -> str:
        return f"EffectLattice(cached={len(self._cache)} functions)"
