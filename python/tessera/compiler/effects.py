"""
tessera.compiler.effects — EffectLattice and Effect type system.

Effects flow upward through the call graph. A function that calls an RNG op
is tagged `random`. A function that calls a `write` collective is tagged `io`.
A @jit(deterministic=True) block FORBIDS `random` unless the function is
also decorated with seed=N.

Lattice order (least → most permissive):
    pure < random < memory < io < top

The EffectLattice walks a function's call graph (in Phase 1: inspects the
function body's AST for known Tessera op calls) and infers the effect level.

Reference: CLAUDE.md §Key Design Contracts — Effect Lattice
           src/programming_model/docs/Tessera_Programming_Model_v1_1_Plan_20250917_212640.md §1.2
"""

from __future__ import annotations
import ast
import enum
import inspect
import textwrap
from typing import Callable, Dict, FrozenSet, List, Optional, Set


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
        pure (0) < random (1) < memory (2) < io (3) < top (4)

    Semantics:
        pure   — no side effects; output depends only on inputs; recompute safe
        random — may call RNG; result varies across identical inputs
        memory — reads or writes mutable state (e.g., KV cache)
        io     — performs collective communication or host I/O
        top    — unknown / unconstrained (conservative fallback)
    """
    pure   = 0
    random = 1
    memory = 2
    io     = 3
    top    = 4

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

# Maps known tessera op names to the effect they introduce.
# Phase 1: conservative static table. Phase 2: derive from op ODS attributes.
_OP_EFFECTS: Dict[str, Effect] = {
    # Pure math ops
    "gemm":        Effect.pure,
    "matmul":      Effect.pure,
    "conv2d":      Effect.pure,
    "layer_norm":  Effect.pure,
    "softmax":     Effect.pure,
    "gelu":        Effect.pure,
    "relu":        Effect.pure,
    "transpose":   Effect.pure,
    "cast":        Effect.pure,
    "fused_epilogue": Effect.pure,

    # Random ops
    "dropout":     Effect.random,
    "randn":       Effect.random,
    "rand":        Effect.random,
    "bernoulli":   Effect.random,
    "normal":      Effect.random,

    # Memory (stateful) ops — KV cache, mutable buffers
    "kv_cache_read":  Effect.memory,
    "kv_cache_write": Effect.memory,
    "flash_attn":     Effect.memory,  # conservative: may read/write KV cache

    # Collective / IO ops
    "all_reduce":      Effect.io,
    "reduce_scatter":  Effect.io,
    "all_gather":      Effect.io,
    "send":            Effect.io,
    "recv":            Effect.io,
    "barrier":         Effect.io,
}

# Attribute / module paths that signal random (numpy, torch, etc.)
_RANDOM_ATTR_PATTERNS: FrozenSet[str] = frozenset({
    "np.random",
    "numpy.random",
    "torch.rand",
    "torch.randn",
    "random.random",
    "random.randint",
})


# ─────────────────────────────────────────────────────────────────────────────
# AST walker for Phase 1 effect inference
# ─────────────────────────────────────────────────────────────────────────────

class _EffectVisitor(ast.NodeVisitor):
    """
    Walks a function's AST and collects the effects of all tessera op calls
    and any random library calls.

    Phase 1 scope: only inspects `tessera.ops.<name>`, `ops.<name>`, and
    known random patterns. Full inter-procedural analysis is Phase 2.
    """

    def __init__(self) -> None:
        self.inferred: Effect = Effect.pure
        self.offending_ops: List[str] = []

    def _record(self, op_name: str, effect: Effect) -> None:
        if effect > Effect.pure:
            self.offending_ops.append(op_name)
        self.inferred = self.inferred.join(effect)

    def visit_Call(self, node: ast.Call) -> None:
        op_name = self._resolve_call_name(node)
        if op_name:
            # Check tessera.ops.<name> or ops.<name>
            bare = op_name.split(".")[-1]
            if bare in _OP_EFFECTS:
                self._record(op_name, _OP_EFFECTS[bare])
            # Check random library patterns
            for pat in _RANDOM_ATTR_PATTERNS:
                if op_name.startswith(pat):
                    self._record(op_name, Effect.random)
                    break
        self.generic_visit(node)

    @staticmethod
    def _resolve_call_name(node: ast.Call) -> Optional[str]:
        """Extract the dotted name of a call, e.g. 'tessera.ops.gemm'."""
        func = node.func
        parts = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        return ".".join(reversed(parts)) if parts else None


# ─────────────────────────────────────────────────────────────────────────────
# EffectLattice
# ─────────────────────────────────────────────────────────────────────────────

class EffectLattice:
    """
    Infers and validates the effect level of a Tessera function.

    Phase 1: AST-based single-function analysis.
    Phase 2: full inter-procedural dataflow over the Graph IR call graph.

    Usage:
        lattice = EffectLattice()
        inferred = lattice.infer(fn)           # infer from source

        # Validate deterministic contract:
        lattice.check_deterministic(fn, seed=42)   # raises if fn has random effect

    The @jit decorator calls this automatically. Users rarely interact with
    EffectLattice directly.
    """

    def __init__(self) -> None:
        # Cache: fn_id → inferred Effect
        self._cache: Dict[int, Effect] = {}

    def infer(self, fn: Callable) -> Effect:
        """
        Infer the effect level of fn by walking its AST.

        Returns:
            Effect — the inferred effect level (pure, random, memory, io, top)

        Note: Functions whose source cannot be retrieved (built-ins, C
        extensions) are conservatively assigned Effect.top.
        """
        fn_id = id(fn)
        if fn_id in self._cache:
            return self._cache[fn_id]

        try:
            source = inspect.getsource(fn)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            # Cannot inspect — conservative fallback
            self._cache[fn_id] = Effect.top
            return Effect.top

        visitor = _EffectVisitor()
        visitor.visit(tree)
        result = visitor.inferred
        self._cache[fn_id] = result
        return result

    def infer_with_ops(self, fn: Callable):
        """
        Like infer(), but also returns the list of offending op names.

        Returns:
            (Effect, List[str]) — effect level and offending ops
        """
        try:
            source = inspect.getsource(fn)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            return Effect.top, ["<uninspectable>"]

        visitor = _EffectVisitor()
        visitor.visit(tree)
        return visitor.inferred, visitor.offending_ops

    def check_deterministic(
        self,
        fn: Callable,
        seed: Optional[int] = None,
    ) -> None:
        """
        Validate that fn satisfies the @jit(deterministic=True) contract.

        A deterministic function must not have `random` effect UNLESS a
        seed is provided (which makes the randomness reproducible).

        Args:
            fn   : the function to validate
            seed : if provided, random ops are allowed (seeded RNG is deterministic)

        Raises:
            TesseraEffectError: if fn has random/io/top effect and no seed is given
        """
        inferred, offending_ops = self.infer_with_ops(fn)

        # io effect is always forbidden under deterministic=True
        if inferred >= Effect.io:
            raise TesseraEffectError(
                fn_name=fn.__name__,
                declared=Effect.pure,
                inferred=inferred,
                offending_ops=offending_ops,
                message=(
                    f"@jit(deterministic=True) function {fn.__name__!r} performs I/O "
                    f"or collective communication ({inferred.name}), which cannot be "
                    f"made deterministic. Remove the deterministic=True flag or "
                    f"eliminate the offending ops: {offending_ops}"
                ),
            )

        # random effect is allowed only if seed is provided
        if inferred >= Effect.random and seed is None:
            raise TesseraEffectError(
                fn_name=fn.__name__,
                declared=Effect.pure,
                inferred=inferred,
                offending_ops=offending_ops,
                message=(
                    f"@jit(deterministic=True) function {fn.__name__!r} calls RNG ops "
                    f"({offending_ops}) without a seed. Either add seed=<int> to "
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
        self._cache.pop(id(fn), None)

    def __repr__(self) -> str:
        return f"EffectLattice(cached={len(self._cache)} functions)"
