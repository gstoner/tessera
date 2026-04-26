"""
tessera.compiler.jit — @jit decorator that drives the Phase 1 compiler pipeline.

The @jit decorator:
  1. Collects constraints registered via tessera.require() in the function body
  2. Checks those constraints against any concrete bindings available at decoration time
  3. Infers effects via EffectLattice
  4. Validates deterministic contracts (deterministic=True + seed)
  5. Emits Graph IR text via GraphIRBuilder
  6. Returns a JitFn wrapper that executes the Python function eagerly (Phase 1)

Phase 3: replace step 6 with compiled kernel dispatch through the MLIR toolchain.

Reference: CLAUDE.md §Key Design Contracts
           CLAUDE.md §Phase 1 Mission
"""

from __future__ import annotations
import functools
import inspect
from typing import Any, Callable, Dict, List, Optional

from .constraints import Constraint, ConstraintSolver, TesseraConstraintError
from .effects import Effect, EffectLattice, TesseraEffectError
from .graph_ir import GraphIRBuilder, GraphIRModule
from .gpu_target import GPUTargetProfile, ISA  # noqa: F401 — re-exported for callers
from .attn_lower import FlashAttnLoweringConfig, SM90_DEFAULT  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Error type
# ─────────────────────────────────────────────────────────────────────────────

class TesseraJitError(Exception):
    """
    Raised by @jit when the compilation pipeline fails for any reason other
    than a constraint or effect violation (those raise their own error types).

    Wraps unexpected errors in the lowering or emission steps.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Global constraint registry
# ─────────────────────────────────────────────────────────────────────────────

# Phase 1: constraints are collected via tessera.require() calls that happen
# *inside* @jit-decorated function bodies. Since we run the function body
# eagerly in Phase 1, we intercept require() at decoration time by parsing
# the AST. At call time, require() is a no-op (constraints already checked).

_ACTIVE_CONSTRAINTS: List[Constraint] = []


def require(constraint: Constraint) -> None:
    """
    Register a structural constraint on the enclosing @jit function.

    At decoration time: collected by ConstraintSolver and checked against
    any concrete dimension bindings extracted from the type signature.

    At call time (Phase 1): no-op. The constraint was already checked.

    Usage:
        @tessera.jit
        def aligned_gemm(A: Tensor["M", "K"], B: Tensor["K", "N"]):
            tessera.require(tessera.constraint.Divisible("K", 64))
            return tessera.ops.gemm(A, B)
    """
    _ACTIVE_CONSTRAINTS.append(constraint)


# ─────────────────────────────────────────────────────────────────────────────
# Constraint extraction from AST
# ─────────────────────────────────────────────────────────────────────────────

import ast
import textwrap


class _ConstraintExtractor(ast.NodeVisitor):
    """
    Walk a @jit function body and extract tessera.require(...) calls,
    instantiating the constraint objects they describe.
    """

    # Map from predicate class name → constructor (imported at module level)
    _PREDICATE_CTORS: Dict[str, type] = {}  # filled after imports below

    def __init__(self) -> None:
        self.constraints: List[Constraint] = []

    def visit_Expr(self, node: ast.Expr) -> None:
        """Handle bare expression statements like `tessera.require(...)`."""
        if isinstance(node.value, ast.Call):
            self._try_extract(node.value)
        self.generic_visit(node)

    def _try_extract(self, call: ast.Call) -> None:
        # Match: require(...) or tessera.require(...)
        func_name = self._resolve_name(call.func)
        if not func_name or not func_name.endswith("require"):
            return
        if not call.args:
            return

        arg = call.args[0]
        if not isinstance(arg, ast.Call):
            return

        pred_name = self._resolve_name(arg.func)
        if not pred_name:
            return
        bare = pred_name.split(".")[-1]

        try:
            pred_args = [ast.literal_eval(a) for a in arg.args]
        except (ValueError, TypeError):
            return  # symbolic args — skip

        ctor = self._PREDICATE_CTORS.get(bare)
        if ctor is not None:
            try:
                self.constraints.append(ctor(*pred_args))
            except Exception:
                pass

    @staticmethod
    def _resolve_name(node: ast.expr) -> Optional[str]:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts)) if parts else None


# Wire up predicate constructors after imports
from .constraints import Divisible, Range, Equal  # noqa: E402
_ConstraintExtractor._PREDICATE_CTORS = {
    "Divisible": Divisible,
    "Range":     Range,
    "Equal":     Equal,
}


def _extract_constraints(fn: Callable) -> List[Constraint]:
    """Parse fn's source and return the list of tessera.require() constraints."""
    try:
        source = inspect.getsource(fn)
        source = textwrap.dedent(source)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return []

    extractor = _ConstraintExtractor()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == fn.__name__:
                for stmt in node.body:
                    extractor.visit(stmt)
                break
    return extractor.constraints


# ─────────────────────────────────────────────────────────────────────────────
# JitFn — the decorated function wrapper
# ─────────────────────────────────────────────────────────────────────────────

class JitFn:
    """
    A @jit-decorated Tessera function.

    Wraps the original Python function. In Phase 1 it executes eagerly
    (plain Python call). In Phase 3 it will invoke the compiled kernel
    through the MLIR lowering chain when target is set.

    Attributes:
        fn              : original Python function
        graph_ir        : emitted GraphIRModule (MLIR text available via .to_mlir())
        inferred_effect : Effect inferred by EffectLattice
        constraints     : ConstraintSolver with registered predicates
        deterministic   : whether @jit(deterministic=True) was set
        seed            : RNG seed (if provided)
        target          : GPUTargetProfile if GPU compilation was requested, else None
        attn_config     : FlashAttnLoweringConfig if flash_attn in body, else None
    """

    def __init__(
        self,
        fn: Callable,
        graph_ir: GraphIRModule,
        inferred_effect: Effect,
        constraints: ConstraintSolver,
        deterministic: bool = False,
        seed: Optional[int] = None,
        target: Optional[GPUTargetProfile] = None,
        attn_config: Optional[FlashAttnLoweringConfig] = None,
    ) -> None:
        self._fn = fn
        self.graph_ir = graph_ir
        self.inferred_effect = inferred_effect
        self.constraints = constraints
        self.deterministic = deterministic
        self.seed = seed
        self.target = target
        self.attn_config = attn_config
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Phase 1: execute the Python function eagerly.
        Phase 3: dispatch through the compiled kernel.
        """
        return self._fn(*args, **kwargs)

    def ir_text(self) -> str:
        """Return the emitted Graph IR as MLIR text."""
        return self.graph_ir.to_mlir()

    @property
    def is_gpu(self) -> bool:
        return self.target is not None

    def __repr__(self) -> str:
        target_str = f" target={self.target!r}" if self.target else ""
        return (
            f"<TesseraJitFn {self._fn.__name__!r} "
            f"effect={self.inferred_effect.name} "
            f"deterministic={self.deterministic}"
            f"{target_str}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# @jit decorator
# ─────────────────────────────────────────────────────────────────────────────

def jit(
    fn: Optional[Callable] = None,
    *,
    deterministic: bool = False,
    seed: Optional[int] = None,
    bindings: Optional[Dict[str, int]] = None,
    target: Optional[GPUTargetProfile] = None,
    attn_config: Optional[FlashAttnLoweringConfig] = None,
) -> Any:
    """
    Tessera JIT decorator — drives the compiler pipeline.

    Can be used with or without arguments:

        @tessera.jit
        def step(W: Region["read"], X: Region["read"], Y: Region["write"]):
            Y[:] = tessera.ops.gemm(X, W)

        @tessera.jit(deterministic=True, seed=42)
        def stable_forward(x: Tensor["B", "D"]):
            return tessera.ops.layer_norm(x)

        # Phase 3: GPU compilation
        @tessera.jit(target=GPUTargetProfile(isa=ISA.SM_90, warps_per_cta=4))
        def flash_attn_fwd(Q, K, V):
            return tessera.ops.flash_attn(Q, K, V, causal=True)

    Args:
        fn           : function to decorate (when used bare without parens)
        deterministic: if True, enforce that the function has no non-seeded
                       random effects (raises TesseraEffectError otherwise)
        seed         : RNG seed; allows random ops under deterministic=True
        bindings     : optional dict of dim_name → concrete size for
                       constraint checking at decoration time
        target       : GPUTargetProfile; when set, routes to GPU lowering
                       pipeline (Phase 3). None = CPU/interpreted path.
        attn_config  : FlashAttnLoweringConfig; when None and target is set with
                       isa >= SM_90, SM90_DEFAULT is used automatically.

    Returns:
        JitFn wrapper around the decorated function.

    Raises:
        TesseraConstraintError : if a structural constraint is violated
        TesseraEffectError     : if a deterministic contract is violated
        TesseraJitError        : if the Graph IR emission pipeline fails
    """

    def _decorate(fn: Callable) -> JitFn:
        # ── Step 1: collect constraints from the function body ──────────────
        solver = ConstraintSolver()
        extracted = _extract_constraints(fn)
        for c in extracted:
            solver.add(c)

        # ── Step 2: check constraints against any known bindings ────────────
        try:
            solver.check(bindings or {})
        except TesseraConstraintError:
            raise

        # ── Step 3: infer effects ────────────────────────────────────────────
        lattice = EffectLattice()
        inferred_effect = lattice.infer(fn)

        # ── Step 4: validate deterministic contract ──────────────────────────
        if deterministic:
            try:
                lattice.check_deterministic(fn, seed=seed)
            except TesseraEffectError:
                raise

        # ── Step 5: resolve attn config for GPU path ────────────────────────
        resolved_attn = attn_config
        if target is not None and target.supports_wgmma and resolved_attn is None:
            resolved_attn = SM90_DEFAULT

        # ── Step 6: emit Graph IR ────────────────────────────────────────────
        try:
            builder = GraphIRBuilder()
            effect_tag = inferred_effect.name if inferred_effect != Effect.pure else None
            # Attach GPU target attrs to the module when target is provided.
            target_attr = target.to_mlir_attr() if target is not None else None
            builder.lower(fn, effect_tag=effect_tag, target_attr=target_attr)
            module = builder.module()
        except Exception as exc:
            raise TesseraJitError(
                f"Graph IR emission failed for {fn.__name__!r}: {exc}"
            ) from exc

        # ── Step 7: wrap and return ──────────────────────────────────────────
        return JitFn(
            fn=fn,
            graph_ir=module,
            inferred_effect=inferred_effect,
            constraints=solver,
            deterministic=deterministic,
            seed=seed,
            target=target,
            attn_config=resolved_attn,
        )

    # Support both @jit and @jit(...) usage
    if fn is not None:
        return _decorate(fn)
    return _decorate
