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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .constraints import Constraint, ConstraintSolver, TesseraConstraintError
from .effects import Effect, EffectLattice, TesseraEffectError
from .graph_ir import GraphIRBuilder, GraphIRModule
from .gpu_target import GPUTargetProfile, ISA  # noqa: F401 — re-exported for callers
from .attn_lower import FlashAttnLoweringConfig, SM90_DEFAULT  # noqa: F401
from .driver import CompileArtifactBundle, compile_graph_module
from .matmul_pipeline import JitDiagnostic, CPUPlan, normalize_target_kind
from .fallback import FallbackReason, TesseraNativeRequiredError


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
# *inside* @jit-decorated function bodies.  Decoration-time collection is done
# by parsing the AST (`_extract_require_calls` below); the runtime `require()`
# function is intentionally a no-op outside an explicit decoration / trace
# scope so that:
#
#   * `@jit`-decorated bodies that fall back to eager Python execution don't
#     mutate process-global state on every call,
#   * cross-test isolation is preserved (a `require()` in one test cannot
#     leak into a later test's collection),
#   * future tracing modes (or external constraint collectors) can opt in
#     by pushing onto ``_ACTIVE_CONSTRAINTS`` via ``collect_constraints()``.
#
# We keep ``_ACTIVE_CONSTRAINTS`` as a thread-local *stack of lists* (not a
# single global list) so concurrent traces in different threads don't clobber
# each other.

import threading


class _ConstraintTLS(threading.local):
    """Thread-local stack of constraint-collection lists.

    Each element of ``stack`` is the list a single ``collect_constraints()``
    context manager appends to.  When the stack is empty, ``require()`` is a
    true no-op (matching the docstring contract).
    """
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[list[Constraint]] = []


_ACTIVE_CONSTRAINTS = _ConstraintTLS()


def require(constraint: Constraint) -> None:
    """
    Register a structural constraint on the enclosing @jit function.

    At decoration time: collected by ConstraintSolver and checked against
    any concrete dimension bindings extracted from the type signature
    (via ``_extract_require_calls`` — a static AST scan, *not* a runtime
    side-effect on this function).

    At call time: **no-op** unless the call is enclosed in a
    :func:`collect_constraints` scope.  This guarantees that ``@jit``
    functions that fall back to eager Python execution don't leak
    constraints into process-global state on every call.

    Usage:
        @tessera.jit
        def aligned_gemm(A: Tensor["M", "K"], B: Tensor["K", "N"]):
            tessera.require(tessera.constraint.Divisible("K", 64))
            return tessera.ops.gemm(A, B)
    """
    stack = _ACTIVE_CONSTRAINTS.stack
    if stack:
        stack[-1].append(constraint)


class collect_constraints:
    """Context manager that opts into runtime ``require()`` collection.

    The collected list is the ``__enter__`` value::

        with collect_constraints() as constraints:
            my_jit_fn(*args)
        # constraints :: list[Constraint]

    Reserved for future tracing modes / external collectors; ordinary
    @jit decoration uses the static AST scan and does not need this.
    """
    def __enter__(self) -> list[Constraint]:
        self._scope: list[Constraint] = []
        _ACTIVE_CONSTRAINTS.stack.append(self._scope)
        return self._scope

    def __exit__(self, exc_type, exc, tb) -> None:
        popped = _ACTIVE_CONSTRAINTS.stack.pop()
        assert popped is self._scope, (
            "collect_constraints scope was popped out of order"
        )


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


def _extract_constraints(fn: Callable, source_text: Optional[str] = None) -> List[Constraint]:
    """Parse fn's source and return the list of tessera.require() constraints."""
    try:
        source = source_text if source_text is not None else inspect.getsource(fn)
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


def _resolve_source_text(
    fn: Callable,
    *,
    source: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Return dedented function source and an origin label for diagnostics."""

    if source is not None and source_path is not None:
        raise TesseraJitError("Pass either source=... or source_path=..., not both")
    if source is not None:
        return textwrap.dedent(source), "explicit"
    if source_path is not None:
        try:
            return textwrap.dedent(Path(source_path).read_text(encoding="utf-8")), f"file:{source_path}"
        except OSError as exc:
            raise TesseraJitError(f"Could not read @jit source_path {source_path!r}: {exc}") from exc
    try:
        return textwrap.dedent(inspect.getsource(fn)), "inspect"
    except (OSError, TypeError):
        return None, "unavailable"


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
        target          : GPUTargetProfile or target string if non-CPU compilation was requested, else None
        attn_config     : FlashAttnLoweringConfig if flash_attn in body, else None
        cpu_plan        : executable CPU lowering plan for supported programs
        compile_bundle  : compiler driver artifacts, diagnostics, and trace
        cpu_tile        : CPU matmul/GEMM tile shape
        source_origin   : where AST source came from: inspect, explicit, file, unavailable
        lowering_diagnostics: developer-facing lowering decision diagnostics
    """

    def __init__(
        self,
        fn: Callable,
        graph_ir: GraphIRModule,
        inferred_effect: Effect,
        constraints: ConstraintSolver,
        deterministic: bool = False,
        seed: Optional[int] = None,
        target: Optional[Any] = None,
        attn_config: Optional[FlashAttnLoweringConfig] = None,
        cpu_plan: Optional[CPUPlan] = None,
        compile_bundle: Optional[CompileArtifactBundle] = None,
        cpu_tile: Tuple[int, int, int] = (128, 128, 64),
        source_origin: str = "inspect",
        lowering_diagnostics: Optional[List[JitDiagnostic]] = None,
        native_required: bool = False,
    ) -> None:
        self._fn = fn
        self.graph_ir = graph_ir
        self.inferred_effect = inferred_effect
        self.constraints = constraints
        self.deterministic = deterministic
        self.seed = seed
        self.target = target
        self.attn_config = attn_config
        self.cpu_plan = cpu_plan
        self.compile_bundle = compile_bundle
        self.cpu_tile = tuple(int(v) for v in cpu_tile)
        self.source_origin = source_origin
        self.lowering_diagnostics = tuple(lowering_diagnostics or [])
        self.native_required = bool(native_required)
        # Last fallback reason (None on a native run).  Inspectable by
        # callers + by CompileReport.fallback_reason emission.
        self.last_fallback_reason: Optional[FallbackReason] = None
        # Phase 8.2 launch-overhead reduction: the artifact + its metadata
        # depend only on immutable construction inputs, so we lazily build
        # them once and reuse on every __call__. Without caching the small-
        # GEMM hot-path is dominated by metadata dict construction + the
        # SHA-256 over the artifact JSON inside `RuntimeArtifact.artifact_hash`.
        self._cached_artifact: Optional["RuntimeArtifact"] = None  # noqa: F821 (forward ref; runtime-import below)
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute through the narrow CPU lowering path when available; otherwise
        fall back to the original Python function.

        Phase A2-followup: before any execution, resolve symbolic dim names
        against the actual argument shapes and re-run the constraint solver
        so violations raise a clear ``TesseraConstraintError`` at first call,
        not a downstream numpy / Accelerate error.

        Step 4 (2026-05-18): auto-emits a :class:`CompileReport` to
        the active sink (no-op when no sink is active).
        """
        self._enforce_call_time_constraints(args, kwargs)
        try:
            if self.cpu_plan is not None and self.cpu_plan.target_kind == "cpu":
                if self.execution_kind == "native_cpu":
                    return self._native_cpu_fast_call(args, kwargs)
                return self.cpu_plan.execute(args, kwargs, self.arg_names)
            if (
                self.cpu_plan is not None
                and self.cpu_plan.target_kind == "apple_cpu"
                and self.compile_bundle is not None
                and self.compile_bundle.executable
            ):
                return self._apple_cpu_fast_call(args, kwargs)
            if (
                self.cpu_plan is not None
                and self.cpu_plan.target_kind == "apple_gpu"
                and self.compile_bundle is not None
                and self.compile_bundle.executable
            ):
                return self._apple_gpu_fast_call(args, kwargs)
            return self._fn(*args, **kwargs)
        finally:
            from . import compile_report as _cr
            if _cr.active_sink_is_capturing():
                _cr.emit_compile_report(self.compile_report())

    def _enforce_call_time_constraints(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        """Resolve symbolic dim names against actual call shapes + re-run the solver.

        Skipped when the function has no dim-annotated args, or when the solver
        was already satisfied at decoration time with concrete ``bindings=``.
        Cached per-shape to avoid re-checking on every call.
        """
        if not self.graph_ir.functions:
            return
        ir_args = self.graph_ir.functions[0].args
        # Resolve dim_name → concrete int by walking positional + keyword args
        resolved: Dict[str, int] = {}
        # Build a name → value map first
        name_to_value: Dict[str, Any] = {}
        for ir_arg, value in zip(ir_args, args):
            name_to_value[ir_arg.name] = value
        for k, v in kwargs.items():
            name_to_value[k] = v

        for ir_arg in ir_args:
            if not ir_arg.dim_names:
                continue
            value = name_to_value.get(ir_arg.name)
            if value is None:
                continue
            shape = getattr(value, "shape", None)
            if shape is None:
                continue
            shape = tuple(shape)
            if len(shape) != len(ir_arg.dim_names):
                continue  # rank mismatch — let downstream surface a clearer error
            for dim_name, concrete in zip(ir_arg.dim_names, shape):
                if not isinstance(dim_name, str):
                    continue
                # Symbolic dim names are typically uppercase (e.g., "M", "K").
                if not dim_name.isidentifier() or dim_name.isnumeric():
                    continue
                prev = resolved.get(dim_name)
                if prev is not None and prev != int(concrete):
                    # Inconsistent binding across args (e.g., K from arg 0 vs. arg 1).
                    # Build a synthetic Equal constraint to get a uniform error type.
                    from .constraints import Equal as _Equal
                    raise TesseraConstraintError(
                        _Equal(dim_name, dim_name),
                        dim_name,
                        actual=int(concrete),
                        message=(
                            f"Inconsistent binding for dim {dim_name!r}: "
                            f"saw {prev} earlier, now {int(concrete)} (arg {ir_arg.name!r})"
                        ),
                    )
                resolved[dim_name] = int(concrete)

        if not resolved:
            return

        # Cache per-shape so repeated calls with the same shape skip the check.
        cache_key = tuple(sorted(resolved.items()))
        cache = getattr(self, "_constraint_cache", None)
        if cache is None:
            cache = set()
            object.__setattr__(self, "_constraint_cache", cache)
        if cache_key in cache:
            return
        # ConstraintSolver.check raises TesseraConstraintError on violation.
        self.constraints.check(resolved)
        cache.add(cache_key)

    def _native_cpu_fast_call(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Dispatch eligible CPU rank-2 f32 GEMM through the native runtime ABI.

        Guard failures fall back to the explicit NumPy reference plan and
        record the reason on ``self.last_fallback_reason`` so callers
        (and the CompileReport emitted by ``runtime_artifact()``) see why
        the native lane wasn't taken.  When the JIT was constructed with
        ``native_required=True``, a launch-time failure raises
        :class:`TesseraNativeRequiredError` instead of falling through.
        """

        from tessera.runtime import _execute_native_cpu_metadata

        if kwargs and args:
            launch_args = {name: value for name, value in zip(self.arg_names, args)}
            launch_args.update(kwargs)
        else:
            launch_args = kwargs if kwargs else args
        try:
            result = _execute_native_cpu_metadata(
                self.runtime_artifact().metadata or {}, launch_args
            )
            # Clear any stale fallback reason from a prior call.
            self.last_fallback_reason = None
            return result
        except Exception as exc:
            self.last_fallback_reason = FallbackReason.CAPABILITY_NOT_READY
            if self.native_required:
                raise TesseraNativeRequiredError(
                    FallbackReason.CAPABILITY_NOT_READY,
                    target="cpu",
                    op_name=getattr(self._fn, "__name__", ""),
                    detail=(
                        f"native CPU launch failed and native_required=True "
                        f"(underlying error: {type(exc).__name__}: {exc})"
                    ),
                ) from exc
            return self.cpu_plan.execute(args, kwargs, self.arg_names)

    def _apple_cpu_fast_call(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Phase 8.2 launch-overhead fast path. Bypasses ``runtime.launch`` by
        calling the metadata dispatcher directly with the cached artifact's
        metadata dict — skipping per-call telemetry events, the artifact
        SHA-256, and the JSON serialization that backs it. The public
        ``launch(mm.runtime_artifact(), ...)`` entry stays unchanged for
        callers who want full telemetry."""

        from tessera.runtime import _execute_apple_cpu_accelerate_metadata

        if kwargs and args:
            launch_args = {name: value for name, value in zip(self.arg_names, args)}
            launch_args.update(kwargs)
        else:
            launch_args = kwargs if kwargs else args

        try:
            return _execute_apple_cpu_accelerate_metadata(
                self.runtime_artifact().metadata or {}, launch_args
            )
        except Exception as exc:
            raise TesseraJitError(f"apple_cpu launch failed: {exc}") from exc

    def _apple_gpu_fast_call(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Phase 8.3 launch-overhead fast path for apple_gpu MPS programs.
        Mirrors `_apple_cpu_fast_call` — bypasses ``runtime.launch`` and calls
        the metadata dispatcher directly with the cached artifact's metadata.
        """

        from tessera.runtime import _execute_apple_gpu_mps_metadata

        if kwargs and args:
            launch_args = {name: value for name, value in zip(self.arg_names, args)}
            launch_args.update(kwargs)
        else:
            launch_args = kwargs if kwargs else args

        try:
            return _execute_apple_gpu_mps_metadata(
                self.runtime_artifact().metadata or {}, launch_args
            )
        except Exception as exc:
            raise TesseraJitError(f"apple_gpu launch failed: {exc}") from exc

    def ir_text(self) -> str:
        """Return the emitted Graph IR as MLIR text."""
        return self.graph_ir.to_mlir()

    def compile_report(self):
        """Synthesize a :class:`CompileReport` from this JitFn's
        current state.

        Step 4 of the 2026-05-18 post-reassessment plan: every JIT
        frontend exposes a uniform CompileReport accessor.  No
        execution happens here — the accessor reads ``graph_ir``,
        ``target``, ``cpu_plan``, and the recent bridge trace.
        """
        from . import compile_report as _cr
        target_kind = (
            self.cpu_plan.target_kind if self.cpu_plan is not None
            else normalize_target_kind(self.target)
        )
        ir_hashes = {"graph_ir": _cr.hash_ir_text(self.ir_text())}
        target_decision = {
            target_kind: (
                f"cpu_plan={self.cpu_plan.target_kind if self.cpu_plan else 'none'}; "
                f"compile_bundle.executable="
                f"{bool(self.compile_bundle and self.compile_bundle.executable)}"
            ),
        }
        # Pick up any routes the bridge captured during the most
        # recent dispatch; the CPU fast path does not produce
        # routes but apple_gpu does.
        routes = _cr.routes_from_thread_trace()
        return _cr.CompileReport(
            program_id=getattr(self._fn, "__qualname__", "<jit_fn>"),
            source=f"@tessera.jit({getattr(self._fn, '__qualname__', '?')})",
            frontend=_cr.FRONTEND_TESSERA_JIT,
            value_kind=_cr.VALUE_KIND_TENSOR,
            target=target_kind,
            ir_hashes=ir_hashes,
            target_decision=target_decision,
            proof_routes=routes,
            # Surface the most recent native-launch fallback reason
            # (set in ``_native_cpu_fast_call``).  ``None`` on a clean
            # native run; populated when the runtime ABI raised and
            # the JIT fell through to ``cpu_plan.execute`` without
            # ``native_required=True``.
            fallback_reason=self.last_fallback_reason,
        )

    @property
    def effect(self) -> Effect:
        """Compatibility alias for the inferred effect."""
        return self.inferred_effect

    @property
    def is_gpu(self) -> bool:
        target_kind = normalize_target_kind(self.target)
        return target_kind.startswith("nvidia") or target_kind in {"rocm", "metalium", "apple_gpu"}

    @property
    def arg_names(self) -> List[str]:
        if not self.graph_ir.functions:
            return []
        return [arg.name for arg in self.graph_ir.functions[0].args]

    @property
    def schedule_ir(self) -> Optional[str]:
        artifact = self.compile_bundle.artifact("schedule") if self.compile_bundle is not None else None
        return artifact.text if artifact is not None else None

    @property
    def tile_ir(self) -> Optional[str]:
        artifact = self.compile_bundle.artifact("tile") if self.compile_bundle is not None else None
        return artifact.text if artifact is not None else None

    @property
    def target_ir(self) -> Optional[str]:
        artifact = self.compile_bundle.artifact("target") if self.compile_bundle is not None else None
        return artifact.text if artifact is not None else None

    @property
    def execution_kind(self) -> str:
        if self.compile_bundle is not None:
            return self.compile_bundle.execution_kind
        if self.cpu_plan is not None and self.cpu_plan.target_kind == "cpu":
            return "reference_cpu"
        return "fallback_eager"

    @property
    def is_executable(self) -> bool:
        return self.execution_kind in {"reference_cpu", "native_cpu", "native_gpu"}

    @property
    def is_reference_execution(self) -> bool:
        return self.execution_kind == "reference_cpu"

    @property
    def is_native_execution(self) -> bool:
        return self.execution_kind in {"native_cpu", "native_gpu"}

    @property
    def has_target_artifacts(self) -> bool:
        return self.cpu_plan is not None

    def lowering_artifacts(self):
        """Return Graph/Schedule/Tile/Target artifacts for the compiled path."""

        if self.cpu_plan is None:
            return ()
        if self.compile_bundle is None:
            return self.cpu_plan.artifacts()
        return self.compile_bundle.lowering_artifacts()

    def lowering_trace(self) -> tuple[dict[str, Any], ...]:
        """Return machine-readable compiler trace events for this JIT function."""

        if self.compile_bundle is None:
            return ()
        return tuple(event.to_dict() for event in self.compile_bundle.trace_events)

    def runtime_artifact(self):
        """Return a RuntimeArtifact for this JIT function's compiler output.

        Cached lazily — the inputs (cpu_plan, compile_bundle,
        lowering_diagnostics) are immutable after JitFn construction, so
        rebuilding the artifact + recomputing its SHA-256 every call is pure
        overhead. The cached artifact is shared between inspection callers and
        the apple_cpu fast path so they observe consistent metadata.
        """

        if self._cached_artifact is not None:
            return self._cached_artifact
        self._cached_artifact = self._build_runtime_artifact()
        return self._cached_artifact

    def _build_runtime_artifact(self):
        """Construct a fresh RuntimeArtifact. Called once per JitFn."""

        from tessera.runtime import RuntimeArtifact

        diagnostics = [d.format() for d in self.lowering_diagnostics]
        bundle_metadata = self.compile_bundle.to_metadata() if self.compile_bundle is not None else {}
        metadata: dict[str, Any] = {
            "target": self.cpu_plan.target_kind if self.cpu_plan is not None else normalize_target_kind(self.target),
            "function_name": self._fn.__name__,
            "source_origin": self.source_origin,
            "effect": self.inferred_effect.name,
            "deterministic": self.deterministic,
            "diagnostics": diagnostics,
            "executable": False,
            "compiler_path": "eager_fallback",
            "execution_kind": "fallback_eager",
            "runtime_status": "unsupported",
            **bundle_metadata,
        }
        if self.cpu_plan is not None and self.cpu_plan.target_kind == "cpu":
            native_cpu = self.execution_kind == "native_cpu"
            metadata.update({
                "executable": True,
                "compiler_path": "jit_cpu_numpy",
                "execution_kind": "native_cpu" if native_cpu else "reference_cpu",
                "runtime_status": "ready",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "input_descriptors": [{"name": name} for name in self.arg_names],
                "output_descriptor": {"name": self.cpu_plan.output_name},
                "cpu_tile": list(self.cpu_plan.tile),
                "ops": [
                    {
                        "op_name": op.op_name,
                        "result": op.result,
                        "operands": [operand[1:] if operand.startswith("%") else operand for operand in op.operands],
                        "kwargs": dict(op.kwargs),
                    }
                    for op in self.cpu_plan.ops
                ],
                "guards": {
                    "dtype": "float32",
                    "rank": 2,
                    "op_count": 1,
                } if native_cpu else {
                    "reference_backend": "numpy",
                },
            })
        elif (
            self.cpu_plan is not None
            and self.cpu_plan.target_kind == "apple_cpu"
            and self.compile_bundle is not None
            and self.compile_bundle.executable
        ):
            ops_payload = [
                {
                    "op_name": op.op_name,
                    "result": op.result,
                    "operands": [operand[1:] if operand.startswith("%") else operand for operand in op.operands],
                    "kwargs": dict(op.kwargs),
                }
                for op in self.cpu_plan.ops
            ]
            accelerate_ops = [
                op["op_name"] for op in ops_payload
                if op["op_name"] in {"tessera.matmul", "tessera.gemm"}
            ]
            single_matmul = (
                len(self.cpu_plan.ops) == 1
                and self.cpu_plan.ops[0].op_name in {"tessera.matmul", "tessera.gemm"}
            )
            # Single matmul keeps the strict f32/rank-2 descriptors and the
            # original guard shape (preserves the Phase 8.2 metadata contract
            # that downstream tooling and tests rely on).
            # Multi-op programs report a relaxed schema: descriptors only
            # carry names because intermediate values can be any dtype/rank
            # (e.g. theta vectors, softmax results), and `accelerate_ops`
            # surfaces which ops will dispatch through Accelerate at launch.
            if single_matmul:
                input_descriptors = [
                    {"name": name, "dtype": "f32", "rank": 2}
                    for name in self.arg_names
                ]
                output_descriptor = {
                    "name": self.cpu_plan.output_name,
                    "dtype": "f32",
                    "rank": 2,
                }
                guards = {
                    "dtype": "float32",
                    "rank": 2,
                    "static_shape_at_launch": True,
                    "op_count": 1,
                }
            else:
                input_descriptors = [{"name": name} for name in self.arg_names]
                output_descriptor = {"name": self.cpu_plan.output_name}
                guards = {
                    "op_count": len(self.cpu_plan.ops),
                    "accelerate_op_count": len(accelerate_ops),
                    "accelerate_dtype": "float32",
                    "accelerate_rank": 2,
                    "fallback_path": "jit_cpu_numpy",
                    "static_shape_at_launch": True,
                }
            metadata.update({
                "executable": True,
                "compiler_path": "apple_cpu_accelerate",
                "execution_kind": "native_cpu",
                "runtime_status": "ready",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "input_descriptors": input_descriptors,
                "output_descriptor": output_descriptor,
                "cpu_tile": list(self.cpu_plan.tile),
                "ops": ops_payload,
                "accelerate_ops": accelerate_ops,
                "guards": guards,
            })
        elif (
            self.cpu_plan is not None
            and self.cpu_plan.target_kind == "apple_gpu"
            and self.compile_bundle is not None
            and self.compile_bundle.executable
        ):
            ops_payload = [
                {
                    "op_name": op.op_name,
                    "result": op.result,
                    "operands": [operand[1:] if operand.startswith("%") else operand for operand in op.operands],
                    "kwargs": dict(op.kwargs),
                }
                for op in self.cpu_plan.ops
            ]
            metadata.update({
                "executable": True,
                "compiler_path": "apple_gpu_mps",
                "execution_kind": "native_gpu",
                "runtime_status": "ready",
                "execution_mode": "metal_runtime",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "input_descriptors": [
                    {"name": name, "dtype": "f32", "rank": 2}
                    for name in self.arg_names
                ],
                "output_descriptor": {
                    "name": self.cpu_plan.output_name,
                    "dtype": "f32",
                    "rank": 2,
                },
                "cpu_tile": list(self.cpu_plan.tile),
                "ops": ops_payload,
                "mps_ops": [op["op_name"] for op in ops_payload],
                "guards": {
                    "dtype": "float32",
                    "rank": 2,
                    "static_shape_at_launch": True,
                    "op_count": 1,
                },
            })
        elif self.cpu_plan is not None:
            metadata.update({
                "compiler_path": "target_ir_artifact",
                "execution_kind": "artifact_only",
                "runtime_status": "artifact_only",
                "reason": "native target execution is not wired",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "cpu_tile": list(self.cpu_plan.tile),
            })

        graph_ir_text = self.graph_ir.to_mlir()
        schedule_ir_text = self.schedule_ir or ""
        tile_ir_text = self.tile_ir or ""
        target_ir_text = self.target_ir or ""

        # Honor TESSERA_DEBUG_IR / TESSERA_DEBUG_DUMP_DIR — write IR snapshots
        # for the configured stages so users can diff before/after a code
        # change without re-instrumenting their source. See debug_env.py.
        from .. import debug_env as _debug_env
        if _debug_env.should_dump():
            _debug_env.dump_artifact(
                symbol=self._fn.__name__,
                graph_ir=graph_ir_text,
                schedule_ir=schedule_ir_text,
                tile_ir=tile_ir_text,
                target_ir=target_ir_text,
            )

        return RuntimeArtifact(
            graph_ir=graph_ir_text,
            schedule_ir=schedule_ir_text,
            tile_ir=tile_ir_text,
            target_ir=target_ir_text,
            metadata=metadata,
            abi_signature=f"tessera.runtime.v1.{metadata['target']}",
        )

    def explain_lowering(self) -> str:
        """Return a human-readable explanation of compile vs fallback status."""

        return "\n".join(d.format() for d in self.lowering_diagnostics)

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
    target: Optional[Any] = None,
    attn_config: Optional[FlashAttnLoweringConfig] = None,
    cpu_tile: Tuple[int, int, int] = (128, 128, 64),
    source: Optional[str] = None,
    source_path: Optional[str] = None,
    native_required: bool = False,
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
        target       : GPUTargetProfile or target string. Supported strings are
                       "rocm", "metalium", "apple_cpu", and "apple_gpu".
                       None = executable CPU/NumPy path.
        attn_config  : FlashAttnLoweringConfig; when None and target is set with
                       isa >= SM_90, SM90_DEFAULT is used automatically.
        cpu_tile     : CPU matmul/GEMM schedule tile `(M, N, K)` for the narrow
                       end-to-end CPU compiler path.
        source       : optional function source text for functions created from
                       stdin/exec where inspect.getsource() cannot recover the
                       function body.
        source_path  : optional path to Python source text for AST lowering.

    Returns:
        JitFn wrapper around the decorated function.

    Raises:
        TesseraConstraintError : if a structural constraint is violated
        TesseraEffectError     : if a deterministic contract is violated
        TesseraJitError        : if the Graph IR emission pipeline fails
    """

    def _decorate(fn: Callable) -> JitFn:
        source_text, source_origin = _resolve_source_text(
            fn,
            source=source,
            source_path=source_path,
        )

        # ── Step 0: refuse to silently fall back when a target was requested ─
        # When @jit(target=...) is set explicitly, the developer expects the
        # named backend to drive execution. Without function source we cannot
        # emit Graph IR, the `compile_bundle` would be empty, and __call__
        # would silently route to plain Python. That looks like the target
        # path is running but produces eager numpy semantics — the worst kind
        # of bug to chase. Fail at decoration time instead.
        #
        # target=None keeps the existing soft-warning behavior so REPL/heredoc
        # exploration of the default eager path is still ergonomic.
        if target is not None and source_text is None:
            raise TesseraJitError(
                f"@jit(target={target!r}) was requested for {fn.__name__!r} "
                f"but its source could not be inspected (source_origin="
                f"{source_origin!r}). Without source, no Graph IR is emitted "
                f"and the call would silently fall back to eager Python — "
                f"giving the appearance of a compiled run while actually "
                f"executing pure Python. Define the function in a file, or "
                f"pass @jit(target=..., source=<source string>) or "
                f"@jit(target=..., source_path=<path>) to enable AST lowering."
            )

        # ── Step 1: collect constraints from the function body ──────────────
        solver = ConstraintSolver()
        extracted = _extract_constraints(fn, source_text=source_text)
        for c in extracted:
            solver.add(c)

        # ── Step 2: check constraints against any known bindings ────────────
        try:
            solver.check(bindings or {})
        except TesseraConstraintError:
            raise

        # ── Step 3: infer effects ────────────────────────────────────────────
        lattice = EffectLattice()
        inferred_effect = lattice.infer(fn, source_text=source_text)

        # ── Step 4: validate deterministic contract ──────────────────────────
        if deterministic:
            try:
                lattice.check_deterministic(fn, seed=seed, source_text=source_text)
            except TesseraEffectError:
                raise

        # ── Step 5: resolve attn config for GPU path ────────────────────────
        resolved_attn = attn_config
        target_kind = normalize_target_kind(target)
        if isinstance(target, GPUTargetProfile) and target.supports_wgmma and resolved_attn is None:
            resolved_attn = SM90_DEFAULT

        # ── Step 6: emit Graph IR ────────────────────────────────────────────
        try:
            builder = GraphIRBuilder()
            effect_tag = inferred_effect.name if deterministic or inferred_effect != Effect.pure else None
            # Attach GPU target attrs to the module when target is provided.
            if isinstance(target, GPUTargetProfile):
                target_attr = target.to_mlir_attr()
            elif target is not None:
                target_attr = f'{{name = "{target_kind}"}}'
            else:
                target_attr = None
            builder.lower(fn, effect_tag=effect_tag, target_attr=target_attr, source_text=source_text)
            module = builder.module()
            diagnostics = []
            for frontend_diag in builder.diagnostics:
                diagnostics.append(JitDiagnostic(
                    frontend_diag.severity,
                    frontend_diag.code,
                    frontend_diag.format(),
                ))
            if source_text is None:
                diagnostics.append(JitDiagnostic(
                    "warning",
                    "JIT_SOURCE_UNAVAILABLE",
                    (
                        "Python source could not be inspected; define the function in a file "
                        "or pass @jit(source=...) / @jit(source_path=...) to enable AST lowering"
                    ),
                ))
            elif source_origin != "inspect":
                diagnostics.append(JitDiagnostic(
                    "info",
                    "JIT_SOURCE_PROVIDED",
                    f"using {source_origin} source for AST lowering",
                ))
            compile_bundle = compile_graph_module(
                module,
                source_origin=source_origin,
                target=target_kind,
                cpu_tile=tuple(int(v) for v in cpu_tile),
                options={
                    "cpu_tile": list(tuple(int(v) for v in cpu_tile)),
                    "deterministic": deterministic,
                    "seed": seed,
                },
            )
            if diagnostics:
                compile_bundle = CompileArtifactBundle(
                    request=compile_bundle.request,
                    graph=compile_bundle.graph,
                    schedule=compile_bundle.schedule,
                    tile=compile_bundle.tile,
                    target_ir=compile_bundle.target_ir,
                    backend=compile_bundle.backend,
                    executable=compile_bundle.executable,
                    runtime_status=compile_bundle.runtime_status,
                    execution_mode=compile_bundle.execution_mode,
                    execution_kind=compile_bundle.execution_kind,
                    diagnostics=tuple(diagnostics) + compile_bundle.diagnostics,
                    trace_events=compile_bundle.trace_events,
                    tool_invocations=compile_bundle.tool_invocations,
                    cpu_plan=compile_bundle.cpu_plan,
                )
            cpu_plan = compile_bundle.cpu_plan
            diagnostics = list(compile_bundle.diagnostics)
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
            cpu_plan=cpu_plan,
            compile_bundle=compile_bundle,
            cpu_tile=tuple(int(v) for v in cpu_tile),
            source_origin=source_origin,
            lowering_diagnostics=diagnostics,
            native_required=native_required,
        )

    # Support both @jit and @jit(...) usage
    if fn is not None:
        return _decorate(fn)
    return _decorate
