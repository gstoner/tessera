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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..runtime import RuntimeArtifact
    from .explain import Explain

from .constraints import Constraint, ConstraintSolver, TesseraConstraintError
from .effects import Effect, EffectLattice
from .graph_ir import GraphIRBuilder, GraphIRModule
from .gpu_target import GPUTargetProfile, ISA  # noqa: F401 — re-exported for callers
from .attn_lower import FlashAttnLoweringConfig, SM90_DEFAULT  # noqa: F401
from .driver import CompileArtifactBundle, compile_graph_module
from .canonical_compile import CompileResult, compile_result_from_bundle
from .matmul_pipeline import JitDiagnostic, CPUPlan, normalize_target_kind
from .diagnostics import JitDiagnosticCode
from .fallback import FallbackReason, TesseraNativeRequiredError


# ─────────────────────────────────────────────────────────────────────────────
# Error type
# ─────────────────────────────────────────────────────────────────────────────
#
# Single canonical class, defined in the low-level `_jit_boundary` (the GraphFn /
# runtime lane). It is re-exported here so the `@jit` decoration lane and the
# GraphFn lane raise the SAME exception — `except TesseraJitError` /
# `pytest.raises(TesseraJitError)` catch both regardless of which module the name
# was imported from. (`_jit_boundary` imports only stdlib + numpy, so this is
# cycle-safe; the base is `RuntimeError`, a subclass of `Exception`, so every
# existing catcher still matches.)
from .._jit_boundary import TesseraJitError  # noqa: E402,F401 — re-exported


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
    """Return function source and its origin, preserving file-backed columns."""

    if source is not None and source_path is not None:
        raise TesseraJitError("Pass either source=... or source_path=..., not both")
    if source is not None:
        return textwrap.dedent(source), "explicit"
    if source_path is not None:
        try:
            path = Path(source_path).resolve()
            return path.read_text(encoding="utf-8"), f"file:{path}"
        except OSError as exc:
            raise TesseraJitError(f"Could not read @jit source_path {source_path!r}: {exc}") from exc
    try:
        return textwrap.dedent(inspect.getsource(fn)), "inspect"
    except (OSError, TypeError):
        return None, "unavailable"


# ─────────────────────────────────────────────────────────────────────────────
# JitFn — the decorated function wrapper
# ─────────────────────────────────────────────────────────────────────────────

# PK8e sentinel — distinguishes "couldn't route this call through the authored
# package" (fall back to the normal path) from a legitimate ``None`` result.
_PKG_FALLBACK: Any = object()

# Phase 4 sentinel — "couldn't route this CPU call through the tessera_jit
# MLIR→LLVM lane" (fall back to the numpy reference plan).
_JIT_FALLBACK: Any = object()


def _resolve_dispatch_via_package(value: "bool | str | None",
                                  target_kind: Optional[str]) -> "bool | str":
    """PK8h — resolve the package-dispatch policy.

    **Deliberate-call note (2026-06-02):** we evaluated making ``"auto"`` the
    *unconditional* default for apple_gpu and rejected it. Under suite-volume
    (hundreds of jitted chains each authoring MTL4 ML pipelines / intermediate
    heaps, co-loaded with the per-test runtime dylibs) the always-on package
    lane drives the Metal runtime to ``SIGABRT``. So auto-routing stays
    **opt-in**, exposed two ways without destabilizing the default path:

    * per-fn — ``@jit(..., dispatch_via_package="auto" | True)``;
    * globally — ``TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE=1`` (``on``/``true``/
      ``yes``) flips the default to ``"auto"`` for recognized apple_gpu fns.

    Default (``value is None``) → ``False`` (live lane), unless the env switch
    is on. An explicit ``True`` / ``"auto"`` / path / ``False`` always wins.
    Non-apple_gpu targets never route (no package lane exists)."""
    if value is not None:
        return value  # explicit per-fn choice always wins
    if target_kind != "apple_gpu":
        return False
    import os
    env = os.environ.get("TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE", "").lower()
    if env in ("1", "on", "true", "yes"):
        return "auto"
    return False


def _rocm_compiled_lane_available() -> bool:
    """True iff the compiler-generated ROCm matmul lane can run on THIS host:
    ``tessera-opt`` is built AND a usable AMD GPU answers the runtime probe.
    Host-gated, so a rocm matmul artifact is stamped ``executable`` only where the
    lane actually runs; elsewhere (e.g. CI with no GPU) it stays ``artifact_only``
    exactly as before — no behavior change off-device."""
    try:
        from .. import runtime as _rt
        return (_rt._tessera_opt_path() is not None
                and _rt._rocm_wmma_runtime_available())
    except Exception:
        return False


def _nvidia_mma_lane_available() -> bool:
    """True iff the shipped NVIDIA mma.sync matmul lane can run on THIS host:
    libtessera_nvidia_gemm.so loads AND a usable NVIDIA GPU answers the runtime
    probe. Host-gated, so an nvidia_sm120 matmul artifact is stamped
    ``executable`` only where the lane actually runs; elsewhere (CI with no GPU)
    it stays ``artifact_only`` — no behavior change off-device."""
    try:
        from .. import runtime as _rt
        return _rt._nvidia_mma_runtime_available()
    except Exception:
        return False


def _signature_ir_args(fn: Callable) -> Tuple[Any, ...]:
    """The call ABI recovered from the Python signature, or () if unreadable.

    Only reached when Graph IR emission produced no function to read the args
    back off. Fail-soft because this recovers a *better* answer than the empty
    tuple it replaces -- if the signature is unreadable too (a builtin, a C
    extension), the old degraded behaviour stands rather than a new failure.
    """
    from .graph_ir import ir_args_from_signature

    try:
        return tuple(ir_args_from_signature(fn))
    except Exception:
        return ()


def _serialized_numeric_policy(op: Any) -> Optional[Dict[str, Any]]:
    """The op's carried `numeric_policy`, as a plain dict for the runtime ABI.

    PR #631 review. `IROp.numeric_policy` is the CANONICAL carrier — populated
    by `numeric_policy_pass.propagate_numeric_policy` so downstream passes need
    not re-derive storage/accum/math_mode from the op name — and every artifact
    builder below serialized only `op.kwargs`. A propagated policy therefore
    never reached the runtime, so the NVIDIA `math_mode` consumer added in this
    branch could only ever see a policy that a caller had happened to encode as
    a raw kwarg. The consumer was right and unreachable.

    Emitted only when a policy is present, so the payload for every
    policy-free op stays byte-identical.
    """
    policy = getattr(op, "numeric_policy", None)
    if policy is None:
        return None
    if isinstance(policy, dict):
        return {k: v for k, v in policy.items() if v is not None}
    fields = ("storage", "accum", "rounding", "scale", "quant_axis",
              "deterministic", "math_mode")
    out = {name: getattr(policy, name) for name in fields
           if getattr(policy, name, None) is not None}
    return out or None


def _op_payload(op: Any) -> Dict[str, Any]:
    """One runtime-ABI operation record. Single writer, so a field added for
    one target cannot go missing on another."""
    payload: Dict[str, Any] = {
        "op_name": op.op_name,
        "result": op.result,
        "operands": [o[1:] if o.startswith("%") else o for o in op.operands],
        "kwargs": dict(op.kwargs),
    }
    policy = _serialized_numeric_policy(op)
    if policy is not None:
        payload["numeric_policy"] = policy
    return payload


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
        compile_result: Optional[CompileResult] = None,
        cpu_tile: Tuple[int, int, int] = (128, 128, 64),
        source_origin: str = "inspect",
        lowering_diagnostics: Optional[List[JitDiagnostic]] = None,
        native_required: bool = False,
        recognized_package: Optional[Any] = None,
        dispatch_via_package: "bool | str" = False,
        differentiation_request: Optional[Any] = None,
        differentiation_provenance: Optional[Any] = None,
        backward_provenance: Optional[Any] = None,
        source_text: Optional[str] = None,
    ) -> None:
        self._fn = fn
        self.graph_ir = graph_ir
        # Decoration-time AST capture is a differential/compatibility oracle,
        # never the post-specialization compiler authority.  The first concrete
        # trace replaces ``graph_ir`` while this immutable candidate remains
        # available only to explicit frontend certificates.
        self._legacy_graph_ir = (
            None
            if graph_ir.module_attrs.get("tessera.frontend.authority") == '"tracer"'
            else graph_ir
        )
        self._frontend_source_text = source_text
        # Call-time constraint binding belongs to the stable Python ABI, not to
        # the post-specialization tracer module.  Tracer authority may replace
        # ``self.graph_ir`` with canonical a0/a1/... arguments after the first
        # call; retain the decoration-time symbolic names/dimensions so every
        # subsequent shape is checked and cached independently.
        #
        # A zero-function module is reachable (apple_gpu trace-defer,
        # auto_batch skip), and an empty ABI is NOT the right answer for it:
        # ``arg_names`` is how keyword arguments are ordered for the tracer and
        # how ``_constraint_ir_args`` re-checks shapes, so falling back to ()
        # silently dropped keyword calls and skipped constraint enforcement.
        # The signature is the source `lower` reads these off anyway, so derive
        # them rather than accept their absence (Decision #30).
        ir_args = (
            tuple(graph_ir.functions[0].args) if graph_ir.functions
            else _signature_ir_args(fn)
        )
        self._call_arg_names = tuple(argument.name for argument in ir_args)
        self._constraint_ir_args = ir_args
        self.inferred_effect = inferred_effect
        self.constraints = constraints
        self.deterministic = deterministic
        self.seed = seed
        self.target = target
        # Workstream B — phase specialization metadata (set by @jit(phase=...)).
        self.phase: Optional[Any] = None
        self.slo: Optional[Any] = None
        self.schedule_policy: Optional[Any] = None
        # Phase-F F5 — surgical tracer gate (supersedes the retired AST bridge).
        # A control-flow apple_gpu function (raw for/if → tessera.scf.* markers,
        # or an explicit tessera.control.* call) routes through the trace-by-
        # running path; pure straight-line functions keep the existing
        # package/auto_batch/canonical path untouched. Detected once at
        # decoration; best-effort (a detect bug must not break decoration).
        self._needs_trace: bool = False
        if target == "apple_gpu":
            try:
                from .trace import function_needs_tracer

                self._needs_trace = function_needs_tracer(graph_ir, fn)
            except Exception:
                self._needs_trace = False
        self.attn_config = attn_config
        self.cpu_plan = cpu_plan
        self.compile_bundle = compile_bundle
        # C.3 — canonical answer (typed artifacts + named gates + executable
        # | reason) from the same compile that produced ``compile_bundle``.
        # ``compile_result.bundle is compile_bundle`` post-retrofit; the new
        # field is the one-typed-surface every consumer should reach for.
        self.compile_result = compile_result
        self.cpu_tile = tuple(int(v) for v in cpu_tile)
        self.source_origin = source_origin
        self.lowering_diagnostics = tuple(lowering_diagnostics or [])
        self.native_required = bool(native_required)
        # Autodiff unification — the validated differentiation request (or None)
        # and its mode-neutral provenance facet. Reverse mode is additionally
        # mirrored onto the compatibility-only ``backward`` surface.
        self.differentiation_request = differentiation_request
        self.differentiation_provenance = differentiation_provenance
        self.backward_provenance = backward_provenance
        # PK8a (2026-06-02) — shape-free RecognizedOp when this module's
        # compute region is an authorable Apple-GPU packaged kernel, else
        # None. ``emit_package`` turns it into a real `.mtlpackage` given
        # concrete example-arg shapes. Populated only for target="apple_gpu".
        self.recognized_package = recognized_package
        self._emitted_package_path: Optional[str] = None
        # PK8e (2026-06-02) — route ``__call__`` through the authored
        # `.mtlpackage` instead of the live MPS/MSL envelope. Value:
        #   False  — never (default; live lane).
        #   True   — always, for any recognized region.
        #   "auto" — PK8g heuristic: only fused chains (``kind=="chain"``),
        #            which the benchmark shows win on the package lane; single
        #            matmul / unary ops stay on the faster live lane.
        # Per-shape caches keyed by (plan.name, plan.dims): authored package
        # paths + prepared Pipelines, so repeated same-shape calls reuse both.
        self.dispatch_via_package = dispatch_via_package
        self._package_path_cache: Dict[Any, str] = {}
        self._package_pipeline_cache: Dict[Any, Any] = {}
        # Last fallback reason (None on a native run).  Inspectable by
        # callers + by CompileReport.fallback_reason emission.
        self.last_fallback_reason: Optional[FallbackReason] = None
        # Phase 8.2 launch-overhead reduction: the artifact + its metadata
        # depend only on immutable construction inputs, so we lazily build
        # them once and reuse on every __call__. Without caching the small-
        # GEMM hot-path is dominated by metadata dict construction + the
        # SHA-256 over the artifact JSON inside `RuntimeArtifact.artifact_hash`.
        self._cached_artifact: Optional["RuntimeArtifact"] = None
        self._autodiff_specializations: Dict[Any, GraphIRModule] = {}
        # E2E-REAL-6: concrete tensor signatures are tracer-owned by default.
        # The decoration-time AST module remains a named candidate until a
        # family differential certificate permits its deletion.
        self._traced_frontend_specializations: Dict[Any, GraphIRModule] = {}
        self.frontend_authority: str = "pending_concrete_trace"
        self.frontend_authority_error: Optional[str] = None
        self.last_frontend_differential: Optional[Any] = None
        self._frontend_differential_certificates: Dict[Any, Any] = {}
        self._frontend_nonreexecuting_certificates: Dict[Any, Any] = {}
        self.last_backward_execution: Optional[Dict[str, Any]] = None
        functools.update_wrapper(self, fn)

    def _ensure_legacy_graph_ir(self) -> GraphIRModule:
        """Materialize the AST oracle only when a differential gate asks for it.

        Two states mean "not materialized", and both must rebuild. ``None`` is
        the tracer-authored module that never stored a candidate. A module with
        no functions is the emission-failure trace-defer / auto_batch skip,
        which stores an EMPTY ``GraphIRModule()`` -- returning that unchanged
        handed the differential gates a module they could only reject
        ("requires one Graph function"), when the oracle they wanted was one
        ``lower`` call away. ``GraphIRBuilder.lower`` returns its function even
        when the module later fails verification, so the recovered candidate is
        exactly the AST behaviour under test -- and a certificate that reports
        it as a MISMATCH is the true answer, where refusing to build one was
        merely an absent answer dressed as a failure.
        """
        if self._legacy_graph_ir is None or not self._legacy_graph_ir.functions:
            builder = GraphIRBuilder()
            builder.lower(self._fn, source_text=self._frontend_source_text,
                          source_origin=self.source_origin)
            self._legacy_graph_ir = builder.module()
        return self._legacy_graph_ir

    # PK8a (2026-06-02) — Graph IR → `.mtlpackage` AOT emission.
    def emit_package(
        self,
        out_path: Optional[Any] = None,
        *,
        example_args: Optional[Any] = None,
    ) -> Optional[str]:
        """Author a production ``.mtlpackage`` for this jitted Apple-GPU
        region and return its path (``None`` if not authorable / authoring
        failed).

        Authorable when ``self.recognized_package`` is set — i.e. the
        compiled region is a matmul, a single MPSGraph-lane op, or a fused
        chain (see :mod:`tessera.compiler.apple_package_author`). The
        packaged kernel needs concrete fp32 shapes:

        * pass ``example_args`` (the tensors you'd call the fn with) — shapes
          are read from their ``.shape`` (this is the realistic AOT path,
          mirroring ``aot.export(fn, *examples)``); or
        * omit them to fall back to static shapes baked into the Graph IR
          (rare — most ``@jit`` IR carries symbolic ``?`` dims).

        ``out_path`` defaults to a temp-cache path keyed by fn name + op +
        shape. The authored package loads + dispatches through PK1-PK7 and is
        positionally bound (``fill_input_at`` / ``read_output_at``).
        """
        rec = self.recognized_package
        if rec is None:
            return None

        if example_args is None:
            # Compile-time shape specialization: when the function's arg
            # annotations are static integers (``Tensor[8, 6]`` → arg
            # ``dim_names`` are all-numeric), derive shapes from them and
            # author with no example tensors. This is what lets
            # ``@jit(target="apple_gpu", emit_package=True)`` fire at compile.
            static_shapes = self._static_input_shapes()
            if static_shapes is not None:
                from .apple_package_author import plan_from_shapes
                plan = plan_from_shapes(rec, static_shapes)
                if plan is not None:
                    return self._author_plan(plan, out_path)
            # Last resort — dims baked into the IR operand types (rare).
            from .apple_package_author import recognize
            static_plan = recognize(self.graph_ir)
            if static_plan is None:
                return None
            return self._author_plan(static_plan, out_path)

        # Derive concrete shapes (+ fp32 check) from the example tensors.
        shapes: List[Tuple[int, ...]] = []
        for a in example_args:
            sh = getattr(a, "shape", None)
            if sh is None:
                return None
            dt = getattr(a, "dtype", None)
            if dt is not None and "float32" not in str(dt) \
                    and "f32" not in str(dt):
                return None  # authoring is fp32-only
            shapes.append(tuple(int(d) for d in sh))

        from .apple_package_author import plan_from_shapes
        plan = plan_from_shapes(rec, shapes)
        if plan is None:
            return None
        return self._author_plan(plan, out_path)

    def _static_input_shapes(self) -> Optional[List[Tuple[int, ...]]]:
        """Concrete input shapes from the function's arg annotations, when
        they are all static integers (``Tensor[8, 6]`` → ``dim_names`` are
        all-numeric). Returns ``None`` if any arg is symbolic (``"M"``) — the
        common case — so the caller knows it can't author without examples."""
        if not self.graph_ir.functions:
            return None
        shapes: List[Tuple[int, ...]] = []
        for arg in self.graph_ir.functions[0].args:
            dim_names = getattr(arg, "dim_names", None)
            if not dim_names or not all(str(d).isdigit() for d in dim_names):
                return None
            shapes.append(tuple(int(d) for d in dim_names))
        return shapes or None

    def _author_plan(self, plan: Any, out_path: Optional[Any]) -> Optional[str]:
        """Resolve a target path (temp cache when ``out_path`` is None) and
        author ``plan`` there. Returns the path on success, else ``None``."""
        if out_path is not None:
            path = str(out_path)
        else:
            import os
            import tempfile
            name = getattr(self._fn, "__name__", "fn")
            dims = "x".join(str(d) for d in plan.dims)
            cache = os.path.join(tempfile.gettempdir(),
                                 "tessera_apple_packages")
            os.makedirs(cache, exist_ok=True)
            path = os.path.join(cache, f"{name}_{plan.name}_{dims}.mtlpackage")
        if plan.author(path):
            self._emitted_package_path = path
            return path
        return None

    # PK8e — execute a call through the authored package (per-shape cache).
    def _ordered_inputs(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Optional[List[Any]]:
        """The positional input tensors in arg order (resolving kwargs by
        name). ``None`` if a declared arg is missing."""
        if not kwargs:
            return list(args)
        names = list(self.arg_names)
        out: List[Any] = []
        for i, nm in enumerate(names):
            if i < len(args):
                out.append(args[i])
            elif nm in kwargs:
                out.append(kwargs[nm])
            else:
                return None
        return out

    def _call_via_package(self, args: Tuple[Any, ...],
                          kwargs: Dict[str, Any]) -> Any:
        """Dispatch this call through the authored `.mtlpackage`. Returns the
        output array, or the ``_PKG_FALLBACK`` sentinel when the call can't be
        routed (non-fp32 / unrecognized shape / runtime unavailable) so the
        caller drops back to the normal MPS/MSL path. Authored packages +
        loaded pipelines are cached per (op, shape)."""
        import numpy as np

        rec = self.recognized_package
        if rec is None:
            return _PKG_FALLBACK
        # Package auto-routing is intentionally narrower than package
        # availability.  Only fused chains are candidates, and a current
        # device/shape-specific characterization report must prove that the
        # package beat the live route with native dispatch + oracle agreement.
        # This prevents a stale one-off benchmark or a host fallback from
        # silently changing generic JIT routing.
        if self.dispatch_via_package == "auto" and \
                getattr(rec, "kind", None) != "chain":
            return _PKG_FALLBACK
        inputs = self._ordered_inputs(args, kwargs)
        if not inputs:
            return _PKG_FALLBACK
        arrs: List[Any] = []
        for v in inputs:
            a = np.asarray(v)
            if a.dtype != np.float32:
                return _PKG_FALLBACK
            arrs.append(np.ascontiguousarray(a))
        shapes = [tuple(int(d) for d in a.shape) for a in arrs]

        from .apple_package_author import plan_from_shapes
        plan = plan_from_shapes(rec, shapes)
        if plan is None or plan.output_shape is None:
            return _PKG_FALLBACK
        if self.dispatch_via_package == "auto":
            import os
            from .apple_route_selector import package_route_selected
            report = os.environ.get("TESSERA_APPLE_GPU_ROUTE_CHARACTERIZATION")
            shape = "x".join(str(d) for d in plan.dims)
            if not package_route_selected(
                report, op=plan.name, shape=shape, dtype="f32",
            ):
                return _PKG_FALLBACK
        key = (plan.name, plan.dims)

        from .. import apple_mlpkg as _mp
        pipe = self._package_pipeline_cache.get(key)
        if pipe is None:
            path = self._package_path_cache.get(key)
            if path is None:
                path = self._author_plan(plan, None)
                if path is None:
                    return _PKG_FALLBACK
                self._package_path_cache[key] = path
            fn_name = _mp.first_function_name(path) or "main"
            pipe = _mp.compile_mlpackage(path, function_name=fn_name)
            if pipe is None or not pipe.prepare_tensors():
                return _PKG_FALLBACK
            self._package_pipeline_cache[key] = pipe

        for i, a in enumerate(arrs):
            if not pipe.fill_input_at(i, a.tobytes()):
                return _PKG_FALLBACK
        if not pipe.dispatch(timeout_ms=30_000):
            return _PKG_FALLBACK
        out_shape = plan.output_shape
        nbytes = int(np.prod(out_shape)) * 4
        raw = pipe.read_output_at(len(arrs), nbytes)
        if raw is None:
            return _PKG_FALLBACK
        return np.frombuffer(raw, dtype=np.float32).reshape(out_shape)

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
        self._enforce_call_time_stochastic_certificate(args, kwargs)
        self._establish_tracer_authority(args, kwargs)
        if self.differentiation_request is not None:
            self._specialized_autodiff_module(args, kwargs)
        try:
            # Phase-F F5 — surgical tracer dispatch (supersedes the AST bridge).
            # ONLY control-flow apple_gpu functions route through the tracer; pure
            # straight-line functions fall through to the existing package /
            # auto_batch / canonical path below, untouched. Raw data-dependent
            # `if`/`while` raises in the tracer ("use tessera.control.*").
            if self.target == "apple_gpu" and self._needs_trace:
                from .trace import jit_trace_enabled, run_jit_traced

                if jit_trace_enabled():
                    return self._run_traced_with_diagnostic(
                        run_jit_traced, args, kwargs)
            if self.cpu_plan is not None and self.cpu_plan.target_kind == "cpu":
                if self.execution_kind == "native_cpu":
                    return self._native_cpu_fast_call(args, kwargs)
                # Phase 4 — run the whole graph through the tessera_jit MLIR→LLVM
                # lane (real codegen) for the covered f32 op set, before the numpy
                # reference plan. A fallback sentinel means the graph is outside
                # the lane (unsupported op / non-f32 / rank) → numpy.
                jit_result = self._try_tessera_jit_call(args, kwargs)
                if jit_result is not _JIT_FALLBACK:
                    return jit_result
                return self.cpu_plan.execute(args, kwargs, self.arg_names)
            if (
                self.cpu_plan is not None
                and self.cpu_plan.target_kind == "apple_cpu"
                and self.compile_bundle is not None
                and self.compile_bundle.executable
            ):
                return self._apple_cpu_fast_call(args, kwargs)
            # PK8e — opt-in: execute through the authored `.mtlpackage`. Tried
            # before the live MPS/MSL path; a fallback sentinel means the call
            # couldn't be routed (non-fp32 / unrecognized shape / runtime
            # down), so we drop through to the normal apple_gpu lane.
            if (
                self.dispatch_via_package
                and self.recognized_package is not None
                and self.cpu_plan is not None
                and self.cpu_plan.target_kind == "apple_gpu"
            ):
                result = self._call_via_package(args, kwargs)
                if result is not _PKG_FALLBACK:
                    return result
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

    def _run_traced_with_diagnostic(
        self, run_jit_traced: Callable, args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Execute the apple_gpu tracer lane behind a stable diagnostic.

        Decision #21: a lowering the backend cannot carry names the op and the
        target -- it never surfaces as a raw Python exception. The tracer runs
        the user body directly, so anything the body does to a ``Tracer`` that
        the tracer does not model (``AttributeError`` for an unmodelled method,
        ``TypeError`` for an unmodelled coercion) escapes as an unhandled
        interpreter error unless it is lifted here.

        Tessera-level errors pass through untouched: ``TesseraTraceError``
        already names the construct and the remedy (``use tessera.control.*``),
        and rewrapping it would break its stability contract. Only a foreign
        exception is lifted, and it carries the decoration-time reason this
        function was routed to the tracer at all -- which is where the
        unlowerable construct is named.
        """
        from .trace import TesseraCallBindingError, TesseraTraceError

        try:
            return run_jit_traced(self, args, kwargs)
        except (TesseraJitError, TesseraTraceError, TesseraCallBindingError):
            # Already a stable Tessera diagnostic, or a caller error that must
            # keep Python's own semantics. Re-wrapping either would bury the
            # message that names the real culprit -- and a misspelled keyword is
            # not a tracer failure.
            raise
        except Exception as exc:
            raise TesseraJitError(
                f"[{JitDiagnosticCode.APPLE_GPU_TRACE_FAILED.value}] "
                f"the apple_gpu tracer could not execute {self._fn.__name__!r}: "
                f"{type(exc).__name__}: {exc}"
                + self._trace_defer_context()
            ) from exc

    def _trace_defer_context(self) -> str:
        """Why this function reached the tracer, quoted from decoration.

        Empty when the tracer is the ordinary route (a control-flow body the
        AST lane lowered fine). Non-empty when AST emission failed, in which
        case the recorded diagnostics name the construct the AST front end
        could not lower -- the fact a reader needs and the raw exception on its
        own never carries.
        """
        reasons = [
            d.format() for d in self.lowering_diagnostics
            if d.code in (
                JitDiagnosticCode.APPLE_GPU_TRACE_DEFERRED.value,
                "PY_FRONTEND_UNSUPPORTED",
            )
        ]
        if not reasons:
            return ""
        joined = "".join(f"\n  - {reason}" for reason in reasons)
        return (
            "\n  this function was routed to the tracer because the AST "
            f"front end could not lower it:{joined}"
        )

    def _enforce_call_time_stochastic_certificate(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        """Certify stochastic identity with the concrete traced-op graph.

        Decoration consumes emitted Graph records and fails closed on unresolved
        calls. A concrete trace additionally resolves aliases and dispatch, so
        an unseeded deterministic request rejects any random dependency before
        the eager fallback can execute it.
        Seeded RNG remains permitted by the established deterministic contract.
        """
        if not self.deterministic or self.seed is not None:
            return
        import numpy as np
        from .effects import TesseraEffectError
        from .stochastic_graph import certify_deterministic
        from .trace import TesseraTraceError, trace

        if kwargs or not all(isinstance(arg, np.ndarray) for arg in args):
            return  # Existing AST gate remains the diagnostic for non-traceable calls.
        try:
            traced = trace(self._fn, *args)
        except TesseraTraceError:
            return  # A tracing limitation must not reject an otherwise valid JIT call.
        deterministic, reason = certify_deterministic(traced.body, traced.outputs)
        if not deterministic:
            raise TesseraEffectError(
                self._fn.__name__, Effect.pure, Effect.random,
                message=(f"@jit(deterministic=True) function {self._fn.__name__!r} "
                         f"has a random traced dependency: {reason}. Add seed=... "
                         "or remove the RNG call."),
            )

    def _enforce_call_time_constraints(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        """Resolve symbolic dim names against actual call shapes + re-run the solver.

        Skipped when the function has no dim-annotated args, or when the solver
        was already satisfied at decoration time with concrete ``bindings=``.
        Cached per-shape to avoid re-checking on every call.
        """
        if not self._constraint_ir_args:
            return
        ir_args = self._constraint_ir_args
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
                # ``dim_name`` is statically typed ``str`` so the
                # ``isidentifier`` / ``isnumeric`` guards below are the
                # only runtime narrowing we need.
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

    def _specialized_autodiff_module(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> GraphIRModule:
        """Specialize an autodiff Graph IR module from concrete call values.

        The decoration-time module remains symbolic. One immutable specialized
        copy is cached per dtype/shape signature and is the input to the paired
        compiler path.
        """
        import numpy as np
        from .graph_ir import specialize_module_from_values

        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None or len(ordered) != len(self.arg_names):
            raise TesseraJitError("autodiff specialization requires every argument")
        signature = tuple(
            (str(np.asarray(value).dtype), tuple(int(d) for d in np.asarray(value).shape))
            for value in ordered
        )
        cached = self._autodiff_specializations.get(signature)
        if cached is not None:
            return cached
        values = dict(zip(self.arg_names, ordered))
        if ordered and all(isinstance(value, np.ndarray) for value in ordered):
            try:
                specialized = self._trace_frontend_capture(args, kwargs)[0]
            except TesseraJitError:
                specialized = specialize_module_from_values(self.graph_ir, values)
        else:
            specialized = specialize_module_from_values(self.graph_ir, values)
        self._autodiff_specializations[signature] = specialized
        return specialized

    def rank_parametric_buckets(self, buckets, *, tessera_opt: str):
        """Opt-in native pre-elaboration analysis; never selects execution.

        Reuses one optimized symbolic recipe across the supplied shape buckets.
        Keep the decoration-time oracle even after concrete tracing replaces
        ``graph_ir``. Unresolved element types fail before invoking MLIR.
        """
        from .parametric_recipe import prepare_recipe
        from .presburger import presburger_system_from_constraints

        from .constraints import Divisible, Equal, Range
        if any(not isinstance(c, (Divisible, Equal, Range)) for c in self.constraints._constraints):
            raise ValueError("parametric rank tier requires integer-affine constraints")
        module = self._ensure_legacy_graph_ir()
        system = presburger_system_from_constraints(self.constraints._constraints)
        return prepare_recipe(module, tessera_opt=tessera_opt, system=system).rank_buckets(buckets)

    def specialized_autodiff_ir(self, *args: Any, **kwargs: Any) -> str:
        """Concrete Graph IR used for this autodiff call signature."""
        if self.differentiation_request is None:
            raise TesseraJitError("specialized_autodiff_ir requires autodiff=...")
        return self._specialized_autodiff_module(args, kwargs).to_mlir()

    def _trace_frontend_capture(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        *,
        require_outputs: bool = False,
    ) -> Tuple[GraphIRModule, Optional[Any]]:
        """Trace one concrete signature into canonical Graph IR.

        E2E-REAL-6 migrates families at this boundary before deleting their AST
        or backend resynthesis lanes. The returned trace retains concrete output
        values for the explicit differential gate; the cached module is the
        default frontend authority for subsequent compiler clients.
        """
        import hashlib
        import numpy as np
        from .trace import trace, to_graph_ir_module

        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None or len(ordered) != len(self.arg_names):
            raise TesseraJitError("traced autodiff specialization requires every argument")
        signature = tuple(
            (str(np.asarray(value).dtype), tuple(int(d) for d in np.asarray(value).shape))
            for value in ordered
        )
        cached = self._traced_frontend_specializations.get(signature)
        if cached is not None and not require_outputs:
            return cached, None
        try:
            traced = trace(self._fn, *ordered)
            # The AST module is a naming convenience here, not an input to the
            # trace: a zero-function one (apple_gpu trace-defer, auto_batch
            # skip) must not fail a capture that never needed it. Both fields
            # already have their fallback in scope -- the qualname hash this
            # line computes anyway, and the function's own name.
            emitted = (
                self.graph_ir.functions[0] if self.graph_ir.functions else None
            )
            source_hash = (
                emitted.source_hash if emitted is not None else None
            ) or hashlib.sha256(
                self._fn.__qualname__.encode("utf-8")
            ).hexdigest()
            module = to_graph_ir_module(
                traced,
                name=emitted.name if emitted is not None else self._fn.__name__,
                source_hash=source_hash,
            )
            if self.differentiation_request is not None:
                intent = self.differentiation_request.module_intent_attrs()
                module.module_attrs.update(intent)
                module.functions[0].fn_attrs.update(intent)
            # FRONTEND-IR-MEDIUM-1 (iii): the tracer rebuilds its arguments from
            # shape and dtype alone, so the region privileges (Decision #2), the
            # ConstraintSolver's facts (Decision #4) and the symbolic dimension
            # names all stop here. Record what was known and declare the debt,
            # so the loss is attributable and machine-checked rather than
            # invisible (Decision #32).
            from .graph_ir import declare_frontier_debt
            declare_frontier_debt(
                module,
                args=self._constraint_ir_args,
                constraints=self.constraints,
            )
        except Exception as exc:
            raise TesseraJitError(f"tracer frontend failed: {exc}") from exc
        self._traced_frontend_specializations[signature] = module
        return module, traced

    def _traced_autodiff_module(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> GraphIRModule:
        """Compatibility name for the tracer-owned compiler specialization."""
        return self._trace_frontend_capture(args, kwargs)[0]

    def _establish_tracer_authority(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        """Make tracing the default capture for ordinary concrete tensor calls.

        Unmigrated execution candidates may continue when tracing cannot yet
        represent a signature, but the fallback is explicit and inspectable;
        no compiler client may mistake the AST module for tracer authority.
        """
        import numpy as np
        from .effects import Effect, infer_graph_effects

        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None or not ordered or not all(
            isinstance(value, np.ndarray) for value in ordered
        ):
            self.frontend_authority = "legacy_candidate_non_tensor_signature"
            return
        # A module with no functions is a REACHABLE state, not a bug: both
        # non-emitting paths in ``_jit_emit_graph_ir`` hand ``JitFn`` an empty
        # ``GraphIRModule()`` -- the apple_gpu emission-failure trace-defer and
        # the auto_batch skip. Reading ``functions[0]`` unguarded turned that
        # into a bare ``IndexError`` out of ``__call__``.
        #
        # Guarding the read is not enough, because there is nothing here to
        # settle either. This routine exists to let the tracer SUPERSEDE an
        # emitted module; with no emitted function there is nothing to
        # supersede, and the probe's only remaining effect is its cost --
        # ``_trace_frontend_capture`` traces by RUNNING the body, which on the
        # auto_batch route is a second GPU execution of a decode chain before
        # the real call. That aborted inside MPSGraph rather than failing.
        # Stop here: both routes trace on their own at call time and never read
        # ``graph_ir``. (``self.graph_ir`` is the gate, not
        # ``_legacy_graph_ir`` -- the latter is ``None`` for a tracer-authored
        # module, which HAS a function and must still be probed.)
        if not self.graph_ir.functions:
            self.frontend_authority = "legacy_candidate_no_emitted_function"
            return
        legacy = self._legacy_graph_ir
        effect = self.inferred_effect
        if legacy is not None and legacy.functions:
            effect, _ = infer_graph_effects(legacy.functions[0].body)
        if effect != Effect.pure:
            self.frontend_authority = "legacy_candidate_effectful_signature"
            return
        try:
            traced_module, _ = self._trace_frontend_capture(args, kwargs)
        except TesseraJitError as exc:
            self.frontend_authority = "legacy_candidate_unmigrated"
            self.frontend_authority_error = str(exc)
            return
        self.graph_ir = traced_module
        self.frontend_authority = "tracer"
        self.frontend_authority_error = None

    def frontend_differential(
        self,
        *args: Any,
        rtol: float = 1e-5,
        atol: float = 1e-6,
        _permitted_effect_ops: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> Any:
        """Prove AST-candidate/tracer parity for one pure tensor signature."""
        import numpy as np
        from .frontend_authority import certify_frontends
        from .graph_ir import specialize_module_from_values

        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None or len(ordered) != len(self.arg_names):
            raise TesseraJitError("frontend differential requires every argument")
        if not all(isinstance(value, np.ndarray) for value in ordered):
            raise TesseraJitError("frontend differential requires tensor arguments")
        signature = tuple(
            (str(value.dtype), tuple(int(dim) for dim in value.shape))
            for value in ordered
        )
        certificate_key = (signature, tuple(sorted(_permitted_effect_ops)))
        cached = self._frontend_differential_certificates.get(certificate_key)
        if cached is not None:
            self.last_frontend_differential = cached
            return cached
        tracer_module, traced = self._trace_frontend_capture(
            args, kwargs, require_outputs=True
        )
        assert traced is not None
        if not traced.output_values or any(value is None for value in traced.output_values):
            raise TesseraJitError("tracer produced no concrete differential outputs")
        legacy_module = specialize_module_from_values(
            self._ensure_legacy_graph_ir(), dict(zip(self.arg_names, ordered))
        )
        legacy_result = self._fn(*args, **kwargs)
        legacy_outputs = legacy_result if isinstance(legacy_result, tuple) else (legacy_result,)
        try:
            certificate = certify_frontends(
                legacy_module=legacy_module,
                tracer_module=tracer_module,
                legacy_outputs=legacy_outputs,
                tracer_outputs=traced.output_values,
                permitted_effect_ops=_permitted_effect_ops,
                rtol=rtol,
                atol=atol,
            )
        except ValueError as exc:
            raise TesseraJitError(str(exc)) from exc
        self.last_frontend_differential = certificate
        self._frontend_differential_certificates[certificate_key] = certificate
        return certificate

    def _frontend_nonreexecuting_certificate(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        *,
        graph_consumers: tuple[str, ...],
    ) -> Any:
        """Certify one stateful family without replaying its source program."""
        import numpy as np

        from .frontend_authority import certify_frontends_non_reexecuting
        from .graph_ir import specialize_module_from_values

        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None or len(ordered) != len(self.arg_names):
            raise TesseraJitError(
                "non-reexecuting frontend proof requires every argument"
            )
        signature = tuple(
            (str(np.asarray(value).dtype), tuple(int(dim) for dim in np.asarray(value).shape))
            for value in ordered
        )
        key = (signature, tuple(sorted(graph_consumers)))
        cached = self._frontend_nonreexecuting_certificates.get(key)
        if cached is not None:
            self.last_frontend_differential = cached
            return cached
        # `_specialized_autodiff_module` establishes and caches this concrete
        # trace before family selection. Reuse it: requesting output values
        # here would execute a stateful source a second time.
        tracer_module, _ = self._trace_frontend_capture(args, kwargs)
        legacy_module = specialize_module_from_values(
            self._ensure_legacy_graph_ir(), dict(zip(self.arg_names, ordered))
        )
        try:
            certificate = certify_frontends_non_reexecuting(
                legacy_module=legacy_module,
                tracer_module=tracer_module,
                graph_consumers=graph_consumers,
            )
        except ValueError as exc:
            raise TesseraJitError(str(exc)) from exc
        self.last_frontend_differential = certificate
        self._frontend_nonreexecuting_certificates[key] = certificate
        return certificate

    def _compile_jvp_module(self, module: GraphIRModule) -> str:
        """Run the sole compiler forward transform for an already-owned module."""
        import re
        import subprocess
        from .scheduled_matmul import find_tessera_opt

        opt = find_tessera_opt()
        if opt is None:
            raise TesseraJitError("tessera-opt not built; cannot emit paired JVP")
        graph_text = re.sub(
            r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir()
        )
        transformed = subprocess.run(
            [str(opt), "--tessera-autodiff-forward", "/dev/stdin"],
            input=graph_text,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if transformed.returncode != 0:
            raise TesseraJitError(
                "forward autodiff transform failed: " + transformed.stderr.strip()
            )
        return transformed.stdout

    def _compile_hvp_module(self, module: GraphIRModule) -> str:
        """Compose paired reverse and forward transforms into an exact HVP."""
        import re
        import subprocess
        from .scheduled_matmul import find_tessera_opt

        opt = find_tessera_opt()
        if opt is None:
            raise TesseraJitError("tessera-opt not built; cannot emit exact HVP")
        graph_text = re.sub(
            r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir()
        )
        transformed = subprocess.run(
            [str(opt), "--tessera-autodiff-hvp-pipeline", "/dev/stdin"],
            input=graph_text,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if transformed.returncode != 0:
            raise TesseraJitError(
                "exact forward-over-reverse HVP transform failed: "
                + transformed.stderr.strip()
            )
        return transformed.stdout

    def compiled_jvp_ir(self, *args: Any, **kwargs: Any) -> str:
        """Return the compiler-emitted paired JVP Graph IR for this signature.

        This is an IR-transformed product ABI, not a native-execution claim.
        The returned ``@f__jvp`` takes all primals followed by tangents for the
        requested ``wrt`` indices and returns primals followed by output
        tangents. Missing tools or an unsupported active op fail closed.
        """
        request = self.differentiation_request
        if request is None or request.mode != "forward":
            raise TesseraJitError(
                "compiled_jvp_ir requires @jit(autodiff='forward' or 'jvp')"
            )
        module = self._specialized_autodiff_module(args, kwargs)
        return self._compile_jvp_module(module)

    def compiled_hvp_ir(self, *args: Any, **kwargs: Any) -> str:
        """Return the exact compiler-owned forward-over-reverse product IR.

        The generated ``@f__bwd__jvp`` ABI takes backward primals (including
        the output-cotangent seed) followed by tangents for the original
        differentiable inputs.  Its tangent results are Hessian-vector
        products. Unsupported second-order operations fail in the compiler;
        this method never substitutes finite differences.
        """
        request = self.differentiation_request
        if request is None or request.mode != "reverse":
            raise TesseraJitError(
                "compiled_hvp_ir requires @jit(autodiff='reverse')"
            )
        module = self._specialized_autodiff_module(args, kwargs)
        return self._compile_hvp_module(module)

    def native_jvp(
        self, *args: Any, tangents: Any, **kwargs: Any
    ) -> tuple[Any, Any]:
        """Execute a compiler-owned native primal/JVP product package.

        The initial product envelope covers single-input linear reductions and
        FFT-family transforms plus non-affine normalization.  Unsupported
        graphs and architectures fail closed; in particular a generic ROCm
        target may not silently run the gfx1151 package on gfx1200/gfx1250.
        """
        import numpy as np
        from tessera.runtime import RuntimeArtifact, launch
        from .native_jvp_plugins import build_native_jvp_family_artifact

        request = self.differentiation_request
        if request is None or request.mode != "forward":
            raise TesseraJitError(
                "native_jvp requires @jit(autodiff='forward' or 'jvp')"
            )
        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None:
            raise TesseraJitError("native_jvp could not bind the compiled inputs")
        tangent_values = tangents if isinstance(tangents, (tuple, list)) else (tangents,)
        if len(tangent_values) != len(request.wrt_indices):
            raise TesseraJitError("native_jvp requires one tangent per active input")
        traced_module = self._traced_autodiff_module(args, kwargs)
        graph_ops = [op for fn in traced_module.functions for op in fn.body]
        permitted_effect_ops: tuple[str, ...] = ()
        if len(graph_ops) == 1 and graph_ops[0].op_name == "tessera.dropout":
            dropout = graph_ops[0]
            if (not bool(dropout.kwargs.get("training", True)) or
                    float(dropout.kwargs.get("p", 0.5)) == 0.0 or
                    dropout.kwargs.get("seed") is not None):
                permitted_effect_ops = ("tessera.dropout",)
        self.frontend_differential(
            *args, _permitted_effect_ops=permitted_effect_ops, **kwargs
        )
        primal_inputs = [np.ascontiguousarray(np.asarray(value)) for value in ordered]
        tangent_inputs = {
            index: np.ascontiguousarray(np.asarray(value))
            for index, value in zip(request.wrt_indices, tangent_values)
        }
        for index, tangent in tangent_inputs.items():
            if index >= len(primal_inputs) or primal_inputs[index].shape != tangent.shape:
                raise TesseraJitError("native JVP primal and tangent shapes must match")
        if len(graph_ops) != 1:
            raise TesseraJitError("native JVP currently requires a single Graph operation")
        source = graph_ops[0]
        target = normalize_target_kind(self.target)
        if target not in {"x86", "rocm", "nvidia_sm120"}:
            raise TesseraJitError(
                "native JVP is currently packaged for x86, ROCm, and exact sm120 only"
            )
        architecture = "zen5_avx512"
        execution_mode = "cpu_avx512"
        if target == "rocm":
            from tessera import runtime as _runtime
            chip = _runtime._rocm_chip()
            if chip != "gfx1151":
                raise TesseraJitError(
                    f"native ROCm JVP requires exact gfx1151; detected {chip!r}"
                )
            architecture = "gfx1151"
            execution_mode = "hip_runtime"
        elif target == "nvidia_sm120":
            architecture = "sm120"
            execution_mode = "cuda_runtime"

        launch_names = tuple(f"primal_{index}" for index in range(len(primal_inputs))) + tuple(
            f"tangent_{index}" for index in request.wrt_indices
        )
        launch_values = tuple(primal_inputs) + tuple(
            tangent_inputs[index] for index in request.wrt_indices
        )
        package_key = (
            target,
            architecture,
            source.op_name,
            tuple((str(value.dtype), tuple(int(dim) for dim in value.shape)) for value in primal_inputs),
            tuple((index, str(tangent_inputs[index].dtype)) for index in request.wrt_indices),
            repr(sorted(source.kwargs.items())),
        )
        package_cache = getattr(self, "_native_jvp_packages", None)
        if package_cache is None:
            package_cache = {}
            self._native_jvp_packages = package_cache
        package = package_cache.get(package_key)
        if package is None:
            paired_ir = self._compile_jvp_module(traced_module)
            try:
                _, package = build_native_jvp_family_artifact(
                    source=source,
                    primal_inputs=primal_inputs,
                    target=target,
                    architecture=architecture,
                    execution_mode=execution_mode,
                    source_graph_ir=traced_module.to_mlir(),
                    paired_jvp_ir=paired_ir,
                    wrt_indices=request.wrt_indices,
                    arg_names=launch_names,
                )
            except ValueError as exc:
                raise TesseraJitError(str(exc)) from exc
            package_cache[package_key] = package
        family = str(package.contract["family"])
        result = launch(
            RuntimeArtifact(metadata=package.runtime_metadata()),
            launch_values,
        )
        if not result.get("ok") or result.get("execution_mode") != execution_mode:
            raise TesseraJitError(
                f"native {target} JVP launch failed: {result.get('reason')}"
            )
        self.last_jvp_execution = {
            "compiler_path": f"{target}_jvp_compiled",
            "execution_kind": "native_cpu" if target == "x86" else "native_gpu",
            "execution_mode": execution_mode,
            "evidence_target": ("x86_avx512" if target == "x86" else
                                "nvidia_sm120" if target == "nvidia_sm120" else
                                "rocm_gfx1151"),
            "artifact_hash": package.artifact_hash,
            "paired_jvp_ir_digest": package.contract["paired_jvp_ir_digest"],
            "source_graph_ir_digest": package.contract["source_graph_ir_digest"],
            "schedule_program_digest": package.contract["schedule_program"]["digest"],
            "tile_program_digest": package.contract["tile_program"]["digest"],
            "frontend_authority": "tracer",
            "family": family,
        }
        primal, tangent = result["output"]
        return primal, tangent

    def native_backward(
        self, *args: Any, out_cotangents: Any, **kwargs: Any
    ) -> tuple[Any, ...]:
        """Compile and launch the paired-pass backward through LLVM JIT on CPU.

        There is no NumPy fallback: missing tools, unsupported lowering, or ABI
        failure raises. Successful execution records the exact compiler path
        and invocation delta in ``last_backward_execution``.
        """
        if self.differentiation_request is None:
            raise TesseraJitError("native_backward requires @jit(autodiff=...)")
        if self.differentiation_request.mode != "reverse":
            raise TesseraJitError(
                "native_backward requires @jit(autodiff='reverse'); "
                "forward requests use compiled_jvp_ir"
            )
        source_module = self._specialized_autodiff_module(args, kwargs)
        target_kind = normalize_target_kind(self.target)
        graph_ops = [
            op for function in source_module.functions for op in function.body
        ]
        frontend_certificate = None
        if len(graph_ops) == 1:
            from .native_vjp_plugins import (
                native_vjp_frontend_proof_policy,
                native_vjp_plugin_available,
            )

            if native_vjp_plugin_available(graph_ops[0].op_name, target_kind):
                proof_policy = native_vjp_frontend_proof_policy(
                    graph_ops[0].op_name, target_kind
                )
                if proof_policy == "non_reexecuting_state_lineage":
                    frontend_certificate = self._frontend_nonreexecuting_certificate(
                        args,
                        kwargs,
                        graph_consumers=(graph_ops[0].op_name,),
                    )
                else:
                    from .effects import infer_graph_effects
                    from .native_vjp_plugins import (
                        native_vjp_differential_effect_exemptions,
                        native_vjp_differential_safe,
                    )

                    source_effect, _ = infer_graph_effects(graph_ops)
                    if not native_vjp_differential_safe(
                        graph_ops[0], target_kind, source_effect.name
                    ):
                        raise TesseraJitError(
                            "native VJP plugin cannot safely run its frontend "
                            "differential certificate for this effect envelope"
                        )
                    exemptions = native_vjp_differential_effect_exemptions(
                        graph_ops[0], target_kind, source_effect.name
                    )
                    frontend_certificate = self.frontend_differential(
                        *args, _permitted_effect_ops=exemptions, **kwargs
                    )
                # The initial specialization is permitted to be the retained
                # AST compatibility candidate so we can discover its family.
                # Once a plugin claims the call, replace it with the certified
                # tracer module; never label the candidate as tracer authority.
                source_module = self._traced_autodiff_module(args, kwargs)
                graph_ops = [
                    op
                    for function in source_module.functions
                    for op in function.body
                ]
        if len(graph_ops) == 1:
            from .native_vjp_plugins import execute_native_vjp_family

            ordered = self._ordered_inputs(args, kwargs)
            if ordered is None:
                raise TesseraJitError(
                    "native backward requires every forward argument"
                )
            request = self.differentiation_request
            if request is None:
                raise TesseraJitError(
                    "native backward requires a differentiation request"
                )
            plugin_result = execute_native_vjp_family(
                source=graph_ops[0],
                target=target_kind,
                ordered_inputs=ordered,
                arg_names=self.arg_names,
                source_arg_names=tuple(
                    argument.name for argument in source_module.functions[0].args
                ),
                out_cotangents=out_cotangents,
                wrt_names=request.wrt,
                source_graph_ir=source_module.to_mlir(
                    canonical=True,
                    target=(
                        "rocm_gfx1151" if target_kind == "rocm" else target_kind
                    ),
                ),
                frontend_certificate=frontend_certificate,
            )
            if plugin_result is not None:
                self.last_backward_execution = dict(plugin_result.execution)
                return plugin_result.gradients
        if target_kind == "rocm":
            if len(graph_ops) == 1:
                raise TesseraJitError(
                    "no registered ROCm native VJP plugin for "
                    f"{graph_ops[0].op_name.removeprefix('tessera.')!r}"
                )
            raise TesseraJitError(
                "ROCm native backward requires one registered Graph op"
            )
        if target_kind == "nvidia_sm120":
            if len(graph_ops) == 1:
                raise TesseraJitError(
                    "no registered NVIDIA SM120 native VJP plugin for "
                    f"{graph_ops[0].op_name.removeprefix('tessera.')!r}"
                )
            raise TesseraJitError(
                "NVIDIA SM120 native backward requires one registered Graph op"
            )
        if target_kind == "x86":
            if len(graph_ops) == 1:
                raise TesseraJitError(
                    "no registered x86 native VJP plugin for "
                    f"{graph_ops[0].op_name.removeprefix('tessera.')!r}"
                )
            raise TesseraJitError(
                "x86 native backward currently requires one registered Graph op"
            )
        if target_kind != "cpu":
            raise TesseraJitError(
                "native_backward currently supports target='cpu'/'x86', "
                "verified ROCm lanes, or verified NVIDIA SM120 lanes")

        import numpy as np
        import re
        import subprocess
        from .. import _jit_boundary as _jb

        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None:
            raise TesseraJitError("native_backward requires every forward argument")
        inputs = [np.ascontiguousarray(np.asarray(value)) for value in ordered]
        cpu_cots = (
            tuple(out_cotangents)
            if isinstance(out_cotangents, (tuple, list))
            else (out_cotangents,)
        )
        cotangents = [np.ascontiguousarray(np.asarray(value)) for value in cpu_cots]
        module = self._specialized_autodiff_module(args, kwargs)
        opt = _jb._find_tessera_opt()
        if opt is None:
            raise TesseraJitError("tessera-opt not built; cannot emit paired backward")
        # GraphIR's human-facing printer uses ``tessera.op(%args)`` while the
        # native MLIR parser accepts either registered custom syntax or generic
        # quoted operations. Use generic form here so every Graph op takes the
        # same parse-stable front door into tessera-opt.
        graph_text = re.sub(
            r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir())
        transformed = subprocess.run(
            [str(opt), "--tessera-autodiff-paired", "/dev/stdin"],
            input=graph_text, capture_output=True, text=True, timeout=60,
        )
        if transformed.returncode != 0:
            raise TesseraJitError(
                "paired autodiff transform failed: " + transformed.stderr.strip())

        from tessera.runtime import RuntimeArtifact, launch

        compiler_path = "cpu_autodiff_matmul_llvm_jit"
        artifact = RuntimeArtifact(metadata={
            "target": "cpu",
            "compiler_path": compiler_path,
            "executable": True,
            "execution_kind": "native_cpu",
            "execution_mode": "mlir_llvm_jit",
            "paired_mlir": transformed.stdout,
            "backward_symbol": self.graph_ir.functions[0].name + "__bwd",
            "output_shapes": [list(value.shape) for value in inputs],
        })
        before = _jb.invocation_count()
        result = launch(artifact, tuple([*inputs, *cotangents]))
        if not result.get("ok"):
            raise TesseraJitError(
                "native backward runtime launch failed: " + str(result.get("reason")))
        after = _jb.invocation_count()
        if after <= before:
            raise TesseraJitError("native backward invocation counter did not advance")
        self.last_backward_execution = {
            "compiler_path": compiler_path,
            "execution_kind": "native_cpu",
            "execution_mode": "mlir_llvm_jit",
            "invocation_delta": after - before,
        }
        return tuple(result["output"])

    def _try_tessera_jit_call(self, args: Tuple[Any, ...],
                              kwargs: Dict[str, Any]) -> Any:
        """Phase 4 — run the whole CPU graph through the tessera_jit MLIR→LLVM
        lane (tessera-to-linalg → bufferize → loops → LLVM, optLevel=2), making
        the real compiler the executed path for the covered f32 op set instead
        of the numpy reference interpreter.

        Returns ``_JIT_FALLBACK`` (defer to numpy) when the graph is outside the
        lane: keyword args (graph args are positional), an unsupported op, a
        non-f32 input, or a shape/rank the GraphFn builder rejects. Correctness
        of the covered ops is proven by the equivalence tests in
        ``tests/unit/test_native_cpu_jit.py`` — a fallback handles "couldn't
        run", never "ran wrong"."""
        import os
        if os.environ.get("TESSERA_DISABLE_CPU_JIT"):
            return _JIT_FALLBACK
        if kwargs:                              # graph args are positional
            return _JIT_FALLBACK
        try:
            metadata = self.runtime_artifact().metadata or {}
        except Exception:                       # noqa: BLE001 — defer to numpy
            return _JIT_FALLBACK
        ops = metadata.get("ops") or []
        arg_names = list(metadata.get("arg_names") or [])
        if not ops or len(args) != len(arg_names):
            return _JIT_FALLBACK

        from .._jit_boundary import (
            _BF16, UnsupportedJitOp, graph_ops_supported, run_graph_ops)
        if not graph_ops_supported(ops):
            return _JIT_FALLBACK

        import numpy as np

        def _elem_for(dt) -> str | None:
            # M1 Max NEON: f32 + f16 (ARMv8.2-A FP16) are native; bf16 is
            # correct but emulated via f32 in-kernel (M1 predates ARMv8.6 BF16).
            # f64 accumulates in f64 throughout (the TesseraToLinalg matmul/reduce
            # low-precision-→f32 rule does not fire) — the exact-precision lane for
            # gradient-checking / numerical validation against the numpy reference.
            if dt == np.float32:
                return "f32"
            if dt == np.float64:
                return "f64"
            if dt == np.float16:
                return "f16"
            if _BF16 is not None and dt == _BF16:
                return "bf16"
            return None

        elem: str | None = None
        arrays: Dict[str, Any] = {}
        for name, value in zip(arg_names, args):
            arr = np.asarray(value)
            this_elem = _elem_for(arr.dtype)
            if this_elem is None:               # unsupported dtype → numpy
                return _JIT_FALLBACK
            if elem is None:
                elem = this_elem
            elif this_elem != elem:             # mixed dtype → numpy
                return _JIT_FALLBACK
            arrays[name] = np.ascontiguousarray(arr)

        try:
            result = run_graph_ops(
                arg_names, ops, metadata.get("output_name"), arrays,
                elem=elem or "f32")
        except (UnsupportedJitOp, TesseraJitError):
            return _JIT_FALLBACK
        self.last_fallback_reason = None
        return result

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

        # ``launch_args`` is typed ``Any`` because the metadata
        # dispatcher accepts both the kwargs-dict form and the
        # raw-args-tuple form depending on which path the
        # ``cpu_plan`` shape selects.
        launch_args: Any
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
            # ``cpu_plan`` is None-checked at the entry point; narrow.
            assert self.cpu_plan is not None
            return self.cpu_plan.execute(args, kwargs, self.arg_names)

    def _apple_cpu_fast_call(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Phase 8.2 launch-overhead fast path. Bypasses ``runtime.launch`` by
        calling the metadata dispatcher directly with the cached artifact's
        metadata dict — skipping per-call telemetry events, the artifact
        SHA-256, and the JSON serialization that backs it. The public
        ``launch(mm.runtime_artifact(), ...)`` entry stays unchanged for
        callers who want full telemetry."""

        from tessera.runtime import _execute_apple_cpu_accelerate_metadata

        launch_args: Any
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

        launch_args: Any
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
        """Return the emitted Graph IR as MLIR text.

        .. note:: ``fn.explain().ir.graph`` returns the same string.
           ``ir_text()`` is kept as a stable lower-level entry point;
           new code should prefer ``.explain()`` for the unified
           view across all four IR layers.
        """
        return self.graph_ir.to_mlir()

    def compile_report(self):
        """Synthesize a :class:`CompileReport` from this JitFn's
        current state.

        Step 4 of the 2026-05-18 post-reassessment plan: every JIT
        frontend exposes a uniform CompileReport accessor.  No
        execution happens here — the accessor reads ``graph_ir``,
        ``target``, ``cpu_plan``, and the recent bridge trace.

        .. note:: ``fn.explain()`` is the front door for developers —
           it consumes ``compile_report`` under the hood and adds
           per-op kernel resolution, IR layers, and next-action
           hints.  Keep using ``compile_report()`` directly for
           benchmark/JSON serialization paths.
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
        return target_kind.startswith("nvidia") or target_kind in {"rocm", "apple_gpu"}

    @property
    def arg_names(self) -> List[str]:
        # The Python call ABI is stable even when the post-specialization
        # tracer canonicalizes parameter symbols to a0/a1/... . Physical
        # package builders receive tracer source names separately.
        return list(self._call_arg_names)

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

    def _uses_rocm_compiled_default(self) -> bool:
        """Stage L4 — True iff this is a rocm single-matmul/gemm AND the
        compiler-generated lane can run on THIS host (tessera-opt + a usable AMD
        GPU). Host-gated, so off-device it is False and nothing changes. Shared by
        ``execution_kind`` and the ``runtime_artifact`` stamping so the two never
        diverge (``is_executable`` agrees with what ``launch()`` actually does)."""
        return (
            self.cpu_plan is not None
            and str(self.cpu_plan.target_kind).startswith("rocm")
            and len(self.cpu_plan.ops) == 1
            and self.cpu_plan.ops[0].op_name in {"tessera.matmul", "tessera.gemm"}
            and _rocm_compiled_lane_available()
        )

    def _uses_rocm_sparse_attn_default(self) -> bool:
        """True iff this is a rocm single ``msa_sparse_attention`` AND the
        compiler-generated block-sparse lane can run on THIS host (tessera-opt +
        a usable AMD GPU). Host-gated exactly like the matmul lane above, so
        off-device it is False and the artifact stays ``artifact_only``. Makes
        ``@jit(target="rocm")`` MSA execute through ``rocm_sparse_attn_compiled``
        (block-sparse WMMA + GPU top-k), matching the Target-IR ``status`` the
        ROCm ``msa_block_sparse`` op reports."""
        return (
            self.cpu_plan is not None
            and str(self.cpu_plan.target_kind).startswith("rocm")
            and len(self.cpu_plan.ops) == 1
            and self.cpu_plan.ops[0].op_name == "tessera.msa_sparse_attention"
            and _rocm_compiled_lane_available()
        )

    def _uses_rocm_grouped_gemm_default(self) -> bool:
        """True for the one-launch grouped-offset ROCm GEMM on a live device."""
        return (
            self.cpu_plan is not None
            and str(self.cpu_plan.target_kind).startswith("rocm")
            and len(self.cpu_plan.ops) == 1
            and self.cpu_plan.ops[0].op_name == "tessera.grouped_gemm"
            and _rocm_compiled_lane_available()
        )

    def _uses_nvidia_mma_default(self) -> bool:
        """sm_120 bring-up — True iff this is an ``nvidia_sm120`` single
        matmul/gemm AND the shipped mma.sync lane can run on THIS host
        (libtessera_nvidia_gemm.so + a usable NVIDIA GPU). Host-gated, so
        off-device it is False and the artifact stays artifact_only. Shared by
        ``execution_kind`` and the ``runtime_artifact`` stamping so the two never
        diverge."""
        return (
            self.cpu_plan is not None
            and str(self.cpu_plan.target_kind) == "nvidia_sm120"
            and len(self.cpu_plan.ops) == 1
            and self.cpu_plan.ops[0].op_name in {"tessera.matmul", "tessera.gemm"}
            and _nvidia_mma_lane_available()
        )

    @property
    def execution_kind(self) -> str:
        if self._uses_rocm_compiled_default():
            return "native_gpu"
        if self._uses_rocm_sparse_attn_default():
            return "native_gpu"
        if self._uses_rocm_grouped_gemm_default():
            return "native_gpu"
        if self._uses_nvidia_mma_default():
            return "native_gpu"
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
        """Return Graph/Schedule/Tile/Target artifacts for the compiled path.

        .. note:: ``fn.explain().ir`` exposes the same four layers as
           strings on a typed namespace (``.graph``/``.schedule``/
           ``.tile``/``.target``).  Use ``lowering_artifacts()`` when
           you need the raw artifact objects (e.g., for hash
           verification); use ``.explain()`` for the human view.
        """

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

        .. note:: ``fn.explain()`` surfaces the artifact's key fields
           (execution kind, target, IR hashes) in a human-readable
           summary.  Continue calling ``runtime_artifact()`` directly
           when you need the raw artifact for ABI/runtime dispatch
           or for ``Apple CPU`` fast-path metadata.
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
                    _op_payload(op)
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
                _op_payload(op)
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
                _op_payload(op)
                for op in self.cpu_plan.ops
            ]
            # The strict f32/rank-2 descriptor schema is the matmul/gemm contract
            # (Phase 8.3) downstream tooling relies on. Single ops that are not a
            # 2-D matmul (e.g. rank-4 conv2d) carry name-only descriptors +
            # relaxed guards, mirroring the apple_cpu multi-op branch — never a
            # dishonest rank-2 descriptor for a rank-4 operand.
            single_matmul = (
                len(self.cpu_plan.ops) == 1
                and self.cpu_plan.ops[0].op_name in {"tessera.matmul", "tessera.gemm"}
            )
            if single_matmul:
                gpu_input_descriptors: list[dict[str, Any]] = [
                    {"name": name, "dtype": "f32", "rank": 2}
                    for name in self.arg_names
                ]
                gpu_output_descriptor: dict[str, Any] = {
                    "name": self.cpu_plan.output_name, "dtype": "f32", "rank": 2}
                gpu_guards: dict[str, Any] = {
                    "dtype": "float32", "rank": 2,
                    "static_shape_at_launch": True, "op_count": 1}
            else:
                gpu_input_descriptors = [{"name": name} for name in self.arg_names]
                gpu_output_descriptor = {"name": self.cpu_plan.output_name}
                gpu_guards = {"op_count": len(self.cpu_plan.ops),
                              "static_shape_at_launch": True}
            metadata.update({
                "executable": True,
                "compiler_path": "apple_gpu_mps",
                "execution_kind": "native_gpu",
                "runtime_status": "ready",
                "execution_mode": "metal_runtime",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "input_descriptors": gpu_input_descriptors,
                "output_descriptor": gpu_output_descriptor,
                "cpu_tile": list(self.cpu_plan.tile),
                "ops": ops_payload,
                "mps_ops": [op["op_name"] for op in ops_payload],
                "guards": gpu_guards,
            })
        elif self._uses_rocm_compiled_default():
            # Stage L4 — the compiler-GENERATED RDNA WMMA GEMM is the default rocm
            # matmul execution lane on a capable host (tessera-opt built + a usable
            # AMD GPU). The hand-written kernel is the reference oracle + the
            # availability fallback (runtime._execute_rocm_compiled_gemm degrades
            # to it). Off-device this branch is skipped and the artifact stays
            # artifact_only (the generic branch below) — no behavior change.
            assert self.cpu_plan is not None  # narrowed by the guard above
            ops_payload = [
                _op_payload(op)
                for op in self.cpu_plan.ops
            ]
            metadata.update({
                "executable": True,
                "compiler_path": "rocm_compiled",
                "execution_kind": "native_gpu",
                "runtime_status": "ready",
                "execution_mode": "hip_runtime",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "ops": ops_payload,
                "cpu_tile": list(self.cpu_plan.tile),
                "rocm_fallback_lane": "rocm_wmma",
            })
        elif self._uses_rocm_sparse_attn_default():
            # @jit(target="rocm") msa_sparse_attention → the compiler-generated
            # block-sparse WMMA + GPU-top-k lane (rocm_sparse_attn_compiled),
            # reached through runtime.launch(). Off-device this branch is skipped
            # and the artifact stays artifact_only — no behavior change.
            assert self.cpu_plan is not None  # narrowed by the guard above
            ops_payload = [
                _op_payload(op)
                for op in self.cpu_plan.ops
            ]
            metadata.update({
                "executable": True,
                "compiler_path": "rocm_sparse_attn_compiled",
                "execution_kind": "native_gpu",
                "runtime_status": "ready",
                "execution_mode": "hip_runtime",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "ops": ops_payload,
                "cpu_tile": list(self.cpu_plan.tile),
            })
        elif self._uses_rocm_grouped_gemm_default():
            assert self.cpu_plan is not None
            ops_payload = [
                _op_payload(op)
                for op in self.cpu_plan.ops
            ]
            metadata.update({
                "executable": True,
                "compiler_path": "rocm_moe_transport_compiled",
                "execution_kind": "native_gpu",
                "runtime_status": "ready",
                "execution_mode": "hip_runtime",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "ops": ops_payload,
                "cpu_tile": list(self.cpu_plan.tile),
                "grouped_argument_layout": "device_offsets[E+1]",
            })
        elif self._uses_nvidia_mma_default():
            # sm_120 bring-up — @jit(target="nvidia_sm120") matmul dispatches to
            # the shipped warp-level mma.sync GEMM (libtessera_nvidia_gemm.so) on
            # a capable host. Off-device this branch is skipped and the artifact
            # stays artifact_only (the generic branch below) — no behavior change.
            assert self.cpu_plan is not None  # narrowed by the guard above
            ops_payload = [
                _op_payload(op)
                for op in self.cpu_plan.ops
            ]
            metadata.update({
                "executable": True,
                "compiler_path": "nvidia_mma",
                "execution_kind": "native_gpu",
                "runtime_status": "ready",
                "execution_mode": "cuda_runtime",
                "arg_names": list(self.arg_names),
                "output_name": self.cpu_plan.output_name,
                "ops": ops_payload,
                "cpu_tile": list(self.cpu_plan.tile),
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
            # rung-2.5 (EVALUATOR_PLAN.md): for an sm_90 NVIDIA matmul, attach the
            # emitted WGMMA PTX assembler text + its structural-validation status,
            # so the emission is first-class metadata (not just Target IR MLIR) and
            # the Evaluator can report EMITS_ASM_TEXT. Skeleton only — assembly is
            # the rung-3 CI gate. Other ops/targets simply omit it (stay rung 1).
            _tgt = str(metadata.get("target", ""))
            if _tgt.startswith("nvidia"):
                from .matmul_pipeline import emit_nvidia_ptx
                _emitted = emit_nvidia_ptx(self.cpu_plan.ops, target_kind=_tgt)
                if _emitted is not None:
                    _ptx, _valid = _emitted
                    metadata["nvidia_ptx"] = _ptx
                    metadata["nvidia_ptx_valid"] = _valid

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

        # Surface the component-aware canonical compile metadata (Sprint A —
        # fusion_groups / shape_envelope / effects / layout_contracts +
        # component_ops) on the user-facing artifact. Merged via
        # ``descriptive_metadata()`` so the executability decision above (owned
        # by the cpu/apple fast paths) is never overridden — additive only.
        if self.compile_result is not None:
            for key, value in self.compile_result.descriptive_metadata().items():
                metadata.setdefault(key, value)

        return RuntimeArtifact(
            graph_ir=graph_ir_text,
            schedule_ir=schedule_ir_text,
            tile_ir=tile_ir_text,
            target_ir=target_ir_text,
            metadata=metadata,
            abi_signature=f"tessera.runtime.v1.{metadata['target']}",
        )

    def explain_lowering(self) -> str:
        """Return a human-readable explanation of compile vs fallback status.

        .. deprecated:: 2026-05-19
           Prefer ``fn.explain()`` — the single front door that unifies
           lowering diagnostics, fallback reasons, IR layers, and
           next-action hints.  This method stays as a data source for
           callers that need only the diagnostic list as text.
        """

        return "\n".join(d.format() for d in self.lowering_diagnostics)

    def explain(self) -> "Explain":
        """Return a single opinionated diagnostic for this JIT function.

        ``print(fn.explain())`` answers four questions in a 5-line
        summary:

          1. What ran?  (``execution_kind``)
          2. Was it native / reference / artifact / fallback?
          3. Why?  (fallback reason, lowering diagnostics)
          4. What should I do next?  (hints with stable IDs)

        Structured fields hang off the :class:`~tessera.compiler.explain.Explain`
        object: ``.ir``, ``.kernels``, ``.diagnostics``,
        ``.next_actions``.  Each is read-only and JSON-serializable
        via ``.as_dict()``.

        This is the front door — the legacy inspection methods
        (``ir_text``, ``schedule_ir``, ``tile_ir``, ``target_ir``,
        ``lowering_artifacts``, ``runtime_artifact``,
        ``compile_report``, ``explain_lowering``) stay as
        underlying data sources but new code should call
        ``.explain()``.
        """

        from . import explain as explain_mod
        return explain_mod.build_explain(self)

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

# Encode-eligible op base names for the apple_gpu one-command-buffer route.
# Mirrors the keys of ``apple_gpu_chain.ENCODE_OP_REGISTRY`` but kept as a plain
# literal so decoration-time auto-detection needs no GPU / runtime import. Drift
# vs the registry is gated by
# ``tests/unit/test_apple_gpu_jit_auto_batch_autodetect.py``.
_APPLE_GPU_ENCODE_OP_NAMES = frozenset({
    "bmm", "layer_norm", "rmsnorm", "softmax", "rope",
    "silu", "gelu", "flash_attn", "conv2d",
})


class _AutoBatchSkipEmission(Exception):
    """Internal control-flow sentinel — raised at the top of the Step 6 try
    block to skip Graph IR emission for the auto_batch route (caught by a
    dedicated handler that installs the deferred state)."""


# Expression / statement AST node types allowed inside a recognized decode
# chain body. The body must be *only* a sequence of op-call assignments and a
# return — no arithmetic (BinOp), subscripts, comparisons, control flow,
# tuples, etc. Anything outside this set means the op results flow into
# non-op computation the one-command-buffer route can't reproduce, so the
# body is conservatively NOT auto-batched.
_DECODE_CHAIN_ALLOWED_NODES = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.arguments, ast.arg,
    ast.Assign, ast.Return, ast.Expr, ast.Pass,
    ast.Call, ast.Attribute, ast.Name, ast.Constant, ast.keyword,
    ast.Load, ast.Store,
)


def _recognized_decode_chain(source_text: Optional[str]) -> bool:
    """True when a function body is a pure chain of ≥2 encode-eligible
    apple_gpu ops and *nothing else* — the exact shape the one-command-buffer
    route batches.

    This is the auto-detection signal for ``@jit(target="apple_gpu")`` with
    the default ``auto_batch=None``: a recognized decode chain runs on one
    command buffer per encode segment (strictly fewer commits, numerically
    identical — see ``test_apple_gpu_jit_auto_batch_canonical``).

    Deliberately conservative. Two gates, both must hold:

    1. Every ``Call`` resolves to an encode-eligible op name — a non-encode
       call (``range``, ``np.exp``, ``tessera.control.*``, a helper) is out.
    2. The body contains *only* whitelisted nodes (op-call assignments + a
       return) — so arithmetic on an op result (``silu(x) * 2``), subscripts,
       comparisons, control flow, or tuple returns all disqualify it, because
       the tracer hands back a ``TraceRef`` the surrounding computation
       couldn't consume the way eager execution would.

    Explicit ``auto_batch=True``/``False`` always overrides detection."""
    if not source_text:
        return False
    try:
        tree = ast.parse(textwrap.dedent(source_text))
    except SyntaxError:
        return False
    funcs = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(funcs) != 1:
        return False
    op_calls = 0
    for node in ast.walk(funcs[0]):
        if not isinstance(node, _DECODE_CHAIN_ALLOWED_NODES):
            return False  # non-chain construct (arithmetic, control flow, ...)
        if isinstance(node, ast.Call):
            func = node.func
            base = (func.attr if isinstance(func, ast.Attribute)
                    else func.id if isinstance(func, ast.Name)
                    else None)
            if base not in _APPLE_GPU_ENCODE_OP_NAMES:
                return False  # a non-encode call disqualifies the chain
            op_calls += 1
    return op_calls >= 2


def _resolve_auto_batch(
    auto_batch: "bool | None",
    target_kind: Optional[str],
    source_text: Optional[str],
) -> bool:
    """Resolve the effective one-command-buffer route flag.

    * ``True`` / ``False`` — explicit; honored verbatim (the non-apple guard
      below still fires for an explicit ``True`` on the wrong target).
    * ``None`` (the default) — auto-detect: on for an apple_gpu body that is a
      recognized decode chain, off otherwise."""
    if auto_batch is not None:
        return bool(auto_batch)
    return target_kind == "apple_gpu" and _recognized_decode_chain(source_text)


# ── @jit decoration stages (extracted from the `jit()` closure, audit
#    2026-06-10 §4) ──────────────────────────────────────────────────────────
# These were inline numbered steps inside the ~325-line `_decorate` closure.
# Hoisted to module-level helpers with explicit inputs/outputs so the closure
# reads as an orchestration of named stages. Behavior is unchanged — this is a
# faithful relocation, gated by the full @jit test surface.


@dataclass
class _FrontendAnalysis:
    """Result of @jit Steps 1-4: constraint solving + effect inference."""
    solver: "ConstraintSolver"
    inferred_effect: Any


@dataclass
class _GraphIREmission:
    """Result of @jit Step 6: AST → Graph IR emission + compile bundle.

    On the auto_batch-skip and apple_gpu trace-defer paths the module is empty
    and ``cpu_plan``/``compile_bundle``/``compile_result`` are None; the
    ``trace_deferred`` flag distinguishes the emission-*failure* defer (which
    forces the surgical tracer) from the auto_batch skip (which does not)."""
    module: Any
    cpu_plan: Any
    compile_bundle: Any
    compile_result: Any
    diagnostics: List["JitDiagnostic"]
    trace_deferred: bool


def _jit_analyze_frontend(
    fn: Callable,
    *,
    source_text: Optional[str],
    bindings: Optional[Dict[str, int]],
    deterministic: bool,
    seed: Optional[int],
) -> _FrontendAnalysis:
    """@jit Steps 1-4: collect structural constraints + check them against any
    known bindings, infer the effect, and validate the deterministic contract.
    Raises TesseraConstraintError / TesseraEffectError on violation (unchanged
    from the inline steps)."""
    # Step 1: collect constraints from the function body.
    solver = ConstraintSolver()
    for c in _extract_constraints(fn, source_text=source_text):
        solver.add(c)

    # Step 2: check constraints against any known bindings.
    solver.check(bindings or {})

    # Step 3: infer effects.
    lattice = EffectLattice()
    inferred_effect = lattice.infer(fn, source_text=source_text)

    # Step 4: reject a proven unseeded random dependency now. Unknown source
    # aliases are not guessed here; the concrete traced-IR certificate in
    # JitFn.__call__ owns that decision once operand shapes are available.
    if deterministic and inferred_effect == Effect.random and seed is None:
        from .effects import TesseraEffectError
        raise TesseraEffectError(
            fn.__name__, Effect.pure, inferred_effect,
            message=(f"@jit(deterministic=True) function {fn.__name__!r} "
                     "contains a registered random Graph operation without "
                     "a seed"),
        )

    return _FrontendAnalysis(solver=solver, inferred_effect=inferred_effect)


def _frontend_jit_diagnostics(builder: GraphIRBuilder) -> List[JitDiagnostic]:
    """Lift a builder's AST-lowering diagnostics into JitDiagnostics."""
    return [
        JitDiagnostic(d.severity, d.code, d.format())
        for d in builder.diagnostics
    ]


def _recover_frontend_diagnostics(
    fn: Callable, *, source_text: Optional[str],
    effect_tag: Optional[str], target_attr: Optional[str],
    prefer_abstract_trace: bool,
    source_origin: str = "inspect",
) -> List[JitDiagnostic]:
    """Re-derive the AST front end's diagnostics on the emission-failure path.

    The Graph IR cache stores the module but not the diagnostics that were
    produced alongside it, so on a cache HIT the frontend diagnostics naming
    the unlowerable construct are simply absent. Deriving them here -- a cold
    path reached only once emission has already failed -- makes the deferral
    message identical on the first decoration and the thousandth, instead of
    silently better the first time.

    Returns an empty list if the re-lowering itself fails: this exists to
    enrich an error that is already being reported, and must never replace it.
    """
    try:
        builder = GraphIRBuilder()
        builder.lower(
            fn, effect_tag=effect_tag, target_attr=target_attr,
            source_text=source_text, source_origin=source_origin,
            prefer_abstract_trace=prefer_abstract_trace,
        )
        return _frontend_jit_diagnostics(builder)
    except Exception:
        return []


def _jit_emit_graph_ir(
    fn: Callable,
    *,
    source_text: Optional[str],
    source_origin: str,
    target: Optional[Any],
    target_kind: str,
    deterministic: bool,
    seed: Optional[int],
    cpu_tile: Tuple[int, int, int],
    inferred_effect: Any,
    constraints: Sequence[Constraint],
    skip_graph_ir: bool,
) -> _GraphIREmission:
    """@jit Step 6: emit Graph IR (with the process-local cache), build the
    compile bundle + canonical result, and handle the two non-emitting paths
    (auto_batch skip, apple_gpu emission-failure trace-defer). A faithful
    relocation of the inline try/except — same control flow and diagnostics."""
    trace_deferred = False
    # Frontend (AST -> Graph IR) diagnostics live OUTSIDE the try because the
    # trace-defer handler below needs them: they are the only record of which
    # construct the AST front end could not lower, and the verifier error that
    # actually raises names only the dangling operand it left behind.
    frontend_diagnostics: list[JitDiagnostic] = []
    # Hoisted out of the try so the trace-defer handler can re-lower with the
    # same inputs rather than recomputing them. Safe to move: both are total
    # over an already-validated ``target`` (``normalize_target_kind`` ran and
    # raised at the call site), and a ``GPUTargetProfile`` always normalizes to
    # ``nvidia_*`` -- never to the ``apple_gpu`` kind that reaches the handler.
    effect_tag = (
        inferred_effect.name
        if deterministic or inferred_effect != Effect.pure
        else None
    )
    # Attach GPU target attrs to the module when target is provided.
    if isinstance(target, GPUTargetProfile):
        target_attr = target.to_mlir_attr()
    elif target is not None:
        target_attr = f'{{name = "{target_kind}"}}'
    else:
        target_attr = None
    try:
        if skip_graph_ir:
            # The auto_batch tracer runs the body directly — the AST Graph IR
            # it would emit here is never consulted, so don't pay to build it.
            raise _AutoBatchSkipEmission
        # G4 memoization (2026-05-19) — process-local cache keyed on
        # source_text + effect_tag + target_attr.
        from . import graph_ir_cache as _gic
        # Identical source at different call sites must not reuse stale locs.
        from .graph_ir import _loc_path
        source_location = (
            f"{source_origin}:{_loc_path(fn.__code__.co_filename)}:{fn.__code__.co_firstlineno}"
        )
        module = _gic.lookup(
            source_text, effect_tag=effect_tag, target_attr=target_attr,
            source_location=source_location)
        if module is None:
            builder = GraphIRBuilder()
            builder.lower(
                fn, effect_tag=effect_tag,
                target_attr=target_attr, source_text=source_text, source_origin=source_origin,
                prefer_abstract_trace=inferred_effect == Effect.pure,
            )
            module = builder.module()
            frontend_diagnostics.extend(
                _frontend_jit_diagnostics(builder))
            _gic.store(
                source_text, module,
                effect_tag=effect_tag, target_attr=target_attr,
                source_location=source_location,
            )
        diagnostics: list[JitDiagnostic] = list(frontend_diagnostics)
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
        from .presburger import (
            attach_presburger_system,
            presburger_system_from_constraints,
        )
        presburger = presburger_system_from_constraints(constraints)
        if presburger is not None:
            for graph_function in module.functions:
                attach_presburger_system(graph_function, presburger)
        compile_bundle = compile_graph_module(
            module,
            source_origin=source_origin,
            target=target_kind,
            cpu_tile=(int(cpu_tile[0]), int(cpu_tile[1]), int(cpu_tile[2])),
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
        compile_result = compile_result_from_bundle(compile_bundle, module=module)
        return _GraphIREmission(
            module=module, cpu_plan=cpu_plan, compile_bundle=compile_bundle,
            compile_result=compile_result, diagnostics=diagnostics,
            trace_deferred=False)
    except _AutoBatchSkipEmission:
        # auto_batch route is on; emission was skipped on purpose. Empty module
        # + no plan/bundle makes __call__ fall through to ``self._fn`` (the
        # auto_batch wrapper). trace_deferred stays False (the wrapper, not the
        # surgical tracer, is the execution path).
        return _GraphIREmission(
            module=GraphIRModule(), cpu_plan=None, compile_bundle=None,
            compile_result=None, trace_deferred=False,
            diagnostics=[JitDiagnostic(
                "info", "JIT_APPLE_GPU_AUTO_BATCH",
                "auto_batch one-command-buffer route active; skipped unused "
                "Graph IR emission (the tracer runs the body directly)")])
    except Exception as exc:
        # An AST Graph-IR emission failure does NOT hard-fail apple_gpu
        # decoration: the tracer runs the function (never reads the AST
        # graph_ir), so a body the AST can't emit still decorates and runs via
        # the tracer at call time. Other targets depend on the IR → re-raise.
        if target_kind != "apple_gpu":
            raise TesseraJitError(
                f"Graph IR emission failed for {fn.__name__!r}: {exc}"
            ) from exc
        # Carry the frontend diagnostics through the defer. They name the
        # construct the AST front end dropped; the verifier error that
        # actually raised names only the dangling operand that drop left
        # behind, which is the symptom, not the cause. `JitFn` quotes these
        # back if the tracer then fails too (JIT_APPLE_GPU_TRACE_FAILED).
        if not frontend_diagnostics:
            frontend_diagnostics = _recover_frontend_diagnostics(
                fn, source_text=source_text, source_origin=source_origin, effect_tag=effect_tag,
                target_attr=target_attr,
                prefer_abstract_trace=inferred_effect == Effect.pure,
            )
        return _GraphIREmission(
            module=GraphIRModule(), cpu_plan=None, compile_bundle=None,
            compile_result=None, trace_deferred=True,
            diagnostics=frontend_diagnostics + [JitDiagnostic(
                "warning", "JIT_APPLE_GPU_TRACE_DEFERRED",
                f"AST Graph IR emission failed ({exc}); deferring to the "
                "Phase-F tracer at call time")])


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
    autodiff: Optional[str] = None,
    wrt: Optional[Sequence[str]] = None,
    auto_batch: "bool | None" = None,
    max_ops_per_cb: Optional[int] = None,
    emit_package: "bool | str" = False,
    dispatch_via_package: "bool | str | None" = None,
    phase: Optional[str] = None,
    slo: Optional[Any] = None,
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
                       "rocm", "apple_cpu", and "apple_gpu".
                       None = executable CPU/NumPy path.
        attn_config  : FlashAttnLoweringConfig; when None and target is set with
                       isa >= SM_90, SM90_DEFAULT is used automatically.
        cpu_tile     : CPU matmul/GEMM schedule tile `(M, N, K)` for the narrow
                       end-to-end CPU compiler path.
        source       : optional function source text for functions created from
                       stdin/exec where inspect.getsource() cannot recover the
                       function body.
        source_path  : optional path to Python source text for AST lowering.
        auto_batch   : apple_gpu one-command-buffer route. ``None`` (default)
                       auto-detects — a recognized decode chain (a body of ≥2
                       encode-eligible ops and nothing else) runs on one
                       command buffer per encode segment, and its unused Graph
                       IR emission is skipped. ``True`` forces the route on,
                       ``False`` forces it off.
        max_ops_per_cb: chunking budget for the auto_batch route — caps
                       encode-eligible ops per command buffer.

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

        # ── Steps 1-4: frontend analysis (constraints + effects) ────────────
        analysis = _jit_analyze_frontend(
            fn,
            source_text=source_text,
            bindings=bindings,
            deterministic=deterministic,
            seed=seed,
        )
        solver = analysis.solver
        inferred_effect = analysis.inferred_effect

        # ── Step 5: resolve attn config for GPU path ────────────────────────
        resolved_attn = attn_config
        target_kind = normalize_target_kind(target)
        if isinstance(target, GPUTargetProfile) and target.supports_wgmma and resolved_attn is None:
            resolved_attn = SM90_DEFAULT

        # P3 (2026-06-09) — resolve the effective one-command-buffer route.
        # ``auto_batch=None`` (default) auto-detects a recognized decode chain;
        # the auto_batch path traces+runs the body and never reads the emitted
        # Graph IR, so when the route is on we skip Graph IR emission entirely
        # (unless ``emit_package`` needs the recognized region). Explicit
        # ``True``/``False`` always override detection.
        _auto_batch = _resolve_auto_batch(auto_batch, target_kind, source_text)
        _skip_graph_ir = (
            _auto_batch and target_kind == "apple_gpu" and not emit_package)

        # ── Step 6: emit Graph IR (incl. auto_batch-skip + trace-defer) ─────
        emission = _jit_emit_graph_ir(
            fn,
            source_text=source_text,
            source_origin=source_origin,
            target=target,
            target_kind=target_kind,
            deterministic=deterministic,
            seed=seed,
            cpu_tile=cpu_tile,
            inferred_effect=inferred_effect,
            constraints=solver.constraints,
            skip_graph_ir=_skip_graph_ir,
        )
        module = emission.module
        cpu_plan = emission.cpu_plan
        compile_bundle = emission.compile_bundle
        compile_result = emission.compile_result
        diagnostics = emission.diagnostics
        _trace_deferred = emission.trace_deferred

        # PK8a wiring (2026-06-02) — recognize whether this module's compute
        # region is an authorable Apple-GPU packaged kernel (matmul / a single
        # MPSGraph-lane op / a fused chain). Pure + device-free: it keys off
        # the op-name sequence only, so it fires even though the live Graph IR
        # carries no static shapes. The shape-free RecognizedOp is paired with
        # concrete example-arg shapes at ``JitFn.emit_package`` time to author
        # the actual `.mtlpackage`. Recognition is recorded on the artifact
        # regardless of host (no GPU touched here).
        recognized_package = None
        if target_kind == "apple_gpu":
            try:
                from .apple_package_author import recognize_op
                recognized_package = recognize_op(module)
            except Exception:
                recognized_package = None

        # ── Step 6b: differentiation request (Phase 1 autodiff unification) ──
        # Validate @jit(autodiff=..., wrt=...) at decoration time, emit the
        # `tessera.autodiff` intent into the Graph IR module (the attribute the
        # C++ autodiff passes key on), and resolve the mode-neutral provenance
        # facet. Python owns validation + diagnostics;
        # the C++ reverse/JVP pass owns the actual Graph transformation.
        from . import autodiff_request as _ad

        diff_request = _ad.build_request(
            fn, autodiff=autodiff, wrt=wrt, native_required=native_required)
        # Phase 4 (A3): source the native-backward hook from the execution
        # matrix via the program's component ops — the backward is native only if
        # every differentiable op has a device-proven backward on this target.
        _bwd_families = (
            tuple(compile_result.component_ops)
            if compile_result is not None else ()
        )
        differentiation_prov = _ad.resolve_differentiation_provenance(
            diff_request, target=target_kind, op_families=_bwd_families)
        backward_prov = (
            differentiation_prov
            if diff_request is not None and diff_request.mode == "reverse"
            else _ad.NOT_REQUESTED
        )
        if diff_request is not None and module is not None:
            intent = diff_request.module_intent_attrs()
            # The C++ --tessera-autodiff pass reads the `tessera.autodiff` marker
            # off the `func.func` (AutodiffPass.cpp: func->getAttrOfType), so the
            # actionable home is the function's fn_attrs. Attach it to the
            # differentiated function (matched by name; all functions if the
            # single-function module has no name match). A module-level marker is
            # also set as a program-level breadcrumb.
            module.module_attrs.update(intent)
            fns = getattr(module, "functions", []) or []
            named = [gf for gf in fns if getattr(gf, "name", None) == fn.__name__]
            for gf in (named or fns):
                gf.fn_attrs.update(intent)
        if diff_request is not None and compile_result is not None:
            import dataclasses as _dc
            compile_result = _dc.replace(
                compile_result,
                differentiation=differentiation_prov,
                backward=(backward_prov if diff_request.mode == "reverse" else None),
            )

        # ── Step 7: wrap and return ──────────────────────────────────────────
        jitfn = JitFn(
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
            compile_result=compile_result,
            cpu_tile=(int(cpu_tile[0]), int(cpu_tile[1]), int(cpu_tile[2])),
            source_origin=source_origin,
            lowering_diagnostics=diagnostics,
            native_required=native_required,
            recognized_package=recognized_package,
            dispatch_via_package=_resolve_dispatch_via_package(
                dispatch_via_package, target_kind),
            differentiation_request=diff_request,
            differentiation_provenance=differentiation_prov,
            backward_provenance=backward_prov,
            source_text=source_text,
        )
        if _trace_deferred:
            # AST emission failed → the tracer is the only execution path. Force
            # the surgical gate on (the empty graph_ir carries no scf markers).
            jitfn._needs_trace = True

        # P1 canonical one-command-buffer route (2026-06-01) — the
        # `auto_batch=True` opt-in wraps the user fn with
        # `apple_gpu_ops.auto_batch`, so every op call inside the body
        # is trace-captured and executed as one cb per encode segment.
        #
        # Both op surfaces route through it: `apple_gpu_ops.*` directly,
        # and `tessera.ops.*` via the interception shim installed
        # globally at import (tessera/__init__.py → apple_gpu_ops_
        # interception.install_apple_gpu_interception). The shim's
        # wrappers check the active trace and forward to apple_gpu_ops
        # when one is live, so a user writing the canonical
        # `tessera.ops.rmsnorm(...)` / `silu(...)` inside a
        # `@jit(target="apple_gpu", auto_batch=True)` decode loop runs
        # the whole chain on one command buffer (Phase 2.1c — landed;
        # no longer the open per-op-adapter problem the old note feared).
        #
        # `max_ops_per_cb` is the chunking budget (Glass-jaw #7): it
        # caps encode-eligible ops per command buffer so a very deep
        # decode chain splits into K cbs transparently instead of
        # hitting the MPSGraph shape × op-count cliff. None = the
        # substrate default (DEFAULT_OPS_PER_CB).
        if max_ops_per_cb is not None and not _auto_batch:
            raise TesseraJitError(
                "@jit(max_ops_per_cb=...) is only meaningful with the "
                "auto_batch one-command-buffer route; got an effective "
                f"auto_batch=False for {getattr(fn, '__name__', '<fn>')!r} "
                f"(auto_batch={auto_batch!r}, target={target_kind!r}).")
        if _auto_batch:
            if target_kind != "apple_gpu":
                raise TesseraJitError(
                    f"@jit(auto_batch=True) currently only supports "
                    f"target='apple_gpu'; got target={target!r} "
                    f"(normalized={target_kind!r}).")
            from .. import apple_gpu_ops as _agpu
            jitfn._fn = _agpu.auto_batch(
                jitfn._fn, max_ops_per_cb=max_ops_per_cb)

        # PK8d (2026-06-02) — compile-time auto-emit. When the caller opts in
        # with ``emit_package=True`` (or a path) AND the region is recognized
        # AND the arg annotations are static integers, author the
        # `.mtlpackage` now — no manual ``emit_package(example_args=...)``
        # call. Misuse guard mirrors ``max_ops_per_cb``. Failure to author
        # (symbolic shapes / runtime unavailable) is silent: the attribute is
        # simply None — auto-emit is a best-effort AOT convenience, never a
        # hard compile error.
        if emit_package:
            if target_kind != "apple_gpu":
                raise TesseraJitError(
                    "@jit(emit_package=...) is only meaningful with "
                    f"target='apple_gpu'; got target={target!r} "
                    f"(normalized={target_kind!r}).")
            out = emit_package if isinstance(emit_package, str) else None
            try:
                jitfn.emit_package(out)
            except Exception:
                pass

        # PK8e — ``dispatch_via_package=True`` routes execution through the
        # authored package (per-shape cache). apple_gpu-only, like the flags
        # above.
        if dispatch_via_package and target_kind != "apple_gpu":
            raise TesseraJitError(
                "@jit(dispatch_via_package=True) is only meaningful with "
                f"target='apple_gpu'; got target={target!r} "
                f"(normalized={target_kind!r}).")

        # Workstream B — phase specialization metadata. Prefill and decode are
        # compiled from the same source but scheduled differently; the
        # PhaseSpecializationPass (compiler/phase_specialization.py) reads these
        # to pick a schedule policy and thread the CacheHandoff. Lightweight
        # passthrough: attached for inspection/consumption, no behavior change.
        if phase is not None:
            from .phase_specialization import Phase, SchedulePolicy
            jitfn.phase = Phase(phase)
            jitfn.slo = slo
            jitfn.schedule_policy = SchedulePolicy.for_phase(jitfn.phase, slo)
        else:
            jitfn.phase = None
            jitfn.slo = slo
            jitfn.schedule_policy = None

        return jitfn

    # Support both @jit and @jit(...) usage
    if fn is not None:
        return _decorate(fn)
    return _decorate
