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
    from .native_gpu_tensor import NativeTensorCall

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



def _rocm_chip() -> str:
    """The chip `target="rocm"` launches on (the runtime pin) — slice 2b."""
    from tessera import runtime as _rt

    return _rt._rocm_chip()

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

    _frontend_batch_policies: tuple[tuple[int | None, ...], ...]

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
        shape_bounds: Optional[Dict[str,int]] = None,
        bounded_source_certificate: Optional[Any] = None,
        bounded_rhs_storage_order: Optional[str] = None,
    ) -> None:
        self._fn = fn
        self._frontend_batch_axes: tuple[int | None, ...] | None = None
        self._frontend_batch_depth: int = 0
        self._frontend_output_permutation: tuple[int, ...] | None = None
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
        self._native_storage_call: Optional["NativeTensorCall"] = None
        self._apple_native_arena: Any = None
        self._native_storage_pair: Any = None
        self._autodiff_specializations: Dict[Any, GraphIRModule] = {}
        # E2E-REAL-6: concrete tensor signatures are tracer-owned by default.
        # The decoration-time AST module remains a named candidate until a
        # family differential certificate permits its deletion.
        self._traced_frontend_specializations: Dict[Any, GraphIRModule] = {}
        self._bounded_lhs = None
        if shape_bounds is not None:
            from .bounded_nvidia_lhs import BoundedLhsDispatcher,SourceCertificate,validate_bounds
            self._bounded_lhs = BoundedLhsDispatcher(self,validate_bounds(shape_bounds),
                bounded_source_certificate or SourceCertificate(fn,source_text),bounded_rhs_storage_order)
        self._nvidia_lhs_program_cache: Dict[str, Any] = {}
        self._nvidia_lhs_prepared_calls: Dict[str, Any] = {}
        self._nvidia_lhs_last_program: Any = None
        self._nvidia_lhs_last_receipts: tuple = ()
        self._nvidia_rhs_program_cache: Dict[str, Any] = {}
        self._rocm_nvfp4_program_cache: Dict[str, Any] = {}
        self._rocm_nvfp4_call_signature = inspect.signature(fn)
        self._rocm_nvfp4_last_program: Any = None
        self._rocm_nvfp4_last_receipts: Any = ()

        self._native_descriptor_specializations: Dict[str, CompileResult] = {}
        self._native_descriptor_artifacts: Dict[str, Any] = {}
        self._native_prepared_movement_calls: Dict[tuple, Any] = {}
        self._native_prepared_matmul_calls: Dict[tuple, Any] = {}
        self._native_descriptor_last_receipt: Any = None
        self._nvidia_rhs_call_signature = (
            inspect.signature(fn) if normalize_target_kind(target) == "nvidia_sm120" else None
        )
        self._nvidia_rhs_last_program: Any = None
        self._nvidia_rhs_last_receipts: tuple = ()
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
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any], *, normalize_batch: bool = True
    ) -> Optional[List[Any]]:
        """The positional input tensors in arg order (resolving kwargs by
        name). ``None`` if a declared arg is missing."""
        from .native_vmap import mixed_batch_policies, normalize_mixed_batch_inputs
        if not kwargs:
            values = list(args)
            return normalize_mixed_batch_inputs(values, self._frontend_batch_policies) if normalize_batch and mixed_batch_policies(self) else values
        names = list(self.arg_names)
        out: List[Any] = []
        for i, nm in enumerate(names):
            if i < len(args):
                out.append(args[i])
            elif nm in kwargs:
                out.append(kwargs[nm])
            else:
                return None
        return normalize_mixed_batch_inputs(out, self._frontend_batch_policies) if normalize_batch and mixed_batch_policies(self) else out

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

    def bind_native_storage_pair(self, package):
        """Bind a compiler-produced primal/JVP or primal/VJP physical ABI."""
        from .native_storage_pair import NativeStoragePair
        pair = NativeStoragePair(package)
        if tuple(pair.signature.parameters) != tuple(inspect.signature(self._fn).parameters):
            pair.close()
            raise ValueError("native pair physical signature disagrees")
        request = self.differentiation_request
        if request is not None and request.mode != pair.contract['mode']:
            pair.close()
            raise ValueError("native pair differentiation mode disagrees")
        self.close_native_storage()
        self._native_storage_call = None
        self._native_storage_jvp = None
        self._native_storage_candidate = None
        self._apple_native_arena = None
        self._native_storage_pair = pair
        self._cached_artifact = None
        return self

    def bind_apple_native_arena(self, package):
        """Bind an explicit compiler-owned Apple tensor ABI, without Graph re-entry."""
        from .apple_native_arena import AppleTensorCall
        if self.differentiation_request is not None:
            raise ValueError("Apple arena binding has no paired differentiation contract")
        binding = AppleTensorCall(package, inspect.signature(self._fn))
        self.close_native_storage()
        self._native_storage_call = None
        self._native_storage_jvp = None
        self._native_storage_candidate = None
        previous = getattr(self, "_apple_native_arena", None)
        if previous is not None:
            previous.close()
        self._native_storage_pair = None
        self._apple_native_arena = binding
        self._cached_artifact = None
        return self

    def bind_native_storage(self, package, specs=None, *, grid=None, block=None):
        """Explicitly bind a native kernel ABI; no Graph re-lowering or fallback.

        The caller supplies the tensor/scalar contract for this native program;
        binding is not an automatic equivalence proof against the Python body.
        """
        from .native_gpu_tensor import NativeTensorCall
        if specs is None:
            if grid is not None or block is not None:
                raise ValueError("generated tensor contracts do not permit geometry overrides")
            from .native_storage_contract import generate_tensor_binding
            binding = generate_tensor_binding(package, inspect.signature(self._fn))
        else:
            if grid is None or block is None:
                raise ValueError("explicit tensor contracts require launch geometry")
            binding = NativeTensorCall(package, inspect.signature(self._fn), tuple(specs),
                                       grid=tuple(grid), block=tuple(block))
        previous = getattr(self, "_native_storage_call", None)
        if previous is not None:
            previous.close()
        pair = getattr(self, "_native_storage_jvp", None)
        if pair is not None:
            pair.close()
        candidate = getattr(self, "_native_storage_candidate", None)
        if candidate is not None:
            candidate.close()
        self._native_storage_candidate = None
        self._native_storage_jvp = None
        apple = getattr(self, "_apple_native_arena", None)
        if apple is not None:
            apple.close()
        self._apple_native_arena = None
        old_pair = getattr(self, "_native_storage_pair", None)
        if old_pair is not None:
            old_pair.close()
        self._native_storage_pair = None
        self._native_storage_call = binding
        self._cached_artifact = None
        return self

    def enable_native_storage_arbiter(self, oracle):
        """Generate a Tier-2 candidate; execution still requires its F4 oracle."""
        from .emit.native_storage_candidate import register_native_storage_candidate
        binding = getattr(self, "_native_storage_call", None)
        if binding is None:
            raise ValueError("no native storage binding")
        candidate = register_native_storage_candidate(binding.package, inspect.signature(self._fn), oracle)
        previous = getattr(self, "_native_storage_candidate", None)
        if previous is not None:
            previous.close()
        self._native_storage_candidate = candidate
        return self

    def bind_native_storage_jvp(self, artifact):
        """Bind an existing compiler-produced paired program to native children."""
        from .native_storage_jvp import NativeStorageJVP
        if self.differentiation_request is not None and self.differentiation_request.mode != "forward":
            raise ValueError("paired native storage currently supports forward AD only")
        pair = NativeStorageJVP(artifact)
        if tuple(pair.signature.parameters) != tuple(inspect.signature(self._fn).parameters):
            pair.close()
            raise ValueError("paired storage signature differs from JIT signature")
        previous = getattr(self, "_native_storage_jvp", None)
        if previous is not None:
            previous.close()
        self.close_native_storage()
        self._native_storage_call = None
        self._native_storage_candidate = None
        self._apple_native_arena = None
        self._native_storage_pair = None
        self._native_storage_jvp = pair
        self._cached_artifact = None
        return self

    def submit_native_storage(self, stream: int, /, *args, **kwargs):
        """Submit the configured native program on a caller-owned stream."""
        binding = getattr(self, "_native_storage_call", None)
        if binding is None:
            raise ValueError("no native storage binding")
        self._native_descriptor_last_receipt = None
        self._enforce_call_time_constraints(args, kwargs)
        self._enforce_call_time_stochastic_certificate(args, kwargs)
        if self.differentiation_request is not None:
            raise ValueError("native storage binding has no paired differentiation contract")
        return binding.submit(stream, *args, **kwargs)

    def close_native_storage(self) -> None:
        """Release the native module; keep the descriptor for lazy rebinding."""
        for name in ("_native_prepared_movement_calls", "_native_prepared_matmul_calls",
                     "_nvidia_lhs_prepared_calls"):
            calls = getattr(self, name, {})
            for call in calls.values():
                call.close()
            calls.clear()

        native_pair = getattr(self, "_native_storage_pair", None)
        if native_pair is not None:
            native_pair.close()
        apple = getattr(self, "_apple_native_arena", None)
        if apple is not None:
            apple.close()
        binding = getattr(self, "_native_storage_call", None)
        if binding is not None:
            binding.close()
        pair = getattr(self, "_native_storage_jvp", None)
        if pair is not None:
            pair.close()
        candidate = getattr(self, "_native_storage_candidate", None)
        if candidate is not None:
            candidate.close()

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
        if self._nvidia_rhs_last_program is not None:
            self._cached_artifact = None
        self._native_descriptor_last_receipt = None
        self._nvidia_lhs_last_program = None
        self._nvidia_lhs_last_receipts = ()
        self._nvidia_rhs_last_program = None
        self._nvidia_rhs_last_receipts = ()
        self._rocm_nvfp4_last_program = None
        self._rocm_nvfp4_last_receipts = ()
        self._enforce_call_time_constraints(args, kwargs)
        self._enforce_call_time_stochastic_certificate(args, kwargs)
        if self._bounded_lhs is not None:
            return self._bounded_lhs(args,kwargs)
        native_pair = getattr(self, "_native_storage_pair", None)
        if native_pair is not None:
            return native_pair(*args, **kwargs)
        apple_storage = getattr(self, "_apple_native_arena", None)
        if apple_storage is not None:
            return apple_storage(*args, **kwargs)
        paired_storage = getattr(self, "_native_storage_jvp", None)
        if paired_storage is not None:
            return paired_storage(*args, **kwargs)
        native_storage = getattr(self, "_native_storage_call", None)
        if native_storage is not None:
            if self.differentiation_request is not None:
                raise ValueError("native storage binding has no paired differentiation contract")
            candidate = getattr(self, "_native_storage_candidate", None)
            if candidate is not None:
                from .emit.candidate import arbitrate
                arguments = native_storage.signature.bind(*args, **kwargs)
                arguments.apply_defaults()
                inputs = tuple(arguments.arguments.values())
                region = candidate.binding.package.binding_digest
                winner = arbitrate(region, candidate.op, candidate.target, inputs=inputs)
                if winner is None:
                    raise ValueError("no oracle-verified native storage candidate")
                return winner.run(region, *inputs)[0]
            return native_storage(*args, **kwargs)
        prepared_matmul = self._try_prepared_matmul_call(args, kwargs)
        if prepared_matmul is not _JIT_FALLBACK:
            from . import compile_report as _cr
            if _cr.active_sink_is_capturing():
                _cr.emit_compile_report(self.compile_report())
            return prepared_matmul
        # Explicit bounded compilation already captured and verified the
        # frontend Graph. Replay binds that immutable native program, as the
        # prepared matmul route above does, instead of re-running the tracer.
        if getattr(self,"_rocm_nvfp4_bounded_program",None) is not None:
            native_ingest=self._try_rocm_nvfp4_program_call(args,kwargs)
            if native_ingest is not _JIT_FALLBACK:
                return native_ingest
        self._establish_tracer_authority(args, kwargs)
        if self.differentiation_request is not None:
            self._specialized_autodiff_module(args, kwargs)
        try:
            native_ingest = self._try_rocm_nvfp4_program_call(args,kwargs)
            if native_ingest is not _JIT_FALLBACK:
                return native_ingest
            native_scaled_program = self._try_rocm_composed_scaled_call(args, kwargs)
            if native_scaled_program is not _JIT_FALLBACK:
                return native_scaled_program
            native_descriptor = self._try_native_descriptor_call(args, kwargs)
            if native_descriptor is not _JIT_FALLBACK:
                return native_descriptor
            native_lhs = self._try_nvidia_lhs_call(args, kwargs)
            if native_lhs is not _JIT_FALLBACK:
                return native_lhs
            native_rhs = self._try_nvidia_rhs_call(args, kwargs)
            if native_rhs is not _JIT_FALLBACK:
                return native_rhs
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
        if self._bounded_lhs is not None:
            # The live/source certificate admits only registered pure named
            # producer/matmul calls. Keep the call-time gate without retracing.
            self._bounded_lhs.certificate.validate()
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
            if hasattr(value, "__cuda_array_interface__"):
                shape = self._resident_frontend_specs((value,))[0][0]
            else:
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

    def _resident_frontend_specs(self,ordered):
        from .resident_nvidia_tensor import cuda_frontend_specs
        from .nvidia_native import requests_attention
        ranks=(1,2,3,4) if requests_attention(self._ensure_legacy_graph_ir()) else (1,2)
        return cuda_frontend_specs(ordered,ranks=ranks)

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
        resident=any(hasattr(value,"__cuda_array_interface__") for value in ordered)
        if resident:
            if normalize_target_kind(self.target)!="nvidia_sm120" or not all(
                    hasattr(value,"__cuda_array_interface__") for value in ordered):
                raise ValueError("resident AD requires all roots on native SM120")
            specs=self._resident_frontend_specs(ordered)
            signature=tuple((str(dtype),shape) for shape,dtype in specs)
        else:
            signature = tuple(
                (str(np.asarray(value).dtype), tuple(int(d) for d in np.asarray(value).shape))
                for value in ordered
            )
        cached = self._autodiff_specializations.get(signature)
        if cached is not None:
            return cached
        values = dict(zip(self.arg_names, ordered))
        if resident:
            specialized=self._trace_frontend_capture(args,kwargs)[0]
        elif ordered and all(isinstance(value, np.ndarray) for value in ordered):
            try:
                specialized = self._trace_frontend_capture(args, kwargs)[0]
            except TesseraJitError:
                if self._frontend_batch_axes is not None:
                    raise
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
        from .trace import trace, to_graph_ir_module, _np_dtype_to_elem

        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None or len(ordered) != len(self.arg_names):
            raise TesseraJitError("traced autodiff specialization requires every argument")
        from .nvfp4_tensor import NVFP4Tensor
        for value in ordered:
            if isinstance(value, NVFP4Tensor):
                value.validate()
        resident = any(hasattr(value, "__cuda_array_interface__") for value in ordered)
        resident_specs = None
        if resident:
            if (normalize_target_kind(self.target) != "nvidia_sm120"
                    or not all(hasattr(value, "__cuda_array_interface__") for value in ordered)):
                raise ValueError("resident frontend requires all roots on native SM120")
            if require_outputs:
                raise ValueError("resident frontend numerical certification requires explicit host oracle inputs")
            resident_specs = self._resident_frontend_specs(ordered)
            signature = tuple((str(dtype),shape) for shape,dtype in resident_specs)
        else:
            signature = tuple(
                (f"{value.dtype}:packed_axis={value.packed_axis}", value.shape) if isinstance(value, NVFP4Tensor)
                else (str(np.asarray(value).dtype), tuple(int(d) for d in np.asarray(value).shape))
                for value in ordered
            )
        cached = self._traced_frontend_specializations.get(signature)
        if cached is not None and not require_outputs:
            return cached, None
        try:
            batch_axes = getattr(self, "_frontend_batch_axes", None)
            if batch_axes is not None:
                from .native_vmap import batch_specs, mixed_batch_policies
                traced = trace(self._fn, *batch_specs(ordered, batch_axes, depth=self._frontend_batch_depth, broadcast_prefix=mixed_batch_policies(self)))
            else:
                trace_inputs = (tuple((shape, _np_dtype_to_elem(dtype)) for shape, dtype in resident_specs)
                                if resident_specs is not None else ordered)
                traced = trace(self._fn, *trace_inputs, evaluate_catalog_outputs=require_outputs)
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
                target=self._legality_target(),
            )
            # Preserve explicit gated storage declarations across tracer SSA
            # renaming. Concrete storage must agree with the source declaration.
            if any(arg.dtype_status is not None for arg in self._constraint_ir_args):
                for captured, declared in zip(
                    module.functions[0].args, self._constraint_ir_args, strict=True
                ):
                    if declared.dtype_status is not None:
                        if captured.ir_type.dtype != declared.ir_type.dtype:
                            raise ValueError("gated Tensor storage differs from declaration")
                        captured.dtype_status = declared.dtype_status
            if batch_axes is not None:
                from .native_vmap import project_batch
                module = project_batch(module, ordered, batch_axes, depth=self._frontend_batch_depth,
                    scale_transpose=(self.differentiation_request is not None
                                     and self.differentiation_request.mode == "reverse"),
                    broadcast_prefix=mixed_batch_policies(self))
                from .native_vmap import project_result_axes
                module = project_result_axes(module, self._frontend_output_permutation)
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
        from .nvfp4_tensor import NVFP4Tensor
        from .effects import Effect, infer_graph_effects

        ordered = self._ordered_inputs(args, kwargs)
        resident = ordered is not None and bool(ordered) and any(hasattr(value, "__cuda_array_interface__") for value in ordered)
        if resident and ordered is not None and (normalize_target_kind(self.target) != "nvidia_sm120"
                         or not all(hasattr(value, "__cuda_array_interface__") for value in ordered)):
            raise ValueError("resident frontend requires all roots on native SM120")
        if ordered is None or not ordered or not all(
            isinstance(value, (np.ndarray, NVFP4Tensor)) or resident for value in ordered
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
        has_nvfp4 = any(isinstance(value, NVFP4Tensor) for value in ordered)
        if has_nvfp4 and normalize_target_kind(self.target) != "nvidia_sm120":
            raise ValueError("logical NVFP4 host bindings require an owning SM120 package")
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
            if has_nvfp4:
                raise
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
        certificate_key = (signature, tuple(sorted(_permitted_effect_ops)), float(rtol), float(atol))
        cached = self._frontend_differential_certificates.get(certificate_key)
        if cached is not None:
            self.last_frontend_differential = cached
            return cached
        if self._frontend_batch_axes is not None:
            from .native_vmap import certify_typed_batch_frontends
            try:
                certificate = certify_typed_batch_frontends(
                    self, self._ordered_inputs(args, kwargs, normalize_batch=False), rtol=rtol, atol=atol)
            except ValueError as exc:
                raise TesseraJitError(str(exc)) from exc
            self.last_frontend_differential = certificate
            self._frontend_differential_certificates[certificate_key] = certificate
            return certificate
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
            r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir(target=normalize_target_kind(self.target))
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
        if any(op.kwargs.get('_region') for fn in module.functions for op in fn.body):
            from .source_control_flow import to_native_autodiff_ir
            graph_text = to_native_autodiff_ir(module)
        else:
            graph_text = re.sub(
                r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir()
            )
        transformed = subprocess.run(
            [str(opt), "--tessera-autodiff-paired=normalize-counted-while=true normalize-data-while=true",
             "--tessera-autodiff-hvp-prepare", "--tessera-autodiff-forward", "/dev/stdin"],
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

    def compile_persistent_device_tape(self, *args, compiler, llvm_bin, backend, chip, **kwargs):
        """Compile split native products with persistent static tensor residuals."""
        import re
        from .native_persistent_tape import materialize_persistent_tape
        request = self.differentiation_request
        if request is None or request.mode != "reverse":
            raise TesseraJitError("persistent device tape requires reverse autodiff")
        module = self._traced_autodiff_module(args, kwargs)
        source = re.sub(r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir())
        return materialize_persistent_tape(source, compiler=compiler, llvm_bin=llvm_bin, backend=backend, chip=chip)

    def compile_sparse_2to4(self, *args, compiler=None, llvm_bin=None, toolkit=None, **kwargs):
        """Trace a single product and compile a checked gfx1201 2:4 specialization.

        This explicit sparse contract returns a reusable compiled package. It
        does not alter ordinary JIT dispatch or silently prune input values.
        AD requests remain attached to the logical JIT function; native_backward
        differentiates that source, never the emitted packing program.
        """
        from .sparse_capture import compile_sparse_graph
        if normalize_target_kind(self.target) != "rocm":
            raise TesseraJitError("sparse capture requires a ROCm function")
        module = self._traced_autodiff_module(args, kwargs)
        return compile_sparse_graph(module, compiler=compiler, llvm_bin=llvm_bin, toolkit=toolkit)

    def package_sparse_2to4(self, *args, selection="checked_2to4", **kwargs):
        """Trace one half matmul and package its checked gfx1201 2:4 specialization
        as a native `RuntimeArtifact` for `runtime.launch` (public admission,
        2026-09-18): Graph -> Schedule (the compiler-built SWMMAC kernel with its
        validity words) -> Tile -> tessera_rocm -> HSACO, on the same boundaries
        as the dense scheduled families. The launch refuses any tile whose A
        block is not 2:4 sparse. AD stays on this logical function."""
        from tessera.runtime import RuntimeArtifact
        from .rocm_native import package_sparse_matmul
        from .scheduled_sparse import lower_scheduled_sparse_matmul
        if normalize_target_kind(self.target) != "rocm":
            raise TesseraJitError("sparse packaging requires a ROCm function")
        module = self._traced_autodiff_module(args, kwargs)
        artifact = lower_scheduled_sparse_matmul(module, selection=selection)
        package = package_sparse_matmul(artifact, pipeline_name="tessera-lower-to-rocm")
        return RuntimeArtifact(
            graph_ir=artifact.graph_ir, tile_ir=package.tile_ir, target_ir=package.target_ir,
            metadata={"target": "rocm_gfx1201", "compiler_path": "rocm_sparse_2to4_scheduled",
                      "arg_names": [artifact.a_name, artifact.b_name], "output_name": artifact.output_name,
                      "sparse_selection": selection, "shape": list(artifact.shape)},
            native_image=package.image, launch_descriptor=package.descriptor)

    def compile_sparse_auto(self, *args, compiler=None, llvm_bin=None, toolkit=None, **kwargs):
        """Compile native per-K-tile sparse/dense selection for one half matmul.

        Both branches execute on gfx1201; dense data is never pruned. This
        explicit policy does not promote the artifact into default dispatch.
        AD stays attached to this function's logical source.
        """
        from .sparse_capture import compile_sparse_graph
        if normalize_target_kind(self.target) != "rocm":
            raise TesseraJitError("sparse selection requires a ROCm function")
        module = self._traced_autodiff_module(args, kwargs)
        return compile_sparse_graph(module, selection="auto_2to4", compiler=compiler,
                                    llvm_bin=llvm_bin, toolkit=toolkit)

    def _native_frontend_signature(self) -> inspect.Signature:
        signature = self._nvidia_rhs_call_signature
        if signature is None:
            raise ValueError("native tensor execution requires a frontend call signature")
        return signature

    def _try_prepared_matmul_call(self, args, kwargs):
        """Reuse only an already verified compiler package and exact host ABI."""
        import os
        if (normalize_target_kind(self.target) != "nvidia_sm120"
                or self.differentiation_request is not None
                or not self._native_prepared_matmul_calls
                or os.environ.get("TESSERA_NVIDIA_PREPARED_MATMUL", "1").lower()
                in {"0", "off", "false"}):
            return _JIT_FALLBACK
        import numpy as np
        bound = self._native_frontend_signature().bind(*args, **kwargs)
        bound.apply_defaults()
        ordered = tuple(bound.arguments[name] for name in self.arg_names)
        if not all(isinstance(value, np.ndarray) for value in ordered):
            return _JIT_FALLBACK
        signature = tuple((value.dtype.str, value.shape, value.strides) for value in ordered)
        call = self._native_prepared_matmul_calls.get(signature)
        if call is None:
            return _JIT_FALLBACK
        if call.pid != os.getpid():
            raise ValueError("prepared matmul cannot cross fork")
        if call.artifact.launch_descriptor != call.descriptor_snapshot:
            raise ValueError("prepared matmul descriptor changed")
        trace_signature = tuple((str(value.dtype), tuple(int(d) for d in value.shape))
                                for value in ordered)
        captured = self._traced_frontend_specializations.get(trace_signature)
        if captured is None or not call.matches(captured):
            call.close()
            del self._native_prepared_matmul_calls[signature]
            return _JIT_FALLBACK
        self.graph_ir = captured
        array, receipt = call(ordered)
        self.compile_result, self.compile_bundle = call.compiled, call.compiled.bundle
        self._native_descriptor_last_receipt = receipt
        self._cached_artifact = call.artifact
        self.last_fallback_reason = None
        return array

    def _try_native_descriptor_call(self, args, kwargs):
        """Bind host tensors to a canonical compiler-owned static descriptor.

        Admits static SM120 FP16/BF16 matmul, ROCm movement/row-softmax and
        gfx1201 checkpoint storage. Python
        allocates declared outputs; native Graph/Schedule/Tile owns arithmetic.
        Compilation and launch errors propagate once this operation is selected.
        """
        target = normalize_target_kind(self.target)
        if target not in {"rocm_gfx1151", "rocm_gfx1201", "nvidia_sm120"}:
            return _JIT_FALLBACK
        from .nvidia_native import supports_f16_matmul, supports_bf16_matmul
        from .nvidia_native import requests_nvfp4_matmul, supports_nvfp4_matmul
        from .nvfp4_tensor import NVFP4Tensor
        nvfp4 = target == "nvidia_sm120" and requests_nvfp4_matmul(self.graph_ir)
        logical_inputs = self._ordered_inputs(args, kwargs)
        if logical_inputs and any(isinstance(value, NVFP4Tensor) for value in logical_inputs) and not nvfp4:
            raise ValueError("logical NVFP4 host bindings require the named scaled matmul Graph contract")
        from .nvidia_native import supports_attention, supports_attention_lse, requests_attention
        attention_saved_lse = False
        attention = (target == "nvidia_sm120" and self.differentiation_request is None
                     and requests_attention(self.graph_ir))
        matmul = (target == "nvidia_sm120" and self.differentiation_request is None
                  and (supports_f16_matmul(self.graph_ir) or supports_bf16_matmul(self.graph_ir)))
        from .rocm_nvfp4_ingest_native import supports_nvfp4_ingest, NVFP4_INGEST_ABI
        from .rocm_mxfp4_storage_native import supports_mxfp4_storage, MXFP4_STORAGE_ABI
        from .rocm_native import (
            requests_paged_kv_read, requests_moe_dispatch,
            _paged_kv_contract, _moe_dispatch_contract,
            GFX_PAGED_KV_F32_ABI, GFX_PAGED_KV_STRIDED_F32_ABI, GFX_MOE_DISPATCH_F32_ABI,
            GFX_SOFTMAX_F32_ABI, requests_softmax,
        )
        from .rocm_math_native import supports_math, requests_math, MATH_ABIS
        native_math = target.startswith("rocm_") and requests_math(self.graph_ir)
        nvidia_softmax = target == "nvidia_sm120" and requests_softmax(self.graph_ir)
        softmax = (target.startswith("rocm_") or nvidia_softmax) and requests_softmax(self.graph_ir)
        movement = target.startswith("rocm_") and ((requests_paged_kv_read(self.graph_ir)
                     and len(self.graph_ir.functions[0].body[0].operands) == 2)
                    or (target == "rocm_gfx1151" and requests_moe_dispatch(self.graph_ir)))
        from .rocm_typed_scaled_native import requests_typed_scaled
        typed_scaled = target == "rocm_gfx1201" and requests_typed_scaled(self.graph_ir)
        checkpoint = target == "rocm_gfx1201" and (
            supports_nvfp4_ingest(self.graph_ir) or supports_mxfp4_storage(self.graph_ir))
        if not movement and not checkpoint and not softmax and not matmul and not attention and not native_math and not nvfp4 and not typed_scaled:
            return _JIT_FALLBACK
        if self.differentiation_request is not None:
            raise ValueError("native descriptor call has no differentiation contract")
        import hashlib
        import os
        import numpy as np
        from .canonical_compile import canonical_compile
        from tessera import runtime as rt
        signature_binding = self._native_frontend_signature() if matmul or attention or nvfp4 or nvidia_softmax else inspect.signature(self._fn)
        bound = signature_binding.bind(*args, **kwargs)
        bound.apply_defaults()
        ordered = tuple(bound.arguments[name] for name in self.arg_names)
        if not all(isinstance(value, (np.ndarray, NVFP4Tensor)) if nvfp4 else isinstance(value, np.ndarray) for value in ordered):
            if movement or matmul:
                return _JIT_FALLBACK
            raise TypeError("native descriptor JIT expects host tensor inputs")
        matmul_signature = tuple((value.dtype.str, value.shape, value.strides) for value in ordered) if matmul else ()
        prepared_matmul = matmul and os.environ.get(
            "TESSERA_NVIDIA_PREPARED_MATMUL", "1").lower() not in {"0", "off", "false"}
        if prepared_matmul:
            lib = rt._load_nvidia_ptx_launch()
            prepared_matmul = lib is not None and hasattr(lib, "tessera_nvidia_matmul_prepare")
        module, _ = self._trace_frontend_capture(ordered, {})
        from .native_vmap import mixed_batch_policies, normalize_mixed_batch_inputs
        if mixed_batch_policies(self):
            ordered = tuple(normalize_mixed_batch_inputs(ordered, self._frontend_batch_policies))
        if attention:
            attention_saved_lse = supports_attention_lse(module)
        if nvfp4:
            if not supports_nvfp4_matmul(module):
                raise ValueError("native NVFP4 JIT requires the exact named static Graph profile")
            op = module.functions[0].body[0]
            values_by_name = dict(zip((arg.name for arg in module.functions[0].args), ordered, strict=True))
            for index, operand in enumerate(op.operands):
                value = values_by_name[operand.removeprefix("%")]
                if index < 2:
                    if not isinstance(value, NVFP4Tensor):
                        raise ValueError("native NVFP4 matrices require explicit logical packed storage")
                    value.validate()
                    transposed = op.kwargs.get("transposeA" if index == 0 else "transposeB", False)
                    packed_from_end = (2 if transposed else 1) if index == 0 else (1 if transposed else 2)
                    if value.packed_axis != len(value.shape) - packed_from_end:
                        raise ValueError("native NVFP4 matrix packing axis differs from the K axis")
                elif not isinstance(value, np.ndarray):
                    raise ValueError("native NVFP4 scales require uint8 ndarray storage")
        if native_math and not supports_math(module):
            raise ValueError("native math requires explicit f32 Graph computation and supported input storage")
        if attention and not (supports_attention(module) or supports_attention_lse(module)):
            return _JIT_FALLBACK
        if matmul:
            # Frontend storage facts, not a Python schedule/algorithm decision.
            # Never mutate the cached caller-owned semantic trace.
            import copy
            module = copy.deepcopy(module)
            function = module.functions[0]
            op = function.body[0]
            values = dict(zip((arg.name for arg in function.args), ordered, strict=True))
            rhs = values[op.operands[1].removeprefix("%")]
            if "rhs_storage_order" not in op.kwargs:
                if rhs.flags.f_contiguous and not rhs.flags.c_contiguous:
                    op.kwargs["rhs_storage_order"] = "col_major"
                elif rhs.flags.c_contiguous:
                    op.kwargs["rhs_storage_order"] = "row_major"
                else:
                    raise ValueError("static native matmul requires compact RHS storage")
        if movement and requests_paged_kv_read(module):
            from .scheduled_paged_kv import project_paged_host_storage
            module = project_paged_host_storage(module, ordered)
        signature = tuple((value.dtype if isinstance(value, NVFP4Tensor) else value.dtype.str,
                           value.shape, *((value.strides,) if movement else ())) for value in ordered)
        prepared_enabled = movement and os.environ.get(
            "TESSERA_ROCM_PREPARED_MOVEMENT", "1").lower() not in {"0", "off", "false"}
        lib = rt._load_rocm_native_movement_runtime() if prepared_enabled else None
        prepared_enabled = prepared_enabled and lib is not None and hasattr(
            lib, "tessera_rocm_movement_prepare")
        if prepared_enabled:
            call = self._native_prepared_movement_calls.get(signature)
            if call is not None:
                if call.matches(module, ordered):
                    array, receipt = call(ordered)
                    self.compile_result = call.compiled
                    self.compile_bundle = call.compiled.bundle
                    self._native_descriptor_last_receipt = receipt
                    self._cached_artifact = call.artifact
                    self.last_fallback_reason = None
                    return array
                call.close()
                del self._native_prepared_movement_calls[signature]
        key = hashlib.sha256(module.to_mlir(target=self.target, canonical=True).encode()).hexdigest()
        compiled = self._native_descriptor_specializations.get(key)
        if compiled is None:
            compiled = canonical_compile(module, target=target,
                source_origin=self.source_origin, enable_tool_validation=False)
            if not compiled.executable:
                raise RuntimeError(f"native descriptor compilation failed: {compiled.reason}")
            self._native_descriptor_specializations[key] = compiled
        descriptor = compiled.launch_descriptor
        if descriptor is None:
            raise ValueError("compilation did not produce its complete static ABI")
        scalars = {}
        scalar_names: tuple[str, ...]
        movement_contract: tuple[str, str, str, tuple[int, ...]] | None
        if movement:
            if descriptor.abi_id in {GFX_PAGED_KV_F32_ABI, GFX_PAGED_KV_STRIDED_F32_ABI}:
                movement_contract = _paged_kv_contract(module)
                scalar_names = ("P", "LP", "PageSize", "H", "D", "Start", "Tokens")
            elif descriptor.abi_id == GFX_MOE_DISPATCH_F32_ABI and target == "rocm_gfx1151":
                movement_contract = _moe_dispatch_contract(module)
                scalar_names = ("T", "S", "H")
            else:
                raise ValueError("movement compilation produced a different ABI")
            if movement_contract is None:
                raise ValueError("traced movement is outside the native tensor contract")
            declared = sorted(descriptor.scalars, key=lambda item: item.ordinal)
            if (tuple(item.name for item in declared) != (scalar_names +
                        (("StrideP", "StridePage", "StrideH", "StrideD")
                         if descriptor.abi_id == GFX_PAGED_KV_STRIDED_F32_ABI else ()))
                    or any(item.dtype != "int64" for item in declared)
                    or tuple(descriptor.provenance.get("shape", ())) != movement_contract[3]):
                raise ValueError("movement descriptor differs from the traced tensor contract")
            scalars = dict(zip(scalar_names, movement_contract[3], strict=True))
            if descriptor.abi_id == GFX_PAGED_KV_STRIDED_F32_ABI:
                from .paged_host_span import checked_page_span
                values = dict(zip((arg.name for arg in module.functions[0].args), ordered, strict=True))
                _, strides = checked_page_span(values[movement_contract[0]])
                scalars.update(zip(("StrideP", "StridePage", "StrideH", "StrideD"), strides, strict=True))
        elif native_math:
            info = descriptor.provenance.get("native_math")
            if descriptor.abi_id not in MATH_ABIS.values() or not isinstance(info, dict):
                raise ValueError("native math compilation did not produce its checked ABI")
            scalars = ({"Rows":info["rows"], "Columns":info["columns"]}
                       if info["family"] == "scan" else {"N":info["elements"]})
        elif softmax:
            import math
            softmax_expected_abi: str | None = GFX_SOFTMAX_F32_ABI
            if nvidia_softmax:
                from .nvidia_native import (
                    SM120_SOFTMAX_F16_ABI, SM120_SOFTMAX_BF16_ABI,
                    SM120_SOFTMAX_F32_ABI,
                )
                softmax_expected_abi = {
                    "fp16": SM120_SOFTMAX_F16_ABI,
                    "bf16": SM120_SOFTMAX_BF16_ABI,
                    "fp32": SM120_SOFTMAX_F32_ABI,
                }.get(module.functions[0].args[0].ir_type.dtype or "")
            if softmax_expected_abi is None or descriptor.abi_id != softmax_expected_abi or len(ordered) != 1:
                raise ValueError("public native row-softmax requires its checked storage ABI")
            shape = ordered[0].shape
            rows, columns = math.prod(shape[:-1]), shape[-1]
            declared = sorted(descriptor.scalars, key=lambda item: item.ordinal)
            if (tuple(item.name for item in declared) != ("Rows", "K")
                    or any(item.dtype != "int64" for item in declared)
                    or tuple(descriptor.provenance.get("shape", ())) != shape
                    or (not nvidia_softmax and descriptor.provenance.get("rows") != rows)
                    or (not nvidia_softmax and descriptor.provenance.get("columns") != columns)
                    or (nvidia_softmax and descriptor.provenance.get("kind") != "softmax")):
                raise ValueError("row-softmax descriptor differs from the typed frontend extent")
            scalars = {"Rows": rows, "K": columns}
        elif typed_scaled:
            from .rocm_typed_scaled_native import contract
            info=contract(module)
            if info is None:raise ValueError("typed scaled primal Graph contract differs")
            shape=info[0]
            declared=sorted(descriptor.scalars,key=lambda item:item.ordinal)
            if tuple(item.name for item in declared)!=("M","N","K") or any(item.dtype!="int64" for item in declared):
                raise ValueError("typed scaled primal scalar ABI differs")
            scalars={"M":shape.m,"N":shape.n,"K":shape.k}
        elif matmul:
            shape = descriptor.provenance.get("shape")
            declared = sorted(descriptor.scalars, key=lambda item: item.ordinal)
            if (descriptor.provenance.get("route") != "canonical_scheduled_tile_consumer"
                    or not isinstance(shape, list) or len(shape) != 3
                    or any(type(d) is not int or d <= 0 for d in shape)
                    or tuple(item.name for item in declared) != ("M", "N", "K")
                    or any(item.dtype != "int64" for item in declared)
                    or descriptor.provenance.get("dynamic_shape_bounds") is not None):
                raise ValueError("matmul descriptor differs from its static native contract")
            scalars = dict(zip(("M", "N", "K"), shape, strict=True))
        elif nvfp4:
            from .nvidia_native import SM120_NVFP4_ABI, SM120_NVFP4_BATCH_ABI
            shape = descriptor.provenance.get("shape")
            batch_rows = descriptor.provenance.get("batch_rows")
            policy = module.functions[0].body[0].kwargs
            independent = policy.get("batching") in {"independent_rhs", "shared_lhs"} or (policy.get("batching") == "shared_rhs_rows" and policy.get("transposeA", False))
            scalar_names = ("M", "N", "K", "BatchRows", "BatchCount") if independent else ("M", "N", "K")
            declared = sorted(descriptor.scalars, key=lambda item: item.ordinal)
            if (descriptor.abi_id != (SM120_NVFP4_BATCH_ABI if independent else SM120_NVFP4_ABI)
                    or not isinstance(shape, list) or len(shape) != 3
                    or any(type(d) is not int or d <= 0 for d in shape)
                    or tuple(item.name for item in declared) != scalar_names
                    or any(item.dtype != "int64" for item in declared)):
                raise ValueError("NVFP4 descriptor differs from its checked static ABI")
            scalar_values = list(shape)
            if independent:
                if not isinstance(batch_rows, list) or len(batch_rows) != 2 or batch_rows[0] * batch_rows[1] != shape[0]:
                    raise ValueError("NVFP4 descriptor batch scalar_values differ")
                scalar_values.extend((batch_rows[1], batch_rows[0]))
            scalars = dict(zip(scalar_names, scalar_values, strict=True))
        elif attention:
            from .nvidia_native import (
                _attention_contract, _attention_lse_contract,
                SM120_ATTN_LSE_F32_ABI, SM120_ATTN_LSE_BIAS_F32_ABI, SM120_ATTN_LSE_BCAST_F32_ABI,
                SM120_ATTN_F16_ABI, SM120_ATTN_BF16_ABI,
                SM120_ATTN_F32_ABI, SM120_ATTN_BIAS_F16_ABI,
                SM120_ATTN_BIAS_BF16_ABI, SM120_ATTN_BIAS_F32_ABI,
            )
            attention_contract=(_attention_lse_contract(module) if attention_saved_lse else _attention_contract(module))
            declared=sorted(descriptor.scalars,key=lambda item:item.ordinal)
            scalar_names=("B","Hq","Hkv","Sq","Sk","D","Dv")
            bias_shape = descriptor.provenance.get("bias_shape", []) if attention_saved_lse else []
            if bias_shape:
                if (not isinstance(bias_shape, list) or len(bias_shape) != 4
                        or any(type(value) is not int or value <= 0 for value in bias_shape)):
                    raise ValueError("saved attention physical bias dimensions are malformed")
                scalar_names += ("BiasB","BiasH","BiasQ","BiasK")
            attention_abis = ({SM120_ATTN_LSE_F32_ABI, SM120_ATTN_LSE_BIAS_F32_ABI, SM120_ATTN_LSE_BCAST_F32_ABI}
                if attention_saved_lse else {SM120_ATTN_F16_ABI,SM120_ATTN_BF16_ABI,SM120_ATTN_F32_ABI,
                    SM120_ATTN_BIAS_F16_ABI,SM120_ATTN_BIAS_BF16_ABI,SM120_ATTN_BIAS_F32_ABI})
            if (attention_contract is None or descriptor.abi_id not in attention_abis
                    or tuple(item.name for item in declared)!=scalar_names
                    or any(item.dtype!="int64" for item in declared)
                    or tuple(descriptor.provenance.get("shape",()))!=attention_contract[1]):
                raise ValueError("attention descriptor differs from its typed frontend contract")
            scalars=dict(zip(scalar_names,(*attention_contract[1], *bias_shape),strict=True))
        elif descriptor.abi_id not in {NVFP4_INGEST_ABI, MXFP4_STORAGE_ABI} or descriptor.scalars:
            raise ValueError("checkpoint compilation did not produce its complete static ABI")
        bindings = sorted(descriptor.buffers, key=lambda item: item.ordinal)
        inputs = [item for item in bindings if item.direction == "input"]
        outputs = [item for item in bindings if item.direction == "output"]
        if len(inputs) != len(ordered) or len(outputs) != len(module.functions[0].result_types):
            raise ValueError("descriptor input/result arity differs from traced frontend")
        arguments = dict(zip((arg.name for arg in module.functions[0].args), ordered, strict=True))
        if set(arguments) != {item.name for item in inputs}:
            raise ValueError("descriptor input bindings differ from traced argument names")
        buffers = {name: value.storage if isinstance(value, NVFP4Tensor) else value
                   for name, value in arguments.items()}
        if typed_scaled and any(not value.flags.c_contiguous for value in buffers.values()):
            from .native_scaled_program import pack_host_view
            library = rt._load_rocm_native_movement_runtime()
            if library is None:
                raise ValueError("native scaled strided storage requires the checked HIP owner")
            # Descriptor layout remains compact. Checked native byte movement
            # materializes alias views before the unchanged launch ABI is bound.
            buffers = {name: pack_host_view(library, value) for name, value in buffers.items()}
        arrays = []
        # Storage preparation only: dimensions and types come from the checked
        # compiler descriptor, never an eager numerical implementation.
        for output in outputs:
            dimensions = {guard.dimension: guard.value
                for guard in descriptor.shape_guards
                if guard.binding == output.name and guard.predicate == "eq"}
            if set(dimensions) != set(range(output.rank)):
                raise ValueError("host tensor call requires fully static output guards")
            dtype = {"uint8": np.uint8, "fp64": np.float64, "fp32": np.float32,
                     **({"fp16": np.float16} if matmul or nvidia_softmax else {})}.get(output.dtype)
            if dtype is None and nvidia_softmax and output.dtype == "bf16":
                import ml_dtypes
                dtype = ml_dtypes.bfloat16
            if dtype is None or output.layout != "row_major":
                raise ValueError("descriptor output storage contract is unsupported")
            # The native prepared owner allocates its independent host result.
            # Keep validating the output contract without allocating an unused
            # second result during first-call preparation.
            if not (prepared_matmul and descriptor.geometry.policy == "sm120_scheduled_typed_16x8_mn"):
                array = np.empty(tuple(dimensions[axis] for axis in range(output.rank)), dtype=dtype)
                arrays.append(array)
                buffers[output.name] = array
        artifact = self._native_descriptor_artifacts.get(key)
        if artifact is None:
            artifact = compiled.to_runtime_artifact()
            if matmul:
                from dataclasses import replace
                artifact = replace(artifact, metadata={
                    **artifact.metadata,
                    "frontend_argument_names": list(self.arg_names),
                    "frontend_input_bindings": [arg.name for arg in module.functions[0].args],
                })
            self._native_descriptor_artifacts[key] = artifact
        if prepared_matmul and descriptor.geometry.policy == "sm120_scheduled_typed_16x8_mn":
            from .prepared_nvidia_matmul import PreparedMatmulCall
            call = PreparedMatmulCall(compiled, artifact, module, self.graph_ir)
            try:
                array, receipt = call(ordered)
            except Exception:
                call.close()
                raise
            if len(self._native_prepared_matmul_calls) >= 24:
                retired = next(iter(self._native_prepared_matmul_calls))
                self._native_prepared_matmul_calls.pop(retired).close()
            self._native_prepared_matmul_calls[matmul_signature] = call
            arrays = [array]
        else:
            receipt = rt.launch(artifact, {"buffers": buffers, "scalars": scalars})
        if receipt.get("ok") is not True or receipt.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"native descriptor launch failed: {receipt}")
        self.compile_result = compiled
        self.compile_bundle = compiled.bundle
        self._native_descriptor_last_receipt = receipt
        self._cached_artifact = artifact
        self.last_fallback_reason = None
        if prepared_enabled:
            from .prepared_rocm_movement import PreparedMovementCall
            self._native_prepared_movement_calls[signature] = PreparedMovementCall(
                compiled, module, artifact, movement_contract, ordered=ordered)
        if attention_saved_lse:
            return tuple(arrays)
        return arrays[0] if movement or softmax or matmul or attention or native_math or nvfp4 or typed_scaled else tuple(arrays)


    def _try_rocm_composed_scaled_call(self,args,kwargs):
        """Compile the full semantic product/sum Graph; native HIP owns execution."""
        if normalize_target_kind(self.target)!="rocm_gfx1201":
            return _JIT_FALLBACK
        from .rocm_typed_scaled_native import requests_composed_typed_scaled, requests_floating_scaled
        floating = requests_floating_scaled(self.graph_ir)
        if self.differentiation_request is not None and not floating:
            return _JIT_FALLBACK
        if not (requests_composed_typed_scaled(self.graph_ir) or floating or
                (self._frontend_output_permutation is not None and
                 self._frontend_output_permutation != tuple(range(len(self._frontend_output_permutation))))):
            return _JIT_FALLBACK
        from .native_scaled_program import package_native_scaled_primal
        from .scheduled_matmul import find_tessera_opt
        from .rocm_native import _tool_digest
        from tessera import runtime as rt
        import numpy as np
        import json
        ordered=self._ordered_inputs(args,kwargs)
        if ordered is None or not all(isinstance(value,np.ndarray) for value in ordered):
            raise ValueError("native composed scaled primal requires explicit host tensors")
        from .paged_host_span import checked_host_span
        for value in ordered:
            checked_host_span(value, min_rank=1)
        self.frontend_differential(*args,**kwargs)
        module=self._traced_autodiff_module(args,kwargs)
        from .rocm_typed_scaled_native import supports_composed_scaled_primal
        if not supports_composed_scaled_primal(module):
            raise ValueError("native composed scaled primal requires its exact product/sum/permutation contract")
        if self.differentiation_request is not None:
            from .rocm_typed_scaled_native import primal_call_module
            module = primal_call_module(module)
        from dataclasses import replace
        module=replace(module,module_attrs={**module.module_attrs,
                       "tessera.target":json.dumps("rocm"),"tessera.arch":json.dumps("gfx1201")})
        graph=module.to_mlir(target="rocm_gfx1201",canonical=True)
        tool=find_tessera_opt()
        if tool is None:raise RuntimeError("native composed scaled primal requires matching tessera-opt")
        key=(graph,_tool_digest(tool))
        cache=getattr(self,"_native_composed_scaled_cache",None)
        if cache is None:cache={};self._native_composed_scaled_cache=cache
        cached=cache.get(key)
        if cached is None:
            package=package_native_scaled_primal(graph)
            artifact=rt.RuntimeArtifact(graph_ir=graph,metadata={
                "target":"rocm","architecture":"gfx1201","evidence_target":"rocm_gfx1201","compiler_path":"rocm_scaled_primal_program_compiled",
                "execution_kind":"native_gpu","execution_mode":"hip_runtime","executable":True,
                "runtime_status":"ready","native_graph_verified":True,
                "arg_names":list(self.arg_names),"native_scaled_program":package.to_manifest()})
            cached=(package,artifact);cache[key]=cached
            if len(cache)>24:del cache[next(iter(cache))]
        package,artifact=cached
        receipt=rt.launch(artifact,ordered)
        if receipt.get("ok") is not True or receipt.get("execution_kind")!="native_gpu":
            raise RuntimeError(f"native composed scaled primal execution failed: {receipt}")
        self._native_composed_scaled_last_program=package
        self._native_descriptor_last_receipt=receipt
        self._cached_artifact=artifact
        self.last_fallback_reason=None
        return receipt["output"]

    def _try_rocm_nvfp4_program_call(self,args,kwargs):
        if normalize_target_kind(self.target)!="rocm_gfx1201":
            return _JIT_FALLBACK
        from .rocm_nvfp4_program import supports_resident_trace,runtime_artifact
        if not supports_resident_trace(self.graph_ir):
            return _JIT_FALLBACK
        if self.differentiation_request is not None:
            raise ValueError("resident lossy checkpoint program has no differentiation contract")
        import hashlib
        import numpy as np
        bound=self._rocm_nvfp4_call_signature.bind(*args,**kwargs)
        bound.apply_defaults()
        ordered=tuple(bound.arguments[name] for name in self.arg_names)
        if len(ordered)!=5 or not all(isinstance(value,np.ndarray) for value in ordered):
            raise TypeError("resident checkpoint JIT requires five explicit host tensors")
        bounded=getattr(self,"_rocm_nvfp4_bounded_program",None)
        if bounded is not None:
            program,artifact=bounded
            from tessera import runtime as rt
            receipt=rt.launch(artifact,ordered)
            if receipt.get("ok") is not True or receipt.get("execution_kind")!="native_gpu":
                raise RuntimeError(f"bounded checkpoint native execution failed: {receipt}")
            self._rocm_nvfp4_last_program=program
            self._rocm_nvfp4_last_receipts=tuple(receipt["component_receipts"])
            self._cached_artifact=None
            self.last_fallback_reason=None
            return receipt["output"]
        module,_=self._trace_frontend_capture(ordered,{})
        if not supports_resident_trace(module):
            raise ValueError("resident checkpoint trace differs from its declared Graph")
        key=hashlib.sha256(module.to_mlir(target=self.target,canonical=True).encode()).hexdigest()
        cached=self._rocm_nvfp4_program_cache.get(key)
        if cached is None:
            program=self.compile_native_nvfp4_program(*ordered)
            artifact=runtime_artifact(program)
            # Retain the compiler product independently of inspection artifacts.
            # Launch still validates its manifest, Graph binding and native ABI.
            cached=(program,artifact)
            self._rocm_nvfp4_program_cache[key]=cached
            if len(self._rocm_nvfp4_program_cache)>24:
                del self._rocm_nvfp4_program_cache[next(iter(self._rocm_nvfp4_program_cache))]
        program,artifact=cached
        from tessera import runtime as rt
        receipt=rt.launch(artifact,ordered)
        if receipt.get("ok") is not True or receipt.get("execution_kind")!="native_gpu":
            raise RuntimeError(f"resident checkpoint native execution failed: {receipt}")
        self._rocm_nvfp4_last_program=program
        self._rocm_nvfp4_last_receipts=tuple(receipt["component_receipts"])
        self._cached_artifact=None
        self.last_fallback_reason=None
        return receipt["output"]

    def compile_native_nvfp4_program(self,*args,m_bound=None,**kwargs):
        from copy import deepcopy
        from dataclasses import replace
        from .rocm_nvfp4_program import package_traced_resident
        if normalize_target_kind(self.target)!="rocm_gfx1201" or self.differentiation_request is not None:
            raise ValueError("native checkpoint program requires primal exact gfx1201")
        bound=self._rocm_nvfp4_call_signature.bind(*args,**kwargs)
        bound.apply_defaults()
        ordered=tuple(bound.arguments[name] for name in self.arg_names)
        module,_=self._trace_frontend_capture(ordered,{})
        if m_bound is not None:
            if type(m_bound) is not int or m_bound<=0:
                raise ValueError("native NVFP4 row bound must be a positive integer")
            module.module_attrs["tessera.native.nvfp4_m_bound"]=f"{m_bound} : i64"
        program=replace(package_traced_resident(module),argument_names=tuple(self.arg_names))
        if m_bound is not None:
            from .rocm_nvfp4_program import runtime_artifact
            self.graph_ir=deepcopy(module)
            self.frontend_authority="tracer"
            self.frontend_authority_error=None
            self._rocm_nvfp4_bounded_program=(program,runtime_artifact(program))
        return program

    def native_nvfp4_packages(self):
        program=self._rocm_nvfp4_last_program
        return () if program is None else (
            program.native.ingest.native,program.native.storage.native,program.native.consumer.package)


    def _try_nvidia_lhs_call(self, args, kwargs):
        """Execute the traced LHS producer on native SM120."""
        if normalize_target_kind(self.target) != "nvidia_sm120" or self.differentiation_request is not None:
            return _JIT_FALLBACK
        import hashlib
        import numpy as np
        bound = self._native_frontend_signature().bind(*args, **kwargs)
        bound.apply_defaults()
        ordered = [bound.arguments[name] for name in self.arg_names]
        resident = any(hasattr(value, "__cuda_array_interface__") for value in ordered)
        if resident and not all(hasattr(value, "__cuda_array_interface__") for value in ordered):
            raise ValueError("resident frontend requires all roots on native SM120")
        if len(ordered) not in {2,3,4} or not all(isinstance(value,np.ndarray) or resident for value in ordered):
            return _JIT_FALLBACK
        module = self._traced_autodiff_module(tuple(ordered), {})
        from .nvidia_tensor_lhs import candidate, project_rhs_storage
        if not candidate(module):
            return _JIT_FALLBACK
        # Once this semantic edge matches, compilation/launch failures propagate.
        # They must never be replaced by eager arithmetic.
        module = project_rhs_storage(module, ordered)
        self._nvidia_lhs_last_receipts = ()
        self._nvidia_lhs_last_program = None
        self._cached_artifact = None
        key = hashlib.sha256(module.to_mlir(target="nvidia_sm120").encode()).hexdigest()
        program = self._nvidia_lhs_program_cache.get(key)
        if program is None:
            program = self.compile_native_lhs_matmul(*ordered)
            self._nvidia_lhs_program_cache[key] = program
        return self._launch_nvidia_lhs_program(program,ordered,key)[0]

    def _launch_nvidia_lhs_program(self,program,ordered,key,*,prepared=None):
        from tessera import runtime as rt
        from .nvidia_tensor_lhs import runtime_artifact
        import os
        lib = rt._load_nvidia_ptx_launch()
        if prepared is None:
            prepared = (os.environ.get("TESSERA_NVIDIA_PREPARED_LHS", "1").lower()
                        not in {"0", "off", "false"} and lib is not None
                        and hasattr(lib, "tessera_nvidia_matmul_attach_producer"))
        if not prepared and any(hasattr(value, "__cuda_array_interface__") for value in ordered):
            raise ValueError("public resident tensor JIT requires its native prepared owner")
        if prepared:
            from .prepared_nvidia_lhs import PreparedLhsCall
            call = self._nvidia_lhs_prepared_calls.get(key)
            if call is None or not call._finalizer.alive:
                call = PreparedLhsCall(program)
                if len(self._nvidia_lhs_prepared_calls) >= 24:
                    oldest = next(iter(self._nvidia_lhs_prepared_calls))
                    self._nvidia_lhs_prepared_calls.pop(oldest).close()
                self._nvidia_lhs_prepared_calls[key] = call
            if any(hasattr(value, "__cuda_array_interface__") for value in ordered):
                _, receipt = call.resident_to_host(ordered)
            else:
                _, receipt = call(ordered)
        else:
            receipt = rt.launch(runtime_artifact(program), tuple(ordered))
        if receipt.get("ok") is not True or receipt.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"native LHS program failed: {receipt}")
        output = receipt["output"]
        self._nvidia_lhs_last_receipts = tuple(receipt["component_receipts"])
        self._nvidia_lhs_last_program = program
        self._cached_artifact = None
        self.last_fallback_reason = None
        return output,receipt

    def native_lhs_packages(self):
        program=self._nvidia_lhs_last_program
        return () if program is None else (*(program.producer_chain or (program.edge.producer,)),*program.rhs_chain,program.edge.consumer)

    def compile_native_lhs_matmul(self,*args,dynamic_axes=(),shape_bounds=None,rhs_storage_order=None,**kwargs):
        from dataclasses import replace
        from .nvidia_tensor_lhs import package_traced_lhs, project_rhs_storage
        if normalize_target_kind(self.target)!="nvidia_sm120" or self.differentiation_request is not None:
            raise TesseraJitError("native LHS matmul requires primal exact nvidia_sm120")
        bound = self._native_frontend_signature().bind(*args, **kwargs)
        bound.apply_defaults()
        ordered = tuple(bound.arguments[name] for name in self.arg_names)
        if shape_bounds is None and self._bounded_lhs is not None:
            shape_bounds=dict(self._bounded_lhs.bounds)
            self._bounded_lhs.certificate.validate()
        if rhs_storage_order is None and self._bounded_lhs is not None:
            rhs_storage_order=self._bounded_lhs.rhs_storage_order
        module = project_rhs_storage(self._traced_autodiff_module(ordered, {}), ordered,
                                     dynamic=bool(dynamic_axes or shape_bounds),
                                     rhs_storage_order=rhs_storage_order)
        return replace(package_traced_lhs(module,dynamic_axes=dynamic_axes,shape_bounds=shape_bounds),
                       argument_names=tuple(self.arg_names))

    def _try_nvidia_rhs_call(self, args, kwargs):
        """Execute the named traced half-storage RHS producer on native SM120."""
        if normalize_target_kind(self.target) != "nvidia_sm120" or self.differentiation_request is not None:
            return _JIT_FALLBACK
        import hashlib
        import numpy as np
        bound = self._native_frontend_signature().bind(*args, **kwargs)
        bound.apply_defaults()
        ordered = [bound.arguments[name] for name in self.arg_names]
        if ordered is None or len(ordered) != 2 or not all(
            isinstance(value, np.ndarray) and value.ndim == 2
            and str(value.dtype) in {"float16", "bfloat16"} for value in ordered
        ):
            return _JIT_FALLBACK
        module = self._traced_autodiff_module(tuple(ordered), {})
        if len(module.functions) != 1:
            return _JIT_FALLBACK
        function = module.functions[0]
        if len(function.body) != 2:
            return _JIT_FALLBACK
        producer, consumer = function.body
        if (producer.op_name not in {"tessera.rmsnorm", "tessera.layer_norm"} or len(producer.operands) != 1
                or consumer.op_name not in {"tessera.matmul", "tessera.gemm"}
                or len(consumer.operands) != 2
                or consumer.operands[1] != "%" + str(producer.result)):
            return _JIT_FALLBACK
        # Once this semantic edge matches, compilation/launch failures propagate.
        # They must never be replaced by eager arithmetic.
        key = hashlib.sha256(module.to_mlir(target="nvidia_sm120").encode()).hexdigest()
        program = self._nvidia_rhs_program_cache.get(key)
        if program is None:
            program = self.compile_native_rhs_matmul(*ordered)
            self._nvidia_rhs_program_cache[key] = program
        from tessera import runtime as rt
        from .nvidia_tensor_rhs import rhs_runtime_artifact
        receipt = rt.launch(rhs_runtime_artifact(program), tuple(ordered))
        if receipt.get("ok") is not True or receipt.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"native RHS program failed: {receipt}")
        output = receipt["output"]
        self._nvidia_rhs_last_receipts = tuple(receipt["component_receipts"])
        self._nvidia_rhs_last_program = program
        self._cached_artifact = None
        self.last_fallback_reason = None
        return output

    def native_rhs_packages(self):
        """Return the two checked packages from the last successful RHS call."""
        program = self._nvidia_rhs_last_program
        return () if program is None else (program.edge.producer, program.edge.consumer)

    def compile_native_rhs_matmul(self, *args, **kwargs):
        """Compile a traced RMSNorm/LayerNorm RHS into two resident native packages.

        The returned program accepts inputs in frontend argument order.
        Native Schedule/Tile owns both kernels; the resident result owns the
        shared edge allocation and stream until close.
        """
        from .nvidia_tensor_rhs import package_traced_norm_rhs
        if normalize_target_kind(self.target) != "nvidia_sm120" or self.differentiation_request is not None:
            raise TesseraJitError("native RHS matmul requires primal exact nvidia_sm120")
        module = self._traced_autodiff_module(args, kwargs)
        from dataclasses import replace
        program = package_traced_norm_rhs(module, pipeline_name="tessera-nvidia-pipeline-sm120")
        return replace(program, argument_names=tuple(self.arg_names))

    def compile_native_attention_vjp(self, *args, compiler, compact_gradients=False, compact_launch="packed_v1", compact_threads=128, sequence_bounds=None, **kwargs):
        """Compile this isolated attention trace into a resident reverse program.

        Capture arguments follow the traced frontend argument order and own
        one native forward O/LSE generation. Backward returns requested Q/K/V
        and supported bias gradients in wrt order from that saved generation.
        """
        import re
        from .native_attention_program import compile_attention_vjp_program
        request = self.differentiation_request
        if request is None or request.mode != "reverse" or normalize_target_kind(self.target) != "nvidia_sm120":
            raise TesseraJitError("native attention VJP requires reverse autodiff on exact nvidia_sm120")
        module = self._specialized_autodiff_module(args, kwargs)
        from dataclasses import replace
        module = replace(module, module_attrs={**module.module_attrs,
                         "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
        if sequence_bounds is not None:
            if (not isinstance(sequence_bounds,(tuple,list)) or len(sequence_bounds)!=2 or
                    any(type(x) is not int or x<=0 for x in sequence_bounds)):
                raise TesseraJitError("attention sequence bounds require positive Sq/Sk capacities")
            module = replace(module,module_attrs={**module.module_attrs,
                "tessera.attention_sequence_bounds": "array<i64: " + ", ".join(map(str,sequence_bounds)) + ">"})
        source = re.sub(r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir())
        return compile_attention_vjp_program(source, request.wrt_indices, compiler=compiler, compact_gradients=compact_gradients, compact_launch=compact_launch, compact_threads=compact_threads)

    def _prepare_native_movement_binding(self, args, kwargs):
        if self.target not in {"rocm_gfx1151", "rocm_gfx1201"}:
            raise ValueError("resident movement requires its owning ROCm target")
        module, _ = self._trace_frontend_capture(args, kwargs)
        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None:
            raise ValueError("resident movement requires every declared input")
        from .scheduled_paged_kv import project_paged_host_storage
        module = project_paged_host_storage(module, ordered)
        call = next((value for value in self._native_prepared_movement_calls.values()
                     if value.matches(module, ordered)), None)
        if call is None:
            result = self._try_native_descriptor_call(args, kwargs)
            if result is None:
                raise ValueError("resident movement requires a supported native paged-read or token-gather Graph")
            call = next((value for value in self._native_prepared_movement_calls.values()
                         if value.matches(module, ordered)), None)
        if call is None:
            raise ValueError("resident movement requires the matching native prepared runtime")
        return call, ordered

    def prepare_native_movement(self, *args, **kwargs):
        """Prepare native-owned static ROCm paged-read or token-gather storage."""
        call, ordered = self._prepare_native_movement_binding(args, kwargs)
        owner = call.resident()
        try:
            owner.upload(ordered)
            return owner
        except BaseException:
            owner.close()
            raise

    def prepare_native_paged_softmax(self, consumer, *args, **kwargs):
        """Bind two native frontend packages with an owned paged-read edge.

        The intermediate allocation remains on the owned HIP stream. This
        explicit static package edge does not infer a generic composed graph.
        """
        import numpy as np
        from .resident_rocm_movement import ResidentMovementCall
        if not isinstance(consumer, JitFn) or consumer.target != self.target:
            raise ValueError("paged softmax requires a JIT consumer on the same target")
        call, ordered = self._prepare_native_movement_binding(args, kwargs)
        consumer(np.zeros(call.output_shape, dtype=np.float32))
        artifact = consumer.runtime_artifact()
        bundle = consumer.compile_bundle
        if (consumer.execution_kind != "native_gpu" or bundle is None or
                any(stage is None for stage in (bundle.schedule, bundle.tile, bundle.target_ir, bundle.backend))):
            raise ValueError("softmax consumer requires native compiler execution")
        stages = (bundle.graph, bundle.schedule, bundle.tile, bundle.target_ir, bundle.backend)
        schedule = bundle.schedule
        if schedule is None:
            raise ValueError("softmax consumer lacks a native schedule")
        verified_stages = tuple(stage for stage in stages if stage is not None)
        if (schedule.producer != "tessera-opt.tessera-graph-to-schedule"
                or any(b.input_digest != a.output_digest
                       for a, b in zip(verified_stages[:-1], verified_stages[1:], strict=True))):
            raise ValueError("softmax consumer lacks native adjacent compiler lineage")
        owner = ResidentMovementCall(call, consumer=artifact)
        try:
            owner.upload(ordered)
            return owner
        except BaseException:
            owner.close()
            raise

    def compile_native_attention_jvp(self, *args, compiler, llvm_bin, **kwargs):
        """Compile an isolated Q/K/V attention JVP from this JIT function's trace.

        The returned program captures resident CUDA inputs and accepts only the
        requested tangent arguments. General JIT graphs remain unsupported.
        """
        import re
        from .native_attention_program import compile_attention_program
        request = self.differentiation_request
        if request is None or request.mode != "forward" or normalize_target_kind(self.target) != "nvidia_sm120":
            raise TesseraJitError("native attention JVP requires forward autodiff on exact nvidia_sm120")
        module = self._specialized_autodiff_module(args, kwargs)
        from dataclasses import replace
        module = replace(module, module_attrs={**module.module_attrs,
                         "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
        source = re.sub(r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir())
        return compile_attention_program(source, request.wrt_indices, compiler=compiler, llvm_bin=llvm_bin,
                                         input_names=tuple(inspect.signature(self._fn).parameters))

    def compile_native_storage_pair(self, *args, compiler, llvm_bin, backend, chip=None, **kwargs):
        """Compile the actual traced forward/reverse pair; Apple returns a host export."""
        import re
        from .native_storage_pair import materialize_storage_pair
        request = self.differentiation_request
        if request is None or request.mode not in ("forward", "reverse"):
            raise TesseraJitError("native pair requires a forward or reverse request")
        module = self._specialized_autodiff_module(args, kwargs)
        graph_text = re.sub(r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir())
        return materialize_storage_pair(graph_text, mode=request.mode, compiler=compiler,
            llvm_bin=llvm_bin, backend=backend, chip=chip)

    def compile_native_storage_jvp(self, *args, compiler, llvm_bin, backend, chip, **kwargs):
        """Generate a native paired storage child from this traced forward program.

        Compilation specializes on host example inputs; execution subsequently
        consumes caller-owned resident tensors through NativeStorageJVP.
        """
        import re
        from .native_storage_jvp import build_native_storage_jvp
        request = self.differentiation_request
        if request is None or request.mode != "forward":
            raise TesseraJitError("native storage JVP requires forward autodiff")
        module = self._specialized_autodiff_module(args, kwargs)
        graph_text = re.sub(r"=\s+(tessera\.[A-Za-z0-9_.]+)\(", r'= "\1"(', module.to_mlir())
        return build_native_storage_jvp(graph_text, compiler=compiler, llvm_bin=llvm_bin,
                                        backend=backend, chip=chip)

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
        the output-cotangent seed and saved residuals) followed by tangents
        for differentiable inputs and continuous residuals. The tensor-only
        ``@f__hvp`` entry captures residuals and their tangents internally.
        Its tangent results are Hessian-vector products. Unsupported second-order operations fail in the compiler;
        this method never substitutes finite differences.
        """
        request = self.differentiation_request
        if request is None or request.mode != "reverse":
            raise TesseraJitError(
                "compiled_hvp_ir requires @jit(autodiff='reverse')"
            )
        module = self._traced_autodiff_module(args, kwargs)
        return self._compile_hvp_module(module)

    def compile_native_hvp(self, *args: Any, compiler, llvm_bin,
                           backend, chip, **kwargs: Any):
        """Materialize a compiler-owned bounded resident CUDA/HIP HVP product."""
        from .native_hvp import materialize_native_hvp
        from .source_control_flow import to_native_autodiff_ir
        request = self.differentiation_request
        if request is None or request.mode != "reverse":
            raise TesseraJitError("compile_native_hvp requires reverse autodiff")
        module = self._traced_autodiff_module(args, kwargs)
        source = (to_native_autodiff_ir(module)
                  if any(op.kwargs.get('_region') for fn in module.functions for op in fn.body)
                  else module.to_mlir(canonical=True))
        return materialize_native_hvp(source, compiler=compiler, llvm_bin=llvm_bin,
                                      backend=backend, chip=chip)

    def native_hvp(self, *args: Any, tangents: Any,
                   out_cotangents: Any, **kwargs: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Execute exact forward-over-reverse AD on the native CPU compiler.

        Returns (gradients, Hessian-vector products), ordered by ``wrt``.
        The output cotangent is held constant: vector-valued functions compute
        the Hessian of that seeded scalarization. Directions for inactive
        arguments are zero. Only static f32 input signatures are admitted;
        compiler rejection never falls back to numerical differentiation.
        """
        import hashlib
        import numpy as np
        from .. import _jit_boundary as boundary

        request = self.differentiation_request
        if request is None or request.mode != "reverse":
            raise TesseraJitError("native_hvp requires @jit(autodiff='reverse')")
        if normalize_target_kind(self.target) != "cpu":
            raise TesseraJitError("native_hvp currently requires target='cpu'")
        self.last_hvp_execution: dict[str, Any] | None = None
        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None:
            raise TesseraJitError("native_hvp requires every forward argument")
        inputs = [np.ascontiguousarray(np.asarray(v)) for v in ordered]
        if any(v.dtype != np.dtype('float32') for v in inputs):
            raise TesseraJitError("native_hvp requires f32 tensor inputs")
        directions = tangents if isinstance(tangents, (tuple, list)) else (tangents,)
        if len(directions) != len(request.wrt_indices):
            raise TesseraJitError("native_hvp requires one tangent per active input")
        seeds = [np.zeros_like(v) for v in inputs]
        for index, direction in zip(request.wrt_indices, directions):
            value = np.ascontiguousarray(np.asarray(direction))
            if value.shape != inputs[index].shape or value.dtype != inputs[index].dtype:
                raise TesseraJitError("native_hvp tangent shape and dtype must match its primal")
            seeds[index] = value
        cots = out_cotangents if isinstance(out_cotangents, (tuple, list)) else (out_cotangents,)
        values = inputs + [np.ascontiguousarray(np.asarray(v)) for v in cots] + seeds
        module = self._traced_autodiff_module(args, kwargs)
        product = self._compile_hvp_module(module)
        symbol = module.functions[0].name + "__hvp"
        handle = boundary.compile_module(product)
        try:
            signature = boundary._function_signature(handle, symbol)
            if signature is None:
                raise TesseraJitError("native_hvp requires the compiler signature ABI")
            arguments, results = signature
            if len(arguments) != len(values) or len(results) != 2 * len(inputs):
                raise TesseraJitError("native_hvp compiler product ABI disagrees with the request")
            # Validate before allocating outputs or entering native code. Shapes
            # and dtypes are projections of the compiled signature, not guesses
            # from the first primal (mixed-precision products need distinct slots).
            for index, (value, entry) in enumerate(zip(values, arguments)):
                boundary._check_array_against_sig("input", index, value, entry)
            outputs = []
            for shape, elem in results:
                if shape is None or any(d is None for d in shape) or elem not in ('f32', 'f64'):
                    raise TesseraJitError("native_hvp requires static f32/f64 tensor results")
                outputs.append(np.empty(shape, dtype='float32' if elem == 'f32' else 'float64'))
            boundary.invoke(handle, symbol, values, outputs)
        finally:
            boundary.destroy(handle)
        self.last_hvp_execution = {
            "execution_kind": "native_cpu", "execution_mode": "mlir_llvm_jit",
            "product_symbol": symbol,
            "product_ir_digest": hashlib.sha256(product.encode()).hexdigest(),
            "differentiation": "exact_forward_over_reverse",
        }
        count = len(inputs)
        return (tuple(outputs[i] for i in request.wrt_indices),
                tuple(outputs[count + i] for i in request.wrt_indices))

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

        self.last_jvp_execution = None
        request = self.differentiation_request
        if request is None or request.mode != "forward":
            raise TesseraJitError(
                "native_jvp requires @jit(autodiff='forward' or 'jvp')"
            )
        self._enforce_call_time_constraints(args, kwargs)
        ordered = self._ordered_inputs(args, kwargs)
        if ordered is None:
            raise TesseraJitError("native_jvp could not bind the compiled inputs")
        tangent_values = tangents if isinstance(tangents, (tuple, list)) else (tangents,)
        if len(tangent_values) != len(request.wrt_indices):
            raise TesseraJitError("native_jvp requires one tangent per active input")
        resident_attention=any(hasattr(value,"__cuda_array_interface__") for value in (*ordered,*tangent_values))
        if resident_attention:
            if normalize_target_kind(self.target)!="nvidia_sm120" or not all(
                    hasattr(value,"__cuda_array_interface__") for value in (*ordered,*tangent_values)):
                raise TesseraJitError("resident attention AD requires all primal/tangent CUDA roots")
            resident_specs=self._resident_frontend_specs(ordered)
            tangent_specs=self._resident_frontend_specs(tangent_values)
            if any(dtype!=np.dtype("float32") for _,dtype in (*resident_specs,*tangent_specs)):
                raise TesseraJitError("resident attention AD requires fp32 storage")
        from .rocm_typed_scaled_native import (
            requests_typed_scaled, requests_composed_typed_scaled, requests_floating_scaled,
        )
        native_scaled_frame = (
            normalize_target_kind(self.target) in {"rocm", "rocm_gfx1201"}
            and (requests_typed_scaled(self.graph_ir)
                 or requests_composed_typed_scaled(self.graph_ir)
                 or requests_floating_scaled(self.graph_ir)
                 or any(op.op_name == "tessera.scaled_matmul"
                        for fn in self.graph_ir.functions for op in fn.body))
        )
        if native_scaled_frame:
            from .paged_host_span import checked_host_span
            # Prove storage before tracing/certification can evaluate a borrowed
            # view. Native preparation later packs bytes and checks its ABI.
            for value in (*ordered, *tangent_values):
                checked_host_span(value, min_rank=1)
            if any(value.dtype != np.dtype("float32") for value in tangent_values):
                raise TesseraJitError("native scaled JVP tangents require fp32 storage")
        traced_module = self._specialized_autodiff_module(args, kwargs)
        from dataclasses import replace
        if normalize_target_kind(self.target) == "nvidia_sm120":
            traced_module = replace(traced_module,module_attrs={**traced_module.module_attrs,
                "tessera.target": '"nvidia_sm120"', "tessera.arch": '"sm_120"'})
        elif normalize_target_kind(self.target) in {"rocm", "rocm_gfx1201"}:
            from tessera import runtime as _runtime
            if (normalize_target_kind(self.target) == "rocm_gfx1201"
                    and _runtime._rocm_chip() != "gfx1201"):
                raise TesseraJitError("native JVP target requires gfx1201")
            traced_module = replace(traced_module,module_attrs={**traced_module.module_attrs,
                "tessera.target": '"rocm"', "tessera.arch": '"' + _runtime._rocm_chip() + '"'})
        graph_ops = [op for fn in traced_module.functions for op in fn.body]
        permitted_effect_ops: tuple[str, ...] = ()
        if len(graph_ops) == 1 and graph_ops[0].op_name == "tessera.dropout":
            dropout = graph_ops[0]
            if (not bool(dropout.kwargs.get("training", True)) or
                    float(dropout.kwargs.get("p", 0.5)) == 0.0 or
                    dropout.kwargs.get("seed") is not None):
                permitted_effect_ops = ("tessera.dropout",)
        if len(graph_ops)==1 and graph_ops[0].op_name=="tessera.flash_attn":
            policy=graph_ops[0].kwargs
            if float(policy.get("dropout_p",0.0))==0.0:
                permitted_effect_ops=("tessera.flash_attn",)
        resident_certificate=None
        if resident_attention:
            if (len(graph_ops)!=1 or graph_ops[0].op_name not in {"tessera.flash_attn","tessera.attention"}
                    or float(graph_ops[0].kwargs.get("dropout_p",0.0))!=0.0):
                raise TesseraJitError("resident JVP requires one zero-dropout native attention Graph")
            from .frontend_authority import certify_resident_frontends
            resident_certificate=certify_resident_frontends(
                legacy_module=self._ensure_legacy_graph_ir(),tracer_module=traced_module,
                signature=resident_specs,graph_consumers=(graph_ops[0].op_name,))
            self.last_frontend_differential=resident_certificate
        else:
            self.frontend_differential(
                *args, _permitted_effect_ops=permitted_effect_ops, **kwargs
            )
        from .native_vmap import mixed_batch_policies, normalize_mixed_batch_inputs
        if mixed_batch_policies(self):
            raw_seed_values = self._ordered_inputs(args, kwargs, normalize_batch=False)
            if raw_seed_values is None:
                raise TesseraJitError("native JVP requires all primal arguments")
            seed_values = list(raw_seed_values)
            for index, value in zip(request.wrt_indices, tangent_values, strict=True):
                if tuple(value.shape) != tuple(seed_values[index].shape):
                    raise TesseraJitError("native JVP primal and tangent shapes must match")
                seed_values[index] = value
            normalized_seeds = normalize_mixed_batch_inputs(seed_values, self._frontend_batch_policies)
            tangent_values = tuple(normalized_seeds[index] for index in request.wrt_indices)
        # The scaled program's checked native owner materializes primal and
        # tangent bytes. Keep alias views at this binding layer; other family
        # ABIs retain their existing compact host-frame contract.
        if resident_attention:
            from types import SimpleNamespace
            primal_inputs=[SimpleNamespace(shape=shape,dtype=dtype) for shape,dtype in resident_specs]
            tangent_inputs={index:SimpleNamespace(shape=shape,dtype=dtype)
                            for index,(shape,dtype) in zip(request.wrt_indices,tangent_specs,strict=True)}
        elif native_scaled_frame:
            # checked_host_span has already established ndarray storage.
            primal_inputs = list(ordered)
            tangent_inputs = dict(zip(request.wrt_indices, tangent_values, strict=True))
        else:
            primal_inputs = [np.ascontiguousarray(np.asarray(value)) for value in ordered]
            tangent_inputs = {
                index: np.ascontiguousarray(np.asarray(value))
                for index, value in zip(request.wrt_indices, tangent_values)
            }
        for index, tangent in tangent_inputs.items():
            if index >= len(primal_inputs) or primal_inputs[index].shape != tangent.shape:
                raise TesseraJitError("native JVP primal and tangent shapes must match")
        if len(graph_ops) != 1:
            from .rocm_typed_scaled_native import supports_composed_scale_jvp
            if (normalize_target_kind(self.target) not in {"rocm", "rocm_gfx1201"}
                    or not supports_composed_scale_jvp(traced_module, request.wrt_indices)):
                raise TesseraJitError("native JVP requires one operation or an admitted scaled product/sum Graph")
        source = (next(op for op in graph_ops if op.op_name == "tessera.scaled_matmul")
                  if len(graph_ops) != 1 else graph_ops[0])
        target = normalize_target_kind(self.target)
        if target == "rocm_gfx1201":
            target = "rocm"
        if target not in {"x86", "rocm", "nvidia_sm120"}:
            raise TesseraJitError(
                "native JVP is currently packaged for x86, ROCm, and exact sm120 only"
            )
        architecture = "zen5_avx512"
        execution_mode = "cpu_avx512"
        if target == "rocm":
            from tessera import runtime as _runtime
            from .native_jvp import architecture_admits
            from .native_jvp_plugins import native_jvp_plugin_declarations

            chip = _runtime._rocm_chip()
            declaration = native_jvp_plugin_declarations().get(
                source.op_name.removeprefix("tessera.")
            )
            family = declaration.family if declaration is not None else ""
            # Admission is per family and exact architecture; gfx1151 does
            # not inherit the gfx1201 scaled-product FP8 program.
            if not architecture_admits("rocm", chip, family):
                raise TesseraJitError(
                    f"native ROCm JVP requires exact gfx1151; detected {chip!r} "
                    f"(gfx1201 requires an exact family admission; rejected "
                    f"{family or source.op_name!r})"
                )
            architecture = chip
            execution_mode = "hip_runtime"
        elif target == "nvidia_sm120":
            architecture = "sm120"
            execution_mode = "cuda_runtime"

        launch_names = tuple(f"primal_{index}" for index in range(len(primal_inputs))) + tuple(
            f"tangent_{index}" for index in request.wrt_indices
        )
        launch_values = ((*ordered,*tangent_values) if resident_attention else
                         tuple(primal_inputs) + tuple(tangent_inputs[index] for index in request.wrt_indices))
        # Every member of a composed Graph participates in package identity;
        # the first product's policy alone cannot identify the native program.
        composed_graph_ir = traced_module.to_mlir(target=target) if len(graph_ops) > 1 else None
        package_key = (
            target,
            architecture,
            source.op_name,
            composed_graph_ir,
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
                    source_graph_ir=composed_graph_ir if composed_graph_ir is not None else traced_module.to_mlir(target=target),
                    paired_jvp_ir=paired_ir,
                    wrt_indices=request.wrt_indices,
                    arg_names=launch_names,
                    input_names=tuple(inspect.signature(self._fn).parameters),
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
                                f"rocm_{_rocm_chip()}"),
            "artifact_hash": package.artifact_hash,
            "paired_jvp_ir_digest": package.contract["paired_jvp_ir_digest"],
            "source_graph_ir_digest": package.contract["source_graph_ir_digest"],
            "schedule_program_digest": package.contract["schedule_program"]["digest"],
            "tile_program_digest": package.contract["tile_program"]["digest"],
            "frontend_authority": "tracer",
            "family": family,
            "host_preparation": ("native_ordered_resident_snapshot" if resident_attention else
                                 "native_checked_view_pack" if native_scaled_frame else "compact_host_frame"),
            **({"frontend_certificate":dict(resident_certificate.contract)} if resident_certificate is not None else {}),
        }
        primal, tangent = result["output"]
        return primal, tangent

    def native_backward_runtime_artifact(self):
        """Return the persisted native-family reverse product after execution."""
        artifact = getattr(self, "_native_backward_artifact", None)
        if artifact is None:
            raise TesseraJitError("no persisted native backward product from the last call")
        return artifact

    def native_backward(
        self, *args: Any, out_cotangents: Any, **kwargs: Any
    ) -> tuple[Any, ...]:
        """Compile and launch the native reverse product for the selected target.

        There is no NumPy fallback: missing tools, unsupported lowering, or ABI
        failure raises. Successful execution records the exact compiler path
        and invocation delta in ``last_backward_execution``.
        """
        self._native_backward_artifact = None
        self.last_backward_execution = None
        if self.differentiation_request is None:
            raise TesseraJitError("native_backward requires @jit(autodiff=...)")
        if self.differentiation_request.mode != "reverse":
            raise TesseraJitError(
                "native_backward requires @jit(autodiff='reverse'); "
                "forward requests use compiled_jvp_ir"
            )
        self._enforce_call_time_constraints(args, kwargs)
        source_module = self._specialized_autodiff_module(args, kwargs)
        target_kind = normalize_target_kind(self.target)
        # Lane selection below is per *family*; the capability registry, the
        # plugin lookup and the emitted Graph IR are per *chip*. Both forms reach
        # here legitimately — `pipeline_gates._normalize_target` documents the
        # same split — and only this comparison rejected the chip-qualified one,
        # so `@jit(target="rocm_gfx1151")` fell through to "native_backward
        # currently supports ..." even though the ROCm lane is exactly what it
        # asked for. Chip-precise names are what make a gfx1151-only capability
        # (the spectral adjoints) requestable without claiming it for every ROCm
        # part, so the family collapse belongs here and nowhere else.
        target_family = (
            "rocm" if target_kind.startswith("rocm")
            else "nvidia_sm120" if target_kind == "nvidia_sm120"
            else target_kind
        )
        graph_ops = [
            op for function in source_module.functions for op in function.body
        ]
        frontend_certificate = None
        composed_scale = False
        if len(graph_ops) > 1 and target_family == "rocm":
            from .rocm_typed_scaled_native import supports_scaled_reverse
            composed_scale = supports_scaled_reverse(
                source_module, self.differentiation_request.wrt_indices)
        if len(graph_ops) == 1 or composed_scale:
            plugin_source = (next(op for op in graph_ops if op.op_name == "tessera.scaled_matmul")
                             if composed_scale else graph_ops[0])
            from .native_vjp_plugins import (
                native_vjp_frontend_proof_policy,
                native_vjp_plugin_available,
            )

            if native_vjp_plugin_available(plugin_source.op_name, target_kind):
                proof_policy = native_vjp_frontend_proof_policy(
                    plugin_source.op_name, target_kind
                )
                certificate_roots=self._ordered_inputs(args,kwargs)
                cotangent_roots=(tuple(out_cotangents) if isinstance(out_cotangents,(tuple,list))
                                 else (out_cotangents,))
                resident_attention=any(hasattr(value,"__cuda_array_interface__")
                    for value in (*(certificate_roots or ()),*cotangent_roots))
                if resident_attention:
                    if (target_kind!="nvidia_sm120" or plugin_source.op_name!="tessera.flash_attn"
                            or float(plugin_source.kwargs.get("dropout_p",0.0))!=0.0
                            or certificate_roots is None or not all(
                                hasattr(value,"__cuda_array_interface__")
                                for value in (*certificate_roots,*cotangent_roots))):
                        raise TesseraJitError("resident reverse requires all CUDA roots and zero-dropout SM120 attention")
                    from .resident_nvidia_tensor import cuda_frontend_specs
                    from .frontend_authority import certify_resident_frontends
                    specs=cuda_frontend_specs(certificate_roots,ranks=(4,))
                    seed_specs=cuda_frontend_specs(cotangent_roots,ranks=(4,))
                    if any(dtype.name!="float32" for _,dtype in (*specs,*seed_specs)):
                        raise TesseraJitError("resident reverse requires fp32 roots and cotangent")
                    from dataclasses import replace
                    source_module=replace(source_module,module_attrs={
                        **source_module.module_attrs,"tessera.target":'"nvidia_sm120"',
                        "tessera.arch":'"sm_120"'})
                    frontend_certificate=certify_resident_frontends(
                        legacy_module=self._ensure_legacy_graph_ir(),tracer_module=source_module,
                        signature=specs,graph_consumers=(plugin_source.op_name,))
                    self.last_frontend_differential=frontend_certificate
                elif proof_policy == "non_reexecuting_state_lineage":
                    frontend_certificate = self._frontend_nonreexecuting_certificate(
                        args,
                        kwargs,
                        graph_consumers=(plugin_source.op_name,),
                    )
                else:
                    from .effects import infer_graph_effects
                    from .native_vjp_plugins import (
                        native_vjp_differential_effect_exemptions,
                        native_vjp_differential_safe,
                    )

                    source_effect, _ = infer_graph_effects(graph_ops)
                    if not native_vjp_differential_safe(
                        plugin_source, target_kind, source_effect.name
                    ):
                        raise TesseraJitError(
                            "native VJP plugin cannot safely run its frontend "
                            "differential certificate for this effect envelope"
                        )
                    exemptions = native_vjp_differential_effect_exemptions(
                        plugin_source, target_kind, source_effect.name
                    )
                    frontend_certificate = self.frontend_differential(
                        *args, _permitted_effect_ops=exemptions, **kwargs
                    )
                # The initial specialization is permitted to be the retained
                # AST compatibility candidate so we can discover its family.
                # Once a plugin claims the call, replace it with the certified
                # tracer module; never label the candidate as tracer authority.
                if not resident_attention:
                    source_module = self._traced_autodiff_module(args, kwargs)
                graph_ops = [
                    op
                    for function in source_module.functions
                    for op in function.body
                ]
        if composed_scale:
            from .rocm_typed_scaled_native import supports_scaled_reverse
            if not supports_scaled_reverse(
                    source_module, self.differentiation_request.wrt_indices):
                raise TesseraJitError("native scale VJP traced Graph differs from its admitted product/sum contract")
        if len(graph_ops) == 1 or composed_scale:
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
            if target_kind == "nvidia_sm120":
                from dataclasses import replace
                source_module = replace(
                    source_module,
                    module_attrs={**source_module.module_attrs,
                                  "tessera.target": '"nvidia_sm120"',
                                  "tessera.arch": '"sm_120"'},
                )
            if target_family == "rocm":
                from dataclasses import replace
                chip = target_kind.removeprefix("rocm_") if target_kind != "rocm" else _rocm_chip()
                source_module = replace(
                    source_module,
                    module_attrs={**source_module.module_attrs,
                                  "tessera.target": '"rocm"',
                                  "tessera.arch": '"' + chip + '"'},
                )
            plugin_result = execute_native_vjp_family(
                source=plugin_source,
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
                        f"rocm_{_rocm_chip()}" if target_kind == "rocm" else target_kind
                    ),
                ),
                frontend_certificate=frontend_certificate,
            )
            if plugin_result is not None:
                self.last_backward_execution = dict(plugin_result.execution)
                self._native_backward_artifact = plugin_result.runtime_artifact
                from .native_vmap import mixed_batch_policies, restore_mapped_gradient
                if mixed_batch_policies(self):
                    raw = self._ordered_inputs(args, kwargs, normalize_batch=False)
                    if raw is None:
                        raise TesseraJitError("native VJP requires all primal arguments")
                    return tuple(restore_mapped_gradient(
                        gradient, raw[index], self._frontend_batch_policies, index
                    ) for gradient, index in zip(
                        plugin_result.gradients, request.wrt_indices, strict=True
                    ))
                return plugin_result.gradients
        if target_family == "rocm":
            if len(graph_ops) == 1:
                raise TesseraJitError(
                    "no registered ROCm native VJP plugin for "
                    f"{plugin_source.op_name.removeprefix('tessera.')!r}"
                )
            raise TesseraJitError(
                "ROCm native backward requires one registered Graph op"
            )
        if target_family == "nvidia_sm120":
            if len(graph_ops) == 1:
                raise TesseraJitError(
                    "no registered NVIDIA SM120 native VJP plugin for "
                    f"{plugin_source.op_name.removeprefix('tessera.')!r}"
                )
            raise TesseraJitError(
                "NVIDIA SM120 native backward requires one registered Graph op"
            )
        if target_family == "x86":
            if len(graph_ops) == 1:
                raise TesseraJitError(
                    "no registered x86 native VJP plugin for "
                    f"{plugin_source.op_name.removeprefix('tessera.')!r}"
                )
            raise TesseraJitError(
                "x86 native backward currently requires one registered Graph op"
            )
        if target_family != "cpu":
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
            if arr.dtype == np.int32:
                # An index operand (target_verify's tokens). It sets no graph
                # element type; run_graph_ops declares it i32 and GraphFn refuses
                # it anywhere but an op's declared index operand (→ numpy).
                arrays[name] = np.ascontiguousarray(arr)
                continue
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
        return self.graph_ir.to_mlir(target=self._legality_target())

    def _legality_target(self) -> str:
        """The target name Graph IR legality is checked against for this jit:
        the string alias when one was given, else the CPU table."""
        return self.target if isinstance(self.target, str) else "cpu"

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
        plan_hash = None
        receipt = self._native_descriptor_last_receipt
        if (receipt is not None and receipt.get("ok") is True
                and receipt.get("execution_kind") == "native_gpu"
                and self.compile_bundle is not None and self.compile_bundle.executable
                and getattr(self._cached_artifact, "native_image", None) is not None):
            for name, stage in (
                ("graph_ir", self.compile_bundle.graph),
                ("schedule_ir", self.compile_bundle.schedule),
                ("tile_ir", self.compile_bundle.tile),
                ("target_ir", self.compile_bundle.target_ir),
            ):
                if stage is not None:
                    ir_hashes[name] = stage.output_digest
            cached_artifact = self._cached_artifact
            if cached_artifact is not None:
                plan_hash = cached_artifact.artifact_hash
        target_decision = {
            target_kind: (
                f"cpu_plan={self.cpu_plan.target_kind if self.cpu_plan else 'none'}; "
                f"compile_bundle.executable="
                f"{bool(self.compile_bundle and self.compile_bundle.executable)}"
            ),
        }
        if self._rocm_nvfp4_last_program is not None:
            target_decision["rocm_gfx1201"] = "canonical_rocm_nvfp4_program; three ordered native packages"
        if self._nvidia_lhs_last_program is not None:
            target_decision["nvidia_sm120"] = "canonical_nvidia_lhs_program; ordered checked native packages"
            receipts = self._nvidia_lhs_last_receipts
            if len(receipts) == len(self.native_lhs_packages()) and all(r.get("ok") and r.get("execution_kind") == "native_gpu"
                                         for r in receipts):
                import json
                program = self._nvidia_lhs_last_program
                ir_hashes = {"graph_ir": _cr.hash_ir_text(program.graph_ir)}
                # These are ordered product fingerprints, not a claim that
                # the two physical modules are one monolithic lowered IR.
                packages = self.native_lhs_packages()
                for layer, values in (
                    ("schedule_ir", [p.descriptor.provenance["schedule_digest"] for p in packages]),
                    ("tile_ir", [p.descriptor.provenance["tile_ir_digest"] for p in packages]),
                    ("target_ir", [p.image.target_ir_digest for p in packages]),
                ):
                    ir_hashes[layer] = _cr.hash_ir_text(json.dumps(values,separators=(",",":")))
                plan_hash = self.runtime_artifact().artifact_hash
                target_decision["nvidia_sm120"] += (
                    "; paired_native_packages.executable=True; stage_hashes=ordered(producer,consumer)")
        if self._nvidia_rhs_last_program is not None:
            target_decision["nvidia_sm120"] = "canonical_nvidia_rhs_program; two ordered checked native packages"
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
            plan_hash=plan_hash,
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
        if (self._nvidia_lhs_last_program is not None
                and len(self._nvidia_lhs_last_receipts) == len(self.native_lhs_packages())
                and all(r.get("ok") and r.get("execution_kind") == "native_gpu"
                        for r in self._nvidia_lhs_last_receipts)):
            return "native_gpu"
        if len(getattr(self, "_nvidia_rhs_last_receipts", ())) == 2:
            return "native_gpu"
        if self._rocm_nvfp4_last_receipts:
            return "native_gpu"
        if getattr(self, "_native_storage_pair", None) is not None or getattr(self, "_apple_native_arena", None) is not None or getattr(self, "_native_storage_call", None) is not None or getattr(self, "_native_storage_jvp", None) is not None:
            return "native_gpu"
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

        ingest_program = getattr(self, "_rocm_nvfp4_last_program", None)
        if ingest_program is not None:
            from .rocm_nvfp4_program import runtime_artifact
            return runtime_artifact(ingest_program)
        lhs_program = getattr(self, "_nvidia_lhs_last_program", None)
        if lhs_program is not None:
            from .nvidia_tensor_lhs import runtime_artifact
            return runtime_artifact(lhs_program)
        rhs_program = getattr(self, "_nvidia_rhs_last_program", None)
        if rhs_program is not None:
            from .nvidia_tensor_rhs import rhs_runtime_artifact
            return rhs_runtime_artifact(rhs_program)
        native_pair = getattr(self, "_native_storage_pair", None)
        if native_pair is not None:
            return RuntimeArtifact(tile_ir=native_pair.package.arena_ir,
                metadata={"target": getattr(native_pair.package, "backend", "apple"), "execution_kind": "native_gpu",
                    "executable": True, "compiler_path": "explicit_native_storage_pair",
                    "package_digest": native_pair.package.binding_digest, "mode": native_pair.contract['mode'],
                    "automatic_selection": False}, abi_signature="tessera.native_storage_pair.v1")
        apple = getattr(self, "_apple_native_arena", None)
        if apple is not None:
            return RuntimeArtifact(tile_ir=apple.package.arena_ir,
                metadata={"target": "apple", "execution_kind": "native_gpu", "executable": True,
                    "compiler_path": "explicit_apple_native_arena", "runtime_status": "ready",
                    "package_digest": apple.package.binding_digest, "python_equivalence": "caller_declared",
                    "automatic_selection": False}, abi_signature="tessera.apple_native_arena.v1")
        pair = getattr(self, "_native_storage_jvp", None)
        if pair is not None:
            return RuntimeArtifact(metadata=pair.artifact.runtime_metadata())
        binding = getattr(self, "_native_storage_call", None)
        if binding is not None:
            return RuntimeArtifact(
                tile_ir=binding.package.arena_ir,
                metadata={"target": binding.package.backend, "execution_kind": "native_gpu",
                    "executable": True, "compiler_path": "explicit_native_storage",
                    "runtime_status": "ready", "binding_digest": binding.binding_digest,
                    "package_digest": binding.package.binding_digest,
                    "python_equivalence": "caller_declared", "automatic_selection": False},
                abi_signature="tessera.native_storage.v1")

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

        # Verify the render against THIS jit's target: the CPU default rejects
        # storage dtypes only the GPU carries (fp8/fp4 on Apple GPU).
        graph_ir_text = self.graph_ir.to_mlir(target=self._legality_target())
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
        from .graph_ir import _loc_path, _scalar_const_env
        source_location = (
            f"{source_origin}:{_loc_path(fn.__code__.co_filename)}:{fn.__code__.co_firstlineno}"
        )
        constant_environment = repr(sorted(_scalar_const_env(fn).items()))
        module = _gic.lookup(
            source_text, effect_tag=effect_tag, target_attr=target_attr,
            source_location=source_location,
            constant_environment=constant_environment)
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
                constant_environment=constant_environment,
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
    source_control_flow: bool = False,
    source_mutable: tuple[int, ...] = (),
    source_fields: tuple = (),
    source_error_specs: tuple = (),
    source_max_steps: int | None = None,
    deterministic: bool = False,
    seed: Optional[int] = None,
    bindings: Optional[Dict[str, int]] = None,
    target: Optional[Any] = None,
    attn_config: Optional[FlashAttnLoweringConfig] = None,
    cpu_tile: Tuple[int, int, int] = (128, 128, 64),
    shape_bounds: Optional[Dict[str,int]] = None,
    rhs_storage_order: Optional[str] = None,
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

    source_control_flow=True selects the bounded native CPU source consumer.
    source_mutable declares input state slots, source_error_specs declares
    floating result shapes for builtin exception transport, and source_max_steps
    bounds recovered loops. This opt-in owner must be closed or used as a context
    manager; it retains at most four native shape/alias specializations.

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
        rhs_storage_order : optional explicit physical row_major/col_major RHS
            for a bounded named tensor program; default preserves column-major packing.
        shape_bounds : optional M/N/K maximum extents for the named straight-line
                       primal SM120 normalization/softmax -> matmul route.
                       Active shapes reuse checked native packages; unspecified
                       axes and storage remain specialization keys.
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

    if rhs_storage_order is not None and (
            (type(rhs_storage_order) is not str or rhs_storage_order not in {"row_major","col_major"}) or shape_bounds is None):
        raise ValueError("rhs_storage_order requires bounded named tensor JIT and row_major/col_major")
    if shape_bounds is not None:
        from .bounded_nvidia_lhs import validate_bounds
        shape_bounds=dict(validate_bounds(shape_bounds))
        if normalize_target_kind(target)!="nvidia_sm120" or autodiff is not None or wrt is not None or source_control_flow:
            raise ValueError("shape_bounds currently requires primal nvidia_sm120 named tensor programs")

    if type(source_control_flow) is not bool:
        raise ValueError('source_control_flow must be boolean')
    if not source_control_flow and (source_mutable or source_fields or source_error_specs or source_max_steps is not None):
        raise ValueError('source options require source_control_flow=True')

    def _decorate(fn: Callable) -> Any:
        if source_control_flow:
            if (target not in (None,'cpu','x86') or deterministic or seed is not None or bindings
                    or attn_config is not None or cpu_tile != (128,128,64) or source is not None or source_path is not None
                    or autodiff is not None or wrt is not None or auto_batch is not None or max_ops_per_cb is not None
                    or emit_package or dispatch_via_package is not None or phase is not None or slo is not None):
                raise ValueError('native source JIT requires its explicit CPU source contract; incompatible options supplied')
            from .native_source_state import NativeSourceJit
            return NativeSourceJit(fn,mutable=source_mutable,error_specs=source_error_specs,max_steps=source_max_steps,object_fields=source_fields)
        source_text, source_origin = _resolve_source_text(
            fn,
            source=source,
            source_path=source_path,
        )

        bounded_certificate=None
        if shape_bounds is not None:
            from .bounded_nvidia_lhs import SourceCertificate
            bounded_certificate=SourceCertificate(fn,source_text)

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
            shape_bounds=shape_bounds,
            bounded_source_certificate=bounded_certificate,
            bounded_rhs_storage_order=rhs_storage_order,
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
