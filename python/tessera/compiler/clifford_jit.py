"""``@clifford_jit(target="apple_gpu")`` — one true compiler-integrated
GA vertical slice.

Closes the compiler/runtime gap end-to-end for a constrained class of
Clifford-only Python functions.  The contract:

  1. **Python entry point.**  A function decorated with
     ``@clifford_jit(target="apple_gpu")``.  The decorator pins a
     small constrained vocabulary (``tessera.ga.*`` ops only) and
     rejects anything else at trace time.
  2. **Op plan.**  Decoration runs the function once under
     ``jit_bridge.jit_context(target)`` with tracing on.  The bridge
     trace becomes the **canonical op plan** — an ordered list of
     ``CliffordOpPlanEntry(op_name, target, status, symbol)`` rows.
  3. **Apple target metadata.**  The wrapped function carries
     :class:`CliffordCompiledArtifact` with the plan, the target
     name (``"apple_gpu"``), the dtype, the manifest sources, and a
     hash of the trace for cache-keyed reuse.
  4. **Runtime dispatch.**  Subsequent calls execute the function
     body inside a fresh ``jit_context(target)`` so every public-API
     GA call routes through the bridge → manifest → shared loader
     path and produces a recorded route.  The artifact lets callers
     pull ``last_routes()`` to prove the executed routes match the
     traced plan.

Why this is the meaningful "compiler gap closed" step:

- Today every `tessera.ga.*` call already routes through the bridge,
  but there's no notion of a **traced + verified function plan**.
- This module adds a 1-call ``CliffordCompiledArtifact`` whose
  presence + match against the manifest is a compile-time guarantee
  that every op in the function has a fused MSL kernel.
- The artifact is also the natural surface for a future AOT exporter
  (the plan is a serializable IR — already JSON-friendly).
- It demonstrates the **Python → manifest → Apple target metadata →
  runtime → benchmark** vertical without committing to the larger
  ``@tessera.jit`` tensor-op machinery (which is tightly coupled to
  ``OP_SPECS``).

Scope of v1:

- Only Cl(3,0) f32 functions.
- Only the 17 GA ops + manifest-resolvable subsets thereof — any
  non-GA call raises :class:`CliffordJitError` at decoration.
- The function may take ``Multivector`` or ``MultivectorField``
  inputs and return any of those or a numpy array (a norm/inner
  scalar).
- ``target="apple_gpu"`` only.  ``"x86"`` / ``"nvidia"`` are reserved
  for future targets; passing them today raises with a precise
  error.
"""

from __future__ import annotations

import functools
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from tessera.compiler import jit_bridge as _bridge
from tessera.compiler import ast_ir as _ast_ir


__all__ = [
    "CliffordOpPlanEntry",
    "CliffordCompiledArtifact",
    "CliffordIRProgram",
    "CliffordIROpCall",
    "CliffordJitError",
    "CliffordCompiledCallable",
    "lower_function_to_ir",
    "clifford_jit",
]


_SUPPORTED_TARGETS = frozenset({"apple_gpu"})


# Whitelist of `tessera.ga.*` attribute names this lowering accepts.
# Each entry maps the public Python name (as it appears in the user's
# source) to its manifest op name (the canonical clifford_*).  Adding
# a new op here is sufficient to extend the AST → IR vocabulary.
_GA_ATTR_TO_OP_NAME: dict[str, str] = {
    "geometric_product":  "clifford_geometric_product",
    "grade_projection":   "clifford_grade_projection",
    "wedge":              "clifford_wedge",
    "left_contraction":   "clifford_left_contraction",
    "inner":              "clifford_inner",
    "reverse":            "clifford_reverse",
    "grade_involution":   "clifford_grade_involution",
    "conjugate":          "clifford_conjugate",
    "norm":               "clifford_norm",
    "norm_squared":       "clifford_norm",  # squared norm decomposes; v1
                                              # routes through norm for the
                                              # bridge-traced symbol.  Note:
                                              # the math is still ||a||^2; the
                                              # IR records the canonical norm
                                              # op for the artifact-level plan.
    "exp_mv":             "clifford_exp",
    "log_mv":             "clifford_log",
    "rotor_sandwich":     "clifford_rotor_sandwich",
    "hodge_star":         "clifford_hodge_star",
}


class CliffordJitError(Exception):
    """Raised when a ``@clifford_jit`` function violates the v1
    contract — e.g., it uses an op with no fused manifest entry, or
    a target other than ``"apple_gpu"``.

    Carries an optional stable :class:`ConstrainedDiagnosticCode` so
    callers (CI, ``.explain()``) can match against the code rather
    than the free-form message text.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        source_position: Any = None,
    ) -> None:
        self.code = code
        self.source_position = source_position
        super().__init__(message)

    def to_diagnostic(self) -> Any:
        """Lift this exception into a
        :class:`tessera.compiler.diagnostics.Diagnostic` for
        ``.explain()`` consumption.  Lane is always ``clifford_jit``."""

        from .diagnostics import (
            ConstrainedDiagnosticCode,
            Diagnostic,
        )

        code = self.code or ConstrainedDiagnosticCode.CLIFFORD_OP_NOT_WHITELISTED.value
        return Diagnostic.from_constrained(
            code=code,
            message=str(self),
            lane="clifford_jit",
            source_position=self.source_position,
        )


# ---------------------------------------------------------------------------
# Graph IR types — small but real: function args + a sequence of op
# calls + a return ref.  This replaces the previous trace-capture
# approach with actual AST → IR lowering.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CliffordIROpCall:
    """One op call in the IR.

    ``op_name`` is the canonical manifest op (e.g., ``clifford_rotor_sandwich``).
    ``operand_refs`` are SSA-style names — either function-argument
    names or auto-generated intermediates (``%t0``, ``%t1``, ...).
    ``result_name`` is the SSA name for this op's output.
    ``python_attr`` is the original ``tessera.ga.X`` attribute name
    for debuggability (e.g., ``"rotor_sandwich"``).
    """
    op_name: str
    operand_refs: tuple[str, ...]
    result_name: str
    python_attr: str = ""


@dataclass(frozen=True)
class CliffordIRProgram:
    """A lowered Clifford-only function.

    The IR is structured (function signature + ordered op-call body +
    return ref).  It's stable enough to be JSON-serialized + used as
    the cache key for re-execution + future AOT export.
    """
    arg_names: tuple[str, ...]
    ops: tuple[CliffordIROpCall, ...]
    return_ref: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "arg_names": list(self.arg_names),
            "ops": [
                {"op": c.op_name, "operands": list(c.operand_refs),
                 "result": c.result_name, "python_attr": c.python_attr}
                for c in self.ops
            ],
            "return_ref": self.return_ref,
        }

    def text(self) -> str:
        """Pretty-printed IR — debug-friendly form."""
        lines = [f"clifford_ir({', '.join(self.arg_names)}):"]
        for c in self.ops:
            lines.append(
                f"  {c.result_name} = {c.op_name}({', '.join(c.operand_refs)})"
                + (f"  # ga.{c.python_attr}" if c.python_attr else "")
            )
        lines.append(f"  return {self.return_ref}")
        return "\n".join(lines)


# M6 Step 1 (2026-05-18): the previous ``_ASTLowerer`` lived here
# but has been generalized into ``tessera.compiler.ast_ir``.  The
# Clifford-specific config (namespace="ga", whitelist, error class)
# is passed in by :func:`lower_function_to_ir` below.


_CLIFFORD_LOWERING_CONFIG = _ast_ir.LoweringConfig(
    namespace="ga",
    attr_to_op_name=_GA_ATTR_TO_OP_NAME,
    error_prefix="clifford_jit",
    error_class=CliffordJitError,
)


def lower_function_to_ir(fn: Callable[..., Any]) -> CliffordIRProgram:
    """Lower a Clifford-only function to :class:`CliffordIRProgram`
    via Python AST walking.

    **M6 Step 1 (2026-05-18):** delegates to the shared
    :func:`tessera.compiler.ast_ir.lower_function`.  The
    Clifford-specific types stay backwards-compatible (same fields,
    same ``text()`` prefix) — only the lowering pass is shared.
    """
    generic = _ast_ir.lower_function(
        fn, _CLIFFORD_LOWERING_CONFIG,
    )
    return CliffordIRProgram(
        arg_names=generic.arg_names,
        ops=tuple(
            CliffordIROpCall(
                op_name=op.op_name,
                operand_refs=op.operand_refs,
                result_name=op.result_name,
                python_attr=op.python_attr,
            )
            for op in generic.ops
        ),
        return_ref=generic.return_ref,
    )


def _execute_ir(
    ir: CliffordIRProgram, args: tuple[Any, ...],
    target: str,
) -> tuple[Any, tuple[_bridge.JitBridgeRoute, ...]]:
    """Walk the IR with concrete operands, dispatching each op via
    the JIT bridge.  Returns ``(result, routes)``.

    The op dispatch happens through ``tessera.ga.<python_attr>``
    (the public API), which is itself routed through the bridge.
    We open a ``jit_context(target)`` span so the recorded routes
    carry the JIT context tag.
    """
    if len(args) != len(ir.arg_names):
        raise CliffordJitError(
            f"clifford_jit IR expects {len(ir.arg_names)} args "
            f"({list(ir.arg_names)}); got {len(args)}"
        )
    # Import the ga namespace lazily — keeps this module light if
    # imported by code that never decorates a GA function.
    import tessera.ga as ga
    env: dict[str, Any] = dict(zip(ir.arg_names, args))

    def _resolve(ref: str) -> Any:
        # M6 Step 1: literal decoding lives in the shared ast_ir
        # module so a future energy_jit reuses the same encoding.
        try:
            return _ast_ir.resolve_operand(ref, env)
        except KeyError as exc:
            raise CliffordJitError(f"executor: {exc}") from exc

    prev_tracing = _bridge.tracing_enabled()
    _bridge.set_tracing_enabled(True)
    _bridge.clear_dispatch_trace()
    try:
        with _bridge.jit_context(target):
            for op in ir.ops:
                # Resolve the Python callable via the public ga.* API.
                fn = getattr(ga, op.python_attr, None)
                if fn is None:
                    raise CliffordJitError(
                        f"executor: tessera.ga has no attribute "
                        f"{op.python_attr!r} (mapped from "
                        f"op_name={op.op_name!r})"
                    )
                operands = [_resolve(ref) for ref in op.operand_refs]
                env[op.result_name] = fn(*operands)
            result = _resolve(ir.return_ref)
    finally:
        routes = tuple(_bridge.take_dispatch_trace())
        _bridge.set_tracing_enabled(prev_tracing)
    return result, routes


@dataclass(frozen=True)
class CliffordOpPlanEntry:
    """One entry in the lowered op plan."""
    op_name: str
    target: str
    status: str
    symbol: str


@dataclass(frozen=True)
class CliffordCompiledArtifact:
    """The Apple target metadata produced at decoration time.

    Attributes
    ----------
    plan : tuple of :class:`CliffordOpPlanEntry`
        Ordered op plan — the canonical IR for the function body.
        Captured from the bridge trace during the one-shot trace
        pass at decoration.
    target : str
        Target name (``"apple_gpu"`` for v1).
    dtype : str
        Canonical dtype string (``"f32"`` for v1).
    manifest_sources : tuple[str, ...]
        Manifest tables consulted (``_CLIFFORD_APPLE_GPU_FUSED`` and
        any others).
    plan_hash : str
        16-char sha256 prefix of the canonical-op-name sequence.
        Used as a cache key for re-execution + AOT export round-trips.
    source_name : str
        ``fn.__qualname__`` of the decorated function (for
        diagnostics + telemetry).
    """
    plan: tuple[CliffordOpPlanEntry, ...]
    target: str
    dtype: str
    manifest_sources: tuple[str, ...] = field(default_factory=tuple)
    plan_hash: str = ""
    source_name: str = ""
    ir: Optional[CliffordIRProgram] = None

    def op_names(self) -> tuple[str, ...]:
        return tuple(e.op_name for e in self.plan)

    def symbols(self) -> tuple[str, ...]:
        return tuple(e.symbol for e in self.plan)

    def as_metadata(self) -> dict[str, Any]:
        """JSON-friendly serialization for benchmark reports + AOT."""
        meta: dict[str, Any] = {
            "target": self.target,
            "dtype": self.dtype,
            "source_name": self.source_name,
            "plan_hash": self.plan_hash,
            "manifest_sources": list(self.manifest_sources),
            "plan": [
                {"op": e.op_name, "target": e.target,
                 "status": e.status, "symbol": e.symbol}
                for e in self.plan
            ],
        }
        if self.ir is not None:
            meta["ir"] = self.ir.as_metadata()
        return meta


def _hash_plan(plan: tuple[CliffordOpPlanEntry, ...]) -> str:
    """Stable hash of the op-name sequence (target-independent)."""
    canonical = "|".join(f"{e.op_name}:{e.symbol}" for e in plan)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class CliffordCompiledCallable:
    """The decorated function, with the artifact + a per-call route
    capture API.

    Calling the instance is identical to calling the original
    function, but every call also captures the route trace so callers
    can prove the execution stayed on the planned path.  Use
    :meth:`last_routes` to read the per-call trace after execution.
    """

    __slots__ = ("_fn", "artifact", "_last_routes_lock", "_last_routes")

    def __init__(
        self,
        fn: Callable[..., Any],
        artifact: CliffordCompiledArtifact,
    ) -> None:
        self._fn = fn
        self.artifact = artifact
        self._last_routes_lock = threading.Lock()
        self._last_routes: tuple[_bridge.JitBridgeRoute, ...] = ()
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the function under a ``jit_context`` span so every
        op routes through the bridge and the trace is captured.

        Auto-emits a CompileReport to the active sink (if any) —
        see :func:`compile_report.capture_compile_reports`.
        """
        # Snapshot the bridge state so we can restore it afterwards.
        prev_tracing = _bridge.tracing_enabled()
        _bridge.set_tracing_enabled(True)
        _bridge.clear_dispatch_trace()
        try:
            with _bridge.jit_context(self.artifact.target):
                result = self._fn(*args, **kwargs)
        finally:
            captured = tuple(_bridge.take_dispatch_trace())
            _bridge.set_tracing_enabled(prev_tracing)
        with self._last_routes_lock:
            self._last_routes = captured
        # Step 4 auto-emit — no-op when no sink is active.
        from . import compile_report as _cr  # local to avoid import cycle
        if _cr.active_sink_is_capturing():
            _cr.emit_compile_report(self.compile_report())
        return result

    def last_routes(self) -> tuple[_bridge.JitBridgeRoute, ...]:
        """The route trace from the most-recent call on this
        thread.  Empty before the first call.

        Each route's ``op_name`` should be one of the
        artifact's planned ops; mismatches indicate the function
        diverged from its traced plan (e.g., a branch that wasn't
        exercised at decoration)."""
        with self._last_routes_lock:
            return self._last_routes

    def plan_matches_routes(self,
                              routes: Optional[tuple[_bridge.JitBridgeRoute, ...]] = None
                              ) -> bool:
        """``True`` iff the route ``op_name`` sequence equals the
        artifact's plan (order matters)."""
        observed = self.last_routes() if routes is None else routes
        observed_ops = tuple(r.op_name for r in observed)
        return observed_ops == self.artifact.op_names()

    def compile_report(self) -> "Any":  # CompileReport forward-ref; runtime-imported inside body
        """Synthesize a :class:`CompileReport` from this callable's
        current artifact + last call's route trace.

        Step 4 of the 2026-05-18 post-reassessment plan: every JIT
        frontend (``@tessera.jit``, textual, ``@clifford_jit``)
        exposes a uniform CompileReport accessor so the M5
        no-silent-native rule applies everywhere.
        """
        from . import compile_report as _cr  # local import to avoid cycle
        ir = self.artifact.ir
        ir_text = ir.text() if ir is not None else ""
        ir_hashes = (
            {"graph_ir": _cr.hash_ir_text(ir_text)} if ir_text else {}
        )
        return _cr.CompileReport(
            program_id=self.artifact.source_name,
            source=f"@clifford_jit({self.artifact.source_name})",
            frontend=_cr.FRONTEND_CLIFFORD_JIT,
            value_kind=_cr.VALUE_KIND_MULTIVECTOR,
            target=self.artifact.target,
            ir_hashes=ir_hashes,
            target_decision={
                self.artifact.target: (
                    f"plan_hash={self.artifact.plan_hash} via "
                    f"{', '.join(self.artifact.manifest_sources) or 'manifest'}"
                ),
            },
            proof_routes=self.last_routes(),
        )


def _validate_target(target: str) -> None:
    if target not in _SUPPORTED_TARGETS:
        from .diagnostics import ConstrainedDiagnosticCode as _Code
        raise CliffordJitError(
            f"clifford_jit target must be one of {sorted(_SUPPORTED_TARGETS)}, "
            f"got {target!r}",
            code=_Code.CLIFFORD_UNSUPPORTED_TARGET.value,
        )


def _validate_plan(plan: tuple[CliffordOpPlanEntry, ...]) -> None:
    """Every traced op must have an ``apple_gpu=fused`` manifest entry.

    This is the **compile-time guarantee**: the wrapped function won't
    raise at runtime due to a missing kernel because every op the
    function calls was manifest-verified at decoration.
    """
    from .diagnostics import ConstrainedDiagnosticCode as _Code
    if not plan:
        raise CliffordJitError(
            "clifford_jit traced an empty op plan — the function did "
            "not call any tessera.ga.* op that routes through the "
            "JIT bridge.  Add at least one supported call (inner, "
            "norm, exp_mv, rotor_sandwich, ...).",
            code=_Code.CLIFFORD_EMPTY_OP_PLAN.value,
        )
    for entry in plan:
        if entry.target != "apple_gpu":
            raise CliffordJitError(
                f"op {entry.op_name!r} routed to target {entry.target!r}; "
                f"clifford_jit(target='apple_gpu') requires every op to "
                f"route to apple_gpu",
                code=_Code.CLIFFORD_TARGET_MISMATCH.value,
            )
        if entry.status != "fused":
            raise CliffordJitError(
                f"op {entry.op_name!r} manifest status is "
                f"{entry.status!r} — clifford_jit requires every op "
                f"to be 'fused' on the target",
                code=_Code.CLIFFORD_TARGET_MISMATCH.value,
            )
        if not entry.op_name.startswith("clifford_"):
            raise CliffordJitError(
                f"op {entry.op_name!r} is not a clifford_* op; "
                f"clifford_jit's v1 scope is Cl(3,0) GA primitives only")


def clifford_jit(
    *, target: str = "apple_gpu", dtype: str = "f32",
    native_required: bool = False,
) -> Callable[[Callable[..., Any]], CliffordCompiledCallable]:
    """Decorator: compile a constrained Clifford-only function to an
    Apple-GPU op plan.

    Parameters
    ----------
    target
        Target backend.  Only ``"apple_gpu"`` is supported in v1.
    dtype
        Element dtype.  Only ``"f32"`` is supported in v1.
    native_required
        **M3 deliverable.**  When ``True``, any condition that would
        otherwise cause a runtime fallback (non-Darwin host, Metal
        unavailable, manifest miss, shape envelope rejection) raises
        :class:`fallback.TesseraNativeRequiredError` at call time with
        a stable :class:`fallback.FallbackReason` code.  When
        ``False`` (the default), the existing behavior is preserved:
        the AST→IR validation still runs at decoration time, but
        runtime conditions can route through the reference path.

    .. code-block:: python

        @clifford_jit(target="apple_gpu")
        def point_cloud_rotor_invariant(rotor, points):
            return ga.norm(ga.rotor_sandwich(rotor, points))

        f = point_cloud_rotor_invariant   # CliffordCompiledCallable
        f.artifact.plan_hash               # stable identifier
        f.artifact.op_names()              # ('clifford_rotor_sandwich', 'clifford_norm')
        out = f(rotor, points)             # GPU dispatch, route-traced
        f.last_routes()                    # per-call route trace
        f.plan_matches_routes()            # True iff routes == plan

    The decorator's lazy form lets the caller supply trace inputs
    via ``CliffordCompiledCallable.compile(...)`` when defaults
    aren't possible (e.g., shape-dependent functions).
    """
    _validate_target(target)
    if dtype != "f32":
        raise CliffordJitError(
            f"clifford_jit v1 only supports dtype='f32', got {dtype!r}")

    def decorator(fn: Callable[..., Any]) -> CliffordCompiledCallable:
        # Preferred path: AST → CliffordIRProgram at decoration time.
        # Validates every op against the manifest before any call runs
        # — that's the real "compile-time guarantee" the v1 contract
        # promises.  If the function's source isn't readable
        # (REPL / exec / lambda defined via dynamic codegen) we fall
        # back to the trace-capture variant.
        wrapper: Any
        try:
            ir = lower_function_to_ir(fn)
        except CliffordJitError as exc:
            if "cannot read source" in str(exc):
                wrapper = _LazyCompiledCallable(
                    fn, target=target, dtype=dtype,
                    native_required=native_required,
                )
                functools.update_wrapper(wrapper, fn)
                return wrapper
            raise
        wrapper = _IRCompiledCallable(
            fn, target=target, dtype=dtype, ir=ir,
            native_required=native_required,
        )
        functools.update_wrapper(wrapper, fn)
        return wrapper

    return decorator


def _manifest_sources_for_plan(
    plan: tuple[CliffordOpPlanEntry, ...],
) -> tuple[str, ...]:
    return tuple(sorted(set(
        "_CLIFFORD_APPLE_GPU_FUSED" if e.op_name.startswith("clifford_")
        else "_EBM_APPLE_GPU_FUSED" if e.op_name.startswith("ebm_")
        else "_UNKNOWN"
        for e in plan
    )))


class _IRCompiledCallable(CliffordCompiledCallable):
    """AST-lowered variant: compiles at decoration time via
    :func:`lower_function_to_ir`, validates every op against the
    manifest, then executes by walking the IR (not by re-running the
    Python source).

    This replaces the previous trace-capture path for source-readable
    functions: the plan is known *before* the first call, the IR is
    the canonical record of the function body, and execution dispatches
    each op directly via the JIT bridge.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        target: str,
        dtype: str,
        ir: CliffordIRProgram,
        native_required: bool = False,
    ) -> None:
        plan_entries: list[CliffordOpPlanEntry] = []
        for op in ir.ops:
            symbol = _bridge.lookup_apple_gpu_symbol(op.op_name)
            if symbol is None:
                raise CliffordJitError(
                    f"clifford_jit({fn.__qualname__}): op {op.op_name!r} "
                    f"(from ga.{op.python_attr}) has no apple_gpu manifest "
                    "entry"
                )
            plan_entries.append(CliffordOpPlanEntry(
                op_name=op.op_name,
                target=target,
                status="fused",
                symbol=symbol,
            ))
        plan = tuple(plan_entries)
        _validate_plan(plan)
        artifact = CliffordCompiledArtifact(
            plan=plan,
            target=target,
            dtype=dtype,
            manifest_sources=_manifest_sources_for_plan(plan),
            plan_hash=_hash_plan(plan),
            source_name=fn.__qualname__,
            ir=ir,
        )
        super().__init__(fn, artifact)
        self._target = target
        self._dtype = dtype
        self._native_required = bool(native_required)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise CliffordJitError(
                f"clifford_jit({self.artifact.source_name}): keyword "
                "arguments are not supported by the AST lowering"
            )
        # M2 Step 3: every positional arg to a clifford_jit function
        # must be a Multivector or a literal already encoded in the
        # IR.  Reject anything else with a source-span diagnostic so
        # users see exactly which arg has the wrong kind.
        from .. import bridge as _bridge_kinds
        if self.artifact.ir is not None:
            arg_names = self.artifact.ir.arg_names
            _bridge_kinds.check_call_kinds(
                args,
                expected="multivector",
                op_name=f"@clifford_jit({self.artifact.source_name})",
                arg_names=arg_names,
            )
        if self._native_required:
            # M3: refuse to dispatch if the host can't actually run
            # native MSL kernels — the alternative (silent reference
            # fallback inside `tessera.ga.*`) is exactly what
            # native_required exists to prevent.
            from .fallback import (
                TesseraNativeRequiredError,
                classify_host,
            )
            import sys as _sys
            is_darwin = _sys.platform == "darwin"
            runtime_ok = True
            if is_darwin:
                try:
                    from .. import _apple_gpu_dispatch as _agd
                    runtime_ok = _agd.apple_gpu_available()
                except Exception:
                    runtime_ok = False
            host_fail = classify_host(
                is_darwin=is_darwin, runtime_available=runtime_ok,
            )
            if host_fail is not None:
                raise TesseraNativeRequiredError(
                    host_fail,
                    target=self.artifact.target,
                    op_name=self.artifact.source_name,
                )
        assert self.artifact.ir is not None  # set by __init__
        result, routes = _execute_ir(
            self.artifact.ir, args, self.artifact.target,
        )
        with self._last_routes_lock:
            self._last_routes = routes
        # Step 4 auto-emit — no-op when no sink is active.
        from . import compile_report as _cr
        if _cr.active_sink_is_capturing():
            _cr.emit_compile_report(self.compile_report())
        return result


class _LazyCompiledCallable(CliffordCompiledCallable):
    """First-call trace variant of :class:`CliffordCompiledCallable`.

    Postpones the artifact build until the first invocation, where
    real input arguments are available.  After that call the
    artifact is frozen and subsequent calls behave like a normal
    :class:`CliffordCompiledCallable`.
    """

    def __init__(self, fn: Callable[..., Any], *,
                 target: str, dtype: str,
                 native_required: bool = False) -> None:
        # Placeholder artifact — replaced on first call.
        placeholder = CliffordCompiledArtifact(
            plan=(), target=target, dtype=dtype,
            manifest_sources=(), plan_hash="", source_name=fn.__qualname__,
        )
        super().__init__(fn, placeholder)
        self._target = target
        self._dtype = dtype
        self._compiled = False
        self._compile_lock = threading.Lock()
        self._native_required = bool(native_required)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._native_required:
            from .fallback import (
                TesseraNativeRequiredError, classify_host,
            )
            import sys as _sys
            is_darwin = _sys.platform == "darwin"
            runtime_ok = True
            if is_darwin:
                try:
                    from .. import _apple_gpu_dispatch as _agd
                    runtime_ok = _agd.apple_gpu_available()
                except Exception:
                    runtime_ok = False
            host_fail = classify_host(
                is_darwin=is_darwin, runtime_available=runtime_ok,
            )
            if host_fail is not None:
                raise TesseraNativeRequiredError(
                    host_fail,
                    target=self.artifact.target,
                    op_name=self.artifact.source_name,
                )
        if not self._compiled:
            with self._compile_lock:
                if not self._compiled:
                    self._compile(args, kwargs)
        return super().__call__(*args, **kwargs)

    def _compile(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Run the function once with the caller's arguments, capture
        the route trace, validate it, and freeze the artifact."""
        prev_tracing = _bridge.tracing_enabled()
        _bridge.set_tracing_enabled(True)
        _bridge.clear_dispatch_trace()
        try:
            with _bridge.jit_context(self._target):
                # Execute solely to capture the op plan — output
                # discarded; the caller's actual call (below in
                # super().__call__) will re-run for the real result.
                self._fn(*args, **kwargs)
        finally:
            captured = tuple(_bridge.take_dispatch_trace())
            _bridge.set_tracing_enabled(prev_tracing)
        plan = tuple(
            CliffordOpPlanEntry(
                op_name=r.op_name, target=r.target,
                status=r.status, symbol=r.symbol,
            )
            for r in captured
        )
        _validate_plan(plan)
        manifest_sources = tuple(sorted(set(
            "_CLIFFORD_APPLE_GPU_FUSED" if e.op_name.startswith("clifford_")
            else "_EBM_APPLE_GPU_FUSED" if e.op_name.startswith("ebm_")
            else "_UNKNOWN"
            for e in plan
        )))
        artifact = CliffordCompiledArtifact(
            plan=plan, target=self._target, dtype=self._dtype,
            manifest_sources=manifest_sources,
            plan_hash=_hash_plan(plan),
            source_name=self._fn.__qualname__,
        )
        # Freeze.
        object.__setattr__(self, "artifact", artifact)
        self._compiled = True
