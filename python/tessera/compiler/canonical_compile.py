"""Canonical compiler driver — audit recommendation **C** (C.1: the wrapper).

The audit's framing:

    Make one canonical compiler driver own the whole ladder: Graph module
    in, typed artifacts plus execution capability out. Right now driver.py,
    matmul_pipeline.py, backend manifests, target maps, and runtime dispatch
    each hold part of the truth.

What's true today:

* ``driver.compile_graph_module(...)`` is structurally already "the real
  ladder runner" — it accepts a ``GraphIRModule`` + target and returns a
  ``CompileArtifactBundle`` with all four IR levels populated, plus
  ``executable``/``runtime_status``/``execution_kind``.
* But the executable answer ``CompileArtifactBundle`` produces is decided
  by ``cpu_plan`` presence inside the driver — it does **not** consult the
  audit-named gate ladder (B) or surface the per-cell proof matrix (A).
* ``runtime.launch()`` separately re-derives an executable answer from
  artifact metadata; ``execution_matrix.executor_for_metadata`` does the
  dispatch lookup; ``backend_manifest.manifest_for`` is the per-target
  kernel truth. **Five owners. One question.**

This module is the canonical wrapper that reconciles them:

    canonical_compile(module, *, target="cpu", ...) → CompileResult

with the audit's contract:

    typed_artifacts + capability_set + (executable | reason)

It is **additive** — every existing caller of ``compile_graph_module`` keeps
working unchanged. New callers (the runtime, tests, examples) should reach
for ``canonical_compile`` because:

* The ``executable`` answer is the AND of the bundle's executable and "no
  gate fails," which is the audit's whole point.
* The ``reason`` string leads with the audit-named first failing gate
  (``"first failing gate `toolchain` — nvcc not on PATH"``) so the caller
  doesn't have to know about the gate evaluator separately.
* The ``CompileResult`` is one typed surface that carries everything the
  five today-owners produce, with a single shared truth.

Drift-guarded by ``tests/unit/test_canonical_compile.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from tessera.compiler import pipeline_gates as _pg
from tessera.compiler.driver import (
    CompileArtifactBundle,
    canonical_compile_options,
    compile_graph_module as _compile_graph_module,
)
from tessera.compiler.graph_ir import GraphIRModule
from tessera.compiler.op_catalog import get_op_spec, normalize_op_name


# --- CompileResult --------------------------------------------------------

@dataclass(frozen=True)
class CompileResult:
    """One typed surface for the canonical compile.

    Carries:

    * ``bundle`` — the full ``CompileArtifactBundle`` from the existing
      driver (Graph / Schedule / Tile / Target IR, diagnostics, trace).
    * ``gate_results`` — the seven named pipeline gates (B layer).
    * ``first_failing_gate`` — the audit-named gate (or None if every gate
      passes).
    * ``executable`` — synthesized AND of bundle + gates. The two must
      agree for the answer to be ``True``.
    * ``reason`` — empty string when executable; the audit-named reason
      otherwise. Leads with ``first failing gate `<name>` — <detail>``.

    Convenience accessors (``graph_ir``, ``schedule_ir``, ``tile_ir``,
    ``target_ir``) return the IR text from the bundle so callers don't have
    to thread through ``.bundle.graph.text`` every time.
    """

    bundle: CompileArtifactBundle
    gate_results: tuple[_pg.GateResult, ...]
    first_failing_gate: Optional[_pg.GateResult]
    executable: bool
    reason: str
    primary_op: Optional[str]
    target: str
    # P1 (2026-06-02) — multi-op program identity. ``component_ops`` is the
    # distinct op vocabulary of the whole program (not just ``primary_op``).
    # ``component_blockers`` lists ``(op, first_failing_gate)`` for every
    # component op that does NOT pass its gates on this target, so the
    # whole-program answer is gated component-by-component rather than
    # trusting the primary op to stand in. ``program_executable`` is the
    # bundle-executable AND "no component blockers" — the honest answer for
    # a real multi-op program (single-op programs collapse to ``executable``).
    component_ops: tuple[str, ...] = ()
    component_blockers: tuple[tuple[str, str], ...] = ()
    program_executable: bool = False
    # Sprint A (2026-06-04) — component-aware metadata contract. These are
    # conservative Graph IR summaries today; later compiler passes can replace
    # the derivation without changing the RuntimeArtifact metadata keys.
    fusion_groups: tuple[dict[str, Any], ...] = ()
    shape_envelope: dict[str, Any] = field(default_factory=dict)
    effects: dict[str, Any] = field(default_factory=dict)
    layout_contracts: dict[str, Any] = field(default_factory=dict)
    # Next Work #1 remainder (2026-06-11) — first-class program outputs: the
    # values each function returns, with producer op + type/shape/dtype/layout.
    # A focused "what does this program emit" view distinct from the verbose
    # per-value shape_envelope.
    outputs: dict[str, Any] = field(default_factory=dict)
    # Phase 1 (autodiff unification, 2026-07-11) — the backward provenance facet.
    # ``None`` when no differentiation was requested; otherwise a
    # ``autodiff_request.BackwardProvenance`` distinguishing IR-transformed /
    # artifact-only / native-executable for gradients, mirroring the forward
    # ``executable`` answer. Typed as ``Any`` to avoid an import cycle
    # (autodiff_request imports the autodiff package, which is heavier).
    backward: Optional[Any] = None

    @property
    def native_image(self):
        """Compiler-produced native image, when packaging reached that stage."""
        return self.bundle.native_image

    @property
    def launch_descriptor(self):
        """Compiler-produced launch contract, never reconstructed by runtime."""
        return self.bundle.launch_descriptor

    @property
    def is_single_op(self) -> bool:
        """True when the program is a single distinct op (component-op
        gating is then identical to the primary-op answer)."""
        return len(self.component_ops) <= 1

    @property
    def graph_ir(self) -> str:
        return self.bundle.graph.text if self.bundle.graph else ""

    @property
    def schedule_ir(self) -> str:
        return self.bundle.schedule.text if self.bundle.schedule else ""

    @property
    def tile_ir(self) -> str:
        return self.bundle.tile.text if self.bundle.tile else ""

    @property
    def target_ir(self) -> str:
        return self.bundle.target_ir.text if self.bundle.target_ir else ""

    @property
    def runtime_status(self) -> str:
        return self.bundle.runtime_status

    @property
    def execution_kind(self) -> str:
        return self.bundle.execution_kind

    @property
    def execution_mode(self) -> str:
        return self.bundle.execution_mode

    @property
    def compiler_path(self) -> str:
        return self.bundle.request.pipeline_name

    def gate_status(self, gate_name: str) -> str:
        """Return the status string for one named gate, or ``"unknown"``."""
        for r in self.gate_results:
            if r.gate == gate_name:
                return r.status
        return "unknown"

    def descriptive_metadata(self) -> dict[str, Any]:
        """The component-aware *descriptive* compile metadata (Sprint A): the
        program's op vocabulary, per-component blockers, and the four
        component-aware summaries (fusion groups, shape envelope, effects,
        layout contracts).

        Deliberately excludes the executability *decision* keys
        (``executable`` / ``canonical_executable`` / ``compiler_path``), so it
        is safe to merge into any ``RuntimeArtifact`` — including the `@jit`
        fast paths that own their own executability call — without overriding
        that decision. This is what makes the canonical metadata reach the
        user-facing ``fn.runtime_artifact().metadata``."""
        return {
            "canonical_primary_op": self.primary_op,
            # P1 (2026-06-02) — multi-op program identity.
            "canonical_component_ops": list(self.component_ops),
            "canonical_program_executable": self.program_executable,
            "canonical_component_blockers": [
                {"op": op, "gate": gate}
                for (op, gate) in self.component_blockers
            ],
            "canonical_fusion_groups": list(self.fusion_groups),
            "canonical_shape_envelope": self.shape_envelope,
            "canonical_effects": self.effects,
            "canonical_layout_contracts": self.layout_contracts,
            "canonical_outputs": self.outputs,
        }

    def to_runtime_artifact(self):
        """Project the canonical answer into a :class:`RuntimeArtifact`.

        The runtime ABI consumes ``RuntimeArtifact``s, not ``CompileResult``s
        (legacy ergonomic). This method stamps the canonical answer
        (``executable`` / ``reason`` / ``first_failing_gate`` / gates) into
        ``RuntimeArtifact.metadata`` so :func:`tessera.runtime.launch` can
        **trust** the upstream answer instead of re-running the gate
        evaluator at launch time.

        The artifact is fully self-describing — every consumer who needs
        the seven-gate truth can read it from the artifact without
        re-importing the compiler. Audit recommendation C.2 in
        ``docs/audit/compiler/COMPILER_AUDIT.md``.
        """
        # Import here to keep the compiler→runtime dependency one-way at
        # import time (canonical_compile is pulled by jit.py very early).
        from tessera.runtime import RuntimeArtifact

        meta: dict[str, Any] = dict(self.bundle.to_metadata())
        meta.update({
            # Audit-named answer (C.1 → C.2). Two channels: a top-level
            # boolean + reason string, and a structured first-failing-gate
            # mirror that the runtime can short-circuit on.
            "canonical_executable": self.executable,
            "canonical_reason": self.reason,
            "canonical_first_failing_gate": (
                self.first_failing_gate.gate
                if self.first_failing_gate else None
            ),
            "canonical_first_failing_gate_detail": (
                self.first_failing_gate.detail
                if self.first_failing_gate else ""
            ),
            "canonical_gates": [
                {"gate": r.gate, "status": r.status, "detail": r.detail}
                for r in self.gate_results
            ],
            # The bundle's executable claim — preserved so callers that
            # need to compare bundle-side vs canonical-side decisions can.
            "bundle_executable": self.bundle.executable,
            # Canonical answer is authoritative for `executable`.
            "executable": self.executable,
        })
        # Component-aware descriptive metadata (Sprint A) — factored so the
        # `@jit` runtime-artifact path can merge it without taking the
        # executability *decision* keys above (those fast paths own that call).
        meta.update(self.descriptive_metadata())
        # Apple Value Target IR (RV-P1, 2026-06-03) — the front door now
        # *consumes* the classifier/extractor instead of leaving them to docs.
        # The runtime artifact records whether the lowered Apple Target IR is
        # the value lane (value-producing call ops) or the artifact lane, and —
        # for the value lane — the dispatch tuples (op_kind/symbol/status) the
        # runtime reads to invoke the named C ABI entry.
        if self.target in ("apple_cpu", "apple_gpu"):
            try:
                from tessera.compiler import driver as _drv

                _ir = self.target_ir or ""
                _kind = _drv.classify_apple_target_ir(_ir)
                meta["apple_target_ir_kind"] = _kind
                if _kind == "value_target_ir":
                    _calls = _drv.extract_apple_value_calls(_ir)
                    meta["apple_value_calls"] = _calls
                    # Sprint 2 (S2-3): route the value lane through the
                    # apple_value_target_ir executor. Preserve the prior path so
                    # consumers can see what it would have been in artifact mode.
                    if _calls:
                        meta["apple_previous_compiler_path"] = meta.get(
                            "compiler_path", "")
                        meta["compiler_path"] = "apple_value_target_ir"
                        # Sprint 8 review (P2) + Stage 16D: EXACT executable
                        # truth. The value executors accept exactly one
                        # supported value call, and support is per C ABI symbol
                        # (plus runtime/probe availability for GPU symbols),
                        # not per op family.
                        _exec_ok = (
                            len(_calls) == 1
                            and _drv.apple_value_call_is_executable(_calls[0])
                        )
                        # The value lane OWNS the executable decision for this
                        # artifact (override the bundle/canonical answer): a
                        # value artifact is launchable iff its single value call
                        # is on the runtime allowlist and any op-specific
                        # runtime probe succeeds. Multi-op, off-allowlist, and
                        # unsupported GPU calls are decisively NOT executable.
                        meta["executable"] = _exec_ok
            except Exception:
                # Classification is metadata-only; never block artifact creation.
                pass
        return RuntimeArtifact(
            graph_ir=self.graph_ir,
            schedule_ir=self.schedule_ir,
            tile_ir=self.tile_ir,
            target_ir=self.target_ir,
            metadata=meta,
            abi_signature=f"tessera.canonical.v1.{self.target}",
            native_image=self.native_image,
            launch_descriptor=self.launch_descriptor,
        )

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable summary — useful for telemetry, dashboards,
        compile-report integration, and tests that want a stable surface.

        For Apple targets, also includes ``apple_gpu_capabilities``
        (per-feature flag dict) and ``apple_gpu_archive`` (MTL4Archive
        cache state) — the Apple-sample Actions 1 + 6 surface. The
        snapshot is a best-effort read; on hosts where the runtime
        isn't available, the keys are present but mostly empty
        (``runtime_available=False``), so consumers don't need to
        special-case the missing case.
        """
        out: dict[str, Any] = {
            "target": self.target,
            "primary_op": self.primary_op,
            "component_ops": list(self.component_ops),
            "program_executable": self.program_executable,
            "component_blockers": [
                {"op": op, "gate": gate}
                for (op, gate) in self.component_blockers
            ],
            "fusion_groups": list(self.fusion_groups),
            "shape_envelope": self.shape_envelope,
            "effects": self.effects,
            "layout_contracts": self.layout_contracts,
            "outputs": self.outputs,
            "compiler_path": self.compiler_path,
            "executable": self.executable,
            "reason": self.reason,
            "runtime_status": self.runtime_status,
            "execution_kind": self.execution_kind,
            "execution_mode": self.execution_mode,
            "first_failing_gate": (self.first_failing_gate.gate
                                   if self.first_failing_gate else None),
            "first_failing_gate_detail": (self.first_failing_gate.detail
                                          if self.first_failing_gate else ""),
            "gates": [
                {"gate": r.gate, "status": r.status, "detail": r.detail}
                for r in self.gate_results
            ],
            "artifact_hashes": self.bundle.artifact_hashes,
            "orchestration_state": self.bundle.orchestration_state,
            "spine_stages": list(self.bundle.spine_stages()),
            "native_image": (
                self.native_image.to_dict() if self.native_image is not None else None
            ),
            "launch_descriptor": (
                self.launch_descriptor.to_dict()
                if self.launch_descriptor is not None else None
            ),
        }
        # Apple-sample Actions 1 + 6 — capability + archive telemetry on
        # Apple targets. The compile result is the natural carrier: a
        # dashboard or test that wants "what's lit up on this host"
        # reads it from one place. Lazy-imported so the canonical
        # compile module stays a pure aggregator over its declared
        # truth sources.
        if self.target in ("apple_gpu", "apple_cpu"):
            try:
                from tessera._apple_gpu_dispatch import (
                    apple_gpu_capabilities_snapshot,
                )
                snap = apple_gpu_capabilities_snapshot()
                out["apple_gpu_capabilities"] = snap.get("capabilities", {})
                out["apple_gpu_capabilities_raw"] = snap.get(
                    "capabilities_raw", 0)
                out["apple_gpu_mtl4_full"] = snap.get("mtl4_full", False)
                out["apple_gpu_archive"] = snap.get("archive", {})
                out["apple_gpu_runtime_available"] = snap.get(
                    "runtime_available", False)
            except Exception:
                # Defensive — capability telemetry must never break
                # ``to_dict``. Leave the keys absent on import failure.
                pass
        return out


# --- Primary-op extraction -----------------------------------------------

def _extract_primary_op(module: GraphIRModule) -> Optional[str]:
    """Pick the op name that the gate evaluator should treat as primary.

    Today: the *first* op in the module's first function. Gate evaluation
    is op-specific for ``codegen`` / ``link`` / ``numerical`` — those want
    a concrete op like ``matmul`` to consult the backend manifest. For
    multi-op programs this is best-effort; the conformance matrix's
    component-ops model is the long-term fit. Returned **without** the
    ``tessera.`` prefix so the gate evaluator's internal registry matches
    (the gate module strips the prefix when it sees it).
    """
    if not module.functions:
        return None
    fn = module.functions[0]
    if not fn.body:
        return None
    op_name = fn.body[0].op_name or ""
    if op_name.startswith("tessera."):
        op_name = op_name[len("tessera."):]
    return op_name or None


def _extract_component_ops(module: GraphIRModule) -> tuple[str, ...]:
    """The distinct op vocabulary of the whole program (P1, 2026-06-02).

    ``_extract_primary_op`` returns only the first op — fine for picking
    a representative for the op-specific gates, but it makes the canonical
    answer single-op-oriented (COMPILER_AUDIT: "program identity is too
    single-op-oriented"). This walks **every** op across **every**
    function and returns the distinct names in first-seen order, prefix-
    stripped, so callers can gate a whole multi-op program component-by-
    component rather than trusting the first op to stand in for all.

    Returns ``()`` for an empty module. A single-op program yields a
    1-tuple equal to ``(primary_op,)``, so existing single-op behavior is
    unchanged.
    """
    seen: dict[str, None] = {}
    for fn in module.functions:
        for node in fn.body:
            name = node.op_name or ""
            if name.startswith("tessera."):
                name = name[len("tessera."):]
            if name:
                seen.setdefault(name, None)
    return tuple(seen)


# --- Sprint A metadata derivation ----------------------------------------

_EFFECT_ORDER: tuple[str, ...] = (
    "pure", "random", "movement", "state", "collective", "memory", "io", "top",
)
_EFFECT_RANK = {name: i for i, name in enumerate(_EFFECT_ORDER)}


def _strip_percent(name: str) -> str:
    return name[1:] if name.startswith("%") else name


def _canonical_op_name(op_name: str) -> str:
    try:
        return normalize_op_name(op_name)
    except Exception:
        if op_name.startswith("tessera."):
            return op_name[len("tessera."):]
        return op_name


def _join_effect(a: str, b: str) -> str:
    return a if _EFFECT_RANK.get(a, 0) >= _EFFECT_RANK.get(b, 0) else b


def _dtype_from_mlir(dtype: str | None) -> str | None:
    if dtype is None:
        return None
    return {
        "f64": "fp64",
        "f32": "fp32",
        "f16": "fp16",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }.get(dtype, dtype)


def _type_metadata(ir_type: Any, *, layout: str | None = None) -> dict[str, Any]:
    """Return a JSON-safe summary of an IR type or MLIR type string."""
    if hasattr(ir_type, "mlir_str"):
        shape = list(getattr(ir_type, "shape", ()) or ())
        dtype = getattr(ir_type, "dtype", None)
        resolved_layout = layout or getattr(ir_type, "layout", None)
        return {
            "type": str(ir_type),
            "shape": shape,
            "rank": None if "*" in shape else len(shape),
            "dtype": dtype,
            "layout": resolved_layout,
        }

    text = str(ir_type or "")
    meta: dict[str, Any] = {
        "type": text,
        "shape": [],
        "rank": None,
        "dtype": None,
        "layout": layout,
    }
    if text.startswith("tensor<") and text.endswith(">"):
        body = text[len("tensor<"):-1]
        parts = body.split("x")
        if len(parts) >= 2:
            dtype = parts[-1]
            dims = parts[:-1]
            if dims == ["*"]:
                parsed_shape: list[str] = ["*"]
            else:
                parsed_shape = dims
            meta.update({
                "shape": parsed_shape,
                "rank": None if "*" in parsed_shape else len(parsed_shape),
                "dtype": _dtype_from_mlir(dtype),
            })
    return meta


def _op_effect(op_name: str, kwargs: Mapping[str, Any]) -> str:
    explicit = kwargs.get("effect")
    if isinstance(explicit, str) and explicit in _EFFECT_RANK:
        return explicit
    spec = get_op_spec(op_name)
    if spec is not None:
        return spec.effect
    return "pure"


def _lowering_family(op_name: str) -> str:
    spec = get_op_spec(op_name)
    return spec.lowering if spec is not None else "unknown"


def _derive_shape_envelope(module: GraphIRModule) -> dict[str, Any]:
    functions: list[dict[str, Any]] = []
    for fn in module.functions:
        values: list[dict[str, Any]] = []
        value_types: dict[str, dict[str, Any]] = {
            arg.name: _type_metadata(arg.ir_type, layout=arg.layout)
            for arg in fn.args
        }
        for op in fn.body:
            result_meta = _type_metadata(op.inferred_type or op.result_type)
            for name in op.result_names:
                value_types[name] = result_meta
                values.append({
                    "name": name,
                    "producer": _canonical_op_name(op.op_name),
                    **result_meta,
                })
        functions.append({
            "name": fn.name,
            "args": [
                {"name": arg.name, **_type_metadata(arg.ir_type, layout=arg.layout)}
                for arg in fn.args
            ],
            "results": [
                {"index": i, **_type_metadata(result_type)}
                for i, result_type in enumerate(fn.result_types)
            ],
            "values": values,
            "returns": [
                {
                    "name": _strip_percent(value),
                    **value_types.get(_strip_percent(value), _type_metadata("")),
                }
                for value in fn.return_values
            ],
        })
    return {"schema": "tessera.compile.shape_envelope.v1", "functions": functions}


def _derive_outputs(module: GraphIRModule) -> dict[str, Any]:
    """Program outputs: each function's returned values with their producer op
    and type/shape/dtype/layout. ``program_outputs`` is the entry (first)
    function's outputs — the program-level "what is emitted" view."""
    functions: list[dict[str, Any]] = []
    for fn in module.functions:
        value_types: dict[str, dict[str, Any]] = {
            arg.name: _type_metadata(arg.ir_type, layout=arg.layout)
            for arg in fn.args
        }
        producers: dict[str, str] = {}
        for op in fn.body:
            result_meta = _type_metadata(op.inferred_type or op.result_type)
            for name in op.result_names:
                value_types[name] = result_meta
                producers[name] = _canonical_op_name(op.op_name)
        outputs: list[dict[str, Any]] = []
        for index, value in enumerate(fn.return_values):
            name = _strip_percent(value)
            outputs.append({
                "index": index,
                "name": name,
                "producer": producers.get(name),   # None when a return is an arg
                **value_types.get(name, _type_metadata("")),
            })
        functions.append({"name": fn.name, "outputs": outputs})
    program_outputs = functions[0]["outputs"] if functions else []
    return {
        "schema": "tessera.compile.outputs.v1",
        "functions": functions,
        "program_outputs": program_outputs,
    }


def _derive_effects(module: GraphIRModule) -> dict[str, Any]:
    functions: list[dict[str, Any]] = []
    module_effect = "pure"
    for fn in module.functions:
        fn_effect = "pure"
        arg_effects = [
            {"name": arg.name, "effect": arg.effect}
            for arg in fn.args if arg.effect
        ]
        for arg in arg_effects:
            fn_effect = _join_effect(fn_effect, str(arg["effect"]))
        op_effects: list[dict[str, str]] = []
        for op in fn.body:
            effect = _op_effect(op.op_name, op.kwargs)
            op_effects.append({
                "op": _canonical_op_name(op.op_name),
                "effect": effect,
            })
            fn_effect = _join_effect(fn_effect, effect)
        module_effect = _join_effect(module_effect, fn_effect)
        functions.append({
            "name": fn.name,
            "summary": fn_effect,
            "args": arg_effects,
            "ops": op_effects,
        })
    return {
        "schema": "tessera.compile.effects.v1",
        "summary": module_effect,
        "functions": functions,
    }


def _derive_layout_contracts(module: GraphIRModule) -> dict[str, Any]:
    functions: list[dict[str, Any]] = []
    for fn in module.functions:
        value_layouts: dict[str, str | None] = {}
        args: list[dict[str, Any]] = []
        for arg in fn.args:
            layout = arg.layout or arg.ir_type.layout
            value_layouts[arg.name] = layout
            args.append({"name": arg.name, "layout": layout})

        ops: list[dict[str, Any]] = []
        for op in fn.body:
            operand_layouts = [
                value_layouts.get(_strip_percent(operand))
                for operand in op.operands
            ]
            result_layout = None
            if op.inferred_type is not None:
                result_layout = op.inferred_type.layout
            if result_layout is None and isinstance(op.kwargs.get("layout"), str):
                result_layout = str(op.kwargs["layout"])
            for name in op.result_names:
                value_layouts[name] = result_layout
            ops.append({
                "op": _canonical_op_name(op.op_name),
                "operands": operand_layouts,
                "result": result_layout,
            })

        functions.append({"name": fn.name, "args": args, "ops": ops})
    return {
        "schema": "tessera.compile.layout_contracts.v1",
        "functions": functions,
    }


# Linear fused chains the Apple GPU runtime collapses into a single kernel —
# mirrors runtime.py's ``_apple_gpu_metadata_is_*_chain`` detectors. Ordered
# longest-first so the attention block matches before its matmul→softmax prefix.
# (SwiGLU is a DAG, not a linear chain — `_match_swiglu_at` handles it inside
# the same scan, tried first because at 4 ops it is the longest fusion.)
_KNOWN_FUSION_CHAINS: tuple[tuple[str, ...], ...] = (
    ("matmul", "softmax", "matmul"),   # attention block O = softmax(A@B)@C
    ("matmul", "softmax"),             # scores → softmax
    ("matmul", "gelu"),                # MLP matmul → gelu
    ("matmul", "rmsnorm"),             # matmul → rmsnorm
    ("matmul", "rmsnorm_safe"),        # matmul → rmsnorm (Gemma soft variant, eps 1e-6)
)


def _chain_canon(op) -> str:
    """Canonical op name with ``gemm`` folded to ``matmul`` for chain matching."""
    cn = _canonical_op_name(op.op_name)
    return "matmul" if cn == "gemm" else cn


def _ssa_connected(body, start: int, length: int) -> bool:
    """Each op in body[start:start+length] consumes a result of its predecessor
    (so the chain is a real data-flow chain, not coincidental adjacency)."""
    for k in range(1, length):
        prev_results = set(body[start + k - 1].result_names)
        operands = {_strip_percent(o) for o in body[start + k].operands}
        if not (prev_results & operands):
            return False
    return True


def _match_swiglu_at(body, canon, i: int) -> bool:
    """SwiGLU is a DAG, not a linear chain (audit 2026-06-10 follow-on):

        gate = matmul(x, Wg); up = matmul(x, Wu)      # both consume the SAME x
        h    = silu_mul(gate, up)
        out  = matmul(h, Wd)

    The executor dispatches the fused kernel from this group alone via the
    ``dispatch`` roles (Phase 0a/0b) — the structural re-matcher this once
    mirrored was deleted in Phase 0c (front-to-back closure plan)."""
    if i + 4 > len(body):
        return False
    if tuple(canon[i:i + 4]) != ("matmul", "matmul", "silu_mul", "matmul"):
        return False
    gate, up, sm, down = body[i], body[i + 1], body[i + 2], body[i + 3]
    gate_operands = [_strip_percent(o) for o in gate.operands]
    up_operands = [_strip_percent(o) for o in up.operands]
    if len(gate_operands) < 2 or len(up_operands) < 2:
        return False
    if gate_operands[0] != up_operands[0]:
        return False  # gate and up must share %x
    if not (gate.result_names and up.result_names and sm.result_names):
        return False
    sm_operands = [_strip_percent(o) for o in sm.operands]
    if len(sm_operands) != 2:
        return False
    if sm_operands[0] != gate.result_names[0] or sm_operands[1] != up.result_names[0]:
        return False
    down_operands = [_strip_percent(o) for o in down.operands]
    return bool(down_operands) and down_operands[0] == sm.result_names[0]


#: Unary activations that can sit on a gate branch (mirrors fusion.EPILOGUE_OPS
#: minus bias). Matched structurally, so order within the 4-op window is free.
_GATE_ACTS: frozenset[str] = frozenset(
    {"silu", "gelu", "relu", "sigmoid", "tanh"})
_GATE_MULS: frozenset[str] = frozenset({"mul", "multiply"})


def _match_gated_matmul_at(body, canon, i: int) -> bool:
    """Gated matmul (SwiGLU gate from PRIMITIVE ops) — a DAG, not a linear chain:

        gate = matmul(x, Wg); up = matmul(x, Wu)   # both consume the SAME x
        gact = f(gate)                              # unary activation on the gate
        out  = mul(gact, up)                        # elementwise combine

    Distinct from ``_match_swiglu_at`` (which has a fused ``silu_mul`` op *and* a
    down-projection). Recognizing this 4-op window as a whole-program known_chain
    routes the program to the ``apple_gpu_mps`` executor, where the runtime
    prepass (``discover_gated_matmul_regions``) synthesizes the fused kernel. The
    window is matched structurally (any op order), since the tracer's emission
    order varies with how the gate expression is spelled."""
    if i + 4 > len(body):
        return False
    idxs = list(range(i, i + 4))
    matmuls = [k for k in idxs if canon[k] == "matmul"]
    acts = [k for k in idxs if canon[k] in _GATE_ACTS]
    muls = [k for k in idxs if canon[k] in _GATE_MULS]
    if len(matmuls) != 2 or len(acts) != 1 or len(muls) != 1:
        return False
    act, mul = acts[0], muls[0]
    if not body[act].operands or not body[act].result_names:
        return False
    act_in = _strip_percent(body[act].operands[0])
    # the activation consumes one matmul's output → that is the gate projection.
    gate_mm = next((k for k in matmuls
                    if body[k].result_names
                    and body[k].result_names[0] == act_in), None)
    if gate_mm is None:
        return False
    up_mm = next(k for k in matmuls if k != gate_mm)
    g_ops = [_strip_percent(o) for o in body[gate_mm].operands]
    u_ops = [_strip_percent(o) for o in body[up_mm].operands]
    if len(g_ops) < 2 or len(u_ops) < 2 or g_ops[0] != u_ops[0]:
        return False                              # the two matmuls must share %x
    if not body[up_mm].result_names:
        return False
    mul_ins = {_strip_percent(o) for o in body[mul].operands}
    # the multiply combines the activation output and the up projection.
    return mul_ins == {body[act].result_names[0], body[up_mm].result_names[0]}


def _chain_result(op) -> str | None:
    names = getattr(op, "result_names", None)
    return _strip_percent(names[0]) if names else None


def _chain_dispatch_roles(
        body, start: int, length: int, fused_kernel: str) -> dict[str, Any] | None:
    """Operand/result roles for a recognized fusion chain, resolved from
    Graph-IR operand order — Phase 0a of the front-to-back closure plan
    (docs/audit/compiler/COMPILER_AUDIT.md).

    Mirrors the inline role extraction in
    ``runtime._execute_apple_gpu_mps_metadata`` but computes it ONCE at compile
    time where SSA names + operand order are unambiguous (the executor's
    value-shape guessing is what blocks attention-region fusion, see
    runtime.py ~:2460). Carried on the group as ``dispatch`` so a later step can
    make the executor authoritative and drop the structural re-matchers.

    Returns ``None`` when roles can't be cleanly resolved, so the group simply
    carries no ``dispatch`` and the executor's existing path is unaffected
    (0a is strictly additive)."""

    def operand(op, idx: int) -> str | None:
        ops = op.operands
        return _strip_percent(ops[idx]) if len(ops) > idx else None

    def tflag(op, axis: str) -> bool:
        # Score/first matmul transpose flag (kwargs `transpose_b`/`transposeB`),
        # carried so the executor honors `Q·Kᵀ` (transpose_b) instead of guessing
        # K's orientation from value shapes — the M2 attention-orientation fix.
        kw = getattr(op, "kwargs", None) or {}
        return bool(kw.get(f"transpose_{axis}") or kw.get(f"transpose{axis.upper()}"))

    chain = body[start:start + length]
    if fused_kernel == "swiglu" and length == 4:
        gate, up, _silu, down = chain
        roles: dict[str, Any] = {
            "x": operand(gate, 0), "wg": operand(gate, 1),
            "wu": operand(up, 1), "wd": operand(down, 1),
            "out": _chain_result(down),
        }
    elif fused_kernel == "matmul_softmax_matmul" and length == 3:
        mm1, _sm, mm2 = chain
        roles = {
            "a": operand(mm1, 0), "b": operand(mm1, 1),
            "c": operand(mm2, 1), "out": _chain_result(mm2),
            "transpose_a": tflag(mm1, "a"), "transpose_b": tflag(mm1, "b"),
        }
    elif fused_kernel in ("matmul_softmax", "matmul_gelu", "matmul_rmsnorm",
                          "matmul_rmsnorm_safe") and length == 2:
        mm, tail = chain
        roles = {
            "a": operand(mm, 0), "b": operand(mm, 1),
            "out": _chain_result(tail),
            "transpose_a": tflag(mm, "a"), "transpose_b": tflag(mm, "b"),
        }
        if fused_kernel in ("matmul_rmsnorm", "matmul_rmsnorm_safe"):
            kwargs = getattr(tail, "kwargs", None) or {}
            eps_default = (1e-6 if str(getattr(tail, "op_name", "")).endswith(
                "rmsnorm_safe") else 1e-5)
            roles["eps"] = float(kwargs.get("eps", eps_default))
    else:
        return None

    # Every tensor role must resolve; ``eps`` is a scalar param, not an SSA name.
    if any(v is None for k, v in roles.items() if k != "eps"):
        return None
    return roles


def _match_known_chains(fn) -> list[dict[str, Any]]:
    body = fn.body
    canon = [_chain_canon(op) for op in body]
    n = len(body)
    groups: list[dict[str, Any]] = []
    i = 0
    while i < n:
        matched_len = 0
        fused_name = ""
        # SwiGLU first — at 4 ops it is the longest fusion, and its DAG shape
        # needs its own connectivity check (gate/up share x; both feed
        # silu_mul) rather than the linear _ssa_connected walk.
        if _match_swiglu_at(body, canon, i):
            matched_len, fused_name = 4, "swiglu"
        elif _match_gated_matmul_at(body, canon, i):
            # gate from primitives (no down-proj) — routes to apple_gpu_mps, the
            # runtime prepass synthesizes the fused gated kernel.
            matched_len, fused_name = 4, "gated_matmul"
        else:
            for chain in _KNOWN_FUSION_CHAINS:             # longest-first
                L = len(chain)
                if i + L <= n and tuple(canon[i:i + L]) == chain and _ssa_connected(body, i, L):
                    matched_len, fused_name = L, "_".join(chain)
                    break
        if matched_len:
            group: dict[str, Any] = {
                "function": fn.name,
                "kind": "known_chain",
                "status": "candidate",
                "fused_kernel": fused_name,
                "ops": [
                    {"index": i + k, "op": canon[i + k]}
                    for k in range(matched_len)
                ],
            }
            # 0a — carry operand/result roles so the executor can dispatch from
            # this group alone (additive: absent when roles don't resolve).
            roles = _chain_dispatch_roles(body, i, matched_len, fused_name)
            if roles is not None:
                group["dispatch"] = roles
            groups.append(group)
            i += matched_len                               # don't overlap chains
        else:
            i += 1
    return groups


def _derive_fusion_groups(module: GraphIRModule) -> tuple[dict[str, Any], ...]:
    """Fusion candidates — metadata, not a scheduling claim. Two signals:

    * **known_chain** — a linear data-flow chain matching a fused kernel the
      backend actually emits (matmul→softmax[→matmul], matmul→gelu/rmsnorm).
    * **producer_consumer** — a conservative fallback: adjacent pure ops in the
      same lowering family that pass an SSA value directly.
    """
    groups: list[dict[str, Any]] = []
    for fn in module.functions:
        groups.extend(_match_known_chains(fn))
        producer_by_value: dict[str, tuple[int, Any]] = {}
        for i, op in enumerate(fn.body):
            effect = _op_effect(op.op_name, op.kwargs)
            family = _lowering_family(op.op_name)
            for operand in op.operands:
                produced = producer_by_value.get(_strip_percent(operand))
                if produced is None:
                    continue
                j, prev = produced
                if j != i - 1:
                    continue
                prev_effect = _op_effect(prev.op_name, prev.kwargs)
                prev_family = _lowering_family(prev.op_name)
                if effect == prev_effect == "pure" and family == prev_family:
                    groups.append({
                        "function": fn.name,
                        "kind": "producer_consumer",
                        "status": "candidate",
                        "ops": [
                            {"index": j, "op": _canonical_op_name(prev.op_name)},
                            {"index": i, "op": _canonical_op_name(op.op_name)},
                        ],
                        "via": _strip_percent(operand),
                        "lowering": family,
                    })
            for result_name in op.result_names:
                producer_by_value[result_name] = (i, op)
    return tuple(groups)


def _derive_compile_metadata(module: GraphIRModule) -> dict[str, Any]:
    return {
        "fusion_groups": _derive_fusion_groups(module),
        "shape_envelope": _derive_shape_envelope(module),
        "effects": _derive_effects(module),
        "layout_contracts": _derive_layout_contracts(module),
        "outputs": _derive_outputs(module),
    }


# Linear fusion chains the Apple Target IR passes re-discover and now also
# consume from a descriptor. (swiglu / mla / nsa lower a pre-fused op — the op
# *is* the descriptor — so they don't need an intent stamp.)
#
# These are the intent *names the C++ consumers read*. The reverse direction
# (every name here is consumed by some Apple pass) is guarded by
# tests/unit/test_fusion_intent_emitter.py::test_emitter_intents_match_cpp_consumers.
_INTENT_KERNELS: frozenset[str] = frozenset({
    "matmul_softmax_matmul", "matmul_softmax", "matmul_gelu", "matmul_rmsnorm",
})

# Producer fused_kernel → the intent name the C++ consumer expects. The Apple
# `MatmulRMSNormFusion` pass handles `rmsnorm` *and* `rmsnorm_safe` and reads a
# single `intent == "matmul_rmsnorm"` for both, so the `matmul_rmsnorm_safe`
# chain (a distinct producer kernel since Phase 0c) maps to that one intent —
# otherwise the C++ would tag rmsnorm_safe fusions `rediscovered` (drift).
# Identity for every other kernel.
_FUSION_INTENT_NAME: dict[str, str] = {
    "matmul_rmsnorm_safe": "matmul_rmsnorm",
}


def stamp_fusion_intents(module: GraphIRModule) -> int:
    """Decision #19 emit-half — stamp ``tessera.fusion.intent`` on the terminal
    op of each recognized linear fusion chain so the Apple Target IR fusion
    passes *consume* the compiler's fusion decision (the emitted call is tagged
    ``source = "descriptor"``) instead of re-discovering the chain. Returns the
    number of chains stamped. Idempotent.

    The terminal op (the highest-index op of the chain — the tail matmul /
    softmax / gelu / rmsnorm) is exactly where each C++ pass reads the intent.
    """
    groups = _derive_fusion_groups(module)
    by_fn = {fn.name: fn for fn in module.functions}
    stamped = 0
    for group in groups:
        if group.get("kind") != "known_chain":
            continue
        kernel = group.get("fused_kernel")
        if not isinstance(kernel, str):
            continue
        # Normalize the producer kernel to the intent name the C++ consumer
        # reads (identity for all but matmul_rmsnorm_safe → matmul_rmsnorm).
        intent_name = _FUSION_INTENT_NAME.get(kernel, kernel)
        if intent_name not in _INTENT_KERNELS:
            continue
        fn_name = group.get("function")
        fn = by_fn.get(fn_name) if isinstance(fn_name, str) else None
        if fn is None:
            continue
        indices = [int(e["index"]) for e in group.get("ops", ())
                   if isinstance(e, Mapping) and isinstance(e.get("index"), int)]
        if not indices:
            continue
        terminal = max(indices)
        if 0 <= terminal < len(fn.body):
            op = fn.body[terminal]
            # Stamp into `attrs` (the MLIR-attribute-only field), NOT `kwargs`:
            # kwargs are forwarded as the op's real call arguments in the
            # reference/runtime execution path, so a descriptor placed there
            # would leak into the numpy op call (e.g. gelu(**kwargs)).
            intent_attr = f'tessera.fusion.intent = "{intent_name}"'
            if not op.attrs:
                op.attrs = intent_attr
            elif "tessera.fusion.intent" not in op.attrs:
                op.attrs = f"{op.attrs}, {intent_attr}"
            stamped += 1
    return stamped


# --- Synthesizers --------------------------------------------------------

def _result_from_bundle(
    bundle: CompileArtifactBundle,
    primary_op: Optional[str],
    component_ops: Optional[tuple[str, ...]] = None,
    compile_metadata: Optional[Mapping[str, Any]] = None,
) -> CompileResult:
    """Shared "bundle + gates → CompileResult" reconciliation. Both
    :func:`canonical_compile` (which runs the ladder first) and
    :func:`compile_result_from_bundle` (for callers like ``@jit`` who
    already have a bundle in hand) flow through here, so the
    executable/reason synthesis lives in exactly one place.

    ``component_ops`` (P1, 2026-06-02) is the whole-program op vocabulary;
    when ``None`` it collapses to ``(primary_op,)`` so single-op callers
    behave exactly as before. The whole-program answer
    (``program_executable``) is gated component-by-component."""
    target = bundle.request.target
    gate_results = _pg.evaluate(target, primary_op)
    first_fail = _pg.first_failing_gate(target, primary_op)

    if bundle.executable and first_fail is None:
        executable = True
        reason = ""
    elif first_fail is not None:
        executable = False
        reason = (
            f"first failing gate `{first_fail.gate}` — {first_fail.detail}."
            f" (see docs/audit/op_target_conformance.md)"
        )
    else:
        executable = False
        reason = (
            f"bundle reports non-executable artifact: "
            f"runtime_status={bundle.runtime_status} "
            f"execution_kind={bundle.execution_kind}"
        )

    # P1 — component-level gating. Gate every distinct op in the program
    # separately so a multi-op program's answer isn't carried by its first
    # op alone. A component "blocks" when its op-specific gates fail.
    comps = component_ops
    if comps is None:
        comps = (primary_op,) if primary_op else ()
    blockers: list[tuple[str, str]] = []
    for op in comps:
        if not op:
            continue
        cfail = _pg.first_failing_gate(target, op)
        if cfail is not None:
            blockers.append((op, cfail.gate))
    program_executable = bool(bundle.executable) and not blockers
    metadata = dict(compile_metadata or {})

    return CompileResult(
        bundle=bundle,
        gate_results=gate_results,
        first_failing_gate=first_fail,
        executable=executable,
        reason=reason,
        primary_op=primary_op,
        target=target,
        component_ops=tuple(comps),
        component_blockers=tuple(blockers),
        program_executable=program_executable,
        fusion_groups=tuple(metadata.get("fusion_groups", ())),
        shape_envelope=dict(metadata.get("shape_envelope", {})),
        effects=dict(metadata.get("effects", {})),
        layout_contracts=dict(metadata.get("layout_contracts", {})),
        outputs=dict(metadata.get("outputs", {})),
    )


def compile_result_from_bundle(
    bundle: CompileArtifactBundle,
    *,
    module: Optional[GraphIRModule] = None,
    primary_op: Optional[str] = None,
) -> CompileResult:
    """Build a :class:`CompileResult` from an already-computed
    :class:`CompileArtifactBundle`.

    Use case (C.3): ``@tessera.jit`` calls ``compile_graph_module`` directly
    today and rewraps the bundle to inject pre-compile diagnostics. Calling
    :func:`canonical_compile` from ``@jit`` would re-run the whole ladder.
    Instead, ``@jit`` finishes its bundle construction and then synthesizes
    the canonical answer from it via this helper — same C.1 truth, no
    duplicated compile work.

    ``primary_op`` may be passed explicitly; otherwise it's extracted from
    ``module``. If neither is provided, the gate evaluation runs
    target-level only (which is still useful — toolchain / hardware_smoke
    don't need an op name).
    """
    if primary_op is None and module is not None:
        primary_op = _extract_primary_op(module)
    component_ops = (
        _extract_component_ops(module) if module is not None else None)
    compile_metadata = _derive_compile_metadata(module) if module is not None else None
    return _result_from_bundle(
        bundle, primary_op, component_ops, compile_metadata)


# --- Canonical compile() -------------------------------------------------

def canonical_compile(
    module: GraphIRModule,
    *,
    target: str = "cpu",
    source_origin: str = "<canonical>",
    options: Mapping[str, Any] | None = None,
    cpu_tile: tuple[int, int, int] = (128, 128, 64),
    enable_tool_validation: bool = True,
) -> CompileResult:
    """Run the full Tessera compile ladder and return a typed result.

    Composes:

    * :func:`tessera.compiler.driver.compile_graph_module` — the existing
      ladder runner (Graph→Schedule→Tile→Target lowering + diagnostics).
    * :func:`tessera.compiler.pipeline_gates.evaluate` — the audit's seven
      named capability gates (B layer).

    Returns a :class:`CompileResult` whose ``executable`` is the
    AND of both layers' answers — the bundle's structural "we made an
    executable artifact" AND "no audit gate fails." When ``executable`` is
    False, ``reason`` leads with the audit-named first failing gate so the
    diagnostic is actionable.

    No new compiler logic lives here — this is a *pure composition* of the
    existing surfaces. Audit recommendation C's whole point: one place to
    look for the answer.
    """
    resolved_options = canonical_compile_options(
        module, target=target, options=options,
    )

    bundle = _compile_graph_module(
        module,
        source_origin=source_origin,
        target=target,
        cpu_tile=cpu_tile,
        options=resolved_options,
        enable_tool_validation=enable_tool_validation,
    )
    return _result_from_bundle(
        bundle,
        _extract_primary_op(module),
        _extract_component_ops(module),
        _derive_compile_metadata(module),
    )


__all__ = [
    "CompileResult",
    "canonical_compile",
    "compile_result_from_bundle",
]
