"""Narrow end-to-end CPU lowering path for supported @jit op graphs.

This module handles straight-line dataflow built from supported Tessera ops:

    @tessera.jit
    def f(...):
        y = tessera.ops.<supported_op>(...)
        return tessera.ops.<supported_op>(y)

It creates inspectable Schedule IR, Tile IR, and Target IR artifacts from the
existing Graph IR text and executes the operation on CPU via NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .graph_ir import GraphIRFunction, GraphIRModule, IROp
from .capabilities import normalize_target as _normalize_target
from .op_catalog import GRAPH_OP_TO_SPEC, LEGACY_GRAPH_OP_ALIASES, SUPPORTED_CPU_OPS, canonical_graph_op_name
from .schedule_planner import SchedulePlanner
from .schedule_ir import lower_graph_to_schedule_ir
from .target_ir import lower_tile_to_target_ir
from .tile_ir import lower_schedule_to_tile_ir


MATMUL_OPS = {"tessera.matmul", "tessera.gemm"}
CONV2D_OPS = {"tessera.conv2d_nhwc", "tessera.conv2d"}
UNARY_OPS = {
    "tessera.layer_norm",
    "tessera.relu",
    "tessera.sigmoid",
    "tessera.sin",
    "tessera.silu",
    "tessera.gelu",
    "tessera.tanh",
    "tessera.rmsnorm",
    "tessera.rmsnorm_safe",
}
REDUCTION_OPS = {"tessera.softmax", "tessera.softmax_safe", "tessera.reduce"}
LAYOUT_OPS = {"tessera.transpose", "tessera.cast"}
STATEFUL_FUNCTIONAL_OPS = {"tessera.adam"}
ROPE_OPS = {"tessera.rope"}


@dataclass
class ReferenceKVCache:
    keys: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)

    def append(self, key: Any, value: Any) -> "ReferenceKVCache":
        self.keys.append(np.asarray(key))
        self.values.append(np.asarray(value))
        return self

    def prune(self, max_entries: Optional[int] = None) -> "ReferenceKVCache":
        if max_entries is not None:
            self.keys = self.keys[-max_entries:]
            self.values = self.values[-max_entries:]
        return self


@dataclass(frozen=True)
class LoweringArtifact:
    """Textual artifact for one compiler layer."""

    level: str
    text: str


@dataclass(frozen=True)
class JitDiagnostic:
    """Developer-facing diagnostic for JIT lowering decisions."""

    severity: str
    code: str
    message: str

    def format(self) -> str:
        return f"{self.severity.upper()}[{self.code}]: {self.message}"


@dataclass(frozen=True)
class CPUPlan:
    """Executable CPU plan for straight-line supported Graph IR ops."""

    function_name: str
    ops: tuple[IROp, ...]
    output_name: str
    tile: tuple[int, int, int]
    target_kind: str
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str
    selected_schedule: dict[str, Any] | None = None

    @property
    def op_name(self) -> str:
        return self.ops[-1].op_name

    @property
    def operand_names(self) -> tuple[str, ...]:
        names = []
        produced = {op.result for op in self.ops if op.result}
        for op in self.ops:
            for operand in op.operands:
                name = _operand_name(operand)
                if name not in produced and name not in names:
                    names.append(name)
        return tuple(names)

    def execute(self, args: Sequence[Any], kwargs: Mapping[str, Any], arg_names: Sequence[str]) -> Any:
        """Walk ops with scf-bracket awareness.

        Followup 1 (audit 2026-05-31): when the body contains
        ``tessera.scf.if.begin / .else / .end`` markers with an
        SSA-operand condition, execute the appropriate branch only —
        skipping the dead branch entirely. Previously every op was
        executed unconditionally; programs that reached this method
        with scf markers in their body were either rejected upstream
        (``build_cpu_plan`` returned ``None``) or executed both
        branches and corrupted SSA bindings. Now scf.if is a real
        first-class control structure in the CPU plan executor.
        """
        values = {name: value for name, value in zip(arg_names, args)}
        values.update(kwargs)
        self._execute_range(0, len(self.ops), values)
        if self.output_name not in values:
            raise ValueError(f"CPU plan did not produce output {self.output_name!r}")
        return values[self.output_name]

    def _execute_range(self, lo: int, hi: int, values: dict[str, Any]) -> None:
        i = lo
        while i < hi:
            op = self.ops[i]
            if op.op_name == "tessera.scf.if.begin":
                # Find matching .else / .end by depth tracking. Body
                # ranges are [begin+1, else_idx) and [else_idx+1, end_idx).
                else_idx, end_idx = _find_scf_if_brackets(self.ops, i)
                # Resolve the condition value. The dynamic-SSA case has
                # the condition as operand[0]; the static-attr case is
                # handled later (kwargs["condition"] = True/False).
                if op.operands:
                    cond_name = _operand_name(op.operands[0])
                    if cond_name not in values:
                        raise ValueError(
                            f"scf.if condition operand {cond_name!r} "
                            f"not bound; CPU plan can't dispatch")
                    cond = bool(_as_value(values[cond_name]))
                else:
                    cond = bool(op.kwargs.get("condition", True))
                # Pick the active branch range and execute it inline.
                if cond:
                    branch_lo, branch_hi = i + 1, else_idx if else_idx is not None else end_idx
                else:
                    branch_lo, branch_hi = (
                        (else_idx + 1, end_idx) if else_idx is not None
                        else (end_idx, end_idx)  # empty else
                    )
                self._execute_range(branch_lo, branch_hi, values)
                i = end_idx + 1
                continue
            if op.op_name in ("tessera.scf.if.end", "tessera.scf.else"):
                # Shouldn't reach here if bracket-matching is correct
                # (the begin handler skips past .end), but defensive.
                i += 1
                continue
            if op.op_name == "tessera.scf.for.begin":
                end_idx = _find_scf_for_end(self.ops, i)
                trip_count = _static_trip_count(op, values)
                induction = str(op.kwargs.get("induction", "_"))
                old_value = values.get(induction, None)
                had_old_value = induction in values
                for idx in range(trip_count):
                    if induction != "_":
                        values[induction] = idx
                    self._execute_range(i + 1, end_idx, values)
                if induction != "_":
                    if had_old_value:
                        values[induction] = old_value
                    else:
                        values.pop(induction, None)
                i = end_idx + 1
                continue
            if op.op_name == "tessera.scf.for.end":
                i += 1
                continue
            self._execute_one(op, values)
            i += 1

    def _execute_one(self, op: IROp, values: dict[str, Any]) -> None:
        """Dispatch a single non-control-flow op."""
        operand_names = tuple(_operand_name(operand) for operand in op.operands)
        missing = [name for name in operand_names if name not in values]
        if missing:
            raise ValueError(
                f"CPU plan requires operand(s): {', '.join(missing)}")
        operands = [_as_value(values[name]) for name in operand_names]
        if op.result is None:
            raise ValueError(
                f"CPU plan cannot execute void op {op.op_name!r}")
        values[op.result] = _execute_op(op.op_name, operands, op.kwargs)

    def artifacts(self) -> tuple[LoweringArtifact, ...]:
        return (
            LoweringArtifact("graph", self.graph_ir),
            LoweringArtifact("schedule", self.schedule_ir),
            LoweringArtifact("tile", self.tile_ir),
            LoweringArtifact("target", self.target_ir),
        )


MatmulCPUPlan = CPUPlan


def build_matmul_cpu_plan(
    module: GraphIRModule,
    *,
    tile: tuple[int, int, int] = (128, 128, 64),
) -> Optional[CPUPlan]:
    """Backward-compatible alias for the general single-op CPU planner."""

    return build_cpu_plan(module, tile=tile)


# Followup 1 (audit 2026-05-31) — scf.if.* markers are now first-class
# control flow in the CPU executor, not generic unknown ops. Listed
# here separately from SUPPORTED_CPU_OPS so the planner can accept
# bodies that mix scf markers with supported leaf ops.
_SUPPORTED_CONTROL_FLOW_OPS = frozenset({
    "tessera.scf.if.begin",
    "tessera.scf.else",
    "tessera.scf.if.end",
    "tessera.scf.for.begin",
    "tessera.scf.for.end",
})


def _scf_body_is_plannable(body: "Sequence[IROp]") -> bool:
    """Followup 1 — True iff every scf op in ``body`` is one the CPU
    plan executor handles: ``scf.if.{begin,else,end}``, each
    ``scf.if.begin`` either carries an SSA operand condition or a
    static literal in ``kwargs["condition"]``; and ``scf.for.{begin,end}``
    with either a static ``trip_count`` attr or an SSA-bound trip-count
    operand. ``scf.while.*`` and text-only loops remain eager-only."""
    for op in body:
        name = _canonical_op_name(op.op_name)
        if not name.startswith("tessera.scf."):
            continue
        if name not in _SUPPORTED_CONTROL_FLOW_OPS:
            return False  # scf.while etc.
        if name == "tessera.scf.if.begin":
            if not op.operands and "condition" not in op.kwargs:
                return False  # text-only condition can't be evaluated
        if name == "tessera.scf.for.begin":
            if op.operands:
                continue  # dynamic SSA trip count
            if "trip_count" not in op.kwargs:
                return False  # text-only / unlowered trip count
            try:
                if int(op.kwargs["trip_count"]) < 0:
                    return False
            except (TypeError, ValueError):
                return False
    return True


def _find_scf_if_brackets(
    ops: "Sequence[IROp]", begin_idx: int,
) -> tuple[Optional[int], int]:
    """For an op at ``begin_idx`` of kind ``tessera.scf.if.begin``,
    return ``(else_idx, end_idx)`` — the matching ``scf.else`` (or
    ``None`` if absent) and the matching ``scf.if.end`` index. Uses
    depth tracking so nested scf.if blocks don't confuse the matcher.
    """
    depth = 1
    else_idx: Optional[int] = None
    for j in range(begin_idx + 1, len(ops)):
        name = ops[j].op_name
        if name == "tessera.scf.if.begin":
            depth += 1
        elif name == "tessera.scf.if.end":
            depth -= 1
            if depth == 0:
                return else_idx, j
        elif name == "tessera.scf.else" and depth == 1:
            else_idx = j
    raise ValueError(
        f"unbalanced scf.if at index {begin_idx} (no matching scf.if.end)")


def _find_scf_for_end(ops: "Sequence[IROp]", begin_idx: int) -> int:
    """For an op at ``begin_idx`` of kind ``tessera.scf.for.begin``,
    return the matching ``scf.for.end`` index."""
    depth = 1
    for j in range(begin_idx + 1, len(ops)):
        name = ops[j].op_name
        if name == "tessera.scf.for.begin":
            depth += 1
        elif name == "tessera.scf.for.end":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError(
        f"unbalanced scf.for at index {begin_idx} (no matching scf.for.end)")


def _static_trip_count(op: IROp, values: Mapping[str, Any]) -> int:
    if "trip_count" in op.kwargs:
        trip_count = int(op.kwargs["trip_count"])
    elif op.operands:
        name = _operand_name(op.operands[0])
        if name not in values:
            raise ValueError(
                f"scf.for trip-count operand {name!r} not bound; "
                "CPU plan can't dispatch")
        trip_count = int(np.asarray(_as_value(values[name])).item())
    else:
        raise ValueError("scf.for requires a static trip_count")
    if trip_count < 0:
        raise ValueError("scf.for trip_count must be non-negative")
    return trip_count


def build_cpu_plan(
    module: GraphIRModule,
    *,
    tile: tuple[int, int, int] = (128, 128, 64),
    target_kind: str = "cpu",
) -> Optional[CPUPlan]:
    """Build a hardware-free lowering artifact plan for supported straight-line dataflow."""

    _validate_tile(tile)
    target_kind = normalize_target_kind(target_kind)
    if len(module.functions) != 1:
        return None
    fn = module.functions[0]
    if not fn.body:
        return None
    for op in fn.body:
        name = _canonical_op_name(op.op_name)
        # Followup 1 accepted scf.if markers. Sprint C added static
        # trip-count scf.for; Sprint D accepts SSA-bound dynamic
        # trip-count scf.for. Text-only loops and scf.while remain
        # eager-only.
        if name in _SUPPORTED_CONTROL_FLOW_OPS:
            if name == "tessera.scf.if.begin":
                # Only operand-condition or static-attr scf.if is
                # plannable. Text-only conditions (D's "case 3", no
                # operand, kwargs["condition_text"] only) need eager
                # Python to evaluate the source — skip the plan.
                if not op.operands and "condition" not in op.kwargs:
                    return None
            if name == "tessera.scf.for.begin":
                if op.operands:
                    continue
                if "trip_count" not in op.kwargs:
                    return None
                try:
                    if int(op.kwargs["trip_count"]) < 0:
                        return None
                except (TypeError, ValueError):
                    return None
            continue
        if name not in SUPPORTED_CPU_OPS or not _valid_arity(op):
            return None
        operand_names = tuple(_operand_name(operand) for operand in op.operands)
        if any(not n or n == "?" for n in operand_names):
            return None
        if op.result is None:
            return None
    # Output: the result of the last non-marker op.
    output_name = None
    for op in reversed(fn.body):
        if op.result is not None:
            output_name = op.result
            break
    if output_name is None:
        return None

    graph_text = module.to_mlir()
    ops = tuple(fn.body)
    selected_schedule = _select_schedule(fn, ops, tile=tile, target_kind=target_kind)
    schedule = _render_schedule_ir(module, fn, ops, tile=tile, target_kind=target_kind)
    tile_ir = _render_tile_ir(module, fn, ops, tile=tile, target_kind=target_kind)
    target = _render_target_ir(module, fn, ops, tile=tile, target_kind=target_kind)
    return CPUPlan(
        function_name=fn.name,
        ops=ops,
        output_name=output_name,
        tile=tile,
        target_kind=target_kind,
        graph_ir=graph_text,
        schedule_ir=schedule,
        tile_ir=tile_ir,
        target_ir=target,
        selected_schedule=selected_schedule,
    )


def explain_cpu_plan(module: GraphIRModule, *, target: str = "cpu") -> JitDiagnostic:
    """Return a diagnostic explaining compile-path or fallback status.

    The ``code`` field of the returned :class:`JitDiagnostic` is a
    string for backwards compatibility, but its values are taken
    verbatim from :class:`tessera.compiler.diagnostics.JitDiagnosticCode`
    so callers can match against ``JitDiagnosticCode.EAGER_FALLBACK_EMPTY.value``
    etc. instead of string literals.
    """

    from .diagnostics import JitDiagnosticCode as _Code

    target = normalize_target_kind(target)
    if not module.functions:
        return JitDiagnostic("warning", _Code.EAGER_FALLBACK_EMPTY.value, "no Graph IR function was emitted")
    fn = module.functions[0]
    if not fn.body:
        return JitDiagnostic("warning", _Code.EAGER_FALLBACK_EMPTY.value, "no Graph IR function body was emitted")
    # A.2 (2026-05-31) — structured control flow markers (scf.if /
    # scf.for / scf.while) are lowered correctly by D.1/D.2/D.3.
    # Followup 1 (2026-05-31) — scf.if with an SSA-operand or static
    # condition is now executable through the CPU plan's branch-aware
    # executor. If the body contains an scf.if that the planner CAN'T
    # handle (text-only condition, or nested scf, or scf.for/while),
    # the eager-fallback diagnostic still fires. Otherwise: no
    # diagnostic — the function compiles through the real plan.
    scf_ops = [op for op in fn.body
               if _canonical_op_name(op.op_name).startswith("tessera.scf.")]
    if scf_ops:
        plannable = _scf_body_is_plannable(fn.body)
        if not plannable:
            seen = scf_ops[0].op_name
            return JitDiagnostic(
                "info",
                _Code.EAGER_FALLBACK_CONTROL_FLOW.value,
                (f"function contains structured control flow ({seen!r} and "
                 f"{len(scf_ops) - 1} other scf op(s)); CPU plan executor "
                 f"handles scf.if with SSA / static conditions and "
                 f"static or SSA-bound trip-count scf.for, but this body "
                 f"has scf.while, text-only scf.for, nested unsupported "
                 f"control flow, or a text-only condition — "
                 f"falling back to eager Python (numerically correct, "
                 f"unoptimized)"),
            )
    unsupported = [
        op for op in fn.body
        if _canonical_op_name(op.op_name) not in SUPPORTED_CPU_OPS
        and _canonical_op_name(op.op_name) not in _SUPPORTED_CONTROL_FLOW_OPS
    ]
    if unsupported:
        names = ", ".join(sorted(SUPPORTED_CPU_OPS))
        seen = unsupported[0].op_name
        return JitDiagnostic(
            "warning",
            _Code.EAGER_FALLBACK_UNSUPPORTED_OP.value,
            f"op {seen!r} is not supported by the CPU compiler path; supported ops: {names}",
        )
    bad_arity = [op for op in fn.body if not _valid_arity(op)]
    if bad_arity:
        return JitDiagnostic(
            "warning",
            _Code.EAGER_FALLBACK_ARITY.value,
            f"op {bad_arity[0].op_name!r} has unsupported operand count",
        )
    unknown = [
        op
        for op in fn.body
        if _canonical_op_name(op.op_name) not in _SUPPORTED_CONTROL_FLOW_OPS
        and (op.result is None or any(_operand_name(operand) in {"", "?"} for operand in op.operands))
    ]
    if unknown:
        return JitDiagnostic(
            "warning",
            _Code.EAGER_FALLBACK_UNSUPPORTED_BODY.value,
            "CPU compiler needs named values for every supported op; using eager Python fallback",
        )
    if target == "cpu":
        return JitDiagnostic(
            "info",
            _Code.COMPILED_CPU.value,
            f"compiled {fn.name} through Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU",
        )
    return JitDiagnostic(
        "info",
        _Code.TARGET_IR_ARTIFACT_ONLY.value,
        (
            f"compiled {fn.name} through Graph IR -> Schedule IR -> Tile IR -> "
            f"{target} Target IR artifact; native execution is not wired"
        ),
    )


def _render_schedule_ir(
    module: GraphIRModule,
    fn: GraphIRFunction,
    ops: Sequence[IROp],
    *,
    tile: tuple[int, int, int],
    target_kind: str,
) -> str:
    return lower_graph_to_schedule_ir(module, tile=tile, target_kind=target_kind).to_mlir()


def _render_tile_ir(
    module: GraphIRModule,
    fn: GraphIRFunction,
    ops: Sequence[IROp],
    *,
    tile: tuple[int, int, int],
    target_kind: str,
) -> str:
    schedule = lower_graph_to_schedule_ir(module, tile=tile, target_kind=target_kind)
    return lower_schedule_to_tile_ir(schedule, target_kind=target_kind).to_mlir()


def _render_target_ir(
    module: GraphIRModule,
    fn: GraphIRFunction,
    ops: Sequence[IROp],
    *,
    tile: tuple[int, int, int],
    target_kind: str,
) -> str:
    if target_kind == "metalium":
        return _render_metalium_target_ir(fn, ops)
    if target_kind in {"cpu", "rocm", "apple_cpu", "apple_gpu"} or target_kind.startswith("nvidia"):
        return _render_object_target_ir(module, tile=tile, target_kind=target_kind)
    lines = [
        'module attributes {tessera.ir.level = "target", target = "cpu"} {',
        '  "tessera.cpu.func"() ({',
    ]
    for idx, op in enumerate(ops):
        op_name = _canonical_op_name(op.op_name)
        lines.append(
            f'    "{_target_op_name(op_name)}"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, abi = "numpy"}} : () -> ()'
        )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_object_target_ir(
    module: GraphIRModule,
    *,
    tile: tuple[int, int, int],
    target_kind: str,
) -> str:
    schedule = lower_graph_to_schedule_ir(module, tile=tile, target_kind=target_kind)
    tile_module = lower_schedule_to_tile_ir(schedule, target_kind=target_kind)
    return lower_tile_to_target_ir(tile_module, target_kind=target_kind).to_mlir()


def _render_rocm_target_ir(fn: GraphIRFunction, ops: Sequence[IROp]) -> str:
    lines = [
        'module attributes {tessera.ir.level = "target", target = "rocm", arch = "gfx90a"} {',
        '  "tessera_rocm.func"() ({',
    ]
    for idx, op in enumerate(ops):
        op_name = _canonical_op_name(op.op_name)
        if op_name in MATMUL_OPS:
            lines.append(
                f'    "tessera_rocm.mfma"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, arch = "gfx90a", shape = "m16n16k16", accum = "f32"}} : () -> ()'
            )
        elif op_name == "tessera.flash_attn":
            lines.append(
                f'    "tessera.target.diagnostic"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, target = "rocm", severity = "unsupported", reason = "flash_attn target kernel contract is not implemented for ROCm in this phase"}} : () -> ()'
            )
        elif op_name.startswith("tessera.kv_cache."):
            lines.append(
                f'    "tessera.target.diagnostic"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, target = "rocm", severity = "unsupported", reason = "KV-cache target lowering is not implemented for ROCm in this phase"}} : () -> ()'
            )
        else:
            lines.append(
                f'    "tessera_rocm.elementwise"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, arch = "gfx90a"}} : () -> ()'
            )
        lines.append(
            f'    "tessera_rocm.async_copy"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, src_space = "global", dst_space = "lds", bytes = 16 : i64}} : () -> ()'
        )
        lines.append(f'    "tessera_rocm.wait"() {{ordinal = {idx} : i64}} : () -> ()')
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_metalium_target_ir(fn: GraphIRFunction, ops: Sequence[IROp]) -> str:
    lines = [
        'module attributes {tessera.ir.level = "target", target = "metalium", arch = "wormhole"} {',
        '  "tessera_metalium.program"() ({',
    ]
    for idx, op in enumerate(ops):
        op_name = _canonical_op_name(op.op_name)
        if op_name in MATMUL_OPS:
            lines.append(
                f'    "tessera_metalium.dma"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, direction = "dram_to_sram", burst = 256 : i64}} : () -> ()'
            )
            lines.append(
                f'    "tessera_metalium.matmul"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, tile = [64, 64, 32], layout = "row_col", accumulate = "f32"}} : () -> ()'
            )
        elif op_name.startswith("tessera.kv_cache."):
            lines.append(
                f'    "tessera.target.diagnostic"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, target = "metalium", severity = "unsupported", reason = "KV-cache paged-buffer target contract is not implemented for Metalium in this phase"}} : () -> ()'
            )
        else:
            lines.append(
                f'    "tessera_metalium.kernel"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, program = "mock_queue"}} : () -> ()'
            )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_apple_cpu_target_ir(fn: GraphIRFunction, ops: Sequence[IROp]) -> str:
    lines = [
        'module attributes {tessera.ir.level = "target", target = "apple_cpu", arch = "arm64-apple-silicon", execution_mode = "cpu_accelerate"} {',
        '  "tessera_apple.cpu.func"() ({',
    ]
    for idx, op in enumerate(ops):
        op_name = _canonical_op_name(op.op_name)
        if op_name in MATMUL_OPS:
            target_op = "tessera_apple.cpu.accelerate_gemm"
            attrs = 'framework = "Accelerate", abi = "cblas_sgemm", dtype = "f32"'
        elif op_name in REDUCTION_OPS:
            target_op = "tessera_apple.cpu.vector_reduce"
            attrs = 'framework = "Accelerate", abi = "vDSP", dtype = "f32"'
        elif op_name in ROPE_OPS:
            target_op = "tessera_apple.cpu.vector_op"
            attrs = 'framework = "Accelerate", abi = "vecLib", pattern = "rotary_pairs", dtype = "f32"'
        else:
            target_op = "tessera_apple.cpu.vector_op"
            attrs = 'framework = "Accelerate", abi = "vecLib", dtype = "f32"'
        lines.append(
            f'    "{target_op}"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, {attrs}}} : () -> ()'
        )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_apple_gpu_target_ir(fn: GraphIRFunction, ops: Sequence[IROp]) -> str:
    lines = [
        'module attributes {tessera.ir.level = "target", target = "apple_gpu", arch = "apple-metal", execution_mode = "metal_artifact"} {',
        '  "tessera_apple.gpu.func"() ({',
    ]
    for idx, op in enumerate(ops):
        op_name = _canonical_op_name(op.op_name)
        if op_name.startswith("tessera.kv_cache."):
            lines.append(
                f'    "tessera_apple.diagnostic"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, severity = "unsupported", reason = "KV-cache target lowering is not implemented for Apple GPU in this phase"}} : () -> ()'
            )
            continue
        if op_name == "tessera.flash_attn":
            kernel = "flash_attn_contract"
            framework = "Metal"
            extra = 'status = "artifact_only", grid = "bhn", threadgroup = "64x1x1", temporary_memory = "scores_lse"'
        elif op_name in MATMUL_OPS:
            kernel = "matmul_contract"
            framework = "MPSGraph"
            extra = 'status = "artifact_only", grid = "mn_tiles", threadgroup = "16x16x1", temporary_memory = "none"'
        elif op_name in REDUCTION_OPS:
            kernel = "softmax_contract"
            framework = "MPSGraph"
            extra = 'status = "artifact_only", grid = "rows", threadgroup = "256x1x1", temporary_memory = "row_max_sum"'
        elif op_name in ROPE_OPS:
            kernel = "rope_contract"
            framework = "Metal"
            extra = 'status = "artifact_only", grid = "tokens_heads", threadgroup = "128x1x1", temporary_memory = "none"'
        else:
            kernel = "elementwise_contract"
            framework = "Metal"
            extra = 'status = "artifact_only", grid = "elements", threadgroup = "256x1x1", temporary_memory = "none"'
        lines.append(
            f'    "tessera_apple.gpu.metal_kernel"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, kernel = "{kernel}", framework = "{framework}", dtype = "f32", {extra}}} : () -> ()'
        )
        lines.append(
            f'    "tessera_apple.gpu.dispatch"() {{ordinal = {idx} : i64, queue = "MTLCommandQueue", artifact = "metallib", execution_mode = "metal_artifact"}} : () -> ()'
        )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_nvidia_target_ir(fn: GraphIRFunction, ops: Sequence[IROp], *, target_kind: str) -> str:
    is_blackwell = target_kind in {"nvidia_sm100", "nvidia_sm120"}
    arch = {
        "nvidia_sm80": "sm_80",
        "nvidia_sm90": "sm_90a",
        "nvidia_sm100": "sm_100a",
        "nvidia_sm120": "sm_120",
    }.get(target_kind, "sm_90a")
    lines = [
        f'module attributes {{tessera.ir.level = "target", target = "{target_kind}", arch = "{arch}"}} {{',
        '  "tessera_nvidia.func"() ({',
    ]
    for idx, op in enumerate(ops):
        op_name = _canonical_op_name(op.op_name)
        if op_name in MATMUL_OPS:
            if is_blackwell:
                lines.append(
                    f'    "tessera_nvidia.tmem_alloc"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, arch = "{arch}", columns = 128 : i64}} : () -> ()'
                )
                lines.append(
                    f'    "tessera_nvidia.tcgen05_mma"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, arch = "{arch}", shape = "m128n128k32", accum = "tmem_f32", cta_group = 2 : i64, block_scaled = true}} : () -> ()'
                )
            else:
                lines.append(
                    f'    "tessera_nvidia.wgmma"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, arch = "{arch}", shape = "m64n64k16", dtype_ab = "bf16", dtype_c = "f32", warpgroup = 4 : i64}} : () -> ()'
                )
                lines.append(
                    f'    "tessera_nvidia.tma_async_copy"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, arch = "{arch}", src_space = "global", dst_space = "shared", bytes = 16 : i64}} : () -> ()'
                )
                lines.append(
                    f'    "tessera_nvidia.mbarrier"() {{ordinal = {idx} : i64, arch = "{arch}", scope = "cta"}} : () -> ()'
                )
        elif op_name == "tessera.flash_attn":
            lines.append(
                f'    "tessera_nvidia.cuda_kernel"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, arch = "{arch}", kernel = "flash_attn_contract", status = "artifact_only"}} : () -> ()'
            )
        else:
            lines.append(
                f'    "tessera_nvidia.cuda_kernel"() {{source = "{op_name}", result = "{op.result}", ordinal = {idx} : i64, arch = "{arch}", kernel = "elementwise_contract", status = "artifact_only"}} : () -> ()'
            )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _operand_name(operand: str) -> str:
    return operand[1:] if operand.startswith("%") else operand


def _validate_tile(tile: tuple[int, int, int]) -> None:
    if len(tile) != 3 or any(int(v) <= 0 for v in tile):
        raise ValueError("CPU matmul tile must be a positive (tile_m, tile_n, tile_k) tuple")


def _valid_arity(op: IROp) -> bool:
    spec = GRAPH_OP_TO_SPEC.get(_canonical_op_name(op.op_name))
    return spec.valid_arity(len(op.operands)) if spec is not None else False


def _execute_op(op_name: str, operands: Sequence[np.ndarray], kwargs: Mapping[str, Any]) -> Any:
    op_name = _canonical_op_name(op_name)
    if op_name in MATMUL_OPS:
        return np.matmul(operands[0], operands[1])
    if op_name in CONV2D_OPS:
        bias = operands[2] if len(operands) > 2 else kwargs.get("bias", None)
        return _conv2d_nhwc(operands[0], operands[1], bias=bias, stride=kwargs.get("stride", 1), padding=kwargs.get("padding", 0))
    if op_name == "tessera.layer_norm":
        x = np.asarray(operands[0])
        eps = float(kwargs.get("eps", 1e-5))
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)
    if op_name == "tessera.relu":
        return np.maximum(0, operands[0])
    if op_name == "tessera.silu":
        x = np.asarray(operands[0])
        return x / (1.0 + np.exp(-x))
    if op_name == "tessera.sigmoid":
        return 1.0 / (1.0 + np.exp(-operands[0]))
    if op_name == "tessera.sin":
        return np.sin(operands[0])
    if op_name == "tessera.gelu":
        x = np.asarray(operands[0])
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    if op_name == "tessera.tanh":
        return np.tanh(operands[0])
    if op_name == "tessera.add":
        rhs = operands[1] if len(operands) > 1 else kwargs.get("scalar", 0.0)
        return np.asarray(operands[0]) + rhs
    if op_name == "tessera.mul":
        rhs = operands[1] if len(operands) > 1 else kwargs.get("scalar", 1.0)
        return np.asarray(operands[0]) * rhs
    if op_name in {"tessera.softmax", "tessera.softmax_safe"}:
        x = operands[0]
        axis = int(kwargs.get("axis", -1))
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    if op_name == "tessera.reduce":
        if str(kwargs.get("op", "sum")) != "sum":
            raise ValueError("CPU compiler path only supports tessera.reduce op='sum'")
        axis = kwargs.get("axis", None)
        if axis is not None:
            axis = int(axis)
        return np.sum(operands[0], axis=axis, keepdims=bool(kwargs.get("keepdims", False)))
    if op_name in {"tessera.rmsnorm", "tessera.rmsnorm_safe"}:
        x = np.asarray(operands[0])
        eps = float(kwargs.get("eps", 1e-5 if op_name == "tessera.rmsnorm" else 1e-6))
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    if op_name == "tessera.transpose":
        axes = kwargs.get("axes", None)
        if isinstance(axes, list):
            axes = tuple(axes)
        return np.transpose(operands[0], axes)
    if op_name == "tessera.cast":
        dtype = str(kwargs.get("dtype", "fp32"))
        cast_map: dict[str, Any] = {"bf16": np.float32, "fp16": np.float16, "fp32": np.float32, "fp64": np.float64}
        return np.asarray(operands[0]).astype(cast_map.get(dtype, np.float32))
    if op_name == "tessera.dropout":
        x = np.asarray(operands[0])
        if not bool(kwargs.get("training", True)):
            return x
        p = float(kwargs.get("p", 0.1))
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout p must be in [0.0, 1.0)")
        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(None if seed is None else int(seed))
        mask = rng.binomial(1, 1.0 - p, x.shape) / (1.0 - p)
        return x * mask
    if op_name == "tessera.flash_attn":
        return _flash_attn_reference(operands[0], operands[1], operands[2], kwargs)
    if op_name in ROPE_OPS:
        return _rope_reference(operands[0], operands[1])
    if op_name in {"tessera.all_reduce", "tessera.reduce_scatter", "tessera.all_gather", "tessera.all_to_all"}:
        return operands[0]
    if op_name in {"tessera.rng_uniform", "tessera.rng_normal"}:
        shape = tuple(kwargs.get("shape", ()))
        dtype = str(kwargs.get("dtype", "fp32"))
        dtype_map: dict[str, Any] = {
            "bf16": np.float32,
            "fp16": np.float16,
            "fp32": np.float32,
            "fp64": np.float64,
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "bool": np.bool_,
        }
        rng = np.random.default_rng(None if kwargs.get("seed", None) is None else int(kwargs["seed"]))
        if op_name == "tessera.rng_uniform":
            out = rng.uniform(float(kwargs.get("lo", 0.0)), float(kwargs.get("hi", 1.0)), shape)
        else:
            out = rng.normal(float(kwargs.get("mean", 0.0)), float(kwargs.get("std", 1.0)), shape)
        return out.astype(dtype_map.get(dtype, np.float32))
    if op_name == "tessera.fused_epilogue":
        x = np.asarray(operands[0])
        bias = operands[1] if len(operands) > 1 else kwargs.get("bias", None)
        if bias is not None:
            x = x + bias
        activation = kwargs.get("activation", "linear")
        if activation == "gelu":
            return _execute_op("tessera.gelu", [x], {})
        if activation == "relu":
            return np.maximum(0, x)
        return x
    if op_name == "tessera.fft":
        return np.fft.fft(operands[0], axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.ifft":
        return np.fft.ifft(operands[0], axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.rfft":
        return np.fft.rfft(operands[0], axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.irfft":
        n = kwargs.get("n", None)
        return np.fft.irfft(operands[0], n=None if n is None else int(n), axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.dct":
        return _dct_reference(operands[0], axis=int(kwargs.get("axis", -1)))
    if op_name == "tessera.spectral_conv":
        x = np.asarray(operands[0])
        w = np.asarray(operands[1])
        n = x.shape[-1] + w.shape[-1] - 1
        nfft = 1 << int(np.ceil(np.log2(n)))
        y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(w, nfft), nfft)
        return y[..., :n]
    if op_name == "tessera.kv_cache.append":
        cache = operands[0] if isinstance(operands[0], ReferenceKVCache) else ReferenceKVCache()
        return cache.append(operands[1], operands[2])
    if op_name == "tessera.kv_cache.prune":
        cache = operands[0] if isinstance(operands[0], ReferenceKVCache) else ReferenceKVCache()
        max_entries = kwargs.get("max_entries", kwargs.get("max_seq", None))
        return cache.prune(None if max_entries is None else int(max_entries))
    if op_name == "tessera.adam":
        param, grad, moment1, moment2 = operands
        beta1 = float(kwargs.get("beta1", 0.9))
        beta2 = float(kwargs.get("beta2", 0.999))
        lr = float(kwargs.get("lr", 1e-3))
        eps = float(kwargs.get("eps", 1e-8))
        step = int(kwargs.get("step", 1))
        new_m = beta1 * moment1 + (1.0 - beta1) * grad
        new_v = beta2 * moment2 + (1.0 - beta2) * (grad * grad)
        m_hat = new_m / (1.0 - beta1**step)
        v_hat = new_v / (1.0 - beta2**step)
        new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
        return new_param, new_m, new_v
    spec = GRAPH_OP_TO_SPEC.get(op_name)
    if spec is not None:
        import tessera

        return tessera.ops.registry.dispatch(spec.public_name, *operands, prefer_runtime=False, **kwargs)
    raise ValueError(f"unsupported CPU op {op_name!r}")


def _tile_op_name(op_name: str) -> str:
    op_name = _canonical_op_name(op_name)
    bare = op_name.split(".")[-1]
    if op_name in MATMUL_OPS:
        return "tile.mma"
    if op_name in CONV2D_OPS:
        return "tile.conv2d"
    if op_name in ROPE_OPS:
        return "tile.rope"
    return f"tile.{bare}"


def _target_op_name(op_name: str) -> str:
    op_name = _canonical_op_name(op_name)
    bare = op_name.split(".")[-1]
    if op_name in MATMUL_OPS:
        bare = "matmul"
    if op_name in CONV2D_OPS:
        bare = "conv2d_nhwc"
    return f"tessera.cpu.{bare}"


def _lowering_kind(op_name: str) -> str:
    spec = GRAPH_OP_TO_SPEC.get(_canonical_op_name(op_name))
    return spec.lowering if spec is not None else "elementwise"


def _canonical_op_name(op_name: str) -> str:
    return canonical_graph_op_name(LEGACY_GRAPH_OP_ALIASES.get(op_name, op_name))


def normalize_target_kind(target: object = "cpu") -> str:
    return _normalize_target(target)


def _select_schedule(
    fn: GraphIRFunction,
    ops: Sequence[IROp],
    *,
    tile: tuple[int, int, int],
    target_kind: str,
) -> dict[str, Any] | None:
    for op in ops:
        if _canonical_op_name(op.op_name) not in MATMUL_OPS:
            continue
        shapes = [_tensor_shape(str(t)) for t in [*op.operand_types[:2], op.result_type]]
        if len(shapes) < 3 or any(len(shape) != 2 for shape in shapes):
            return {
                "op_name": "tessera.matmul",
                "target": target_kind,
                "config": {"tile_m": tile[0], "tile_n": tile[1], "tile_k": tile[2], "num_warps": 4, "num_stages": 2},
                "method": "manual",
                "reason": "dynamic_or_unknown_shape",
            }
        m, k = shapes[0]
        _, n = shapes[1]
        if not all(isinstance(v, int) and v > 0 for v in (m, n, k)):
            return None
        return SchedulePlanner().plan_gemm(m=m, n=n, k=k, target=target_kind).to_dict()
    return None


def _tensor_shape(type_text: str | None) -> tuple[int, ...]:
    if not type_text or not type_text.startswith("tensor<") or "x" not in type_text:
        return ()
    body = type_text.removeprefix("tensor<").removesuffix(">")
    parts = body.split("x")[:-1]
    dims: list[int] = []
    for part in parts:
        try:
            dims.append(int(part))
        except ValueError:
            return ()
    return tuple(dims)


def _as_value(value: Any) -> Any:
    if isinstance(value, ReferenceKVCache):
        return value
    if hasattr(value, "numpy") and callable(value.numpy):
        return np.asarray(value.numpy())
    if hasattr(value, "_data"):
        return np.asarray(value._data)
    return np.asarray(value)


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv2d_nhwc(x: Any, weight: Any, *, bias: Any = None, stride: Any = 1, padding: Any = 0) -> np.ndarray:
    x = np.asarray(x)
    weight = np.asarray(weight)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d reference expects NHWC input and HWIO weights")
    x_pad = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    batch, in_h, in_w, _ = x_pad.shape
    k_h, k_w, _, out_c = weight.shape
    out_h = (in_h - k_h) // stride_h + 1
    out_w = (in_w - k_w) // stride_w + 1
    out = np.zeros((batch, out_h, out_w, out_c), dtype=np.result_type(x, weight))
    for i in range(out_h):
        for j in range(out_w):
            window = x_pad[:, i * stride_h:i * stride_h + k_h, j * stride_w:j * stride_w + k_w, :]
            out[:, i, j, :] = np.tensordot(window, weight, axes=([1, 2, 3], [0, 1, 2]))
    if bias is not None:
        out = out + np.asarray(bias)
    return out


def _flash_attn_reference(q: Any, k: Any, v: Any, kwargs: Mapping[str, Any]) -> np.ndarray:
    q = np.asarray(q)
    k = np.asarray(k)
    v = np.asarray(v)
    dropout_p = float(kwargs.get("dropout_p", 0.0))
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0.0, 1.0)")
    scale = kwargs.get("scale", None)
    scale = 1.0 / np.sqrt(q.shape[-1]) if scale is None else float(scale)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    if bool(kwargs.get("causal", False)):
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask = np.triu(np.ones((q_len, k_len), dtype=bool), k=1 + max(k_len - q_len, 0))
        scores = np.where(mask, -np.inf, scores)
    weights = _execute_op("tessera.softmax", [scores], {})
    if dropout_p > 0.0:
        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(None if seed is None else int(seed))
        keep = rng.binomial(1, 1.0 - dropout_p, weights.shape)
        weights = weights * keep / (1.0 - dropout_p)
    return np.matmul(weights, v)


def _dct_reference(x: Any, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)
    n = x.shape[axis]
    y = np.concatenate([x, np.flip(x, axis=axis)], axis=axis)
    spec = np.fft.fft(y, axis=axis)
    slicer = [slice(None)] * spec.ndim
    slicer[axis] = slice(0, n)
    return np.real(spec[tuple(slicer)])


def _rope_reference(x: Any, theta: Any) -> np.ndarray:
    x = np.asarray(x)
    theta = np.asarray(theta)
    if x.shape[-1] % 2 != 0:
        raise ValueError("rope requires an even innermost dimension")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    if theta.shape[-1] == x.shape[-1]:
        theta = theta[..., 0::2]
    cos = np.cos(theta)
    sin = np.sin(theta)
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


__all__ = [
    "LoweringArtifact",
    "JitDiagnostic",
    "MATMUL_OPS",
    "CPUPlan",
    "MatmulCPUPlan",
    "ReferenceKVCache",
    "SUPPORTED_CPU_OPS",
    "build_cpu_plan",
    "build_matmul_cpu_plan",
    "explain_cpu_plan",
    "normalize_target_kind",
]
