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

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .graph_ir import GraphIRFunction, GraphIRModule, IROp


MATMUL_OPS = {"tessera.matmul", "tessera.gemm"}
UNARY_OPS = {"tessera.relu", "tessera.sigmoid", "tessera.sin"}
REDUCTION_OPS = {"tessera.softmax"}
LAYOUT_OPS = {"tessera.transpose"}
STATEFUL_FUNCTIONAL_OPS = {"tessera.adam"}
SUPPORTED_CPU_OPS = MATMUL_OPS | UNARY_OPS | REDUCTION_OPS | LAYOUT_OPS | STATEFUL_FUNCTIONAL_OPS


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
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str

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
        values = {name: value for name, value in zip(arg_names, args)}
        values.update(kwargs)
        for op in self.ops:
            operand_names = tuple(_operand_name(operand) for operand in op.operands)
            missing = [name for name in operand_names if name not in values]
            if missing:
                raise ValueError(f"CPU plan requires operand(s): {', '.join(missing)}")
            operands = [_as_numpy(values[name]) for name in operand_names]
            if op.result is None:
                raise ValueError(f"CPU plan cannot execute void op {op.op_name!r}")
            values[op.result] = _execute_op(op.op_name, operands, op.kwargs)
        if self.output_name not in values:
            raise ValueError(f"CPU plan did not produce output {self.output_name!r}")
        return values[self.output_name]

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


def build_cpu_plan(
    module: GraphIRModule,
    *,
    tile: tuple[int, int, int] = (128, 128, 64),
) -> Optional[CPUPlan]:
    """Build a CPU plan if the Graph IR module is supported straight-line dataflow."""

    _validate_tile(tile)
    if len(module.functions) != 1:
        return None
    fn = module.functions[0]
    if not fn.body:
        return None
    for op in fn.body:
        if op.op_name not in SUPPORTED_CPU_OPS or not _valid_arity(op):
            return None
        operand_names = tuple(_operand_name(operand) for operand in op.operands)
        if any(not name or name == "?" for name in operand_names):
            return None
        if op.result is None:
            return None
    output_name = fn.body[-1].result
    if output_name is None:
        return None

    graph_text = module.to_mlir()
    ops = tuple(fn.body)
    schedule = _render_schedule_ir(fn, ops, tile=tile)
    tile_ir = _render_tile_ir(fn, ops)
    target = _render_target_ir(fn, ops)
    return CPUPlan(
        function_name=fn.name,
        ops=ops,
        output_name=output_name,
        tile=tile,
        graph_ir=graph_text,
        schedule_ir=schedule,
        tile_ir=tile_ir,
        target_ir=target,
    )


def explain_cpu_plan(module: GraphIRModule, *, target: str = "cpu") -> JitDiagnostic:
    """Return a diagnostic explaining compile-path or fallback status."""

    if target != "cpu":
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_TARGET",
            f"native target {target!r} execution is not wired; using eager Python fallback",
        )
    if not module.functions:
        return JitDiagnostic("warning", "JIT_EAGER_FALLBACK_EMPTY", "no Graph IR function was emitted")
    fn = module.functions[0]
    if not fn.body:
        return JitDiagnostic("warning", "JIT_EAGER_FALLBACK_EMPTY", "no Graph IR function body was emitted")
    unsupported = [op for op in fn.body if op.op_name not in SUPPORTED_CPU_OPS]
    if unsupported:
        names = ", ".join(sorted(SUPPORTED_CPU_OPS))
        seen = unsupported[0].op_name
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_UNSUPPORTED_OP",
            f"op {seen!r} is not supported by the CPU compiler path; supported ops: {names}",
        )
    bad_arity = [op for op in fn.body if not _valid_arity(op)]
    if bad_arity:
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_ARITY",
            f"op {bad_arity[0].op_name!r} has unsupported operand count",
        )
    unknown = [
        op
        for op in fn.body
        if op.result is None or any(_operand_name(operand) in {"", "?"} for operand in op.operands)
    ]
    if unknown:
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_UNSUPPORTED_BODY",
            "CPU compiler needs named values for every supported op; using eager Python fallback",
        )
    return JitDiagnostic(
        "info",
        "JIT_COMPILED_CPU",
        f"compiled {fn.name} through Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU",
    )


def _render_schedule_ir(
    fn: GraphIRFunction,
    ops: Sequence[IROp],
    *,
    tile: tuple[int, int, int],
) -> str:
    tile_m, tile_n, tile_k = tile
    lines = [
        'module attributes {tessera.ir.level = "schedule"} {',
        f'  "tessera.schedule.func"() ({{',
    ]
    for idx, op in enumerate(ops):
        operand_names = tuple(_operand_name(operand) for operand in op.operands)
        operand_attr = ", ".join(f'"{name}"' for name in operand_names)
        if op.op_name in MATMUL_OPS:
            lines.append(
                f'    "tessera.schedule.tile"() {{source = "{op.op_name}", result = "{op.result}", ordinal = {idx} : i64, tile_m = {tile_m} : i64, tile_n = {tile_n} : i64, tile_k = {tile_k} : i64}} : () -> ()'
            )
        else:
            lines.append(
                f'    "tessera.schedule.elementwise"() {{source = "{op.op_name}", result = "{op.result}", ordinal = {idx} : i64, vectorize = true}} : () -> ()'
            )
        lines.append(
            f'    "tessera.schedule.layout"() {{operands = [{operand_attr}], layout = "row_major", ordinal = {idx} : i64}} : () -> ()'
        )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}", target = "cpu"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_tile_ir(fn: GraphIRFunction, ops: Sequence[IROp]) -> str:
    lines = [
        'module attributes {tessera.ir.level = "tile"} {',
        f'  "tessera.tile.func"() ({{',
    ]
    for idx, op in enumerate(ops):
        lines.append(
            f'    "{_tile_op_name(op.op_name)}"() {{source = "{op.op_name}", result = "{op.result}", ordinal = {idx} : i64, lowering = "{_lowering_kind(op.op_name)}", vectorize = true}} : () -> ()'
        )
    lines.extend([
        f'  }}) {{sym_name = "{fn.name}", target = "cpu"}} : () -> ()',
        "}",
    ])
    return "\n".join(lines)


def _render_target_ir(fn: GraphIRFunction, ops: Sequence[IROp]) -> str:
    lines = [
        'module attributes {tessera.ir.level = "target", target = "cpu"} {',
        f'  "tessera.cpu.func"() ({{',
    ]
    for idx, op in enumerate(ops):
        lines.append(
            f'    "{_target_op_name(op.op_name)}"() {{source = "{op.op_name}", result = "{op.result}", ordinal = {idx} : i64, abi = "numpy"}} : () -> ()'
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
    if op.op_name in MATMUL_OPS:
        return len(op.operands) == 2
    if op.op_name in UNARY_OPS | REDUCTION_OPS | LAYOUT_OPS:
        return len(op.operands) == 1
    if op.op_name in STATEFUL_FUNCTIONAL_OPS:
        return len(op.operands) == 4
    return False


def _execute_op(op_name: str, operands: Sequence[np.ndarray], kwargs: Mapping[str, Any]) -> Any:
    if op_name in MATMUL_OPS:
        return np.matmul(operands[0], operands[1])
    if op_name == "tessera.relu":
        return np.maximum(0, operands[0])
    if op_name == "tessera.sigmoid":
        return 1.0 / (1.0 + np.exp(-operands[0]))
    if op_name == "tessera.sin":
        return np.sin(operands[0])
    if op_name == "tessera.softmax":
        x = operands[0]
        axis = int(kwargs.get("axis", -1))
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    if op_name == "tessera.transpose":
        axes = kwargs.get("axes", None)
        if isinstance(axes, list):
            axes = tuple(axes)
        return np.transpose(operands[0], axes)
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
    raise ValueError(f"unsupported CPU op {op_name!r}")


def _tile_op_name(op_name: str) -> str:
    bare = op_name.split(".")[-1]
    if op_name in MATMUL_OPS:
        bare = "matmul"
    return f"tessera.tile.{bare}"


def _target_op_name(op_name: str) -> str:
    bare = op_name.split(".")[-1]
    if op_name in MATMUL_OPS:
        bare = "matmul"
    return f"tessera.cpu.{bare}"


def _lowering_kind(op_name: str) -> str:
    if op_name in MATMUL_OPS:
        return "loop_nest"
    if op_name == "tessera.softmax":
        return "stable_reduction"
    if op_name == "tessera.adam":
        return "functional_optimizer_step"
    if op_name == "tessera.transpose":
        return "layout_transform"
    return "elementwise"


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy") and callable(value.numpy):
        return np.asarray(value.numpy())
    if hasattr(value, "_data"):
        return np.asarray(value._data)
    return np.asarray(value)


__all__ = [
    "LoweringArtifact",
    "JitDiagnostic",
    "MATMUL_OPS",
    "CPUPlan",
    "MatmulCPUPlan",
    "SUPPORTED_CPU_OPS",
    "build_cpu_plan",
    "build_matmul_cpu_plan",
    "explain_cpu_plan",
]
