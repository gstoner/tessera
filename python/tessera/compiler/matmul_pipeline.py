"""Narrow end-to-end CPU lowering path for simple @jit ops.

This module intentionally handles one shape of program at a time:

    @tessera.jit
    def f(...):
        return tessera.ops.<supported_op>(...)

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
STATEFUL_FUNCTIONAL_OPS = {"tessera.adam"}
SUPPORTED_CPU_OPS = MATMUL_OPS | UNARY_OPS | REDUCTION_OPS | STATEFUL_FUNCTIONAL_OPS


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
    """Executable CPU plan for a single supported Graph IR op."""

    function_name: str
    op_name: str
    operand_names: tuple[str, ...]
    tile: tuple[int, int, int]
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str

    def execute(self, args: Sequence[Any], kwargs: Mapping[str, Any], arg_names: Sequence[str]) -> Any:
        values = {name: value for name, value in zip(arg_names, args)}
        values.update(kwargs)
        missing = [name for name in self.operand_names if name not in values]
        if missing:
            raise ValueError(f"CPU plan requires operand(s): {', '.join(missing)}")
        operands = [_as_numpy(values[name]) for name in self.operand_names]
        return _execute_op(self.op_name, operands)

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
    """Build a CPU plan if the Graph IR module is a supported single-op shape."""

    _validate_tile(tile)
    if len(module.functions) != 1:
        return None
    fn = module.functions[0]
    supported_ops = [op for op in fn.body if op.op_name in SUPPORTED_CPU_OPS]
    if len(supported_ops) != 1 or len(fn.body) != 1:
        return None
    op = supported_ops[0]
    if not _valid_arity(op):
        return None
    operand_names = tuple(_operand_name(operand) for operand in op.operands)
    if any(not name or name == "?" for name in operand_names):
        return None

    graph_text = module.to_mlir()
    schedule = _render_schedule_ir(fn, op, operand_names, tile=tile)
    tile_ir = _render_tile_ir(fn, op)
    target = _render_target_ir(fn, op)
    return CPUPlan(
        function_name=fn.name,
        op_name=op.op_name,
        operand_names=operand_names,
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
    supported = [op for op in fn.body if op.op_name in SUPPORTED_CPU_OPS]
    if len(fn.body) != 1:
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_UNSUPPORTED_BODY",
            "CPU compiler currently supports one returned op; using eager Python fallback",
        )
    if not supported:
        names = ", ".join(sorted(SUPPORTED_CPU_OPS))
        seen = fn.body[0].op_name if fn.body else "<none>"
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_UNSUPPORTED_OP",
            f"op {seen!r} is not supported by the CPU compiler path; supported ops: {names}",
        )
    if not _valid_arity(supported[0]):
        return JitDiagnostic(
            "warning",
            "JIT_EAGER_FALLBACK_ARITY",
            f"op {supported[0].op_name!r} has unsupported operand count",
        )
    return JitDiagnostic(
        "info",
        "JIT_COMPILED_CPU",
        f"compiled {fn.name} through Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU",
    )


def _render_schedule_ir(
    fn: GraphIRFunction,
    op: IROp,
    operand_names: Sequence[str],
    *,
    tile: tuple[int, int, int],
) -> str:
    operand_attr = ", ".join(f'"{name}"' for name in operand_names)
    tile_m, tile_n, tile_k = tile
    tile_line = (
        f'    "tessera.schedule.tile"() {{source = "{op.op_name}", tile_m = {tile_m} : i64, tile_n = {tile_n} : i64, tile_k = {tile_k} : i64}} : () -> ()'
        if op.op_name in MATMUL_OPS
        else f'    "tessera.schedule.elementwise"() {{source = "{op.op_name}", vectorize = true}} : () -> ()'
    )
    return "\n".join(
        [
            'module attributes {tessera.ir.level = "schedule"} {',
            f'  "tessera.schedule.func"() ({{',
            tile_line,
            f'    "tessera.schedule.layout"() {{operands = [{operand_attr}], layout = "row_major"}} : () -> ()',
            f'  }}) {{sym_name = "{fn.name}", target = "cpu"}} : () -> ()',
            "}",
        ]
    )


def _render_tile_ir(fn: GraphIRFunction, op: IROp) -> str:
    return "\n".join(
        [
            'module attributes {tessera.ir.level = "tile"} {',
            f'  "tessera.tile.func"() ({{',
            f'    "{_tile_op_name(op.op_name)}"() {{source = "{op.op_name}", lowering = "{_lowering_kind(op.op_name)}", vectorize = true}} : () -> ()',
            f'  }}) {{sym_name = "{fn.name}", target = "cpu"}} : () -> ()',
            "}",
        ]
    )


def _render_target_ir(fn: GraphIRFunction, op: IROp) -> str:
    return "\n".join(
        [
            'module attributes {tessera.ir.level = "target", target = "cpu"} {',
            f'  "tessera.cpu.func"() ({{',
            f'    "{_target_op_name(op.op_name)}"() {{source = "{op.op_name}", abi = "numpy"}} : () -> ()',
            f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
            "}",
        ]
    )


def _operand_name(operand: str) -> str:
    return operand[1:] if operand.startswith("%") else operand


def _validate_tile(tile: tuple[int, int, int]) -> None:
    if len(tile) != 3 or any(int(v) <= 0 for v in tile):
        raise ValueError("CPU matmul tile must be a positive (tile_m, tile_n, tile_k) tuple")


def _valid_arity(op: IROp) -> bool:
    if op.op_name in MATMUL_OPS:
        return len(op.operands) == 2
    if op.op_name in UNARY_OPS | REDUCTION_OPS:
        return len(op.operands) == 1
    if op.op_name in STATEFUL_FUNCTIONAL_OPS:
        return len(op.operands) == 4
    return False


def _execute_op(op_name: str, operands: Sequence[np.ndarray]) -> Any:
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
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)
    if op_name == "tessera.adam":
        param, grad, moment1, moment2 = operands
        beta1 = 0.9
        beta2 = 0.999
        lr = 1e-3
        eps = 1e-8
        step = 1
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
