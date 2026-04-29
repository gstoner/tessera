"""Narrow end-to-end CPU lowering path for @jit matmul.

This module intentionally handles one shape of program:

    @tessera.jit
    def mm(A, B):
        return tessera.ops.matmul(A, B)

It creates inspectable Schedule IR, Tile IR, and Target IR artifacts from the
existing Graph IR text and executes the operation on CPU via NumPy. The goal is
to give developers a real end-to-end spine while the general compiler pipeline
is still maturing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .graph_ir import GraphIRFunction, GraphIRModule, IROp


MATMUL_OPS = {"tessera.matmul", "tessera.gemm"}


@dataclass(frozen=True)
class LoweringArtifact:
    """Textual artifact for one compiler layer."""

    level: str
    text: str


@dataclass(frozen=True)
class MatmulCPUPlan:
    """Executable CPU plan for a single Graph IR matmul/gemm op."""

    function_name: str
    op_name: str
    lhs_name: str
    rhs_name: str
    graph_ir: str
    schedule_ir: str
    tile_ir: str
    target_ir: str

    def execute(self, args: Sequence[Any], kwargs: Mapping[str, Any], arg_names: Sequence[str]) -> Any:
        values = {name: value for name, value in zip(arg_names, args)}
        values.update(kwargs)
        if self.lhs_name not in values or self.rhs_name not in values:
            raise ValueError(
                f"matmul CPU plan requires operands {self.lhs_name!r} and {self.rhs_name!r}"
            )
        lhs = _as_numpy(values[self.lhs_name])
        rhs = _as_numpy(values[self.rhs_name])
        return np.matmul(lhs, rhs)

    def artifacts(self) -> tuple[LoweringArtifact, ...]:
        return (
            LoweringArtifact("graph", self.graph_ir),
            LoweringArtifact("schedule", self.schedule_ir),
            LoweringArtifact("tile", self.tile_ir),
            LoweringArtifact("target", self.target_ir),
        )


def build_matmul_cpu_plan(module: GraphIRModule) -> Optional[MatmulCPUPlan]:
    """Build a CPU plan if the Graph IR module is the supported matmul shape."""

    if len(module.functions) != 1:
        return None
    fn = module.functions[0]
    matmul_ops = [op for op in fn.body if op.op_name in MATMUL_OPS]
    if len(matmul_ops) != 1:
        return None
    op = matmul_ops[0]
    if len(op.operands) != 2:
        return None
    lhs_name = _operand_name(op.operands[0])
    rhs_name = _operand_name(op.operands[1])
    if not lhs_name or not rhs_name:
        return None

    graph_text = module.to_mlir()
    schedule = _render_schedule_ir(fn, op, lhs_name, rhs_name)
    tile = _render_tile_ir(fn, op)
    target = _render_target_ir(fn, op)
    return MatmulCPUPlan(
        function_name=fn.name,
        op_name=op.op_name,
        lhs_name=lhs_name,
        rhs_name=rhs_name,
        graph_ir=graph_text,
        schedule_ir=schedule,
        tile_ir=tile,
        target_ir=target,
    )


def _render_schedule_ir(fn: GraphIRFunction, op: IROp, lhs_name: str, rhs_name: str) -> str:
    return "\n".join(
        [
            'module attributes {tessera.ir.level = "schedule"} {',
            f'  "tessera.schedule.func"() ({{',
            f'    "tessera.schedule.tile"() {{source = "{op.op_name}", tile_m = 128 : i64, tile_n = 128 : i64, tile_k = 64 : i64}} : () -> ()',
            f'    "tessera.schedule.layout"() {{lhs = "{lhs_name}", rhs = "{rhs_name}", layout = "row_major"}} : () -> ()',
            f'  }}) {{sym_name = "{fn.name}", target = "cpu"}} : () -> ()',
            "}",
        ]
    )


def _render_tile_ir(fn: GraphIRFunction, op: IROp) -> str:
    return "\n".join(
        [
            'module attributes {tessera.ir.level = "tile"} {',
            f'  "tessera.tile.func"() ({{',
            f'    "tessera.tile.matmul"() {{source = "{op.op_name}", lowering = "loop_nest", vectorize = true}} : () -> ()',
            f'  }}) {{sym_name = "{fn.name}", target = "cpu"}} : () -> ()',
            "}",
        ]
    )


def _render_target_ir(fn: GraphIRFunction, op: IROp) -> str:
    return "\n".join(
        [
            'module attributes {tessera.ir.level = "target", target = "cpu"} {',
            f'  "tessera.cpu.func"() ({{',
            f'    "tessera.cpu.matmul"() {{source = "{op.op_name}", abi = "numpy"}} : () -> ()',
            f'  }}) {{sym_name = "{fn.name}"}} : () -> ()',
            "}",
        ]
    )


def _operand_name(operand: str) -> str:
    return operand[1:] if operand.startswith("%") else operand


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy") and callable(value.numpy):
        return np.asarray(value.numpy())
    if hasattr(value, "_data"):
        return np.asarray(value._data)
    return np.asarray(value)


__all__ = [
    "LoweringArtifact",
    "MATMUL_OPS",
    "MatmulCPUPlan",
    "build_matmul_cpu_plan",
]
