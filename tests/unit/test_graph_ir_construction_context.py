from __future__ import annotations

import sys

import pytest

from tessera.compiler.graph_ir import (
    GraphIRConstructionContext,
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    TENSOR_FP32,
    GraphIRVerificationError,
    construct_mlir_module,
)


def _module() -> GraphIRModule:
    fn = GraphIRFunction(
        name="mm",
        args=[IRArg("A", TENSOR_FP32), IRArg("B", TENSOR_FP32)],
        body=[
            IROp(
                result="C",
                op_name="tessera.matmul",
                operands=["%A", "%B"],
                operand_types=[str(TENSOR_FP32), str(TENSOR_FP32)],
                result_type=str(TENSOR_FP32),
                inferred_type=TENSOR_FP32,
            )
        ],
        result_types=[TENSOR_FP32],
        return_values=["%C"],
    )
    return GraphIRModule(functions=[fn])


def test_construction_context_verifies_before_serialization():
    ctx = GraphIRConstructionContext(module=_module(), validate_native_mlir=False)

    text = ctx.to_mlir()

    assert ctx.serialization_count == 1
    assert "tessera.matmul" in text
    assert ctx.construct_mlir_module().graph_module is ctx.module
    assert ctx.serialization_count == 2


def test_construction_context_rejects_invalid_object_before_text():
    fn = GraphIRFunction(name="broken", return_values=["%missing"], result_types=[TENSOR_FP32])
    ctx = GraphIRConstructionContext(module=GraphIRModule(functions=[fn]))

    with pytest.raises(GraphIRVerificationError):
        ctx.to_mlir()

    assert ctx.serialization_count == 0


def test_missing_mlir_bindings_still_returns_object_backed_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlir", None)

    mlir_module = construct_mlir_module(_module())

    assert mlir_module.graph_module.functions[0].name == "mm"
    assert not mlir_module.is_native
