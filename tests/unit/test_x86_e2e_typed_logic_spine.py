"""X86-E2E-2 typed compare, logical, and bitwise descriptor slices."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.canonical_compile import canonical_compile
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.x86_native import (
    X86_BITWISE_I32_ABI,
    X86_BITWISE_KINDS,
    X86_COMPARE_F32_ABI,
    X86_COMPARE_KINDS,
    X86_LOGICAL_I8_ABI,
    X86_LOGICAL_KINDS,
    emit_elementwise_tile_ir,
    package_elementwise,
    supports_elementwise,
    tools_available,
)


def _module(op_name: str, shape: tuple[int, ...] = (3, 17)) -> GraphIRModule:
    extent = "x".join(map(str, shape))
    if op_name in X86_COMPARE_KINDS:
        input_dtype, input_mlir, output_dtype, output_mlir = "fp32", "f32", "bool", "i1"
        binary = True
    elif op_name in X86_LOGICAL_KINDS:
        input_dtype = output_dtype = "bool"
        input_mlir = output_mlir = "i1"
        binary = X86_LOGICAL_KINDS[op_name] != "not"
    else:
        input_dtype = output_dtype = "int32"
        input_mlir = output_mlir = "i32"
        binary = X86_BITWISE_KINDS[op_name] not in {"not", "popcount"}
    source = IRType(f"tensor<{extent}x{input_mlir}>", tuple(map(str, shape)), input_dtype)
    result = IRType(f"tensor<{extent}x{output_mlir}>", tuple(map(str, shape)), output_dtype)
    args = [IRArg("a", source)] + ([IRArg("b", source)] if binary else [])
    operands = ["%a"] + (["%b"] if binary else [])
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_typed_logic", args=args, result_types=[result],
        body=[IROp(
            result="o", op_name=op_name, operands=operands,
            operand_types=[str(source)] * len(operands), result_type=str(result), kwargs={},
        )], return_values=["%o"],
    )])


@pytest.mark.parametrize(
    "family,kind,storage,output_storage",
    [("compare", "lt", "f32", "i8"), ("logical", "not", "i8", "i8"),
     ("bitwise", "popcount", "i32", "i32")],
)
def test_tile_carrier_preserves_family_dtype_and_arity(family, kind, storage, output_storage) -> None:
    text = emit_elementwise_tile_ir(entry="entry", family=family, kind=kind)
    assert f'family = "{family}"' in text and f'kind = "{kind}"' in text
    assert f'storage = "{storage}"' in text
    assert f'output_storage = "{output_storage}"' in text
    assert "tessera_x86_avx512" not in text


@pytest.mark.parametrize(
    "op_name,abi",
    [("tessera.lt", X86_COMPARE_F32_ABI),
     ("tessera.logical_not", X86_LOGICAL_I8_ABI),
     ("tessera.popcount", X86_BITWISE_I32_ABI)],
)
def test_explicit_package_has_stable_abi_and_provenance(monkeypatch, op_name, abi) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    module = _module(op_name)
    assert supports_elementwise(module)
    package = package_elementwise(module, pipeline_name="tessera-lower-to-x86")
    assert package.descriptor.abi_id == abi
    assert package.descriptor.provenance["work_item"] == "X86-E2E-2"


def test_contract_rejects_cross_family_dtype() -> None:
    logical = _module("tessera.logical_and")
    logical.functions[0].args[0].ir_type = IRType("tensor<3x17xi8>", ("3", "17"), "int8")
    assert not supports_elementwise(logical)
    bitwise = _module("tessera.bitwise_and")
    bitwise.functions[0].result_types[0] = IRType("tensor<3x17xi64>", ("3", "17"), "int64")
    assert not supports_elementwise(bitwise)


def _fake_lower(tile_ir: str, symbol: str):
    assert "tile.elementwise_kernel" in tile_ir
    return f"module {{ func.call @{symbol}() : () -> () }}", b"\x7fELF-x86-e2e2-logic", "compiler", "toolchain"


@pytest.mark.parametrize(
    "op_name,shape,abi",
    [("tessera.lt", (32768,), X86_COMPARE_F32_ABI),
     ("tessera.logical_and", (130,), X86_LOGICAL_I8_ABI),
     ("tessera.bitwise_xor", (32768,), X86_BITWISE_I32_ABI)],
)
def test_measured_canonical_selector_promotes_eligible_rows(monkeypatch, op_name, shape, abi) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    result = canonical_compile(_module(op_name, shape), target="x86", enable_tool_validation=False)
    assert result.launch_descriptor is not None
    assert result.launch_descriptor.abi_id == abi


@pytest.mark.parametrize(
    "op_name,shape", [("tessera.lt", (8192,)), ("tessera.bitwise_xor", (16384,))],
)
def test_measured_canonical_selector_retains_small_rows(monkeypatch, op_name, shape) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    result = canonical_compile(_module(op_name, shape), target="x86", enable_tool_validation=False)
    assert result.launch_descriptor is None


def _launch(op_name: str, inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    package = package_elementwise(_module(op_name, inputs[0].shape), pipeline_name="tessera-lower-to-x86")
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"}, tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    )
    output_dtype = np.int32 if package.descriptor.abi_id == X86_BITWISE_I32_ABI else np.bool_
    output = np.zeros(inputs[0].shape, dtype=output_dtype)
    names = [binding.name for binding in package.descriptor.buffers]
    values = {name: value for name, value in zip(names[:-1], inputs)}
    values[names[-1]] = output
    values["N"] = output.size
    result = rt.launch(artifact, values)
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    return output


_COMPARE_REFS = {
    "tessera.eq": np.equal, "tessera.ne": np.not_equal, "tessera.lt": np.less,
    "tessera.le": np.less_equal, "tessera.gt": np.greater, "tessera.ge": np.greater_equal,
}
_LOGICAL_REFS = {
    "tessera.logical_and": np.logical_and, "tessera.logical_or": np.logical_or,
    "tessera.logical_xor": np.logical_xor, "tessera.logical_not": np.logical_not,
}
_BITWISE_REFS = {
    "tessera.bitwise_and": np.bitwise_and, "tessera.bitwise_or": np.bitwise_or,
    "tessera.bitwise_xor": np.bitwise_xor, "tessera.bitwise_not": np.bitwise_not,
    "tessera.popcount": lambda a: np.vectorize(lambda x: int(x).bit_count(), otypes=[np.int32])(
        a.view(np.uint32)
    ),
}


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("op_name", tuple(_COMPARE_REFS))
def test_compare_descriptor_matches_numpy_including_nan(op_name) -> None:
    a = np.array([0.0, 1.0, np.nan, -2.0, np.inf] * 11, dtype=np.float32)
    b = np.array([0.0, 2.0, np.nan, -3.0, np.inf] * 11, dtype=np.float32)
    np.testing.assert_array_equal(_launch(op_name, (a, b)), _COMPARE_REFS[op_name](a, b))


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("op_name", tuple(_LOGICAL_REFS))
def test_logical_descriptor_matches_numpy(op_name) -> None:
    rng = np.random.default_rng(8630)
    a = np.ascontiguousarray(rng.random((3, 37)) < 0.5)
    binary = op_name != "tessera.logical_not"
    inputs = (a, np.ascontiguousarray(rng.random(a.shape) < 0.5)) if binary else (a,)
    np.testing.assert_array_equal(_launch(op_name, inputs), _LOGICAL_REFS[op_name](*inputs))


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("op_name", tuple(_BITWISE_REFS))
def test_bitwise_descriptor_matches_numpy(op_name) -> None:
    rng = np.random.default_rng(8631)
    a = np.ascontiguousarray(rng.integers(-(1 << 30), 1 << 30, (3, 37), dtype=np.int32))
    unary = op_name in {"tessera.bitwise_not", "tessera.popcount"}
    inputs = (a,) if unary else (a, np.ascontiguousarray(rng.integers(-(1 << 30), 1 << 30, a.shape, dtype=np.int32)))
    np.testing.assert_array_equal(_launch(op_name, inputs), _BITWISE_REFS[op_name](*inputs))
