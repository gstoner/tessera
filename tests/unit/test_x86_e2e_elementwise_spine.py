"""X86-E2E-2 typed flat-elementwise artifact and launch contracts."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.canonical_compile import canonical_compile
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.x86_native import (
    X86_BINARY_F32_ABI,
    X86_BINARY_KINDS,
    X86_PREDICATE_F32_ABI,
    X86_PREDICATE_KINDS,
    X86_UNARY_F32_ABI,
    X86_UNARY_KINDS,
    emit_elementwise_tile_ir,
    package_elementwise,
    supports_elementwise,
    tools_available,
)


def _module(op_name: str, shape: tuple[int, ...] = (3, 17)) -> GraphIRModule:
    extent = "x".join(map(str, shape))
    source = IRType(f"tensor<{extent}xf32>", tuple(map(str, shape)), "fp32")
    predicate = op_name in X86_PREDICATE_KINDS
    result = IRType(
        f"tensor<{extent}x{'i1' if predicate else 'f32'}>",
        tuple(map(str, shape)), "bool" if predicate else "fp32",
    )
    binary = op_name in X86_BINARY_KINDS
    args = [IRArg("a", source)] + ([IRArg("b", source)] if binary else [])
    operands = ["%a"] + (["%b"] if binary else [])
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_elementwise", args=args, result_types=[result],
        body=[IROp(
            result="o", op_name=op_name, operands=operands,
            operand_types=[str(source)] * len(operands),
            result_type=str(result), kwargs={},
        )], return_values=["%o"],
    )])


def _fake_lower(tile_ir: str, symbol: str):
    assert "tile.elementwise_kernel" in tile_ir
    return f"module {{ func.call @{symbol}() : () -> () }}", b"\x7fELF-x86-e2e2", "compiler", "toolchain"


@pytest.mark.parametrize(
    "family,kind,symbol",
    [
        ("unary", "abs", "tessera_x86_avx512_unary_f32"),
        ("binary", "add", "tessera_x86_avx512_binary_f32"),
        ("predicate", "isfinite", "tessera_x86_avx512_predicate_f32"),
    ],
)
def test_elementwise_tile_carrier_is_backend_neutral(family, kind, symbol) -> None:
    text = emit_elementwise_tile_ir(entry="entry", family=family, kind=kind)
    assert "tile.elementwise_kernel" in text
    assert f'family = "{family}"' in text and f'kind = "{kind}"' in text
    assert symbol not in text


@pytest.mark.parametrize(
    "op_name,abi",
    [
        ("tessera.absolute", X86_UNARY_F32_ABI),
        ("tessera.add", X86_BINARY_F32_ABI),
        ("tessera.isfinite", X86_PREDICATE_F32_ABI),
    ],
)
def test_elementwise_package_and_canonical_selector(monkeypatch, op_name, abi) -> None:
    module = _module(op_name, (128, 128) if op_name == "tessera.add" else (3, 17))
    assert supports_elementwise(module)
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    package = package_elementwise(module, pipeline_name="tessera-lower-to-x86")
    assert package.descriptor.abi_id == abi
    assert package.descriptor.provenance["work_item"] == "X86-E2E-2"
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    result = canonical_compile(module, target="x86", enable_tool_validation=False)
    assert result.launch_descriptor is not None
    assert result.launch_descriptor.abi_id == abi
    assert result.execution_kind == "native_cpu"


def test_small_binary_stays_retained_but_explicit_package_remains_available(monkeypatch) -> None:
    module = _module("tessera.add", (32, 257))
    assert supports_elementwise(module)
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    assert package_elementwise(module, pipeline_name="tessera-lower-to-x86").descriptor.abi_id == X86_BINARY_F32_ABI
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    result = canonical_compile(module, target="x86", enable_tool_validation=False)
    assert result.launch_descriptor is None


def test_elementwise_contract_rejects_dynamic_shape_and_dtype() -> None:
    wrong_dtype = _module("tessera.add")
    wrong_dtype.functions[0].args[1].ir_type = IRType(
        "tensor<3x17xf16>", ("3", "17"), "fp16",
    )
    assert not supports_elementwise(wrong_dtype)
    dynamic = _module("tessera.absolute")
    dynamic.functions[0].args[0].ir_type = IRType(
        "tensor<?x17xf32>", ("?", "17"), "fp32",
    )
    assert not supports_elementwise(dynamic)


def _launch(module: GraphIRModule, inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    package = package_elementwise(module, pipeline_name="tessera-lower-to-x86")
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
        target_ir=package.target_ir,
    )
    output_dtype = np.bool_ if package.descriptor.abi_id == X86_PREDICATE_F32_ABI else np.float32
    output = np.zeros(inputs[0].shape, dtype=output_dtype)
    names = [binding.name for binding in package.descriptor.buffers]
    values = {name: value for name, value in zip(names[:-1], inputs)}
    values[names[-1]] = output
    values["N"] = output.size
    result = rt.launch(artifact, values)
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    return output


_UNARY_REFS = {
    "tessera.sqrt": np.sqrt, "tessera.rsqrt": lambda x: 1.0 / np.sqrt(x),
    "tessera.reciprocal": lambda x: 1.0 / x, "tessera.absolute": np.abs,
    "tessera.sign": np.sign, "tessera.floor": np.floor, "tessera.ceil": np.ceil,
    "tessera.trunc": np.trunc, "tessera.round": np.round,
}
_BINARY_REFS = {
    "tessera.sub": np.subtract, "tessera.div": np.divide,
    "tessera.maximum": np.maximum, "tessera.minimum": np.minimum,
    "tessera.add": np.add, "tessera.mul": np.multiply, "tessera.mod": np.mod,
    "tessera.floor_div": np.floor_divide,
}
_PREDICATE_REFS = {
    "tessera.isnan": np.isnan, "tessera.isinf": np.isinf,
    "tessera.isfinite": np.isfinite,
}


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("op_name", tuple(_UNARY_REFS))
def test_elementwise_unary_descriptor_matches_oracle(op_name) -> None:
    rng = np.random.default_rng(8621)
    x = np.ascontiguousarray(rng.uniform(0.25, 4.0, 37), dtype=np.float32)
    got = _launch(_module(op_name, (37,)), (x,))
    np.testing.assert_allclose(got, _UNARY_REFS[op_name](x), rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("op_name", tuple(_BINARY_REFS))
def test_elementwise_binary_descriptor_matches_oracle(op_name) -> None:
    rng = np.random.default_rng(8622)
    a = np.ascontiguousarray(rng.uniform(-4.0, 4.0, (3, 17)), dtype=np.float32)
    b = np.ascontiguousarray(rng.uniform(0.25, 3.0, (3, 17)), dtype=np.float32)
    got = _launch(_module(op_name), (a, b))
    np.testing.assert_allclose(got, _BINARY_REFS[op_name](a, b), rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("op_name", tuple(_PREDICATE_REFS))
def test_elementwise_predicate_descriptor_matches_oracle(op_name) -> None:
    x = np.array([[0.0, np.nan, np.inf, -np.inf, 3.5] * 4], dtype=np.float32)
    got = _launch(_module(op_name, x.shape), (x,))
    np.testing.assert_array_equal(got, _PREDICATE_REFS[op_name](x))
