"""X86-E2E-2 typed reduction, normalization, and positional cohort."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.x86_native import (
    X86_ALIBI_F32_ABI,
    X86_ARGREDUCE_F32_ABI,
    X86_NORM_F32_ABI,
    X86_ROPE_F32_ABI,
    X86_SCAN_F32_ABI,
    emit_cohort2_tile_ir,
    package_cohort2,
    supports_cohort2,
    tools_available,
)


def _type(shape: tuple[int, ...], dtype: str = "fp32") -> IRType:
    spelling = "f32" if dtype == "fp32" else "i32"
    prefix = "x".join(map(str, shape))
    return IRType(f"tensor<{prefix + 'x' if prefix else ''}{spelling}>", tuple(map(str, shape)), dtype)


def _module(op_name: str) -> GraphIRModule:
    if op_name in {"tessera.argmax", "tessera.cumsum", "tessera.rmsnorm", "tessera.layer_norm"}:
        source = _type((3, 17))
        result = _type((3,), "int32") if op_name == "tessera.argmax" else source
        kwargs = {"axis": -1, "keepdims": False} if op_name == "tessera.argmax" else {"axis": -1}
        if op_name in {"tessera.rmsnorm", "tessera.layer_norm"}:
            kwargs = {"eps": 1e-5}
        args, operands = [IRArg("x", source)], ["%x"]
    elif op_name == "tessera.rope":
        source, result = _type((3, 18)), _type((3, 18))
        args, operands, kwargs = [IRArg("x", source), IRArg("theta", source)], ["%x", "%theta"], {}
    else:
        source, result = _type((4,)), _type((4, 7, 7))
        args, operands, kwargs = [IRArg("slopes", source)], ["%slopes"], {"num_heads": 4, "seq_len": 7}
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_cohort2", args=args, result_types=[result],
        body=[IROp(result="o", op_name=op_name, operands=operands,
                   operand_types=[str(arg.ir_type) for arg in args],
                   result_type=str(result), kwargs=kwargs)],
        return_values=["%o"],
    )])


@pytest.mark.parametrize("family,kind,carrier", [
    ("argreduce", "argmax", "tile.argreduce_kernel"),
    ("scan", "sum", "tile.scan_kernel"),
    ("norm", "rmsnorm", "tile.norm_kernel"),
    ("rope", "rope", "tile.rope_kernel"),
    ("alibi", "alibi", "tile.alibi_kernel"),
])
def test_cohort2_carriers_are_backend_neutral(family, kind, carrier) -> None:
    text = emit_cohort2_tile_ir(entry="entry", family=family, kind=kind, eps=1e-5)
    assert carrier in text
    assert "tessera_x86_" not in text


@pytest.mark.parametrize("op_name,abi", [
    ("tessera.argmax", X86_ARGREDUCE_F32_ABI),
    ("tessera.cumsum", X86_SCAN_F32_ABI),
    ("tessera.rmsnorm", X86_NORM_F32_ABI),
    ("tessera.layer_norm", X86_NORM_F32_ABI),
    ("tessera.rope", X86_ROPE_F32_ABI),
    ("tessera.alibi", X86_ALIBI_F32_ABI),
])
def test_cohort2_contract_and_package(monkeypatch, op_name, abi) -> None:
    module = _module(op_name)
    assert supports_cohort2(module)

    def fake_lower(tile_ir: str, symbol: str):
        assert "tile." in tile_ir
        return f"module {{ func.call @{symbol}() : () -> () }}", b"\x7fELF-x86-c2", "compiler", "toolchain"

    monkeypatch.setattr("tessera.compiler.x86_native._lower", fake_lower)
    package = package_cohort2(module, pipeline_name="tessera-lower-to-x86")
    assert package.descriptor.abi_id == abi
    assert package.descriptor.provenance["work_item"] == "X86-E2E-2"


def _launch(op_name: str, inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    package = package_cohort2(_module(op_name), pipeline_name="tessera-lower-to-x86")
    output_shape = tuple(package.descriptor.provenance["output_shape"])
    output_dtype = np.int32 if package.descriptor.abi_id == X86_ARGREDUCE_F32_ABI else np.float32
    output = np.zeros(output_shape, dtype=output_dtype)
    values = {binding.name: value for binding, value in zip(package.descriptor.buffers[:-1], inputs)}
    values[package.descriptor.buffers[-1].name] = output
    family = str(package.descriptor.provenance["family"])
    if family == "alibi":
        values.update({"H": package.descriptor.provenance["rows"], "S": package.descriptor.provenance["cols"]})
    else:
        values.update({"Rows": package.descriptor.provenance["rows"], "Cols": package.descriptor.provenance["cols"]})
    if family == "norm":
        values["Epsilon"] = package.descriptor.provenance["eps"]
    artifact = rt.RuntimeArtifact(metadata={"target": "x86"}, native_image=package.image,
                                  launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
                                  target_ir=package.target_ir)
    result = rt.launch(artifact, values)
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    return output


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
def test_cohort2_descriptors_match_oracles() -> None:
    rng = np.random.default_rng(86220)
    x = np.ascontiguousarray(rng.normal(size=(3, 17)), dtype=np.float32)
    np.testing.assert_array_equal(_launch("tessera.argmax", (x,)), np.argmax(x, axis=-1).astype(np.int32))
    np.testing.assert_allclose(_launch("tessera.cumsum", (x,)), np.cumsum(x, axis=-1, dtype=np.float32), rtol=2e-6, atol=2e-6)
    rms = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-5))
    np.testing.assert_allclose(_launch("tessera.rmsnorm", (x,)), rms, rtol=3e-6, atol=3e-6)
    mean = np.mean(x, axis=-1, keepdims=True)
    layer = (x - mean) / np.sqrt(np.mean((x - mean) ** 2, axis=-1, keepdims=True) + np.float32(1e-5))
    np.testing.assert_allclose(_launch("tessera.layer_norm", (x,)), layer, rtol=3e-6, atol=3e-6)

    rope_x = np.ascontiguousarray(rng.normal(size=(3, 18)), dtype=np.float32)
    theta = np.ascontiguousarray(rng.normal(size=(3, 18)), dtype=np.float32)
    expected = np.empty_like(rope_x)
    expected[:, 0::2] = rope_x[:, 0::2] * np.cos(theta[:, 0::2]) - rope_x[:, 1::2] * np.sin(theta[:, 0::2])
    expected[:, 1::2] = rope_x[:, 0::2] * np.sin(theta[:, 0::2]) + rope_x[:, 1::2] * np.cos(theta[:, 0::2])
    np.testing.assert_allclose(_launch("tessera.rope", (rope_x, theta)), expected, rtol=3e-5, atol=3e-5)

    slopes = np.ascontiguousarray(np.linspace(0.1, 0.4, 4), dtype=np.float32)
    positions = np.arange(7, dtype=np.float32)
    expected_alibi = slopes[:, None, None] * (positions[None, None, :] - positions[None, :, None])
    np.testing.assert_allclose(_launch("tessera.alibi", (slopes,)), expected_alibi, rtol=0.0, atol=1e-6)
