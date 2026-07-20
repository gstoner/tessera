from __future__ import annotations

import math

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.canonical_compile import canonical_compile
from tessera.compiler.x86_native import (
    X86_BINARY_MATH_F32_ABI,
    X86_TRANSCENDENTAL_F32_ABI,
    X86_WHERE_F32_ABI,
    package_elementwise,
    supports_elementwise,
    tools_available,
)


def _module(op_name: str, shape: tuple[int, ...]) -> GraphIRModule:
    extent = "x".join(map(str, shape))
    f32 = IRType(f"tensor<{extent}xf32>", tuple(map(str, shape)), "fp32")
    boolean = IRType(f"tensor<{extent}xi1>", tuple(map(str, shape)), "bool")
    if op_name == "tessera.where":
        args = [IRArg("c", boolean), IRArg("a", f32), IRArg("b", f32)]
        operands = ["%c", "%a", "%b"]
    elif op_name in {"tessera.pow", "tessera.silu_mul"}:
        args = [IRArg("a", f32), IRArg("b", f32)]
        operands = ["%a", "%b"]
    else:
        args = [IRArg("x", f32)]
        operands = ["%x"]
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_flat_followon", args=args, result_types=[f32],
        body=[IROp(result="o", op_name=op_name, operands=operands,
                   operand_types=[str(arg.ir_type) for arg in args],
                   result_type=str(f32), kwargs={})], return_values=["%o"],
    )])


@pytest.mark.parametrize(
    "op_name,abi",
    [("tessera.where", X86_WHERE_F32_ABI),
     ("tessera.exp", X86_TRANSCENDENTAL_F32_ABI),
     ("tessera.pow", X86_BINARY_MATH_F32_ABI),
     ("tessera.silu_mul", X86_BINARY_MATH_F32_ABI)],
)
def test_followon_flat_packages_lower_to_stable_abi(
    monkeypatch, op_name: str, abi: str,
) -> None:
    monkeypatch.setattr(
        "tessera.compiler.x86_native._lower",
        lambda tile_ir, symbol: (
            f"module {{ func.call @{symbol}() : () -> () }}",
            b"\x7fELF-x86-e2e2-flat",
            "compiler",
            "toolchain",
        ),
    )
    module = _module(op_name, (3, 17))
    assert supports_elementwise(module)
    package = package_elementwise(module, pipeline_name="tessera-lower-to-x86")
    assert package.descriptor.abi_id == abi
    assert package.descriptor.provenance["work_item"] == "X86-E2E-2"
    assert f"call @{package.descriptor.entry_symbol}" in package.target_ir


@pytest.mark.parametrize(
    "op_name,shape,promoted",
    [("tessera.exp", (130,), True), ("tessera.pow", (130,), False),
     ("tessera.pow", (32, 257), True), ("tessera.where", (32, 257), False),
     ("tessera.where", (1024, 1024), True)],
)
def test_measured_flat_followon_selector_policy(monkeypatch, op_name, shape, promoted) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    if promoted:
        def fake_lower(tile_ir: str, symbol: str):
            return f"module {{ func.call @{symbol}() : () -> () }}", b"\x7fELF-flat", "compiler", "toolchain"
        monkeypatch.setattr("tessera.compiler.x86_native._lower", fake_lower)
    result = canonical_compile(_module(op_name, shape), target="x86", enable_tool_validation=False)
    assert (result.launch_descriptor is not None) is promoted


def _launch(op_name: str, values: dict[str, np.ndarray]) -> np.ndarray:
    shape = next(iter(values.values())).shape
    package = package_elementwise(_module(op_name, shape), pipeline_name="tessera-lower-to-x86")
    output = np.zeros(shape, dtype=np.float32)
    launch_values = dict(values)
    launch_values[package.descriptor.buffers[-1].name] = output
    launch_values["N"] = output.size
    result = rt.launch(rt.RuntimeArtifact(
        metadata={"target": "x86"}, tile_ir=package.tile_ir, target_ir=package.target_ir,
        native_image=package.image, launch_descriptor=package.descriptor,
    ), launch_values)
    assert result["ok"] is True, result.get("reason")
    return output


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
def test_followon_flat_descriptors_match_oracles() -> None:
    rng = np.random.default_rng(8640)
    a = np.ascontiguousarray(rng.uniform(0.2, 2.0, (3, 19)), dtype=np.float32)
    b = np.ascontiguousarray(rng.uniform(-2.0, 2.0, a.shape), dtype=np.float32)
    c = np.ascontiguousarray(rng.random(a.shape) > 0.5)
    np.testing.assert_array_equal(_launch("tessera.where", {"c": c, "a": a, "b": b}), np.where(c, a, b))
    np.testing.assert_allclose(_launch("tessera.exp", {"x": b}), np.exp(b), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(_launch("tessera.pow", {"a": a, "b": b}), np.power(a, b), rtol=2e-5, atol=2e-5)
    ref = a / (1.0 + np.exp(-a)) * b
    np.testing.assert_allclose(_launch("tessera.silu_mul", {"a": a, "b": b}), ref, rtol=2e-5, atol=2e-5)
