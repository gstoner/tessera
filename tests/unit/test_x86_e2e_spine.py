"""X86-E2E-1 typed softmax/reduction artifact and launch contracts."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.canonical_compile import canonical_compile
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.x86_native import (
    X86_ATTENTION_EXT_F32_ABI,
    X86_ATTENTION_F32_ABI,
    X86_MATMUL_F32_ABI,
    X86_REDUCE_F32_ABI,
    X86_SOFTMAX_F32_ABI,
    emit_attention_tile_ir,
    emit_matmul_tile_ir,
    emit_reduce_tile_ir,
    emit_softmax_tile_ir,
    package_attention,
    package_matmul,
    package_reduction,
    package_softmax,
    supports_attention,
    supports_matmul,
    supports_native_package,
    supports_reduction,
    supports_softmax,
    tools_available,
)


def _softmax_module(shape: tuple[int, ...] = (3, 17)) -> GraphIRModule:
    extents = "x".join(map(str, shape))
    tensor = IRType(f"tensor<{extents}xf32>", tuple(map(str, shape)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_softmax", args=[IRArg("x", tensor)], result_types=[tensor],
        body=[IROp(
            result="o", op_name="tessera.softmax", operands=["%x"],
            operand_types=[str(tensor)], result_type=str(tensor), kwargs={"axis": -1},
        )], return_values=["%o"],
    )])


def _reduction_module(
    *, shape: tuple[int, ...] = (3, 17), kind: str = "sum", keepdims: bool = False,
) -> GraphIRModule:
    extents = "x".join(map(str, shape))
    source = IRType(f"tensor<{extents}xf32>", tuple(map(str, shape)), "fp32")
    output_shape = shape[:-1] + ((1,) if keepdims else ())
    output_extents = "x".join(map(str, output_shape))
    output = IRType(f"tensor<{output_extents}xf32>", tuple(map(str, output_shape)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_reduce", args=[IRArg("x", source)], result_types=[output],
        body=[IROp(
            result="o", op_name=f"tessera.{kind}", operands=["%x"],
            operand_types=[str(source)], result_type=str(output),
            kwargs={"axis": -1, "keepdims": keepdims},
        )], return_values=["%o"],
    )])


def _matmul_module(shape: tuple[int, int, int] = (5, 17, 9)) -> GraphIRModule:
    m, k, n = shape
    a = IRType(f"tensor<{m}x{k}xf32>", (str(m), str(k)), "fp32")
    b = IRType(f"tensor<{k}x{n}xf32>", (str(k), str(n)), "fp32")
    output = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_matmul", args=[IRArg("a", a), IRArg("b", b)], result_types=[output],
        body=[IROp(
            result="o", op_name="tessera.matmul", operands=["%a", "%b"],
            operand_types=[str(a), str(b)], result_type=str(output), kwargs={},
        )], return_values=["%o"],
    )])


def _attention_module(*, extended: bool = False) -> GraphIRModule:
    b, h, sq, sk, d, dv = 1, 2, 5, 7, 4, 3
    q = IRType(f"tensor<{b}x{h}x{sq}x{d}xf32>", tuple(map(str, (b, h, sq, d))), "fp32")
    key = IRType(f"tensor<{b}x{h}x{sk}x{d}xf32>", tuple(map(str, (b, h, sk, d))), "fp32")
    value = IRType(f"tensor<{b}x{h}x{sk}x{dv}xf32>", tuple(map(str, (b, h, sk, dv))), "fp32")
    output = IRType(f"tensor<{b}x{h}x{sq}x{dv}xf32>", tuple(map(str, (b, h, sq, dv))), "fp32")
    args = [IRArg("q", q), IRArg("k", key), IRArg("v", value)]
    operands = ["%q", "%k", "%v"]
    operand_types = [str(q), str(key), str(value)]
    kwargs = {"scale": 0.5, "causal": False}
    if extended:
        bias = IRType(f"tensor<{b}x{h}x{sq}x{sk}xf32>", tuple(map(str, (b, h, sq, sk))), "fp32")
        args.append(IRArg("bias", bias))
        operands.append("%bias")
        operand_types.append(str(bias))
        kwargs.update({"window": 3, "softcap": 4.0})
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_attention", args=args, result_types=[output],
        body=[IROp(
            result="o", op_name="tessera.flash_attn", operands=operands,
            operand_types=operand_types, result_type=str(output), kwargs=kwargs,
        )], return_values=["%o"],
    )])


def _fake_lower(tile_ir: str, symbol: str):
    assert "tile." in tile_ir
    return f"module {{ func.call @{symbol}() : () -> () }}", b"\x7fELF-x86", "compiler", "toolchain"


def test_x86_emitters_use_shared_typed_envelopes() -> None:
    softmax = emit_softmax_tile_ir(entry="softmax")
    reduction = emit_reduce_tile_ir(entry="reduce", kind="mean", axis=1, keepdims=True)
    assert "tile.softmax_kernel" in softmax
    assert 'storage = "f32", accum = "f32", axis = -1' in softmax
    assert "tile.reduce_kernel" in reduction
    assert 'kind = "mean"' in reduction and "inner_is_one = true" in reduction
    assert "tessera_x86_avx512" not in softmax + reduction

    matmul = emit_matmul_tile_ir(entry="matmul")
    attention = emit_attention_tile_ir(
        entry="attention", scale=0.5, causal=False, bias=True, window=3, softcap=4.0,
    )
    assert "tile.matmul_kernel" in matmul and 'a = "f32"' in matmul
    assert "tile.attention_kernel" in attention and "dropout_p = 0.0" in attention
    assert "tessera_x86_" not in matmul + attention


def test_x86_contracts_reject_dtype_axis_and_result_drift() -> None:
    assert supports_softmax(_softmax_module())
    assert supports_reduction(_reduction_module())
    wrong_axis = _reduction_module()
    wrong_axis.functions[0].body[0].kwargs["axis"] = 0
    assert not supports_reduction(wrong_axis)
    wrong_dtype = _softmax_module()
    wrong_dtype.functions[0].args[0].ir_type = IRType("tensor<3x17xf16>", ("3", "17"), "fp16")
    assert not supports_softmax(wrong_dtype)
    assert supports_matmul(_matmul_module())
    assert supports_attention(_attention_module())
    assert supports_attention(_attention_module(extended=True))
    gqa = _attention_module()
    gqa.functions[0].args[1].ir_type = IRType("tensor<1x1x7x4xf32>", ("1", "1", "7", "4"), "fp32")
    assert not supports_attention(gqa)
    assert supports_native_package(_softmax_module())
    assert not supports_native_package(wrong_dtype)


@pytest.mark.parametrize(
    "module,abi",
    [
        (_softmax_module(), X86_SOFTMAX_F32_ABI),
        (_reduction_module(), X86_REDUCE_F32_ABI),
        (_matmul_module(), X86_MATMUL_F32_ABI),
        (_attention_module(), X86_ATTENTION_F32_ABI),
        (_attention_module(extended=True), X86_ATTENTION_EXT_F32_ABI),
    ],
)
def test_canonical_x86_selector_defaults_to_descriptor(
    monkeypatch, module, abi,
) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    result = canonical_compile(
        module, target="x86", enable_tool_validation=False,
    )
    assert result.bundle.execution_mode == "native_descriptor"
    assert result.bundle.execution_kind == "native_cpu"
    assert result.launch_descriptor is not None
    assert result.launch_descriptor.abi_id == abi


def test_canonical_x86_selector_preserves_opt_out_and_unsupported_route(
    monkeypatch,
) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    monkeypatch.setattr("tessera.compiler.x86_native.tools_available", lambda: True)
    opted_out = canonical_compile(
        _softmax_module(), target="x86", options={"package_native": False},
        enable_tool_validation=False,
    )
    assert opted_out.launch_descriptor is None

    unsupported = _softmax_module()
    # Axis 0 is legal for the retained x86 softmax route, but deliberately
    # outside the current stable descriptor ABI (last axis only).
    unsupported.functions[0].body[0].kwargs["axis"] = 0
    fallback = canonical_compile(
        unsupported, target="x86", enable_tool_validation=False,
    )
    assert fallback.launch_descriptor is None


@pytest.mark.parametrize("family,abi", [("softmax", X86_SOFTMAX_F32_ABI), ("reduction", X86_REDUCE_F32_ABI)])
def test_x86_packages_own_shared_object_and_typed_descriptor(monkeypatch, family, abi) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    package = (
        package_softmax(_softmax_module(), pipeline_name="tessera-lower-to-x86")
        if family == "softmax"
        else package_reduction(_reduction_module(), pipeline_name="tessera-lower-to-x86")
    )
    assert package.image.target == "x86"
    assert package.image.architecture == "x86_64_avx512"
    assert package.image.binary_format == "shared_object"
    assert package.descriptor.abi_id == abi
    assert package.descriptor.provenance["work_item"] == "X86-E2E-1"


def test_driver_joins_x86_native_package(monkeypatch) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    bundle = compile_graph_module(
        _softmax_module(), source_origin="unit", target="x86",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.execution_kind == "native_cpu"
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.abi_id == X86_SOFTMAX_F32_ABI


@pytest.mark.parametrize("abi", [
    X86_SOFTMAX_F32_ABI, X86_REDUCE_F32_ABI, X86_MATMUL_F32_ABI,
    X86_ATTENTION_F32_ABI, X86_ATTENTION_EXT_F32_ABI,
])
def test_x86_builtin_launcher_registers_each_pilot_abi_in_isolation(abi) -> None:
    rt.unregister_native_launcher("x86")
    try:
        rt._ensure_builtin_native_launcher("x86", abi)
        assert "x86" in rt._native_launchers
        registration = rt._native_launchers["x86"]
        assert registration.binary_formats == ("shared_object",)
        assert registration.submit is rt._submit_x86_native
    finally:
        rt.unregister_native_launcher("x86")


@pytest.mark.parametrize(
    "module,packager,abi",
    [
        (_matmul_module(), package_matmul, X86_MATMUL_F32_ABI),
        (_attention_module(), package_attention, X86_ATTENTION_F32_ABI),
        (_attention_module(extended=True), package_attention, X86_ATTENTION_EXT_F32_ABI),
    ],
)
def test_x86_next_slices_package_typed_descriptors(monkeypatch, module, packager, abi) -> None:
    monkeypatch.setattr("tessera.compiler.x86_native._lower", _fake_lower)
    package = packager(module, pipeline_name="tessera-lower-to-x86")
    assert package.descriptor.abi_id == abi
    assert package.image.entry_points[0].abi_id == abi
    assert package.descriptor.provenance["work_item"] == "X86-E2E-1"


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("shape", [(1, 1), (3, 17), (4, 256)])
def test_x86_softmax_descriptor_matches_oracle(shape) -> None:
    package = package_softmax(_softmax_module(shape), pipeline_name="tessera-lower-to-x86")
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    rng = np.random.default_rng(8601)
    x = np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
    output = np.zeros_like(x)
    result = rt.launch(artifact, {"x": x, "o": output, "Rows": int(np.prod(shape[:-1])), "K": shape[-1]})
    expected = np.exp(x - np.max(x, axis=-1, keepdims=True))
    expected /= np.sum(expected, axis=-1, keepdims=True)
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("kind", ["sum", "mean", "max"])
def test_x86_reduction_descriptor_matches_oracle(kind) -> None:
    module = _reduction_module(shape=(5, 257), kind=kind, keepdims=True)
    package = package_reduction(module, pipeline_name="tessera-lower-to-x86")
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    rng = np.random.default_rng(8602)
    x = np.ascontiguousarray(rng.standard_normal((5, 257)), dtype=np.float32)
    output = np.zeros((5, 1), dtype=np.float32)
    result = rt.launch(artifact, {"x": x, "o": output, "Outer": 5, "AxisExtent": 257, "Inner": 1})
    expected = getattr(np, kind)(x, axis=-1, keepdims=True)
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("shape", [(1, 1, 1), (5, 17, 9), (16, 31, 19)])
def test_x86_matmul_descriptor_matches_oracle(shape) -> None:
    package = package_matmul(_matmul_module(shape), pipeline_name="tessera-lower-to-x86")
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    m, k, n = shape
    rng = np.random.default_rng(8604)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32)
    b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float32)
    output = np.zeros((m, n), dtype=np.float32)
    result = rt.launch(artifact, {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k})
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    np.testing.assert_allclose(output, a @ b, rtol=3e-5, atol=3e-5)


def _attention_oracle(q, key, value, *, scale, bias=None, window=-1, softcap=0.0):
    scores = np.einsum("bhqd,bhkd->bhqk", q, key) * scale
    if softcap:
        scores = softcap * np.tanh(scores / softcap)
    if bias is not None:
        scores += bias
    if window >= 0:
        sq, sk = q.shape[-2], key.shape[-2]
        offset = max(sk - sq, 0)
        for i in range(sq):
            lo, hi = i + offset - window // 2, i + offset + window // 2
            scores[..., i, :max(0, lo)] = -np.inf
            scores[..., i, min(sk, hi + 1):] = -np.inf
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, value)


@pytest.mark.skipif(not tools_available(), reason="x86 compiler/shared library unavailable")
@pytest.mark.parametrize("extended", [False, True])
def test_x86_attention_descriptor_matches_oracle(extended) -> None:
    module = _attention_module(extended=extended)
    package = package_attention(module, pipeline_name="tessera-lower-to-x86")
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir,
    )
    rng = np.random.default_rng(8605)
    q = np.ascontiguousarray(rng.standard_normal((1, 2, 5, 4)), dtype=np.float32)
    key = np.ascontiguousarray(rng.standard_normal((1, 2, 7, 4)), dtype=np.float32)
    value = np.ascontiguousarray(rng.standard_normal((1, 2, 7, 3)), dtype=np.float32)
    output = np.zeros((1, 2, 5, 3), dtype=np.float32)
    args = {
        "q": q, "k": key, "v": value, "o": output,
        "B": 1, "Hq": 2, "Hkv": 2, "Sq": 5, "Sk": 7, "D": 4, "Dv": 3,
    }
    bias = None
    if extended:
        bias = np.ascontiguousarray(rng.standard_normal((1, 2, 5, 7)) * 0.1, dtype=np.float32)
        args["bias"] = bias
    result = rt.launch(artifact, args)
    expected = _attention_oracle(
        q, key, value, scale=0.5, bias=bias,
        window=3 if extended else -1, softcap=4.0 if extended else 0.0,
    )
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_cpu"
    np.testing.assert_allclose(output, expected, rtol=3e-5, atol=3e-5)
