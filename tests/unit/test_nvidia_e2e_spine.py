from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.capabilities import supports_op
from tessera.compiler.nvidia_native import (
    _link_cuda_device_library_if_needed,
    emit_f16_matmul_tile_ir,
    emit_attention_backward_tile_ir,
    emit_attention_tile_ir,
    emit_f16_softmax_tile_ir,
    emit_f32_softmax_tile_ir,
    emit_nvfp4_matmul_tile_ir,
    emit_reduce_tile_ir,
    emit_matmul_tile_ir,
    emit_mx_matmul_tile_ir,
    emit_paged_kv_read_tile_ir,
    package_nvfp4_matmul,
    requests_mx_matmul,
    requests_attention,
    requests_attention_backward,
    requests_softmax,
    requests_reduction,
    requests_nvfp4_matmul,
    requests_paged_kv_read,
    supports_bf16_matmul,
    supports_attention,
    supports_attention_backward,
    supports_f16_matmul,
    supports_f16_softmax,
    supports_f32_softmax,
    supports_fp64_matmul,
    supports_fp8_matmul,
    supports_int8_matmul,
    supports_mx_matmul,
    supports_nvfp4_matmul,
    supports_paged_kv_read,
    supports_reduction,
    supports_tf32_matmul,
)
from tessera.compiler.primitive_coverage import NumericPolicy


def test_cuda_device_library_linking_is_demand_driven_and_fail_closed() -> None:
    passthrough = b"define void @kernel() { ret void }\n"
    linked, libraries = _link_cuda_device_library_if_needed(
        passthrough,
        llvm_link=Path("unused"),
        libdevice=None,
    )
    assert linked == passthrough
    assert libraries == ()

    unresolved = b"declare float @__nv_expf(float)\n"
    with pytest.raises(RuntimeError, match="requires CUDA libdevice"):
        _link_cuda_device_library_if_needed(
            unresolved,
            llvm_link=Path("unused"),
            libdevice=None,
        )


def _matmul_module(
    dtype: str = "fp16",
    *,
    math_mode: str | None = None,
) -> GraphIRModule:
    mlir_dtype = {
        "fp64": "f64",
        "fp16": "f16",
        "bf16": "bf16",
        "fp32": "f32",
        "fp8_e4m3": "f8E4M3FN",
        "fp8_e5m2": "f8E5M2",
        "int8": "i8",
    }[dtype]
    a = IRType(f"tensor<32x48x{mlir_dtype}>", ("32", "48"), dtype)
    b = IRType(f"tensor<48x24x{mlir_dtype}>", ("48", "24"), dtype)
    result_dtype = "int32" if dtype == "int8" else "fp64" if dtype == "fp64" else "fp32"
    result_ir = "i32" if dtype == "int8" else "f64" if dtype == "fp64" else "f32"
    c = IRType(f"tensor<32x24x{result_ir}>", ("32", "24"), result_dtype)
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="sm120_matmul",
                args=[IRArg("a", a), IRArg("b", b)],
                result_types=[c],
                body=[
                    IROp(
                        result="c",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(c),
                        numeric_policy=(
                            NumericPolicy(storage="fp32", accum="fp32", math_mode=math_mode)
                            if math_mode is not None
                            else None
                        ),
                    )
                ],
                return_values=["%c"],
            )
        ]
    )


def test_sm120_f16_native_packager_owns_typed_tile_kernel() -> None:
    module = _matmul_module()
    assert supports_f16_matmul(module)
    assert not supports_f16_matmul(_matmul_module("fp32"))

    source = emit_f16_matmul_tile_ir(
        entry="tessera_tile_matmul_shared_f16",
        schedule="shared",
    )
    assert "tile.matmul_kernel" in source
    assert 'a = "f16", b = "f16", acc = "f32"' in source
    assert 'warps = 4 : i64, staging = "shared"' in source
    assert "llvm.func @tessera_tile_matmul_shared_f16" in source


def test_sm120_bf16_uses_same_canonical_seam_with_distinct_abi_storage() -> None:
    module = _matmul_module("bf16")
    assert supports_bf16_matmul(module)
    assert not supports_f16_matmul(module)
    source = emit_matmul_tile_ir(
        entry="tessera_tile_matmul_shared_bf16",
        storage="bf16",
        schedule="shared",
    )
    assert 'a = "bf16", b = "bf16", acc = "f32"' in source
    assert 'warps = 4 : i64, staging = "shared"' in source
    with pytest.raises(ValueError, match="canonical matmul storage"):
        emit_matmul_tile_ir(entry="bad", storage="fp6")


def test_sm120_tf32_and_fp8_use_explicit_physical_storage_contracts() -> None:
    tf32 = _matmul_module("fp32", math_mode="tf32")
    assert supports_tf32_matmul(tf32)
    assert not supports_tf32_matmul(_matmul_module("fp32"))
    tf32_tile = emit_matmul_tile_ir(
        entry="tessera_tile_matmul_direct_tf32",
        storage="tf32",
        schedule="direct",
    )
    assert 'k = 8, a = "tf32", b = "tf32", acc = "f32"' in tf32_tile
    for dtype, physical in (("fp8_e4m3", "e4m3"), ("fp8_e5m2", "e5m2")):
        assert supports_fp8_matmul(_matmul_module(dtype))
        source = emit_matmul_tile_ir(
            entry=f"tessera_tile_matmul_direct_{physical}",
            storage=physical,
            schedule="direct",
        )
        assert f'k = 32, a = "{physical}", b = "{physical}"' in source
    with pytest.raises(ValueError, match="requires direct schedule"):
        emit_matmul_tile_ir(entry="bad", storage="e4m3", schedule="shared")

    assert supports_int8_matmul(_matmul_module("int8"))
    int8_tile = emit_matmul_tile_ir(
        entry="tessera_tile_matmul_direct_s8",
        storage="s8",
        schedule="direct",
    )
    assert 'k = 32, a = "s8", b = "s8", acc = "s32"' in int8_tile
    assert 'output = "i32"' in int8_tile

    assert supports_fp64_matmul(_matmul_module("fp64"))
    fp64_tile = emit_matmul_tile_ir(
        entry="tessera_tile_matmul_direct_f64",
        storage="f64",
        schedule="direct",
    )
    assert 'm = 8, n = 8, k = 4, a = "f64", b = "f64", acc = "f64"' in fp64_tile
    assert 'output = "f64"' in fp64_tile


def _softmax_module(
    shape: tuple[int, ...] = (2, 3, 17),
    *,
    dtype: str = "fp32",
    axis: int = -1,
) -> GraphIRModule:
    mlir_dtype = "f32" if dtype == "fp32" else "f16"
    dims = "x".join(str(dim) for dim in shape)
    x = IRType(f"tensor<{dims}x{mlir_dtype}>", tuple(map(str, shape)), dtype)
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="sm120_softmax",
                args=[IRArg("x", x)],
                result_types=[x],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.softmax",
                        operands=["%x"],
                        operand_types=[str(x)],
                        result_type=str(x),
                        kwargs={"axis": axis},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def test_sm120_softmax_packager_owns_typed_precision_and_axis_contract() -> None:
    module = _softmax_module()
    assert requests_softmax(module)
    assert supports_op("nvidia_sm120", "tessera.softmax", dtype="fp16").supported
    assert supports_op("nvidia_sm120", "tessera.softmax_safe", dtype="fp16").supported
    assert supports_f32_softmax(module)
    assert supports_f16_softmax(_softmax_module(dtype="fp16"))
    assert not supports_f32_softmax(_softmax_module(dtype="fp16"))
    assert not supports_f16_softmax(module)
    assert not supports_f32_softmax(_softmax_module(axis=0))
    source = emit_f32_softmax_tile_ir(entry="tessera_tile_softmax_f32")
    assert "tile.softmax_kernel %x, %o, %rows, %columns" in source
    assert 'exp_mode = "approx_exp2"' in source
    assert "ftz = false" in source
    assert 'storage = "f32", accum = "f32", axis = -1' in source
    assert "llvm.func @tessera_tile_softmax_f32" in source
    f16_source = emit_f16_softmax_tile_ir(entry="tessera_tile_softmax_f16")
    assert 'storage = "f16", accum = "f32", axis = -1' in f16_source
    assert "llvm.func @tessera_tile_softmax_f16" in f16_source


def _attention_module(
    *, dtype: str = "fp32", hq: int = 4, hkv: int = 2, causal: bool = False,
    bias: bool = False, window: tuple[int, int] | None = None,
    softcap: float = 0.0, dropout_p: float = 0.0,
) -> GraphIRModule:
    storage_ir = "f16" if dtype == "fp16" else "f32"
    q = IRType(f"tensor<1x{hq}x5x8x{storage_ir}>", ("1", str(hq), "5", "8"), dtype)
    k = IRType(f"tensor<1x{hkv}x7x8x{storage_ir}>", ("1", str(hkv), "7", "8"), dtype)
    v = IRType(f"tensor<1x{hkv}x7x6x{storage_ir}>", ("1", str(hkv), "7", "6"), dtype)
    o = IRType(f"tensor<1x{hq}x5x6xf32>", ("1", str(hq), "5", "6"), "fp32")
    bias_type = IRType(f"tensor<1x{hq}x5x7xf32>", ("1", str(hq), "5", "7"), "fp32")
    args = [IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    operands = ["%q", "%k", "%v"]
    operand_types = [str(q), str(k), str(v)]
    if bias:
        args.append(IRArg("bias", bias_type))
        operands.append("%bias")
        operand_types.append(str(bias_type))
    return GraphIRModule(functions=[GraphIRFunction(
        name="sm120_attention", args=args,
        result_types=[o], body=[IROp(
            result="o", op_name="tessera.flash_attn", operands=operands,
            operand_types=operand_types, result_type=str(o),
            kwargs={"scale": 0.3535533905932738, "causal": causal,
                    "window": window, "softcap": softcap,
                    "dropout_p": dropout_p, "dropout_seed": 17},
        )], return_values=["%o"],
    )])


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
@pytest.mark.parametrize("causal", [False, True])
def test_sm120_attention_owns_storage_accum_scale_causal_and_gqa(dtype, causal) -> None:
    module = _attention_module(dtype=dtype, causal=causal)
    assert requests_attention(module)
    assert supports_attention(module)
    source = emit_attention_tile_ir(
        entry="attention", storage="f16" if dtype == "fp16" else "f32",
        scale=0.3535533905932738, causal=causal,
    )
    assert "tile.attention_kernel" in source
    assert 'accum = "f32"' in source
    assert f"causal = {str(causal).lower()}" in source
    assert not supports_attention(_attention_module(hq=3, hkv=2))
    with pytest.raises(ValueError, match="finite and positive"):
        emit_attention_tile_ir(entry="bad", storage="f32", scale=0.0, causal=False)


def test_sm120_attention_owns_bias_window_softcap_and_dropout_contract() -> None:
    module = _attention_module(
        bias=True, causal=True, window=(3, 1), softcap=1.7, dropout_p=0.25
    )
    assert supports_attention(module)
    source = emit_attention_tile_ir(
        entry="attention_full", storage="f32", scale=0.5, causal=True,
        bias=True, window_left=3, window_right=1, softcap=1.7,
        dropout_p=0.25, dropout_seed=17,
    )
    assert "%bias: !llvm.ptr" in source
    assert "bias = true" in source
    assert "window_left = 3" in source
    assert "window_right = 1" in source
    assert "softcap = 1.7" in source
    assert "dropout_p = 0.25" in source
    with pytest.raises(ValueError, match="dropout_p"):
        emit_attention_tile_ir(
            entry="bad", storage="f32", scale=0.5, causal=False, dropout_p=1.0
        )


def _attention_backward_module(*, bias: bool = False) -> GraphIRModule:
    q = IRType("tensor<1x2x3x4xf32>", ("1", "2", "3", "4"), "fp32")
    k = IRType("tensor<1x1x4x4xf32>", ("1", "1", "4", "4"), "fp32")
    v = IRType("tensor<1x1x4x3xf32>", ("1", "1", "4", "3"), "fp32")
    do = IRType("tensor<1x2x3x3xf32>", ("1", "2", "3", "3"), "fp32")
    bias_type = IRType("tensor<1x2x3x4xf32>", ("1", "2", "3", "4"), "fp32")
    args = [IRArg("do", do), IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    operands = ["%do", "%q", "%k", "%v"]
    operand_types = [str(do), str(q), str(k), str(v)]
    if bias:
        args.append(IRArg("bias", bias_type))
        operands.append("%bias")
        operand_types.append(str(bias_type))
    return GraphIRModule(functions=[GraphIRFunction(
        name="sm120_attention_backward", args=args, result_types=[q, k, v],
        body=[IROp(
            result="dq,dk,dv", op_name="tessera.flash_attn_bwd",
            operands=operands, operand_types=operand_types,
            result_type=f"({q}, {k}, {v})",
            kwargs={"scale": 0.5, "causal": True, "window": (2, 1),
                    "softcap": 1.7, "route": "deterministic_direct",
                    "deterministic": True, "workspace_limit_bytes": 0},
        )], return_values=["%dq", "%dk", "%dv"],
    )])


def test_sm120_attention_backward_owns_determinism_and_workspace_contract() -> None:
    module = _attention_backward_module(bias=True)
    assert requests_attention_backward(module)
    assert supports_attention_backward(module)
    source = emit_attention_backward_tile_ir(
        entry="attention_backward", scale=0.5, causal=True, bias=True,
        window_left=2, window_right=1, softcap=1.7,
    )
    assert "tile.attention_backward_kernel" in source
    assert 'route = "deterministic_direct"' in source
    assert "deterministic = true" in source
    assert "workspace_bytes = 0 : i64" in source
    bad = _attention_backward_module()
    bad.functions[0].body[0].kwargs["route"] = "atomic"
    assert not supports_attention_backward(bad)


def _paged_kv_module(*, start: int = 3, end: int = 10) -> GraphIRModule:
    pages = IRType("tensor<4x4x3x8xf32>", ("4", "4", "3", "8"), "fp32")
    table = IRType("tensor<4xi32>", ("4",), "int32")
    out = IRType(f"tensor<{end-start}x3x8xf32>", (str(end-start), "3", "8"), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="paged_kv", args=[IRArg("pages", pages), IRArg("page_table", table)],
        result_types=[out], body=[IROp(
            result="slice", op_name="tessera.kv_cache.read",
            operands=["%pages", "%page_table"], operand_types=[str(pages), str(table)],
            result_type=str(out), kwargs={"start": start, "end": end},
        )], return_values=["%slice"],
    )])


def test_sm120_paged_kv_owns_typed_direct_descriptor_contract() -> None:
    module = _paged_kv_module()
    assert requests_paged_kv_read(module)
    assert supports_paged_kv_read(module)
    assert not supports_paged_kv_read(_paged_kv_module(start=-1, end=2))
    source = emit_paged_kv_read_tile_ir(entry="paged")
    assert "tile.paged_kv_read_kernel" in source
    assert 'table_storage = "i32"' in source
    assert 'route = "direct"' in source


def _reduction_module(
    *,
    dtype: str = "fp32",
    kind: str = "sum",
    axis: int = -1,
    keepdims: bool = False,
) -> GraphIRModule:
    storage_ir = "f16" if dtype == "fp16" else "f32"
    x = IRType(f"tensor<3x5x17x{storage_ir}>", ("3", "5", "17"), dtype)
    normalized_axis = axis % 3
    input_shape = ("3", "5", "17")
    output_shape = tuple(
        "1" if keepdims and dim == normalized_axis else extent
        for dim, extent in enumerate(input_shape)
        if keepdims or dim != normalized_axis
    )
    output_dims = "x".join(output_shape)
    out = IRType(f"tensor<{output_dims}xf32>", output_shape, "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="sm120_reduce",
                args=[IRArg("x", x)],
                result_types=[out],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.reduce" if kind == "sum" else f"tessera.{kind}",
                        operands=["%x"],
                        operand_types=[str(x)],
                        result_type=str(out),
                        kwargs={"axis": axis, "keepdims": keepdims},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


@pytest.mark.parametrize("kind", ["sum", "mean", "max", "amax"])
@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_sm120_reduction_owns_kind_precision_axis_and_nan_contract(kind, dtype) -> None:
    module = _reduction_module(dtype=dtype, kind=kind)
    assert requests_reduction(module)
    assert supports_reduction(module)
    physical_kind = "max" if kind == "amax" else kind
    source = emit_reduce_tile_ir(
        entry=f"reduce_{physical_kind}",
        storage="f16" if dtype == "fp16" else "f32",
        kind=physical_kind,
    )
    assert "tile.reduce_kernel" in source
    assert f'kind = "{physical_kind}"' in source
    assert 'accum = "f32"' in source
    assert 'nan_mode = "propagate"' in source
    assert supports_reduction(_reduction_module(axis=0))
    assert supports_reduction(_reduction_module(axis=1, keepdims=True))


def test_sm120_nvfp4_native_packager_owns_scales_and_k64_contract() -> None:
    a = IRType("tensor<33x129x!tessera.nvfp4>", ("33", "129"), "nvfp4")
    b = IRType("tensor<129x19x!tessera.nvfp4>", ("129", "19"), "nvfp4")
    sa = IRType("tensor<33x9xi8>", ("33", "9"), "uint8")
    sb = IRType("tensor<9x19xi8>", ("9", "19"), "uint8")
    c = IRType("tensor<33x19xf32>", ("33", "19"), "fp32")
    module = GraphIRModule(
        functions=[
            GraphIRFunction(
                name="sm120_nvfp4",
                args=[
                    IRArg("a", a),
                    IRArg("b", b),
                    IRArg("scale_a", sa),
                    IRArg("scale_b", sb),
                ],
                result_types=[c],
                body=[
                    IROp(
                        result="c",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(c),
                        kwargs={"scale_a": "%scale_a", "scale_b": "%scale_b"},
                    )
                ],
                return_values=["%c"],
            )
        ]
    )
    assert supports_nvfp4_matmul(module)
    assert supports_op("nvidia_sm120", "tessera.matmul", dtype="nvfp4").supported
    assert not module.verify().ok
    assert module.verify(target="nvidia_sm120").ok
    assert "tessera.matmul" in module.to_mlir(target="nvidia_sm120")

    source = emit_nvfp4_matmul_tile_ir(entry="tessera_tile_matmul_nvfp4")
    assert "tile.matmul_kernel %a, %b, %scale_a, %scale_b, %d" in source
    assert 'm = 16, n = 8, k = 64, a = "nvfp4", b = "nvfp4"' in source
    assert 'warps = 1 : i64, staging = "global"' in source


def _mx_matmul_module(dtype: str, *, scale_k: int = 2) -> GraphIRModule:
    mlir_dtype = {
        "fp6_e2m3": "!tessera.fp6_e2m3",
        "fp6_e3m2": "!tessera.fp6_e3m2",
        "fp4_e2m1": "!tessera.fp4_e2m1",
    }[dtype]
    a = IRType(f"tensor<19x37x{mlir_dtype}>", ("19", "37"), dtype)
    b = IRType(f"tensor<37x13x{mlir_dtype}>", ("37", "13"), dtype)
    sa = IRType(f"tensor<19x{scale_k}xi8>", ("19", str(scale_k)), "uint8")
    sb = IRType(f"tensor<{scale_k}x13xi8>", (str(scale_k), "13"), "uint8")
    c = IRType("tensor<19x13xf32>", ("19", "13"), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="sm120_mx",
                args=[
                    IRArg("a", a),
                    IRArg("b", b),
                    IRArg("scale_a", sa),
                    IRArg("scale_b", sb),
                ],
                result_types=[c],
                body=[
                    IROp(
                        result="c",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(c),
                        kwargs={"scale_a": "%scale_a", "scale_b": "%scale_b"},
                    )
                ],
                return_values=["%c"],
            )
        ]
    )


@pytest.mark.parametrize(
    ("dtype", "physical", "fragment_k"),
    [
        ("fp6_e2m3", "e2m3", 32),
        ("fp6_e3m2", "e3m2", 32),
        ("fp4_e2m1", "fp4_e2m1", 64),
    ],
)
def test_sm120_mx_packager_contract_is_distinct_from_nvfp4(
    dtype: str,
    physical: str,
    fragment_k: int,
) -> None:
    module = _mx_matmul_module(dtype)
    assert requests_mx_matmul(module)
    assert supports_mx_matmul(module)
    source = emit_mx_matmul_tile_ir(entry="mx", storage=physical)
    assert 'family = "mma_sync"' in source
    assert f'k = {fragment_k}, a = "{physical}", b = "{physical}"' in source
    assert "scale_a" in source and "scale_b" in source
    assert not requests_nvfp4_matmul(module)
    assert not supports_mx_matmul(_mx_matmul_module(dtype, scale_k=3))

    with pytest.raises(ValueError, match="MX matmul storage"):
        emit_mx_matmul_tile_ir(entry="bad", storage="nvfp4")


@pytest.mark.parametrize(
    ("kwargs", "scale_shape", "scale_dtype"),
    [
        ({}, (16, 4), "uint8"),
        ({"scale_a": "%scale_a", "scale_b": "%scale_b"}, (16, 3), "uint8"),
        ({"scale_a": "%scale_a", "scale_b": "%scale_b", "bias": "%bias"}, (16, 4), "uint8"),
        ({"scale_a": "%scale_a", "scale_b": "%scale_b"}, (16, 4), "fp32"),
    ],
)
def test_sm120_nvfp4_packager_rejects_invalid_scale_and_epilogue_contracts(
    kwargs: dict[str, object],
    scale_shape: tuple[int, int],
    scale_dtype: str,
) -> None:
    a = IRType("tensor<16x64x!tessera.nvfp4>", ("16", "64"), "nvfp4")
    b = IRType("tensor<64x8x!tessera.nvfp4>", ("64", "8"), "nvfp4")
    sa = IRType("tensor<16x4xi8>", tuple(str(x) for x in scale_shape), scale_dtype)
    sb = IRType("tensor<4x8xi8>", ("4", "8"), scale_dtype)
    c = IRType("tensor<16x8xf32>", ("16", "8"), "fp32")
    module = GraphIRModule(
        functions=[
            GraphIRFunction(
                name="invalid_nvfp4",
                args=[
                    IRArg("a", a),
                    IRArg("b", b),
                    IRArg("scale_a", sa),
                    IRArg("scale_b", sb),
                ],
                result_types=[c],
                body=[
                    IROp(
                        result="c",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(c),
                        kwargs=kwargs,
                    )
                ],
                return_values=["%c"],
            )
        ]
    )
    assert requests_nvfp4_matmul(module)
    assert not supports_nvfp4_matmul(module)
    with pytest.raises(ValueError, match="logical scale_a/scale_b views"):
        package_nvfp4_matmul(module, pipeline_name="tessera-nvidia-pipeline-sm120")
