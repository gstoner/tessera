from __future__ import annotations

import numpy as np
import pytest

from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.primitive_coverage import NumericPolicy
from tessera.runtime import launch
from tessera.runtime import _submit_nvidia_sm120_native
from tessera.compiler.nvidia_native import package_paged_attention
from tests._support.nvidia import nvidia_cuda_host_ready


def _module(
    m: int,
    n: int,
    k: int,
    *,
    storage: str = "fp16",
) -> GraphIRModule:
    graph_storage, mlir_dtype = {
        "fp64": ("fp64", "f64"),
        "fp16": ("fp16", "f16"),
        "bf16": ("bf16", "bf16"),
        "tf32": ("fp32", "f32"),
        "fp8_e4m3": ("fp8_e4m3", "f8E4M3FN"),
        "fp8_e5m2": ("fp8_e5m2", "f8E5M2"),
        "int8": ("int8", "i8"),
    }[storage]
    a = IRType(f"tensor<{m}x{k}x{mlir_dtype}>", (str(m), str(k)), graph_storage)
    b = IRType(f"tensor<{k}x{n}x{mlir_dtype}>", (str(k), str(n)), graph_storage)
    result_storage = "int32" if storage == "int8" else "fp64" if storage == "fp64" else "fp32"
    result_ir = "i32" if storage == "int8" else "f64" if storage == "fp64" else "f32"
    c = IRType(
        f"tensor<{m}x{n}x{result_ir}>",
        (str(m), str(n)),
        result_storage,
    )
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"canonical_sm120_{mlir_dtype}",
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
                            NumericPolicy(storage="fp32", accum="fp32", math_mode="tf32") if storage == "tf32" else None
                        ),
                    )
                ],
                return_values=["%c"],
            )
        ]
    )


def _nvfp4_module(m: int, n: int, k: int) -> GraphIRModule:
    sk = (k + 15) // 16
    a = IRType(f"tensor<{m}x{k}x!tessera.nvfp4>", (str(m), str(k)), "nvfp4")
    b = IRType(f"tensor<{k}x{n}x!tessera.nvfp4>", (str(k), str(n)), "nvfp4")
    sa = IRType(f"tensor<{m}x{sk}xi8>", (str(m), str(sk)), "uint8")
    sb = IRType(f"tensor<{sk}x{n}xi8>", (str(sk), str(n)), "uint8")
    c = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="canonical_sm120_nvfp4",
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


def _mx_module(m: int, n: int, k: int, storage: str) -> GraphIRModule:
    sk = (k + 31) // 32
    mlir_dtype = {
        "fp6_e2m3": "!tessera.fp6_e2m3",
        "fp6_e3m2": "!tessera.fp6_e3m2",
        "fp4_e2m1": "!tessera.fp4_e2m1",
    }[storage]
    a = IRType(f"tensor<{m}x{k}x{mlir_dtype}>", (str(m), str(k)), storage)
    b = IRType(f"tensor<{k}x{n}x{mlir_dtype}>", (str(k), str(n)), storage)
    sa = IRType(f"tensor<{m}x{sk}xi8>", (str(m), str(sk)), "uint8")
    sb = IRType(f"tensor<{sk}x{n}xi8>", (str(sk), str(n)), "uint8")
    c = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"canonical_sm120_{storage}",
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


def _softmax_module(shape: tuple[int, ...], dtype: str) -> GraphIRModule:
    dims = "x".join(str(dim) for dim in shape)
    mlir_dtype = "f16" if dtype == "fp16" else "f32"
    x = IRType(f"tensor<{dims}x{mlir_dtype}>", tuple(map(str, shape)), dtype)
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="canonical_sm120_softmax",
                args=[IRArg("x", x)],
                result_types=[x],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.softmax",
                        operands=["%x"],
                        operand_types=[str(x)],
                        result_type=str(x),
                        kwargs={"axis": -1},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _attention_module(
    shape: tuple[int, int, int, int, int, int, int],
    storage: str,
    causal: bool,
    *,
    bias: bool = False,
    window: tuple[int, int] | None = None,
    softcap: float = 0.0,
    dropout_p: float = 0.0,
    dropout_seed: int = 0,
) -> GraphIRModule:
    b, hq, hkv, sq, sk, d, dv = shape
    graph_storage = "fp16" if storage == "fp16" else "fp32"
    ir_storage = "f16" if storage == "fp16" else "f32"
    q = IRType(f"tensor<{b}x{hq}x{sq}x{d}x{ir_storage}>", tuple(map(str, (b, hq, sq, d))), graph_storage)
    k = IRType(f"tensor<{b}x{hkv}x{sk}x{d}x{ir_storage}>", tuple(map(str, (b, hkv, sk, d))), graph_storage)
    v = IRType(f"tensor<{b}x{hkv}x{sk}x{dv}x{ir_storage}>", tuple(map(str, (b, hkv, sk, dv))), graph_storage)
    o = IRType(f"tensor<{b}x{hq}x{sq}x{dv}xf32>", tuple(map(str, (b, hq, sq, dv))), "fp32")
    bias_type = IRType(f"tensor<{b}x{hq}x{sq}x{sk}xf32>", tuple(map(str, (b, hq, sq, sk))), "fp32")
    args = [IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    operands = ["%q", "%k", "%v"]
    operand_types = [str(q), str(k), str(v)]
    if bias:
        args.append(IRArg("bias", bias_type))
        operands.append("%bias")
        operand_types.append(str(bias_type))
    return GraphIRModule(functions=[GraphIRFunction(
        name="canonical_sm120_attention",
        args=args, result_types=[o],
        body=[IROp(
            result="o", op_name="tessera.flash_attn", operands=operands,
            operand_types=operand_types, result_type=str(o),
            kwargs={"scale": 1.0 / np.sqrt(float(d)), "causal": causal,
                    "window": window, "softcap": softcap,
                    "dropout_p": dropout_p, "dropout_seed": dropout_seed},
        )], return_values=["%o"],
    )])


def _attention_backward_module(*, bias: bool = False, dtype: str = "fp32",
                               dropout_p: float = 0.0, dropout_seed: int = 0) -> GraphIRModule:
    ir = "f16" if dtype == "fp16" else "f32"
    q = IRType(f"tensor<1x2x3x4x{ir}>", ("1", "2", "3", "4"), dtype)
    k = IRType(f"tensor<1x1x4x4x{ir}>", ("1", "1", "4", "4"), dtype)
    v = IRType(f"tensor<1x1x4x3x{ir}>", ("1", "1", "4", "3"), dtype)
    do = IRType(f"tensor<1x2x3x3x{ir}>", ("1", "2", "3", "3"), dtype)
    bias_type = IRType("tensor<1x2x3x4xf32>", ("1", "2", "3", "4"), "fp32")
    args = [IRArg("do", do), IRArg("q", q), IRArg("k", k), IRArg("v", v)]
    operands = ["%do", "%q", "%k", "%v"]
    operand_types = [str(do), str(q), str(k), str(v)]
    if bias:
        args.append(IRArg("bias", bias_type))
        operands.append("%bias")
        operand_types.append(str(bias_type))
    return GraphIRModule(functions=[GraphIRFunction(
        name="canonical_sm120_attention_backward", args=args,
        result_types=[q, k, v], body=[IROp(
            result="dq,dk,dv", op_name="tessera.flash_attn_bwd",
            operands=operands, operand_types=operand_types,
            result_type=f"({q}, {k}, {v})",
            kwargs={"scale": 0.5, "causal": True, "window": (2, 1),
                    "softcap": 1.7, "route": "deterministic_direct",
                    "deterministic": True, "workspace_limit_bytes": 0,
                    "dropout_p": dropout_p, "dropout_seed": dropout_seed},
        )], return_values=["%dq", "%dk", "%dv"],
    )])


def _paged_kv_module(p: int, page_size: int, heads: int, dim: int, logical_pages: int,
                     start: int, end: int) -> GraphIRModule:
    pages = IRType(f"tensor<{p}x{page_size}x{heads}x{dim}xf32>",
                   tuple(map(str, (p, page_size, heads, dim))), "fp32")
    table = IRType(f"tensor<{logical_pages}xi32>", (str(logical_pages),), "int32")
    out = IRType(f"tensor<{end-start}x{heads}x{dim}xf32>",
                 tuple(map(str, (end-start, heads, dim))), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="canonical_sm120_paged_kv", args=[IRArg("pages", pages), IRArg("page_table", table)],
        result_types=[out], body=[IROp(
            result="slice", op_name="tessera.kv_cache.read",
            operands=["%pages", "%page_table"], operand_types=[str(pages), str(table)],
            result_type=str(out), kwargs={"start": start, "end": end},
        )], return_values=["%slice"],
    )])


def _reduction_module(shape: tuple[int, ...], dtype: str, kind: str, *,
                      axis: int = -1, keepdims: bool = False) -> GraphIRModule:
    dims = "x".join(str(dim) for dim in shape)
    mlir_dtype = "f16" if dtype == "fp16" else "f32"
    x = IRType(f"tensor<{dims}x{mlir_dtype}>", tuple(map(str, shape)), dtype)
    normalized_axis = axis + len(shape) if axis < 0 else axis
    output_shape = shape[:normalized_axis] + ((1,) if keepdims else ()) + shape[normalized_axis + 1:]
    output_dims = "x".join(str(dim) for dim in output_shape)
    out = IRType(f"tensor<{output_dims}xf32>", tuple(map(str, output_shape)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="canonical_sm120_reduce",
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


def _epilogue_module(
    m: int,
    n: int,
    k: int,
    storage: str,
    *,
    bias: bool,
    activation: str,
    residual: bool,
) -> GraphIRModule:
    graph_storage, ir = {
        "fp16": ("fp16", "f16"), "bf16": ("bf16", "bf16"),
        "tf32": ("fp32", "f32"), "fp8_e4m3": ("fp8_e4m3", "f8E4M3FN"),
        "fp8_e5m2": ("fp8_e5m2", "f8E5M2"),
    }[storage]
    a = IRType(f"tensor<{m}x{k}x{ir}>", (str(m), str(k)), graph_storage)
    b = IRType(f"tensor<{k}x{n}x{ir}>", (str(k), str(n)), graph_storage)
    bias_type = IRType(f"tensor<{n}xf32>", (str(n),), "fp32")
    residual_type = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    out = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    args = [IRArg("a", a), IRArg("b", b)]
    kwargs: dict[str, object] = {"activation": activation}
    if bias:
        args.append(IRArg("bias", bias_type))
        kwargs["bias"] = "%bias"
    if residual:
        args.append(IRArg("residual", residual_type))
        kwargs["residual"] = "%residual"
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name="canonical_sm120_epilogue",
                args=args,
                result_types=[out],
                body=[
                    IROp(
                        result="c",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(out),
                        kwargs=kwargs,
                        numeric_policy=(
                            NumericPolicy(storage="fp32", accum="fp32", math_mode="tf32")
                            if storage == "tf32" else None
                        ),
                    )
                ],
                return_values=["%c"],
            )
        ]
    )


def _pack_nvfp4(codes: np.ndarray, axis: int) -> np.ndarray:
    if codes.shape[axis] % 2:
        padding = [(0, 0)] * codes.ndim
        padding[axis] = (0, 1)
        codes = np.pad(codes, padding)
    lo = np.take(codes, np.arange(0, codes.shape[axis], 2), axis=axis)
    hi = np.take(codes, np.arange(1, codes.shape[axis], 2), axis=axis)
    return np.ascontiguousarray(lo | (hi << np.uint8(4)))


def _decode_e2m1(codes: np.ndarray) -> np.ndarray:
    table = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    )
    return table[codes]


def _decode_ue4m3(codes: np.ndarray) -> np.ndarray:
    exponent = ((codes >> 3) & 15).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    normal = np.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    subnormal = np.ldexp(mantissa / 8.0, -6)
    return np.where(exponent == 0, subnormal, normal).astype(np.float32)


def _decode_ue8m0(codes: np.ndarray) -> np.ndarray:
    return np.ldexp(np.ones(codes.shape, np.float32), codes.astype(np.int32) - 127)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("shape", [(16, 8, 16), (37, 29, 23)])
def test_canonical_sm120_request_packages_registers_launches_and_compares(shape) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    m, n, k = shape
    module = _module(m, n, k)
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-1",
        target="nvidia_sm120",
        options={"package_native": True, "nvidia_schedule": "shared"},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None
    assert bundle.native_image.resource_record is not None
    assert "registers_per_thread" in bundle.native_image.resource_record.metrics
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.native_image.resource_record.metrics["spill_load_bytes"] == 0
    assert "static_shared_memory_bytes" in bundle.native_image.resource_record.metrics
    assert bundle.launch_descriptor is not None
    assert "tile.matmul_kernel" in bundle.tile.text
    assert "nvvm.mma.sync" in bundle.target_ir.text
    assert ".visible .entry tessera_tile_matmul_shared_f16" in bundle.backend.text

    warm = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-1",
        target="nvidia_sm120",
        options={"package_native": True, "nvidia_schedule": "shared"},
        enable_tool_validation=False,
    )
    assert warm.native_image is not None
    assert warm.native_image.compile_state == "warm_cache"
    assert warm.native_image.image_digest == bundle.native_image.image_digest
    assert warm.launch_descriptor == bundle.launch_descriptor

    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    rng = np.random.default_rng(17)
    a = np.ascontiguousarray((rng.standard_normal((m, k)) * 0.25).astype(np.float16))
    b = np.asfortranarray((rng.standard_normal((k, n)) * 0.25).astype(np.float16))
    c = np.zeros((m, n), dtype=np.float32)
    result = launch(
        artifact,
        {
            "a": a,
            "b": b,
            "c": c,
            "M": m,
            "N": n,
            "K": k,
        },
    )
    assert result["ok"] is True, result.get("reason")
    assert result["execution_mode"] == "descriptor"
    assert result["output"] is c
    reference = a.astype(np.float32) @ b.astype(np.float32)
    assert np.max(np.abs(c - reference)) < 2e-2


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("shape", [(16, 8, 16), (37, 29, 23)])
def test_canonical_sm120_bf16_packages_launches_and_compares(shape) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    m, n, k = shape
    module = _module(m, n, k, storage="bf16")
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-2",
        target="nvidia_sm120",
        options={"package_native": True, "nvidia_schedule": "shared"},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.native_image.resource_record.metrics["spill_load_bytes"] == 0
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.buffers[0].dtype == "bf16"
    assert bundle.launch_descriptor.provenance["storage"] == "bf16"
    assert "tile.matmul_kernel" in bundle.tile.text
    assert "__tessera_sm120_ab_stage_bf16" in bundle.target_ir.text
    assert "nvvm.mma.sync" in bundle.target_ir.text
    assert ".visible .entry tessera_tile_matmul_shared_bf16" in bundle.backend.text

    rng = np.random.default_rng(120_016 + k)
    a = np.ascontiguousarray((rng.standard_normal((m, k)) * 0.25).astype(ml_dtypes.bfloat16))
    b = np.asfortranarray((rng.standard_normal((k, n)) * 0.25).astype(ml_dtypes.bfloat16))
    c = np.zeros((m, n), dtype=np.float32)
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(
        artifact,
        {
            "a": a,
            "b": b,
            "c": c,
            "M": m,
            "N": n,
            "K": k,
        },
    )
    assert result["ok"] is True, result.get("reason")
    reference = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(c, reference, atol=4e-2, rtol=3e-3)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(
    "storage",
    ["fp64", "tf32", "fp8_e4m3", "fp8_e5m2", "int8"],
)
@pytest.mark.parametrize("shape", [(16, 8, 32), (19, 13, 37)])
def test_canonical_sm120_tf32_fp8_packages_launches_and_compares(
    storage,
    shape,
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    m, n, k = shape
    module = _module(m, n, k, storage=storage)
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-2",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.native_image.resource_record.metrics["spill_load_bytes"] == 0
    assert bundle.launch_descriptor is not None
    physical = {
        "fp64": "f64",
        "tf32": "tf32",
        "fp8_e4m3": "e4m3",
        "fp8_e5m2": "e5m2",
        "int8": "s8",
    }[storage]
    assert bundle.launch_descriptor.provenance["storage"] == physical
    assert bundle.launch_descriptor.provenance["schedule"] == "direct"
    assert f"tessera_tile_matmul_direct_{physical}" in bundle.backend.text

    rng = np.random.default_rng(120_032 + k)
    numpy_dtype = {
        "fp64": np.float64,
        "tf32": np.float32,
        "fp8_e4m3": ml_dtypes.float8_e4m3fn,
        "fp8_e5m2": ml_dtypes.float8_e5m2,
        "int8": np.int8,
    }[storage]
    if storage == "int8":
        a = np.ascontiguousarray(rng.integers(-7, 8, size=(m, k), dtype=np.int8))
        b = np.asfortranarray(rng.integers(-7, 8, size=(k, n), dtype=np.int8))
    else:
        scale = 0.125 if storage != "fp8_e5m2" else 0.25
        a = np.ascontiguousarray((rng.standard_normal((m, k)) * scale).astype(numpy_dtype))
        b = np.asfortranarray((rng.standard_normal((k, n)) * scale).astype(numpy_dtype))
    output_dtype = np.int32 if storage == "int8" else np.float64 if storage == "fp64" else np.float32
    c = np.zeros((m, n), dtype=output_dtype)
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(
        artifact,
        {
            "a": a,
            "b": b,
            "c": c,
            "M": m,
            "N": n,
            "K": k,
        },
    )
    assert result["ok"] is True, result.get("reason")
    if storage == "int8":
        reference = a.astype(np.int32) @ b.astype(np.int32)
        np.testing.assert_array_equal(c, reference)
    elif storage == "fp64":
        reference = a @ b
        np.testing.assert_allclose(c, reference, atol=1e-12, rtol=1e-12)
    else:
        reference = a.astype(np.float32) @ b.astype(np.float32)
        tolerance = (2e-2, 3e-3) if storage == "tf32" else (3e-2, 3e-3)
        np.testing.assert_allclose(
            c,
            reference,
            atol=tolerance[0],
            rtol=tolerance[1],
        )


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("shape", [(1, 16), (8, 64), (4, 300), (2, 3, 48)])
@pytest.mark.parametrize(
    ("storage", "numpy_dtype", "atol"),
    [("fp32", np.float32, 3e-6), ("fp16", np.float16, 5e-4)],
)
def test_canonical_sm120_softmax_packages_launches_and_compares(
    shape,
    storage,
    numpy_dtype,
    atol,
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    module = _softmax_module(shape, storage)
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-2",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.native_image.resource_record.metrics["spill_load_bytes"] == 0
    assert bundle.launch_descriptor is not None
    storage_ir = "f16" if storage == "fp16" else "f32"
    assert bundle.launch_descriptor.provenance["storage"] == storage_ir
    assert bundle.launch_descriptor.provenance["accum"] == "f32"
    assert "tile.softmax_kernel" in bundle.tile.text
    assert f".visible .entry tessera_tile_softmax_{storage_ir}" in bundle.backend.text

    warm = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-2",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert warm.native_image is not None
    assert warm.native_image.compile_state == "warm_cache"
    assert warm.native_image.image_digest == bundle.native_image.image_digest
    assert warm.launch_descriptor == bundle.launch_descriptor

    rng = np.random.default_rng(120_700 + shape[-1])
    x = np.ascontiguousarray((rng.standard_normal(shape) * 5.0).astype(numpy_dtype))
    if shape == (1, 16):
        x[0] = np.linspace(-1000.0, 1000.0, shape[-1], dtype=numpy_dtype)
    output = np.zeros_like(x)
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(artifact, {"x": x, "o": output, "Rows": rows, "K": shape[-1]})
    assert result["ok"] is True, result.get("reason")
    assert result["execution_mode"] == "descriptor"
    assert result["output"] is output
    x_f32 = x.astype(np.float32)
    shifted = x_f32 - x_f32.max(axis=-1, keepdims=True)
    reference = np.exp(shifted)
    reference /= reference.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(output.astype(np.float32), reference, atol=atol, rtol=0)
    np.testing.assert_allclose(
        output.astype(np.float32).sum(axis=-1),
        1.0,
        atol=atol * 3,
        rtol=0,
    )
    if shape == (1, 16):
        malformed = launch(
            artifact,
            {
                "x": x,
                "o": np.zeros((1, 15), numpy_dtype),
                "Rows": rows,
                "K": shape[-1],
            },
        )
        assert malformed["ok"] is False
        assert malformed["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("shape", [(8, 64), (4, 130), (2, 3, 17)])
@pytest.mark.parametrize("storage", ["fp16", "fp32"])
@pytest.mark.parametrize("kind", ["sum", "mean", "max"])
def test_canonical_sm120_reduction_packages_launches_and_compares(
    shape,
    storage,
    kind,
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    module = _reduction_module(shape, storage, kind)
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-2",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.native_image.resource_record.metrics["spill_load_bytes"] == 0
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["kind"] == kind
    assert bundle.launch_descriptor.provenance["nan_mode"] == "propagate"
    assert f"tessera_tile_reduce_{kind}_" in bundle.backend.text

    rng = np.random.default_rng(120_900 + shape[-1])
    numpy_dtype = np.float16 if storage == "fp16" else np.float32
    x = np.ascontiguousarray((rng.standard_normal(shape) * 1.5).astype(numpy_dtype))
    if kind == "max" and shape == (8, 64):
        x[0, 3] = np.nan
        x[1, 4] = np.inf
    output = np.zeros(shape[:-1], np.float32)
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    outer = int(np.prod(shape[:-1]))
    result = launch(artifact, {"x": x, "o": output, "Outer": outer,
                               "AxisExtent": shape[-1], "Inner": 1})
    assert result["ok"] is True, result.get("reason")
    reference = {
        "sum": np.sum,
        "mean": np.mean,
        "max": np.max,
    }[kind](x.astype(np.float32), axis=-1)
    np.testing.assert_allclose(
        output,
        reference,
        rtol=0,
        atol=2e-5 if storage == "fp32" else 2e-3,
        equal_nan=True,
    )


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["fp16", "fp32"])
@pytest.mark.parametrize("kind,axis", [("sum", 0), ("mean", 1), ("max", 2)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("schedule", ["serial", "cooperative_128"])
def test_canonical_sm120_reduction_axis_keepdims_and_cooperative_candidate(
    storage, kind, axis, keepdims, schedule,
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    shape = (5, 131, 7)
    module = _reduction_module(shape, storage, kind, axis=axis, keepdims=keepdims)
    bundle = compile_graph_module(
        module, source_origin="NVIDIA-E2E-2", target="nvidia_sm120",
        options={"package_native": True, "nvidia_reduction_schedule": schedule},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    descriptor = bundle.launch_descriptor
    assert descriptor is not None and bundle.native_image is not None
    assert descriptor.provenance["axis"] == axis
    assert descriptor.provenance["keepdims"] is keepdims
    assert descriptor.provenance["schedule"] == schedule
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    rng = np.random.default_rng(121_700 + axis + int(keepdims))
    dtype = np.float16 if storage == "fp16" else np.float32
    x = np.ascontiguousarray((rng.standard_normal(shape) * 0.5).astype(dtype))
    output_shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1:]
    output = np.zeros(output_shape, np.float32)
    outer = int(np.prod(shape[:axis])) if axis else 1
    inner = int(np.prod(shape[axis + 1:])) if axis + 1 < len(shape) else 1
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(artifact, {"x": x, "o": output, "Outer": outer,
        "AxisExtent": shape[axis], "Inner": inner})
    assert result["ok"], result.get("reason")
    reference = {"sum": np.sum, "mean": np.mean, "max": np.max}[kind](
        x.astype(np.float32), axis=axis, keepdims=keepdims,
    )
    np.testing.assert_allclose(output, reference, rtol=0,
        atol=2e-3 if storage == "fp16" else 2e-5)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["fp16", "bf16"])
@pytest.mark.parametrize("activation", ["none", "relu", "gelu", "silu"])
@pytest.mark.parametrize("bias,residual", [(False, False), (True, False), (False, True), (True, True)])
def test_canonical_sm120_fused_epilogue_matrix(
    storage,
    activation,
    bias,
    residual,
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    from tessera.compiler.fusion import FusedRegion

    m, n, k = 19, 23, 29
    module = _epilogue_module(
        m,
        n,
        k,
        storage,
        bias=bias,
        activation=activation,
        residual=residual,
    )
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-2",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.launch_descriptor.provenance["epilogue"] == {
        "bias": bias,
        "activation": activation,
        "residual": residual,
        "order": ["matmul", "bias", "activation", "residual"],
    }
    rng = np.random.default_rng(121_100 + int(bias) * 7 + int(residual) * 11)
    dtype = np.float16 if storage == "fp16" else ml_dtypes.bfloat16
    a = np.ascontiguousarray((rng.standard_normal((m, k)) * 0.15).astype(dtype))
    b = np.asfortranarray((rng.standard_normal((k, n)) * 0.15).astype(dtype))
    bias_value = np.ascontiguousarray((rng.standard_normal(n) * 0.05).astype(np.float32)) if bias else None
    residual_value = np.ascontiguousarray((rng.standard_normal((m, n)) * 0.05).astype(np.float32)) if residual else None
    output = np.zeros((m, n), np.float32)
    bindings = {"a": a, "b": b, "c": output, "M": m, "N": n, "K": k}
    if bias_value is not None:
        bindings["bias"] = bias_value
    if residual_value is not None:
        bindings["residual"] = residual_value
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(artifact, bindings)
    assert result["ok"] is True, result.get("reason")
    epilogue = (() if not bias else ("bias",)) + (() if activation == "none" else (activation,))
    if not epilogue and not residual:
        reference = a.astype(np.float32) @ b.astype(np.float32)
    else:
        region = FusedRegion(
            epilogue=epilogue,
            residual=residual,
            storage_dtype="f16" if storage == "fp16" else "bf16",
        )
        reference = region.reference(
            a.astype(np.float32),
            b.astype(np.float32),
            bias_value,
            residual_value,
        )
    np.testing.assert_allclose(output, reference, atol=5e-3 if storage == "fp16" else 5e-2, rtol=3e-3)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["tf32", "fp8_e4m3", "fp8_e5m2"])
@pytest.mark.parametrize("activation", ["none", "relu", "gelu", "silu"])
@pytest.mark.parametrize("bias,residual", [(False, False), (True, False), (False, True), (True, True)])
def test_canonical_sm120_low_precision_fused_epilogue_matrix(
    storage, activation, bias, residual,
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    m, n, k = 11, 13, 17
    module = _epilogue_module(
        m, n, k, storage, bias=bias, activation=activation, residual=residual,
    )
    bundle = compile_graph_module(
        module, source_origin="NVIDIA-E2E-2", target="nvidia_sm120",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.launch_descriptor.provenance["schedule"] == "direct"
    rng = np.random.default_rng(121_300 + len(activation) + 7 * int(bias) + 11 * int(residual))
    dtype = {
        "tf32": np.float32,
        "fp8_e4m3": ml_dtypes.float8_e4m3fn,
        "fp8_e5m2": ml_dtypes.float8_e5m2,
    }[storage]
    scale = 0.12 if storage != "fp8_e5m2" else 0.2
    a = np.ascontiguousarray((rng.standard_normal((m, k)) * scale).astype(dtype))
    b = np.asfortranarray((rng.standard_normal((k, n)) * scale).astype(dtype))
    bias_value = np.ascontiguousarray((rng.standard_normal(n) * 0.03).astype(np.float32)) if bias else None
    residual_value = np.ascontiguousarray((rng.standard_normal((m, n)) * 0.03).astype(np.float32)) if residual else None
    output = np.zeros((m, n), np.float32)
    bindings = {"a": a, "b": b, "c": output, "M": m, "N": n, "K": k}
    if bias_value is not None: bindings["bias"] = bias_value
    if residual_value is not None: bindings["residual"] = residual_value
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(artifact, bindings)
    assert result["ok"], result.get("reason")
    reference = a.astype(np.float32) @ b.astype(np.float32)
    if bias_value is not None: reference = reference + bias_value
    def bounded_tanh(value):
        z = np.clip(value, -5.0, 5.0); z2 = z * z
        numerator = 135135.0 + 17325.0*z2 + 378.0*z2*z2 + z2*z2*z2
        denominator = 135135.0 + 62370.0*z2 + 3150.0*z2*z2 + 28.0*z2*z2*z2
        return z * numerator / denominator
    if activation == "relu": reference = np.maximum(reference, 0.0)
    elif activation == "silu": reference = reference * 0.5 * (1.0 + bounded_tanh(0.5 * reference))
    elif activation == "gelu":
        inner = 0.7978845608028654 * (reference + 0.044715 * reference**3)
        reference = 0.5 * reference * (1.0 + bounded_tanh(inner))
    if residual_value is not None: reference = reference + residual_value
    np.testing.assert_allclose(output, reference, atol=5e-2, rtol=4e-3)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("shape", [(16, 8, 64), (33, 19, 129), (7, 5, 31)])
def test_canonical_sm120_nvfp4_general_shape_scales_and_ragged(shape) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    m, n, k = shape
    module = _nvfp4_module(m, n, k)
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-1",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None
    assert bundle.native_image.resource_record is not None
    assert "registers_per_thread" in bundle.native_image.resource_record.metrics
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.native_image.resource_record.metrics["spill_load_bytes"] == 0
    assert "static_shared_memory_bytes" in bundle.native_image.resource_record.metrics
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["scale_vector_size"] == 16
    assert "scale_a" in bundle.tile.text and "scale_b" in bundle.tile.text
    assert "mxf4nvf4.block_scale" in bundle.target_ir.text

    warm = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-1",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert warm.native_image is not None
    assert warm.native_image.compile_state == "warm_cache"
    assert warm.native_image.image_digest == bundle.native_image.image_digest
    assert warm.launch_descriptor == bundle.launch_descriptor

    rng = np.random.default_rng(120_500 + m + n + k)
    a_codes = rng.integers(0, 16, size=(m, k), dtype=np.uint8)
    b_codes = rng.integers(0, 16, size=(k, n), dtype=np.uint8)
    sk = (k + 15) // 16
    choices = np.asarray([0x30, 0x38, 0x40], np.uint8)
    scale_a = np.ascontiguousarray(choices[(np.arange(m)[:, None] + np.arange(sk)[None, :]) % 3])
    scale_b = np.ascontiguousarray(choices[(2 * np.arange(sk)[:, None] + np.arange(n)[None, :]) % 3])
    a_packed = _pack_nvfp4(a_codes, 1)
    b_packed = _pack_nvfp4(b_codes, 0)
    c = np.zeros((m, n), np.float32)
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(
        artifact,
        {
            "a": a_packed,
            "b": b_packed,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "c": c,
            "M": m,
            "N": n,
            "K": k,
        },
    )
    assert result["ok"] is True, result.get("reason")
    a = _decode_e2m1(a_codes) * np.repeat(_decode_ue4m3(scale_a), 16, axis=1)[:, :k]
    b = _decode_e2m1(b_codes) * np.repeat(_decode_ue4m3(scale_b), 16, axis=0)[:k, :]
    np.testing.assert_allclose(c, a @ b, rtol=0, atol=2e-3)

    if shape == (16, 8, 64):
        malformed = launch(
            artifact,
            {
                "a": a_packed,
                "b": b_packed,
                "scale_a": np.ascontiguousarray(scale_a[:, :-1]),
                "scale_b": scale_b,
                "c": c,
                "M": m,
                "N": n,
                "K": k,
            },
        )
        assert malformed["ok"] is False
        assert malformed["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"
        assert "shape guard failed" in malformed["reason"]


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["fp6_e2m3", "fp6_e3m2", "fp4_e2m1"])
@pytest.mark.parametrize("shape", [(16, 8, 64), (19, 13, 69)])
def test_canonical_sm120_mx_general_shape_scales_and_ragged(storage, shape) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    m, n, k = shape
    module = _mx_module(m, n, k, storage)
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-2",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.native_image.resource_record.metrics["spill_load_bytes"] == 0
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["scale_dtype"] == "ue8m0"
    assert bundle.launch_descriptor.provenance["scale_vector_size"] == 32
    assert "mma.sync.aligned" in bundle.backend.text
    assert "block_scale" in bundle.backend.text

    value_dtype = {
        "fp6_e2m3": ml_dtypes.float6_e2m3fn,
        "fp6_e3m2": ml_dtypes.float6_e3m2fn,
        "fp4_e2m1": ml_dtypes.float4_e2m1fn,
    }[storage]
    rng = np.random.default_rng(120_800 + m + n + k)
    a_values = (rng.standard_normal((m, k)) * 1.25).astype(value_dtype)
    b_values = (rng.standard_normal((k, n)) * 1.25).astype(value_dtype)
    a_codes = a_values.view(np.uint8)
    b_codes = b_values.view(np.uint8)
    if storage == "fp4_e2m1":
        a_physical = _pack_nvfp4(a_codes, 1)
        b_physical = _pack_nvfp4(b_codes, 0)
    else:
        a_physical = np.ascontiguousarray(a_codes)
        b_physical = np.ascontiguousarray(b_codes)
    sk = (k + 31) // 32
    scale_codes = np.asarray([126, 127, 128], np.uint8)
    scale_a = np.ascontiguousarray(scale_codes[(np.arange(m)[:, None] + np.arange(sk)[None, :]) % 3])
    scale_b = np.ascontiguousarray(scale_codes[(2 * np.arange(sk)[:, None] + np.arange(n)[None, :]) % 3])
    c = np.zeros((m, n), np.float32)
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(
        artifact,
        {
            "a": a_physical,
            "b": b_physical,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "c": c,
            "M": m,
            "N": n,
            "K": k,
        },
    )
    assert result["ok"] is True, result.get("reason")
    a = a_values.astype(np.float32) * np.repeat(_decode_ue8m0(scale_a), 32, axis=1)[:, :k]
    b = b_values.astype(np.float32) * np.repeat(_decode_ue8m0(scale_b), 32, axis=0)[:k, :]
    np.testing.assert_allclose(c, a @ b, rtol=2e-3, atol=2e-2)


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("storage", ["fp16", "fp32"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "shape",
    [(1, 4, 4, 5, 7, 8, 6), (1, 4, 1, 6, 6, 7, 5)],
    ids=["mha_rectangular", "mqa_ragged"],
)
def test_canonical_sm120_attention_packages_launches_and_compares(
    storage, causal, shape
) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    module = _attention_module(shape, storage, causal)
    bundle = compile_graph_module(
        module,
        source_origin="NVIDIA-E2E-2",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.orchestration_state == "launchable"
    assert bundle.native_image is not None
    assert bundle.native_image.resource_record is not None
    assert bundle.native_image.resource_record.metrics["spill_store_bytes"] == 0
    assert bundle.native_image.resource_record.metrics["spill_load_bytes"] == 0
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["accum"] == "f32"
    assert bundle.launch_descriptor.provenance["causal"] is causal
    assert "tile.attention_kernel" in bundle.tile.text
    assert ".visible .entry tessera_tile_attention_" in bundle.backend.text

    b, hq, hkv, sq, sk, d, dv = shape
    rng = np.random.default_rng(121_100 + sum(shape) + int(causal))
    numpy_dtype = np.float16 if storage == "fp16" else np.float32
    q = np.ascontiguousarray((rng.standard_normal((b, hq, sq, d)) * 0.5).astype(numpy_dtype))
    k = np.ascontiguousarray((rng.standard_normal((b, hkv, sk, d)) * 0.5).astype(numpy_dtype))
    v = np.ascontiguousarray((rng.standard_normal((b, hkv, sk, dv)) * 0.5).astype(numpy_dtype))
    output = np.zeros((b, hq, sq, dv), np.float32)
    scalars = dict(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), shape, strict=True))
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(artifact, {"q": q, "k": k, "v": v, "o": output, **scalars})
    assert result["ok"] is True, result.get("reason")

    reference = np.zeros_like(output)
    scale = 1.0 / np.sqrt(float(d))
    ratio = hq // hkv
    for batch in range(b):
        for query_head in range(hq):
            kv_head = query_head // ratio
            scores = q[batch, query_head].astype(np.float32) @ k[batch, kv_head].astype(np.float32).T
            scores *= scale
            if causal:
                scores = np.where(
                    np.arange(sk)[None, :] <= np.arange(sq)[:, None], scores, -np.inf
                )
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights /= weights.sum(axis=-1, keepdims=True)
            reference[batch, query_head] = weights @ v[batch, kv_head].astype(np.float32)
    tolerance = 2e-3 if storage == "fp16" else 2e-5
    np.testing.assert_allclose(output, reference, atol=tolerance, rtol=tolerance)


@pytest.mark.hardware_nvidia
def test_canonical_sm120_attention_advanced_forward_contract_and_dropout_replay() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    shape = (1, 2, 1, 5, 7, 8, 6)
    module = _attention_module(
        shape, "fp32", True, bias=True, window=(2, 1), softcap=1.7,
        dropout_p=0.25, dropout_seed=12345,
    )
    bundle = compile_graph_module(
        module, source_origin="NVIDIA-E2E-2", target="nvidia_sm120",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.native_image and bundle.launch_descriptor
    assert bundle.launch_descriptor.provenance["dropout_rng"] == "lcg32_counter_v1"
    b, hq, hkv, sq, sk, d, dv = shape
    rng = np.random.default_rng(121_303)
    q = np.ascontiguousarray(rng.standard_normal((b, hq, sq, d)).astype(np.float32) * 0.4)
    k = np.ascontiguousarray(rng.standard_normal((b, hkv, sk, d)).astype(np.float32) * 0.4)
    v = np.ascontiguousarray(rng.standard_normal((b, hkv, sk, dv)).astype(np.float32) * 0.4)
    bias = np.ascontiguousarray(rng.standard_normal((b, hq, sq, sk)).astype(np.float32) * 0.1)
    first = np.zeros((b, hq, sq, dv), np.float32)
    second = np.zeros_like(first)
    scalars = dict(zip(("B", "Hq", "Hkv", "Sq", "Sk", "D", "Dv"), shape, strict=True))
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    for output in (first, second):
        result = launch(
            artifact, {"q": q, "k": k, "v": v, "bias": bias, "o": output, **scalars}
        )
        assert result["ok"], result.get("reason")
    np.testing.assert_array_equal(first, second)

    def pade_tanh(x):
        z = np.clip(x, -5.0, 5.0)
        z2 = z * z
        numerator = 135135.0 + 17325.0 * z2 + 378.0 * z2 * z2 + z2 * z2 * z2
        denominator = 135135.0 + 62370.0 * z2 + 3150.0 * z2 * z2 + 28.0 * z2 * z2 * z2
        return z * numerator / denominator

    reference = np.zeros_like(first)
    ratio = hq // hkv
    threshold = int(0.25 * 4294967296.0)
    for batch in range(b):
        for head in range(hq):
            kv_head = head // ratio
            for query in range(sq):
                scores = q[batch, head, query] @ k[batch, kv_head].T
                scores = scores / np.sqrt(float(d)) + bias[batch, head, query]
                scores = 1.7 * pade_tanh(scores / 1.7)
                keys = np.arange(sk)
                legal = (keys <= query) & (keys >= query - 2) & (keys <= query + 1)
                scores = np.where(legal, scores, -np.inf)
                weights = np.exp(scores - np.max(scores))
                weights /= np.sum(weights)
                counter = (((batch * hq + head) * sq + query) * sk + keys)
                hashes = (counter * 1664525 + 12345 + 1013904223) & 0xFFFFFFFF
                weights = weights * (hashes >= threshold) / 0.75
                reference[batch, head, query] = weights @ v[batch, kv_head]
    np.testing.assert_allclose(first, reference, atol=4e-4, rtol=4e-4)

    malformed = launch(
        artifact,
        {"q": q, "k": k, "v": v, "bias": bias[..., :-1], "o": first, **scalars},
    )
    assert malformed["ok"] is False
    assert malformed["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"


@pytest.mark.hardware_nvidia
def test_canonical_sm120_attention_backward_is_deterministic_and_oracle_proven() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    module = _attention_backward_module(bias=True)
    bundle = compile_graph_module(
        module, source_origin="NVIDIA-PARITY-ATTN-BWD", target="nvidia_sm120",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.native_image and bundle.launch_descriptor
    descriptor = bundle.launch_descriptor
    assert descriptor.provenance["route"] == "deterministic_direct"
    assert descriptor.provenance["dk_dv_reduction"] == "single_owner_fixed_order"
    assert descriptor.workspace.bytes == 0
    rng = np.random.default_rng(121_500)
    q = (rng.standard_normal((1, 2, 3, 4)) * 0.2).astype(np.float32)
    k = (rng.standard_normal((1, 1, 4, 4)) * 0.2).astype(np.float32)
    v = (rng.standard_normal((1, 1, 4, 3)) * 0.2).astype(np.float32)
    do = (rng.standard_normal((1, 2, 3, 3)) * 0.2).astype(np.float32)
    bias = (rng.standard_normal((1, 2, 3, 4)) * 0.05).astype(np.float32)
    dq = np.zeros_like(q); dk = np.zeros_like(k); dv = np.zeros_like(v)
    scalars = {"B": 1, "Hq": 2, "Hkv": 1, "Sq": 3, "Sk": 4, "D": 4, "Dv": 3}
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    launch_args = {"do": do, "q": q, "k": k, "v": v, "bias": bias,
                   "dq": dq, "dk": dk, "dv": dv, **scalars}
    first = launch(artifact, launch_args)
    assert first["ok"], first.get("reason")
    observed = tuple(x.copy() for x in first["output"])
    dq.fill(0); dk.fill(0); dv.fill(0)
    second = launch(artifact, launch_args)
    assert second["ok"], second.get("reason")
    for a, b in zip(observed, second["output"]):
        np.testing.assert_array_equal(a, b)

    def pade_tanh(x):
        z = np.clip(x, -5.0, 5.0)
        z2 = z * z
        numerator = 135135.0 + 17325.0*z2 + 378.0*z2*z2 + z2*z2*z2
        denominator = 135135.0 + 62370.0*z2 + 3150.0*z2*z2 + 28.0*z2*z2*z2
        return z * numerator / denominator

    ref_dq = np.zeros_like(q); ref_dk = np.zeros_like(k); ref_dv = np.zeros_like(v)
    keys = np.arange(4)
    for head in range(2):
        for query in range(3):
            raw = q[0, head, query] @ k[0, 0].T * 0.5 + bias[0, head, query]
            t = pade_tanh(raw / 1.7)
            scores = 1.7 * t
            legal = (keys <= query) & (keys >= query - 2) & (keys <= query + 1)
            scores = np.where(legal, scores, -np.inf)
            p = np.exp(scores - np.max(scores)); p /= np.sum(p)
            delta = do[0, head, query] @ (p @ v[0, 0])
            for key in np.flatnonzero(legal):
                ds = p[key] * (do[0, head, query] @ v[0, 0, key] - delta)
                ds *= 1.0 - t[key] * t[key]
                ref_dq[0, head, query] += ds * 0.5 * k[0, 0, key]
                ref_dk[0, 0, key] += ds * 0.5 * q[0, head, query]
                ref_dv[0, 0, key] += p[key] * do[0, head, query]
    for actual, expected in zip(observed, (ref_dq, ref_dk, ref_dv)):
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    malformed = launch(artifact, {**launch_args, "dk": dk[..., :-1]})
    assert malformed["ok"] is False
    assert malformed["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"


@pytest.mark.hardware_nvidia
def test_canonical_sm120_fp16_attention_dropout_backward_replays_mask() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    dropout_p, seed = 0.25, 771
    module = _attention_backward_module(
        dtype="fp16", dropout_p=dropout_p, dropout_seed=seed
    )
    bundle = compile_graph_module(
        module, source_origin="NVIDIA-E2E-2", target="nvidia_sm120",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.native_image and bundle.launch_descriptor
    assert bundle.launch_descriptor.provenance["storage"] == "f16"
    assert bundle.launch_descriptor.provenance["dropout_rng"] == "lcg32_counter_v1"
    rng = np.random.default_rng(121_771)
    q = (rng.standard_normal((1, 2, 3, 4)) * 0.2).astype(np.float16)
    k = (rng.standard_normal((1, 1, 4, 4)) * 0.2).astype(np.float16)
    v = (rng.standard_normal((1, 1, 4, 3)) * 0.2).astype(np.float16)
    do = (rng.standard_normal((1, 2, 3, 3)) * 0.2).astype(np.float16)
    dq = np.zeros_like(q); dk = np.zeros_like(k); dv = np.zeros_like(v)
    scalars = {"B": 1, "Hq": 2, "Hkv": 1, "Sq": 3, "Sk": 4, "D": 4, "Dv": 3}
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    args = {"do": do, "q": q, "k": k, "v": v,
            "dq": dq, "dk": dk, "dv": dv, **scalars}
    first = launch(artifact, args); assert first["ok"], first.get("reason")
    observed = tuple(value.copy() for value in first["output"])
    dq.fill(0); dk.fill(0); dv.fill(0)
    second = launch(artifact, args); assert second["ok"], second.get("reason")
    for lhs, rhs in zip(observed, second["output"]):
        np.testing.assert_array_equal(lhs, rhs)

    def pade_tanh(x):
        z = np.clip(x, -5.0, 5.0); z2 = z * z
        return z * (135135.0 + 17325.0*z2 + 378.0*z2*z2 + z2*z2*z2) / (
            135135.0 + 62370.0*z2 + 3150.0*z2*z2 + 28.0*z2*z2*z2)

    qf, kf, vf, dof = (value.astype(np.float32) for value in (q, k, v, do))
    refs = [np.zeros_like(qf), np.zeros_like(kf), np.zeros_like(vf)]
    keys = np.arange(4); threshold = int(dropout_p * 4294967296.0)
    for head in range(2):
        for query in range(3):
            raw = qf[0, head, query] @ kf[0, 0].T * 0.5
            t = pade_tanh(raw / 1.7); scores = 1.7 * t
            legal = (keys <= query) & (keys >= query - 2) & (keys <= query + 1)
            scores = np.where(legal, scores, -np.inf)
            p = np.exp(scores - np.max(scores)); p /= p.sum()
            counter = ((head * 3 + query) * 4 + keys)
            hashes = (counter * 1664525 + seed + 1013904223) & 0xFFFFFFFF
            dropout_scale = (hashes >= threshold).astype(np.float32) / (1.0 - dropout_p)
            dp = dropout_scale * (vf[0, 0] @ dof[0, head, query])
            delta = np.sum(p * dp)
            ds = p * (dp - delta) * (1.0 - t*t)
            ds = np.where(legal, ds, 0.0)
            refs[0][0, head, query] += (ds[:, None] * 0.5 * kf[0, 0]).sum(0)
            refs[1][0, 0] += ds[:, None] * 0.5 * qf[0, head, query]
            refs[2][0, 0] += (p * dropout_scale)[:, None] * dof[0, head, query]
    for actual, expected in zip(observed, refs):
        np.testing.assert_allclose(actual.astype(np.float32), expected, rtol=3e-3, atol=3e-3)


@pytest.mark.hardware_nvidia
def test_canonical_sm120_fused_paged_attention_owns_causal_offset() -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    rng = np.random.default_rng(120_712)
    p, lp, ps, h, qlen, tokens, d = 4, 4, 4, 2, 3, 13, 8
    offset = tokens - qlen
    table = np.array([2, 0, 3, 1], np.int32)
    indices = np.arange(tokens, dtype=np.int64)
    logical_k = (rng.standard_normal((lp * ps, h, d)) * 0.1).astype(np.float32)
    logical_v = (rng.standard_normal((lp * ps, h, d)) * 0.1).astype(np.float32)
    kp = np.empty((p, ps, h, d), np.float32); vp = np.empty_like(kp)
    for logical, physical in enumerate(table):
        kp[physical] = logical_k[logical * ps:(logical + 1) * ps]
        vp[physical] = logical_v[logical * ps:(logical + 1) * ps]
    q = (rng.standard_normal((h, qlen, d)) * 0.1).astype(np.float32)
    o = np.zeros_like(q)
    package = package_paged_attention(
        physical_pages=p, logical_pages=lp, page_size=ps, heads=h,
        query_length=qlen, tokens=tokens, dim=d, scale=d ** -0.5,
        causal=True, causal_offset=offset,
        pipeline_name="tessera-nvidia-pipeline-sm120",
    )
    got = _submit_nvidia_sm120_native(
        package.image, package.descriptor,
        {"Q": q, "K_pages": kp, "V_pages": vp, "page_table": table,
         "token_indices": indices, "O": o},
        {"P": p, "LP": lp, "PageSize": ps, "H": h, "QueryLength": qlen,
         "Tokens": tokens, "D": d, "CausalOffset": offset}, None,
    )
    expected = np.zeros_like(q)
    for head in range(h):
        for query in range(qlen):
            scores = q[head, query] @ logical_k[indices, head].T * d ** -0.5
            scores[np.arange(tokens) > query + offset] = -np.inf
            probs = np.exp(scores - np.max(scores)); probs /= probs.sum()
            expected[head, query] = probs @ logical_v[indices, head]
    np.testing.assert_allclose(got, expected, rtol=3e-5, atol=3e-5)
    assert package.descriptor.provenance["causal_offset"] == offset
    assert package.descriptor.provenance["route"] == "fused_direct"

    bad = table.copy(); bad[1] = p
    with pytest.raises((ValueError, RuntimeError)):
        _submit_nvidia_sm120_native(
            package.image, package.descriptor,
            {"Q": q, "K_pages": kp, "V_pages": vp, "page_table": bad,
             "token_indices": indices, "O": o},
            {"P": p, "LP": lp, "PageSize": ps, "H": h, "QueryLength": qlen,
             "Tokens": tokens, "D": d, "CausalOffset": offset}, None,
        )


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize(("start", "end"), [(0, 1), (3, 10), (7, 13), (15, 16)])
def test_canonical_sm120_paged_kv_remap_boundaries_and_invalid_table(start, end) -> None:
    if not nvidia_cuda_host_ready():
        pytest.skip("host WSL CUDA device/toolchain unavailable")
    p, page_size, heads, dim, logical_pages = 4, 4, 3, 8, 4
    module = _paged_kv_module(p, page_size, heads, dim, logical_pages, start, end)
    bundle = compile_graph_module(
        module, source_origin="NVIDIA-E2E-2", target="nvidia_sm120",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.native_image and bundle.launch_descriptor
    assert bundle.launch_descriptor.provenance["route"] == "direct"
    rng = np.random.default_rng(121_400 + start + end)
    logical = rng.standard_normal((logical_pages * page_size, heads, dim)).astype(np.float32)
    table = np.array([2, 0, 3, 1], np.int32)
    pages = np.empty((p, page_size, heads, dim), np.float32)
    for logical_page, physical_page in enumerate(table):
        pages[physical_page] = logical[logical_page * page_size:(logical_page + 1) * page_size]
    output = np.zeros((end - start, heads, dim), np.float32)
    dims = {"P": p, "LP": logical_pages, "PageSize": page_size, "H": heads,
            "D": dim, "Start": start, "Tokens": end - start}
    artifact = compile_result_from_bundle(bundle, module=module).to_runtime_artifact()
    result = launch(artifact, {"pages": pages, "page_table": table, "slice": output, **dims})
    assert result["ok"], result.get("reason")
    np.testing.assert_array_equal(output, logical[start:end])

    remapped = np.array([1, 3, 0, 2], np.int32)
    remapped_pages = np.empty_like(pages)
    for logical_page, physical_page in enumerate(remapped):
        remapped_pages[physical_page] = logical[
            logical_page * page_size:(logical_page + 1) * page_size
        ]
    output.fill(0)
    result = launch(
        artifact, {"pages": remapped_pages, "page_table": remapped, "slice": output, **dims}
    )
    assert result["ok"], result.get("reason")
    np.testing.assert_array_equal(output, logical[start:end])

    invalid = table.copy()
    invalid[0] = p
    rejected = launch(
        artifact, {"pages": pages, "page_table": invalid, "slice": output, **dims}
    )
    assert rejected["ok"] is False
    assert "invalid physical page" in rejected["reason"]
