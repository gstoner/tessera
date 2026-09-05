from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import apple_native, nvidia_native, rocm_native, scheduled_matmul, x86_native
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import ScheduledMatmulArtifact
from tests._support.nvidia import nvidia_cuda_host_ready


from tests._support.compiler_tool import missing_dialects, tessera_opt_path

# Tool presence is not capability. These lower SM120 scheduled matmul, which
# the boundary rejects with "requires the registered NVIDIA Target IR dialect"
# on a driver built without `TESSERA_BUILD_NVIDIA_BACKEND` -- a build-selection
# fact, not a defect in the code under test.
_opt = tessera_opt_path()
requires_tessera_opt = pytest.mark.skipif(
    scheduled_matmul.find_tessera_opt() is None,
    reason="requires the production tessera-opt compiler",
)
requires_nvidia_target_ir = pytest.mark.skipif(
    _opt is None or bool(missing_dialects(_opt, "tessera_nvidia")),
    reason="tessera-opt does not register the tessera_nvidia Target IR dialect "
           "(configure TESSERA_BUILD_NVIDIA_BACKEND, or point TESSERA_OPT at "
           "a driver that has it)",
)


def _module(
    *,
    target: str,
    shape: tuple[int, int, int] = (17, 19, 23),
    dtype: str | None = None,
    output_dtype: str = "fp32",
    activation: str = "none",
    bias: bool = False,
    residual: bool = False,
) -> GraphIRModule:
    m, k, n = shape
    inferred_dtype, element = (
        ("fp16", "f16")
        if target in ("rocm", "apple_gpu_f16", "nvidia_sm120")
        else ("fp32", "f32")
    )
    dtype = dtype or inferred_dtype
    element = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    a = IRType(f"tensor<{m}x{k}x{element}>", (str(m), str(k)), dtype)
    b = IRType(f"tensor<{k}x{n}x{element}>", (str(k), str(n)), dtype)
    output_element = {"fp16": "f16", "fp32": "f32"}[output_dtype]
    output = IRType(
        f"tensor<{m}x{n}x{output_element}>", (str(m), str(n)), output_dtype
    )
    args = [IRArg("a", a), IRArg("b", b)]
    operands = ["%a", "%b"]
    operand_types = [str(a), str(b)]
    kwargs: dict[str, object] = {"activation": activation}
    if bias:
        bias_type = IRType(f"tensor<{n}xf32>", (str(n),), "fp32")
        args.append(IRArg("bias", bias_type))
        kwargs["bias"] = "%bias"
        operands.append("%bias")
        operand_types.append(str(bias_type))
    if residual:
        residual_type = IRType(
            f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32"
        )
        args.append(IRArg("residual", residual_type))
        kwargs["residual"] = "%residual"
        operands.append("%residual")
        operand_types.append(str(residual_type))
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"{target}_scheduled_matmul",
                args=args,
                result_types=[output],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.matmul",
                        operands=operands,
                        operand_types=operand_types,
                        result_type=str(output),
                        kwargs=kwargs,
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


def _dynamic_module(
    *, bounds: tuple[int, int, int] = (32, 24, 32), dtype: str = "fp16",
    target: str = "nvidia_sm120", activation: str = "none",
    bias: bool = False, residual: bool = False,
) -> GraphIRModule:
    m, n, k = bounds
    module = _module(
        target=target, shape=(m, k, n), dtype=dtype,
        activation=activation, bias=bias, residual=residual,
    )
    fn = module.functions[0]
    element = "f16" if dtype == "fp16" else "bf16"
    fn.args[0].ir_type = IRType(f"tensor<?x?x{element}>", ("?", "?"), dtype)
    fn.args[1].ir_type = IRType(f"tensor<?x?x{element}>", ("?", "?"), dtype)
    fn.result_types[0] = IRType("tensor<?x?xf32>", ("?", "?"), "fp32")
    op = fn.body[0]
    op.operand_types[:2] = [str(fn.args[0].ir_type), str(fn.args[1].ir_type)]
    op.result_type = str(fn.result_types[0])
    op.inferred_type = fn.result_types[0]
    op.kwargs["shape_bounds"] = [m, n, k]
    return module


@pytest.mark.parametrize(
    ("dynamic_operand", "expected"),
    (("b_k", (False, False, True)), ("out_m", (True, False, False))),
)
def test_linked_secondary_dimension_dynamicity_is_retained(
    dynamic_operand: str, expected: tuple[bool, bool, bool]
) -> None:
    module = _module(target="nvidia_sm120", shape=(32, 32, 24))
    fn = module.functions[0]
    op = fn.body[0]
    if dynamic_operand == "b_k":
        fn.args[1].ir_type = IRType("tensor<?x24xf16>", ("?", "24"), "fp16")
        op.operand_types[1] = str(fn.args[1].ir_type)
    else:
        fn.result_types[0] = IRType("tensor<?x24xf32>", ("?", "24"), "fp32")
        op.result_type = str(fn.result_types[0])
        op.inferred_type = fn.result_types[0]
    op.kwargs["shape_bounds"] = [32, 24, 32]
    assert scheduled_matmul._graph_contract(module, "nvidia_sm120")[-3:] == expected


_ARTIFACT_CONTRACT = {
    "x86": ("x86", "zen5-avx512", "fp32", "f32", 16, 16),
    "rocm": ("rocm", "gfx1151", "fp16", "f16", 32, 64),
    "apple_gpu": ("apple_gpu", "apple7", "fp32", "f32", 16, 16),
    "apple_gpu_f16": ("apple_gpu", "apple7", "fp16", "f16", 32, 32),
    "nvidia_sm120": ("nvidia_sm120", "sm_120", "fp16", "f16", 128, 128),
}


def _artifact(*, target: str) -> ScheduledMatmulArtifact:
    (
        compiler_target,
        architecture,
        a_dtype,
        storage,
        macro_tile_m,
        macro_tile_n,
    ) = _ARTIFACT_CONTRACT[target]
    digest = hashlib.sha256(b"schedule decision").hexdigest()
    tile_ir = f'''module {{
  llvm.func @{target}_scheduled_matmul(%a: !llvm.ptr, %b: !llvm.ptr,
      %o: !llvm.ptr, %m: i64, %n: i64, %k: i64) {{
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {{
      tessera.schedule_hash = "{digest}", storage = "{storage}", accum = "f32",
      tessera.macro_tile_m = {macro_tile_m} : i64,
      tessera.macro_tile_n = {macro_tile_n} : i64
    }} : !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64
    llvm.return
  }}
}}
'''
    return ScheduledMatmulArtifact(
        graph_ir=f"module attributes {{tessera.target = \"{compiler_target}\"}} {{}}",
        schedule_ir=f'''module {{
  %graph = tessera.matmul %a, %b {{schedule.artifact_hash = "{digest}"}}
  %scheduled = schedule.matmul %graph {{artifact_hash = "{digest}"}}
  schedule.artifact {{hash = "{digest}"}}
}}
''',
        tile_ir=tile_ir,
        target=compiler_target,
        architecture=architecture,
        function_name=f"{target}_scheduled_matmul",
        a_name="a",
        b_name="b",
        output_name="o",
        m=17,
        n=23,
        k=19,
        a_dtype=a_dtype,
        b_dtype=a_dtype,
        output_dtype="fp32",
        storage=storage,
        accum="f32",
        macro_tile_m=macro_tile_m,
        macro_tile_n=macro_tile_n,
        schedule_digest=digest,
    )


def test_nvidia_sm120_uses_shared_scheduled_matmul_contract() -> None:
    module = _module(target="nvidia_sm120")
    assert scheduled_matmul.supports_scheduled_matmul(
        module, target="nvidia_sm120"
    )
    artifact = _artifact(target="nvidia_sm120")
    assert artifact.target == "nvidia_sm120"
    assert artifact.architecture == "sm_120"
    artifact.validate()


def test_nvidia_sm120_bf16_uses_shared_scheduled_matmul_contract() -> None:
    module = _module(target="nvidia_sm120", dtype="bf16")
    assert scheduled_matmul.supports_scheduled_matmul(
        module, target="nvidia_sm120"
    )


@requires_tessera_opt
@requires_nvidia_target_ir
def test_nvidia_bounded_dynamic_graph_emits_strided_typed_carrier() -> None:
    artifact = scheduled_matmul.lower_scheduled_matmul(
        _dynamic_module(), target="nvidia_sm120"
    )
    assert (artifact.dynamic_m, artifact.dynamic_n, artifact.dynamic_k) == (
        True, True, True
    )
    assert artifact.function_name.endswith("_kernel")
    assert "_macro_kernel" not in artifact.function_name
    assert artifact.tile_ir.count("leading_dim = 0") == 3
    assert artifact.tile_ir.count("tile.materialize_composed_layout") == 2
    assert "i64, i64, i64, i64, i64, i64)" in artifact.tile_ir


@requires_tessera_opt
def test_rocm_bounded_dynamic_graph_emits_runtime_extent_wmma_carrier() -> None:
    artifact = scheduled_matmul.lower_scheduled_matmul(
        _dynamic_module(target="rocm", bounds=(64, 64, 48)),
        target="rocm_gfx1151",
    )
    assert (artifact.dynamic_m, artifact.dynamic_n, artifact.dynamic_k) == (
        True, True, True
    )
    assert artifact.target == "rocm"
    assert artifact.architecture == "gfx1151"
    assert artifact.tile_ir.count("tile.matmul_kernel") == 1
    assert artifact.tile_ir.count("tensor.dim") >= 3
    assert artifact.tile_ir.count("arith.index_cast") >= 3


@requires_tessera_opt
@requires_nvidia_target_ir
def test_nvidia_large_bounded_dynamic_graph_selects_alignment_safe_macro_cta() -> None:
    artifact = scheduled_matmul.lower_scheduled_matmul(
        _dynamic_module(bounds=(512, 512, 256)), target="nvidia_sm120"
    )
    assert artifact.function_name.endswith("_macro_kernel")
    assert artifact.tile_ir.count("tessera_nvidia.macro_cta_matmul") == 1
    assert 'staging = "masked_scalar_shared_ab_16bit"' in artifact.tile_ir
    assert 'completion = "cta_barrier"' in artifact.tile_ir
    assert 'stages = 1 : i64' in artifact.tile_ir


@pytest.mark.skipif(
    not nvidia_cuda_host_ready()
    or not nvidia_native.tools_available()
    or scheduled_matmul.find_tessera_opt() is None,
    reason="requires the SM120 CUDA compiler, PTX bridge, and RTX host",
)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_sm120_bounded_dynamic_strided_matmul_exact_device(dtype) -> None:
    storage_type = np.float16 if dtype == "fp16" else pytest.importorskip("ml_dtypes").bfloat16
    bundle = compile_graph_module(
        _dynamic_module(dtype=dtype),
        source_origin="sm120-scheduled-dynamic-strided-exact-device",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    assert bundle.schedule is not None
    assert bundle.schedule.input_digest == bundle.graph.output_digest
    assert bundle.tile.input_digest == bundle.schedule.output_digest
    assert bundle.target_ir.input_digest == bundle.tile.output_digest
    assert bundle.tile.producer == "tessera-opt.tessera-schedule-to-tile"
    assert [scalar.name for scalar in bundle.launch_descriptor.scalars] == [
        "M", "N", "K", "LDA", "LDB", "LDD"
    ]
    m, n, k = 17, 13, 19
    lda, ldb, ldd = 29, 31, 23
    rng = np.random.default_rng(17_013_019)
    a_storage = np.zeros((m, lda), dtype=storage_type)
    a = a_storage[:, :k]
    a[...] = rng.standard_normal((m, k)).astype(storage_type)
    b_storage = np.zeros((ldb, n), dtype=storage_type, order="F")
    b = b_storage[:k, :]
    b[...] = rng.standard_normal((k, n)).astype(storage_type)
    d_storage = np.full((m, ldd), -123.0, dtype=np.float32)
    output = d_storage[:, :n]
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    result = rt.launch(runtime_artifact, {
        "a": a, "b": b, "o": output, "M": m, "N": n, "K": k,
        "LDA": lda, "LDB": ldb, "LDD": ldd,
    })
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(
        output, a.astype(np.float32) @ b.astype(np.float32),
        rtol=2e-4, atol=2e-4,
    )
    np.testing.assert_array_equal(d_storage[:, n:], -123.0)


@pytest.mark.skipif(
    not nvidia_cuda_host_ready()
    or not nvidia_native.tools_available()
    or scheduled_matmul.find_tessera_opt() is None,
    reason="requires the SM120 CUDA compiler, PTX bridge, and RTX host",
)
def test_sm120_dynamic_fused_strided_matmul_exact_device() -> None:
    bundle = compile_graph_module(
        _dynamic_module(activation="relu", bias=True, residual=True),
        source_origin="sm120-scheduled-dynamic-fused-strided-exact-device",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    m, n, k = 17, 13, 19
    lda, ldb, ldd = 29, 31, 23
    rng = np.random.default_rng(117_113_119)
    a_storage = np.zeros((m, lda), dtype=np.float16)
    a = a_storage[:, :k]
    a[...] = rng.standard_normal((m, k)).astype(np.float16)
    b_storage = np.zeros((ldb, n), dtype=np.float16, order="F")
    b = b_storage[:k, :]
    b[...] = rng.standard_normal((k, n)).astype(np.float16)
    bias = rng.standard_normal(n).astype(np.float32)
    residual_storage = np.zeros((m, ldd), dtype=np.float32)
    residual = residual_storage[:, :n]
    residual[...] = rng.standard_normal((m, n)).astype(np.float32)
    d_storage = np.full((m, ldd), -321.0, dtype=np.float32)
    output = d_storage[:, :n]
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text if bundle.tile else None,
        target_ir=bundle.target_ir.text if bundle.target_ir else None,
    )
    result = rt.launch(runtime_artifact, {
        "a": a, "b": b, "bias": bias, "residual": residual, "o": output,
        "M": m, "N": n, "K": k, "LDA": lda, "LDB": ldb, "LDD": ldd,
    })
    assert result["ok"] is True, result.get("reason")
    expected = np.maximum(a.astype(np.float32) @ b.astype(np.float32) + bias, 0)
    expected += residual
    np.testing.assert_allclose(output, expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_array_equal(d_storage[:, n:], -321.0)


@pytest.mark.skipif(
    not nvidia_cuda_host_ready()
    or not nvidia_native.tools_available()
    or scheduled_matmul.find_tessera_opt() is None,
    reason="requires the SM120 CUDA compiler, PTX bridge, and RTX host",
)
def test_sm120_dynamic_macro_cta_strided_matmul_exact_device() -> None:
    bundle = compile_graph_module(
        _dynamic_module(bounds=(512, 512, 128)),
        source_origin="sm120-scheduled-dynamic-macro-strided-exact-device",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None
    assert "tessera_nvidia.macro_cta_matmul" in bundle.tile.text
    assert bundle.launch_descriptor.geometry.policy == "sm120_scheduled_macro_cta_32x32_mn"
    assert bundle.launch_descriptor.provenance["physical_route"].startswith(
        "macro_cta_masked_scalar_shared_ab_"
    )
    m, n, k = 257, 259, 127
    lda, ldb, ldd = 139, 137, 269
    rng = np.random.default_rng(257_259_127)
    a_storage = np.zeros((m, lda), dtype=np.float16)
    a = a_storage[:, :k]
    a[...] = rng.standard_normal((m, k)).astype(np.float16)
    b_storage = np.zeros((ldb, n), dtype=np.float16, order="F")
    b = b_storage[:k, :]
    b[...] = rng.standard_normal((k, n)).astype(np.float16)
    d_storage = np.zeros((m, ldd), dtype=np.float32)
    output = d_storage[:, :n]
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text if bundle.target_ir else None,
    )
    result = rt.launch(runtime_artifact, {
        "a": a, "b": b, "o": output, "M": m, "N": n, "K": k,
        "LDA": lda, "LDB": ldb, "LDD": ldd,
    })
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(
        output, a.astype(np.float32) @ b.astype(np.float32),
        rtol=2e-4, atol=2e-4,
    )


@requires_tessera_opt
@requires_nvidia_target_ir
def test_nvidia_sm120_scheduled_epilogue_and_reduced_output_are_retained() -> None:
    module = _module(
        target="nvidia_sm120", shape=(256, 256, 512), output_dtype="fp16",
        activation="gelu", bias=True, residual=True,
    )
    artifact = scheduled_matmul.lower_scheduled_matmul(
        module, target="nvidia_sm120"
    )
    assert artifact.bias_name == "bias"
    assert artifact.residual_name == "residual"
    assert artifact.activation == "gelu"
    assert '_fused_f16_gelu_b1_r1_outf16_macro_kernel' in artifact.function_name
    assert 'bias = true' in artifact.tile_ir
    assert 'activation = "gelu"' in artifact.tile_ir
    assert 'residual = true' in artifact.tile_ir
    assert 'output = "f16"' in artifact.tile_ir


def test_public_matmul_explicit_epilogue_matches_scheduled_order() -> None:
    """The public reference and scheduled CUDA contract share one order."""
    import tessera

    a = np.array([[1.0, -2.0]], dtype=np.float32)
    b = np.array([[2.0, -1.0], [3.0, 4.0]], dtype=np.float32)
    bias = np.array([0.5, -0.5], dtype=np.float32)
    residual = np.array([[7.0, 11.0]], dtype=np.float32)
    actual = tessera.ops.matmul(
        a, b, bias=bias, activation="relu", residual=residual
    )
    expected = np.maximum(a @ b + bias, 0.0) + residual
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((16, 32, 32), False),
        ((32, 32, 16), False),
        ((32, 32, 32), False),
        ((64, 128, 96), False),
        ((48, 64, 24), False),
        ((256, 128, 256), False),
        ((256, 256, 128), False),
        ((256, 256, 256), False),
        ((256, 512, 256), True),
        ((512, 256, 512), True),
        ((257, 512, 257), True),
    ],
)
def test_sm120_macro_cta_admission_is_exact(shape, expected) -> None:
    m, k, n = shape
    assert scheduled_matmul._uses_sm120_macro_cta(
        m, n, k, "f16", "f32"
    ) is expected
    assert scheduled_matmul._uses_sm120_macro_cta(
        m, n, k, "bf16", "f32"
    ) is expected


def test_scheduled_artifact_rejects_graph_reentry() -> None:
    artifact = _artifact(target="x86")
    artifact.validate()
    with pytest.raises(ValueError, match="must not retain Graph or Schedule"):
        ScheduledMatmulArtifact(
            **{**artifact.__dict__, "tile_ir": artifact.tile_ir + "\n schedule.yield"}
        ).validate()


def test_x86_packages_the_exact_scheduled_tile_artifact(monkeypatch) -> None:
    artifact = _artifact(target="x86")

    def fake_lower(tile_ir: str, symbol: str, family: str):
        assert tile_ir == artifact.tile_ir
        assert symbol == "tessera_x86_avx512_gemm_f32"
        return f"module {{ call @{symbol} }}", b"x86-image", "compiler", "toolchain"

    monkeypatch.setattr(x86_native, "_lower", fake_lower)
    package = x86_native.package_scheduled_matmul(
        artifact,
        pipeline_name="tessera-lower-to-x86",
    )

    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest
    assert package.descriptor.provenance["tile_ir_digest"] == artifact.tile_digest
    assert package.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"


def test_rocm_packages_the_exact_scheduled_tile_artifact(monkeypatch) -> None:
    artifact = _artifact(target="rocm")

    def fake_compile(tile_ir: str):
        assert tile_ir == artifact.tile_ir
        return (
            'module { "tessera_rocm.wmma"() : () -> () }',
            "module { gpu.binary @gfx1151 }",
            b"hsaco-image",
            "compiler",
            "toolchain",
            (),
            "cold",
        )

    monkeypatch.setattr(rocm_native, "_compile_scheduled_matmul_tile_ir", fake_compile)
    package = rocm_native.package_scheduled_matmul(
        artifact,
        pipeline_name="tessera-lower-to-rocm",
    )

    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest
    assert package.descriptor.provenance["tile_ir_digest"] == artifact.tile_digest
    assert package.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert package.descriptor.provenance["physical_route"] == (
        "gfx1151_multiwave_lds_wmma_2x4"
    )


def test_apple_gpu_packages_the_exact_scheduled_tile_artifact(monkeypatch, tmp_path) -> None:
    from tessera.compiler import apple_native

    artifact = _artifact(target="apple_gpu")
    fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    fake_dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)

    package = apple_native.package_scheduled_matmul(
        artifact,
        pipeline_name="tessera-lower-to-apple_gpu",
    )

    # The package consumes the shared launch tile text verbatim (no Graph re-entry).
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.entry_symbol == "tessera_apple_gpu_bmm_f32"
    assert package.descriptor.provenance["route"] == "apple_gpu_bmm_f32_batch1"
    assert package.descriptor.provenance["batch"] == 1
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest
    assert package.descriptor.provenance["tile_ir_digest"] == artifact.tile_digest
    # E2E §0.2 point 5 — the delegated route names the decision it does not carry.
    assert package.descriptor.provenance["dropped_reason"] == "delegated_to_mps_bmm"
    assert package.descriptor.provenance["dropped_macro_tile"] == [16, 16]


def test_apple_gpu_packages_scheduled_simdgroup_f16(monkeypatch, tmp_path) -> None:
    from tessera.compiler import apple_native

    artifact = _artifact(target="apple_gpu_f16")
    fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    fake_dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)

    package = apple_native.package_scheduled_matmul(
        artifact, pipeline_name="tessera-lower-to-apple_gpu",
    )
    # f16 routes to the compiler-emitted simdgroup MSL GEMM, not the MPS BMM.
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.entry_symbol == "tessera_apple_gpu_tile_simdgroup_gemm_f16"
    assert package.descriptor.provenance["route"] == "apple_gpu_simdgroup_gemm_f16"
    assert package.descriptor.provenance["block"] == [32, 32, 16]
    # Compiler-emitted MSL has a device timer, so it is not DEVICE-EVENT-1 gated.
    assert package.descriptor.provenance["device_time_promotion"] == "eligible"
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest
    assert package.descriptor.provenance["tile_ir_digest"] == artifact.tile_digest


def test_apple_gpu_scheduled_matmul_rejects_non_apple_contract(monkeypatch, tmp_path) -> None:
    from tessera.compiler import apple_native

    fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    fake_dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)
    # An x86 f32 artifact is structurally valid but not the apple7 contract.
    with pytest.raises(ValueError, match="apple7 contract"):
        apple_native.package_scheduled_matmul(
            _artifact(target="x86"),
            pipeline_name="tessera-lower-to-apple_gpu",
        )


def test_rocm_native_packaging_uses_typed_family_pipeline(monkeypatch) -> None:
    pipelines: list[str] = []
    monkeypatch.setattr(rocm_native, "_tessera_opt", lambda: Path("/fake/tessera-opt"))
    monkeypatch.setattr(rocm_native, "_driver_selected_device_libraries", lambda: ())
    monkeypatch.setattr(rocm_native, "_extract_hsaco", lambda text: b"hsaco")
    monkeypatch.setattr(rocm_native, "_version_fingerprint", lambda tool: "fingerprint")
    monkeypatch.setattr(rocm_native, "_rocm_clang", lambda path: None)

    def fake_run(tool: Path, source: str, pipeline: str) -> str:
        pipelines.append(pipeline)
        if "output=target" in pipeline:
            return (
                'module attributes {tessera.pipeline.target_ir_consumer = '
                '"tessera_rocm"} { "tessera_rocm.test"() : () -> () }'
            )
        return "module { gpu.binary @gfx1151 }"

    monkeypatch.setattr(rocm_native, "_run_opt", fake_run)
    rocm_native._compile_native_tile_ir(
        "legacy-tile",
        directive="tessera_rocm.test",
        family="softmax",
    )
    softmax_target, softmax_native = pipelines
    assert "tessera-rocm-executable{" in softmax_target
    assert "family=softmax" in softmax_target
    assert "output=target" in softmax_target
    assert "output=binary" in softmax_native

    pipelines.clear()
    rocm_native._compile_native_tile_ir(
        "scheduled-matmul-tile",
        directive="tessera_rocm.test",
        family="matmul",
    )
    for pipeline in pipelines:
        assert "family=matmul" in pipeline
        assert "input=tile" in pipeline
        assert "generate-wmma-gemm-kernel" not in pipeline
        assert "lower-tile-to-rocm" not in pipeline


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151", "apple_gpu"])
def test_driver_records_adjacent_scheduled_matmul_lineage(
    monkeypatch, tmp_path, target: str
) -> None:
    artifact_target = {
        "x86": "x86",
        "rocm_gfx1151": "rocm",
        "apple_gpu": "apple_gpu",
    }[target]
    module_target = {
        "x86": "x86",
        "rocm_gfx1151": "rocm",
        "apple_gpu": "apple_gpu",
    }[target]
    artifact = _artifact(target=artifact_target)
    module = _module(target=module_target)
    monkeypatch.setattr(
        scheduled_matmul,
        "lower_scheduled_matmul",
        lambda module, *, target: artifact,
    )
    # This lineage proof is host-free: the lowering is stubbed above, so pin the
    # Apple scheduled-boundary availability too. Otherwise the assertion would
    # silently depend on whether the runner happens to have a built tessera-opt.
    from tessera.compiler import driver as _driver

    monkeypatch.setattr(_driver, "_apple_scheduled_boundary_available", lambda: True)
    if target == "x86":
        monkeypatch.setattr(
            x86_native,
            "_lower",
            lambda tile_ir, symbol, family: (
                f"module {{ call @{symbol} }}",
                b"x86-image",
                "compiler",
                "toolchain",
            ),
        )
    elif target == "rocm_gfx1151":
        monkeypatch.setattr(
            rocm_native,
            "_compile_scheduled_matmul_tile_ir",
            lambda tile_ir: (
                'module { "tessera_rocm.wmma"() : () -> () }',
                "module { gpu.binary @gfx1151 }",
                b"hsaco-image",
                "compiler",
                "toolchain",
                (),
                "cold",
            ),
        )
    else:
        from tessera.compiler import apple_native

        fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
        fake_dylib.write_bytes(b"apple-runtime-image")
        monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)

    bundle = compile_graph_module(
        module,
        source_origin="test",
        target=target,
        options={"package_native": True},
        enable_tool_validation=False,
    )

    assert bundle.request.graph_ir == artifact.graph_ir
    assert bundle.schedule is not None and bundle.schedule.text == artifact.schedule_ir
    assert bundle.tile is not None and bundle.tile.text == artifact.tile_ir
    assert bundle.lineage_complete
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-3"


@pytest.mark.skipif(
    not x86_native.tools_available() or scheduled_matmul.find_tessera_opt() is None,
    reason="x86 compiler/image unavailable",
)
@pytest.mark.parametrize("shape", [(1, 1, 1), (5, 17, 9), (16, 31, 19)])
def test_x86_scheduled_matmul_executes_exact_artifact(shape) -> None:
    m, k, n = shape
    bundle = compile_graph_module(
        _module(target="x86", shape=shape),
        source_origin="e2e-real-3-exact-device",
        target="x86",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(3103)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32)
    b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float32)
    output = np.zeros((m, n), dtype=np.float32)

    result = rt.launch(
        artifact,
        {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k},
    )

    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(output, a @ b, rtol=3e-5, atol=3e-5)


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("shape", [(32, 32, 32), (17, 19, 23)])
@pytest.mark.skipif(
    not rocm_native.native_packaging_available(),
    reason="ROCm compiler/device libraries unavailable",
)
def test_gfx1151_scheduled_matmul_executes_exact_artifact(shape) -> None:
    m, k, n = shape
    bundle = compile_graph_module(
        _module(target="rocm", shape=shape),
        source_origin="e2e-real-3-exact-device",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    artifact = rt.RuntimeArtifact(
        metadata={"target": "rocm_gfx1151"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(3104)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float16)
    b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float16)
    output = np.zeros((m, n), dtype=np.float32)

    result = rt.launch(
        artifact,
        {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k},
    )

    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(output, a.astype(np.float32) @ b.astype(np.float32), rtol=2e-2, atol=2e-2)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    not rocm_native.native_packaging_available(),
    reason="ROCm compiler/device libraries unavailable",
)
def test_gfx1151_bounded_dynamic_scheduled_matmul_executes_exact_artifact() -> None:
    bundle = compile_graph_module(
        _dynamic_module(target="rocm", bounds=(64, 64, 48)),
        source_origin="layout-alg-1-gfx1151-bounded-dynamic-exact-device",
        target="rocm_gfx1151",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    assert bundle.launch_descriptor.provenance["shape_policy"] == "bounded_dynamic"
    assert all(guard.predicate == "max" for guard in bundle.launch_descriptor.shape_guards)
    artifact = rt.RuntimeArtifact(
        metadata={"target": "rocm_gfx1151"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    m, n, k = 37, 29, 35
    rng = np.random.default_rng(37_029_035)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float16)
    b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float16)
    output = np.zeros((m, n), dtype=np.float32)
    result = rt.launch(
        artifact,
        {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k},
    )
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(
        output, a.astype(np.float32) @ b.astype(np.float32),
        rtol=2e-2, atol=2e-2,
    )


@pytest.mark.parametrize(
    "shape",
    [
        (16, 16, 8),
        (16, 32, 8),
        (16, 32, 32),
        (32, 32, 16),
        (48, 64, 24),
    ],
)
@pytest.mark.skipif(
    not nvidia_cuda_host_ready()
    or not nvidia_native.tools_available()
    or scheduled_matmul.find_tessera_opt() is None,
    reason="requires the SM120 CUDA compiler, PTX bridge, and RTX host",
)
def _sm120_typed_scheduled_matmul_executes_exact_artifact(
    shape: tuple[int, int, int],
) -> None:
    """Prove the canonical composed-layout -> typed-MMA package on RTX 5070."""
    m, k, n = shape
    bundle = compile_graph_module(
        _module(target="nvidia_sm120", shape=shape),
        source_origin="sm120-scheduled-typed-mma-exact-device",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    tile = bundle.tile.text
    assert tile.count("tile.materialize_composed_layout") == 2
    assert tile.count("tile.view") == 2
    assert tile.count("tile.fragment_pack") == 2
    assert "tile.matmul_kernel" not in tile
    assert "tile.mma" in tile
    assert "tessera_nvidia.block_coordinate" in tile
    artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=tile,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(3106)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32).astype(np.float16)
    # The SM120 B-fragment contract is column-major, which is part of the
    # descriptor and therefore deliberately exercised rather than copied away.
    b = np.asfortranarray(rng.standard_normal((k, n)), dtype=np.float32).astype(np.float16)
    output = np.zeros((m, n), dtype=np.float32)
    result = rt.launch(artifact, {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k})
    assert result["ok"] is True, result.get("reason")
    assert result.get("execution_kind") == "native_gpu", result
    np.testing.assert_allclose(output, a.astype(np.float32) @ b.astype(np.float32), rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize(
    "shape", [(256, 512, 256), (512, 256, 512), (257, 512, 257)]
)
@pytest.mark.skipif(
    not nvidia_cuda_host_ready()
    or not nvidia_native.tools_available()
    or scheduled_matmul.find_tessera_opt() is None,
    reason="requires the SM120 CUDA compiler, PTX bridge, and RTX host",
)
def _sm120_macro_cta_reuses_shared_panels_exact_device(
    shape: tuple[int, int, int],
) -> None:
    m, k, n = shape
    bundle = compile_graph_module(
        _module(target="nvidia_sm120", shape=shape),
        source_origin="sm120-scheduled-macro-cta-exact-device",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    assert "tessera_nvidia.macro_cta_matmul" in bundle.tile.text
    assert "tile.mma" not in bundle.tile.text
    assert "__tessera_sm120_ab_stage_f16" in bundle.target_ir.text
    assert bundle.target_ir.text.count("nvvm.barrier") >= 2
    assert "nvvm.mma.sync" in bundle.target_ir.text
    assert bundle.launch_descriptor.geometry.policy == "sm120_scheduled_macro_cta_32x32_mn"
    assert bundle.launch_descriptor.provenance["physical_route"] == (
        "macro_cta_cp_async_2stage_shared_ab_f16"
    )

    artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(32_032 + m + k + n)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32).astype(np.float16)
    b = np.asfortranarray(rng.standard_normal((k, n)), dtype=np.float32).astype(np.float16)
    reference = a.astype(np.float32) @ b.astype(np.float32)
    output = np.zeros((m, n), dtype=np.float32)
    result = rt.launch(
        artifact, {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k}
    )
    assert result["ok"] is True, result.get("reason")
    assert result.get("execution_kind") == "native_gpu", result
    max_abs_error = float(np.max(np.abs(output - reference)))
    tolerance = 2e-4 + 2e-4 * float(np.max(np.abs(reference)))
    assert max_abs_error <= tolerance


@pytest.mark.parametrize(
    "shape,expected_route,expects_macro_target",
    [
        ((257, 513, 257), "macro_cta_masked_scalar_shared_ab_f16", False),
        ((257, 520, 257), "macro_cta_cp_async_2stage_shared_ab_f16", True),
    ],
)
@pytest.mark.skipif(
    not nvidia_cuda_host_ready()
    or not nvidia_native.tools_available()
    or scheduled_matmul.find_tessera_opt() is None,
    reason="requires the SM120 CUDA compiler, PTX bridge, and RTX host",
)
def _sm120_macro_cta_k_tail_exact_device(
    shape: tuple[int, int, int],
    expected_route: str,
    expects_macro_target: bool,
) -> None:
    """Prove both alignment-safe K-tail routes against an independent oracle."""
    m, k, n = shape
    bundle = compile_graph_module(
        _module(target="nvidia_sm120", shape=shape),
        source_origin="sm120-scheduled-k-tail-exact-device",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    assert ("tessera_nvidia.macro_cta_matmul" in bundle.tile.text) is expects_macro_target
    assert bundle.launch_descriptor.provenance["physical_route"] == expected_route

    artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(32_034 + k)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32).astype(np.float16)
    b = np.asfortranarray(rng.standard_normal((k, n)), dtype=np.float32).astype(np.float16)
    output = np.zeros((m, n), dtype=np.float32)
    result = rt.launch(
        artifact, {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k}
    )
    assert result["ok"] is True, result.get("reason")
    reference = a.astype(np.float32) @ b.astype(np.float32)
    tolerance = 2e-4 + 2e-4 * float(np.max(np.abs(reference)))
    assert float(np.max(np.abs(output - reference))) <= tolerance


@pytest.mark.skipif(
    not nvidia_cuda_host_ready()
    or not nvidia_native.tools_available()
    or scheduled_matmul.find_tessera_opt() is None,
    reason="requires the SM120 CUDA compiler, PTX bridge, and RTX host",
)
def _sm120_scheduled_epilogue_reduced_output_exact_device() -> None:
    """Exercise the widened A/B/bias/residual/D ABI and f16 store rounding."""
    m, k, n = (256, 256, 512)
    bundle = compile_graph_module(
        _module(
            target="nvidia_sm120",
            shape=(m, k, n),
            output_dtype="fp16",
            activation="relu",
            bias=True,
            residual=True,
        ),
        source_origin="sm120-scheduled-epilogue-exact-device",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    assert bundle.launch_descriptor.abi_id.endswith(".f16.out_f16.v2")
    assert [binding.name for binding in bundle.launch_descriptor.buffers] == [
        "a", "b", "bias", "residual", "o"
    ]

    artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(32_035)
    a = np.ascontiguousarray(rng.standard_normal((m, k)) * 0.1, dtype=np.float32).astype(np.float16)
    b = np.asfortranarray(rng.standard_normal((k, n)) * 0.1, dtype=np.float32).astype(np.float16)
    bias = np.ascontiguousarray(rng.standard_normal(n) * 0.1, dtype=np.float32)
    residual = np.ascontiguousarray(rng.standard_normal((m, n)) * 0.1, dtype=np.float32)
    output = np.zeros((m, n), dtype=np.float16)
    result = rt.launch(
        artifact,
        {
            "a": a, "b": b, "bias": bias, "residual": residual, "o": output,
            "M": m, "N": n, "K": k,
        },
    )
    assert result["ok"] is True, result.get("reason")
    reference = np.maximum(
        a.astype(np.float32) @ b.astype(np.float32) + bias[None, :], 0.0
    ) + residual
    # Tensor-core reduction order is not NumPy's BLAS reduction order.  The
    # two paths may therefore straddle an f16 rounding boundary even though
    # both accumulate in f32; use the same scale-aware f32-accumulator bound
    # as the non-fused package, plus the final f16 store rounding.
    expected = reference.astype(np.float16)
    max_abs_error = float(
        np.max(np.abs(output.astype(np.float32) - expected.astype(np.float32)))
    )
    tolerance = 5e-4 + 2e-4 * float(np.max(np.abs(reference)))
    assert max_abs_error <= tolerance


@pytest.mark.skipif(
    not nvidia_cuda_host_ready()
    or not nvidia_native.tools_available()
    or scheduled_matmul.find_tessera_opt() is None,
    reason="requires the SM120 CUDA compiler, PTX bridge, and RTX host",
)
def _sm120_macro_cta_bf16_exact_device() -> None:
    ml_dtypes = pytest.importorskip("ml_dtypes")
    m, k, n = (257, 512, 257)
    bundle = compile_graph_module(
        _module(target="nvidia_sm120", shape=(m, k, n), dtype="bf16"),
        source_origin="sm120-scheduled-macro-cta-bf16-exact-device",
        target="nvidia_sm120",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    assert 'storage = "bf16"' in bundle.tile.text
    assert "__tessera_sm120_ab_stage_bf16" in bundle.target_ir.text
    assert bundle.launch_descriptor.provenance["physical_route"] == (
        "macro_cta_cp_async_2stage_shared_ab_bf16"
    )
    artifact = rt.RuntimeArtifact(
        metadata={"target": "nvidia_sm120"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(32_033)
    a = np.ascontiguousarray(
        (rng.standard_normal((m, k)) * 0.25).astype(ml_dtypes.bfloat16)
    )
    b = np.asfortranarray(
        (rng.standard_normal((k, n)) * 0.25).astype(ml_dtypes.bfloat16)
    )
    reference = a.astype(np.float32) @ b.astype(np.float32)
    output = np.zeros((m, n), dtype=np.float32)
    result = rt.launch(
        artifact, {"a": a, "b": b, "o": output, "M": m, "N": n, "K": k}
    )
    assert result["ok"] is True, result.get("reason")
    assert result.get("execution_kind") == "native_gpu", result
    max_abs_error = float(np.max(np.abs(output - reference)))
    tolerance = 2e-2 + 2e-2 * float(np.max(np.abs(reference)))
    assert max_abs_error <= tolerance


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("shape", [(16, 16, 16), (17, 19, 23)])
@pytest.mark.skipif(
    not apple_native.tools_available() or scheduled_matmul.find_tessera_opt() is None,
    reason="Apple GPU runtime dylib / tessera-opt unavailable",
)
def test_apple_gpu_scheduled_matmul_executes_exact_artifact(shape) -> None:
    m, k, n = shape
    bundle = compile_graph_module(
        _module(target="apple_gpu", shape=shape),
        source_origin="e2e-real-3-exact-device",
        target="apple_gpu",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    # The compiler boundary consumed the shared launch tile, not a re-classified
    # Graph module: the delegated route and its dropped decision are recorded.
    assert bundle.launch_descriptor.provenance["route"] == "apple_gpu_bmm_f32_batch1"
    assert bundle.launch_descriptor.provenance["dropped_reason"] == "delegated_to_mps_bmm"
    artifact = rt.RuntimeArtifact(
        metadata={"target": "apple_gpu"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(3105)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32)
    b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float32)
    output = np.zeros((m, n), dtype=np.float32)

    result = rt.launch(artifact, {"a": a, "b": b, "o": output})

    assert result["ok"] is True, result.get("reason")
    # Placement must be positively proven native, never a CPU fallback pass.
    assert result.get("execution_kind") == "native_gpu", result
    np.testing.assert_allclose(output, a @ b, rtol=3e-5, atol=3e-5)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("shape", [(16, 16, 16), (48, 32, 80)])
@pytest.mark.skipif(
    not apple_native.tools_available() or scheduled_matmul.find_tessera_opt() is None,
    reason="Apple GPU runtime dylib / tessera-opt unavailable",
)
def test_apple_gpu_scheduled_simdgroup_f16_executes_exact_artifact(shape) -> None:
    m, k, n = shape
    bundle = compile_graph_module(
        _module(target="apple_gpu_f16", shape=shape),
        source_origin="e2e-real-3-exact-device",
        target="apple_gpu",
        options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    # The compiler-emitted simdgroup route, not the delegated MPS BMM.
    assert bundle.launch_descriptor.provenance["route"] == "apple_gpu_simdgroup_gemm_f16"
    assert bundle.launch_descriptor.provenance["device_time_promotion"] == "eligible"
    artifact = rt.RuntimeArtifact(
        metadata={"target": "apple_gpu"},
        native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(3106)
    a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float16)
    b = np.ascontiguousarray(rng.standard_normal((k, n)), dtype=np.float16)
    output = np.zeros((m, n), dtype=np.float32)

    result = rt.launch(artifact, {"a": a, "b": b, "o": output})

    assert result["ok"] is True, result.get("reason")
    assert result.get("execution_kind") == "native_gpu", result
    np.testing.assert_allclose(
        output, a.astype(np.float32) @ b.astype(np.float32), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_nvidia_scheduled_package_compiles_once_without_graph(monkeypatch, dtype, dynamic):
    from dataclasses import replace

    artifact = _artifact(target="nvidia_sm120")
    artifact = replace(
        artifact, a_dtype=dtype, b_dtype=dtype,
        storage="f16" if dtype == "fp16" else "bf16", dynamic_m=dynamic,
        tile_ir=artifact.tile_ir.replace('storage = "f16"', f'storage = "{"f16" if dtype == "fp16" else "bf16"}"'),
    )
    fields = (
        f'storage = "{artifact.storage}", accum = "f32", output = "f32", '
        'activation = "none", arch = "sm_120", a_layout = "row_major", '
        'b_layout = "col_major", bias = false, residual = false, '
    )
    artifact = replace(
        artifact,
        schedule_ir=artifact.schedule_ir.replace(
            'schedule.matmul %graph {', 'schedule.matmul %graph {' + fields
        ).replace('schedule.artifact {',
                  f'schedule.artifact {{shape_key = "M=17;N=23;K=19;dtype={artifact.storage}", '),
        tile_ir=artifact.tile_ir.replace('%k: i64)',
            '%k: i64, %lda: i64, %ldb: i64, %ldd: i64)' if dynamic else '%k: i64)'),
    )
    calls = []

    def compile_tile(text, entry):
        calls.append((text, entry))
        return (text, "// PTX", {}, "compiler", "toolchain", (), "cold")

    def forbidden(*args, **kwargs):
        raise AssertionError("scheduled package re-entered Graph packaging")

    monkeypatch.setattr(nvidia_native, "_compile_tile_ir", compile_tile)
    monkeypatch.setattr(nvidia_native, "package_matmul", forbidden)
    package = nvidia_native.package_scheduled_matmul(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert calls == [(artifact.tile_ir, artifact.function_name)]
    expected_abi = (
        nvidia_native.SM120_STRIDED_F16_ABI if dtype == "fp16" else nvidia_native.SM120_STRIDED_BF16_ABI
    ) if dynamic else (
        nvidia_native.SM120_F16_ABI if dtype == "fp16" else nvidia_native.SM120_BF16_ABI
    )
    assert package.descriptor.abi_id == expected_abi
    assert [b.layout for b in package.descriptor.buffers] == (
        ["strided"] * 3 if dynamic else ["row_major", "col_major", "row_major"]
    )
    assert {g.predicate for g in package.descriptor.shape_guards} == ({"max"} if dynamic else {"eq"})
    # Graph text is provenance only once the scheduled artifact exists.
    detached = replace(artifact, graph_ir="discarded frontend")
    assert nvidia_native.package_scheduled_matmul(detached, pipeline_name="tessera-nvidia-pipeline-sm120") == package
    changed = replace(artifact, tile_ir=artifact.tile_ir + "\n// changed compiler input")
    assert nvidia_native.package_scheduled_matmul(changed, pipeline_name="tessera-nvidia-pipeline-sm120").image.image_digest != package.image.image_digest

    for changed in (replace(artifact, m=99), replace(artifact, output_dtype="fp16"),
                    replace(artifact, dynamic_m=not dynamic),
                    replace(artifact, function_name="missing_entry")):
        before = len(calls)
        with pytest.raises(ValueError, match="disagrees"):
            nvidia_native.package_scheduled_matmul(
                changed, pipeline_name="tessera-nvidia-pipeline-sm120"
            )
        assert len(calls) == before
