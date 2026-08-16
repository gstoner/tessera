from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import apple_native, rocm_native, scheduled_matmul, x86_native
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import ScheduledMatmulArtifact


def _module(
    *,
    target: str,
    shape: tuple[int, int, int] = (17, 19, 23),
) -> GraphIRModule:
    m, k, n = shape
    dtype, element = (
        ("fp16", "f16") if target in ("rocm", "apple_gpu_f16") else ("fp32", "f32")
    )
    a = IRType(f"tensor<{m}x{k}x{element}>", (str(m), str(k)), dtype)
    b = IRType(f"tensor<{k}x{n}x{element}>", (str(k), str(n)), dtype)
    output = IRType(f"tensor<{m}x{n}xf32>", (str(m), str(n)), "fp32")
    return GraphIRModule(
        functions=[
            GraphIRFunction(
                name=f"{target}_scheduled_matmul",
                args=[IRArg("a", a), IRArg("b", b)],
                result_types=[output],
                body=[
                    IROp(
                        result="o",
                        op_name="tessera.matmul",
                        operands=["%a", "%b"],
                        operand_types=[str(a), str(b)],
                        result_type=str(output),
                        kwargs={},
                    )
                ],
                return_values=["%o"],
            )
        ]
    )


_ARTIFACT_CONTRACT = {
    "x86": ("x86", "zen5-avx512", "fp32", "f32", 16, 16),
    "rocm": ("rocm", "gfx1151", "fp16", "f16", 32, 64),
    "apple_gpu": ("apple_gpu", "apple7", "fp32", "f32", 16, 16),
    "apple_gpu_f16": ("apple_gpu", "apple7", "fp16", "f16", 32, 32),
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
