from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_native, scheduled_matmul, x86_native
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import ScheduledMatmulArtifact


def _module(
    *,
    target: str,
    shape: tuple[int, int, int] = (17, 19, 23),
) -> GraphIRModule:
    m, k, n = shape
    dtype, element = ("fp32", "f32") if target == "x86" else ("fp16", "f16")
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


def _artifact(*, target: str) -> ScheduledMatmulArtifact:
    compiler_target = "x86" if target == "x86" else "rocm"
    architecture = "zen5-avx512" if target == "x86" else "gfx1151"
    a_dtype = "fp32" if target == "x86" else "fp16"
    storage = "f32" if target == "x86" else "f16"
    digest = hashlib.sha256(b"schedule decision").hexdigest()
    tile_ir = f'''module {{
  llvm.func @{target}_scheduled_matmul(%a: !llvm.ptr, %b: !llvm.ptr,
      %o: !llvm.ptr, %m: i64, %n: i64, %k: i64) {{
    tile.matmul_kernel %a, %b, %o, %m, %n, %k {{
      tessera.schedule_hash = "{digest}", storage = "{storage}", accum = "f32",
      tessera.macro_tile_m = {16 if target == "x86" else 32} : i64,
      tessera.macro_tile_n = {16 if target == "x86" else 64} : i64
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
        macro_tile_m=16 if target == "x86" else 32,
        macro_tile_n=16 if target == "x86" else 64,
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

    def fake_lower(tile_ir: str, symbol: str):
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


def test_rocm_generator_order_changes_only_for_scheduled_matmul(monkeypatch) -> None:
    pipelines: list[str] = []
    monkeypatch.setattr(rocm_native, "_tessera_opt", lambda: Path("/fake/tessera-opt"))
    monkeypatch.setattr(rocm_native, "_driver_selected_device_libraries", lambda: ())
    monkeypatch.setattr(rocm_native, "_extract_hsaco", lambda text: b"hsaco")
    monkeypatch.setattr(rocm_native, "_version_fingerprint", lambda tool: "fingerprint")
    monkeypatch.setattr(rocm_native, "_rocm_clang", lambda path: None)

    def fake_run(tool: Path, source: str, pipeline: str) -> str:
        pipelines.append(pipeline)
        return 'module { "tessera_rocm.test"() : () -> () }'

    monkeypatch.setattr(rocm_native, "_run_opt", fake_run)
    rocm_native._compile_native_tile_ir(
        "legacy-tile",
        directive="tessera_rocm.test",
        generator="legacy-generator",
    )
    legacy_target, legacy_native = pipelines
    assert "legacy-generator" not in legacy_target
    assert legacy_native.index("lower-tile-to-rocm") < legacy_native.index("legacy-generator")

    pipelines.clear()
    rocm_native._compile_native_tile_ir(
        "scheduled-matmul-tile",
        directive="tessera_rocm.test",
        generator="generate-wmma-gemm-kernel",
        generator_before_lowering=True,
    )
    for pipeline in pipelines:
        assert pipeline.index("generate-wmma-gemm-kernel") < pipeline.index(
            "lower-tile-to-rocm"
        )


@pytest.mark.parametrize("target", ["x86", "rocm_gfx1151"])
def test_driver_records_adjacent_scheduled_matmul_lineage(monkeypatch, target: str) -> None:
    artifact_target = "x86" if target == "x86" else "rocm"
    artifact = _artifact(target=artifact_target)
    module = _module(target="x86" if target == "x86" else "rocm")
    monkeypatch.setattr(
        scheduled_matmul,
        "lower_scheduled_matmul",
        lambda module, *, target: artifact,
    )
    if target == "x86":
        monkeypatch.setattr(
            x86_native,
            "_lower",
            lambda tile_ir, symbol: (
                f"module {{ call @{symbol} }}",
                b"x86-image",
                "compiler",
                "toolchain",
            ),
        )
    else:
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


@pytest.mark.skipif(not x86_native.tools_available(), reason="x86 compiler/image unavailable")
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
