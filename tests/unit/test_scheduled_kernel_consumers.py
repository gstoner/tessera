from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_native, scheduled_kernel, x86_native
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_kernel import ScheduledKernelArtifact
from tessera.compiler.scheduled_matmul import find_tessera_opt


def _module(*, family: str, target: str) -> GraphIRModule:
    source = IRType("tensor<2x3x5xf32>", ("2", "3", "5"), "fp32")
    if family == "softmax":
        output = source
        op_name = "tessera.softmax"
        kwargs = {"axis": -1}
    else:
        output = (
            IRType("tensor<2x3xf32>", ("2", "3"), "fp32")
            if target == "x86"
            else IRType("tensor<2x5xf32>", ("2", "5"), "fp32")
        )
        op_name = "tessera.mean"
        kwargs = {"axis": -1 if target == "x86" else 1, "keepdims": False}
    return GraphIRModule(functions=[GraphIRFunction(
        name=f"{target}_{family}",
        args=[IRArg("x", source)],
        result_types=[output],
        body=[IROp(
            result="o", op_name=op_name, operands=["%x"],
            operand_types=[str(source)], result_type=str(output), kwargs=kwargs,
        )],
        return_values=["%o"],
    )])


def _artifact(*, family: str, target: str) -> ScheduledKernelArtifact:
    digest = hashlib.sha256(f"{family}-{target}".encode()).hexdigest()
    compiler_target = "x86" if target == "x86" else "rocm"
    architecture = "zen5-avx512" if target == "x86" else "gfx1151"
    workgroup = 1 if target == "x86" else 256
    if family == "softmax":
        tile_op = f'''tile.softmax_kernel %x, %o, %rows, %cols {{
      storage = "f32", accum = "f32", axis = -1 : i64,
      exp_mode = "accurate", ftz = false,
      tessera.schedule_hash = "{digest}",
      tessera.workgroup_size = {workgroup} : i64
    }} : !llvm.ptr, !llvm.ptr, i64, i64'''
        input_shape = output_shape = (2, 3, 5)
        kind, axis, rows, columns, outer, extent, inner = "softmax", -1, 6, 5, 1, 1, 1
    else:
        tile_op = f'''tile.reduce_kernel %x, %o, %outer, %extent, %inner {{
      storage = "f32", accum = "f32", kind = "mean", axis = 1 : i64,
      keepdims = false, schedule = "serial", nan_mode = "propagate",
      inner_is_one = false, tessera.schedule_hash = "{digest}",
      tessera.workgroup_size = {workgroup} : i64
    }} : !llvm.ptr, !llvm.ptr, i64, i64, i64'''
        input_shape, output_shape = (2, 3, 5), (2, 5)
        kind, axis, rows, columns, outer, extent, inner = "mean", 1, 1, 1, 2, 3, 5
    return ScheduledKernelArtifact(
        graph_ir=f'module attributes {{tessera.target = "{compiler_target}"}} {{}}',
        schedule_ir=f'''module {{
  %graph = "tessera.{"softmax" if family == "softmax" else "reduce"}"() {{schedule.artifact_hash = "{digest}"}} : () -> tensor<2x3x5xf32>
  %scheduled = schedule.{family} %graph {{artifact_hash = "{digest}"}} : tensor<2x3x5xf32> -> tensor<2x3x5xf32>
  schedule.artifact {{hash = "{digest}"}}
}}''',
        tile_ir=f'''module {{
  llvm.func @{target}_{family}(%x: !llvm.ptr, %o: !llvm.ptr) {{
    {tile_op}
    llvm.return
  }}
}}''',
        target=compiler_target,
        architecture=architecture,
        function_name=f"{target}_{family}",
        family=family,
        kind=kind,
        input_name="x",
        output_name="o",
        input_shape=input_shape,
        output_shape=output_shape,
        dtype="fp32",
        storage="f32",
        accum="f32",
        axis=axis,
        keepdims=False,
        rows=rows,
        columns=columns,
        outer=outer,
        axis_extent=extent,
        inner=inner,
        workgroup_size=workgroup,
        schedule_digest=digest,
    )


@pytest.mark.skipif(find_tessera_opt() is None, reason="production tessera-opt unavailable")
@pytest.mark.parametrize(
    "family,target,tile_op",
    [("softmax", "x86", "tile.softmax_kernel"), ("reduce", "rocm_gfx1151", "tile.reduce_kernel")],
)
def test_graph_lowers_through_content_addressed_semantic_kernel(family, target, tile_op) -> None:
    artifact = scheduled_kernel.lower_scheduled_kernel(
        _module(family=family, target="x86" if target == "x86" else "rocm"),
        target=target,
    )
    assert tile_op in artifact.tile_ir
    assert "schedule." not in artifact.tile_ir
    assert "tessera.softmax" not in artifact.tile_ir
    assert "tessera.reduce" not in artifact.tile_ir
    if family == "reduce":
        assert "tessera.mean" not in artifact.graph_ir
        assert "tessera.reduce" in artifact.graph_ir


def test_x86_packages_exact_scheduled_softmax(monkeypatch) -> None:
    artifact = _artifact(family="softmax", target="x86")
    monkeypatch.setattr(
        x86_native, "_lower",
        lambda tile_ir, symbol: (
            f"module {{ call @{symbol} }}", b"x86-image", "compiler", "toolchain"
        ) if tile_ir == artifact.tile_ir else pytest.fail("Tile artifact was resynthesized"),
    )
    package = x86_native.package_scheduled_kernel(
        artifact, pipeline_name="tessera-lower-to-x86"
    )
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.provenance["work_item"] == "E2E-REAL-5"
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest


def test_rocm_packages_exact_scheduled_reduction(monkeypatch) -> None:
    artifact = _artifact(family="reduce", target="rocm")

    def fake_compile(tile_ir: str):
        assert tile_ir == artifact.tile_ir
        return "rocm-target", "rocm-backend", b"hsaco", "compiler", "toolchain", (), "cold"

    monkeypatch.setattr(rocm_native, "_compile_reduction_tile_ir", fake_compile)
    package = rocm_native.package_scheduled_kernel(
        artifact, pipeline_name="tessera-lower-to-rocm"
    )
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.provenance["work_item"] == "E2E-REAL-5"
    assert package.descriptor.provenance["tile_ir_digest"] == artifact.tile_digest


@pytest.mark.parametrize("target,family", [("x86", "softmax"), ("rocm_gfx1151", "reduce")])
def test_driver_records_adjacent_semantic_kernel_lineage(monkeypatch, target, family) -> None:
    artifact_target = "x86" if target == "x86" else "rocm"
    artifact = _artifact(family=family, target=artifact_target)
    module = _module(family=family, target=artifact_target)
    monkeypatch.setattr(
        scheduled_kernel, "lower_scheduled_kernel", lambda module, *, target: artifact
    )
    if target == "x86":
        monkeypatch.setattr(
            x86_native, "_lower",
            lambda tile_ir, symbol: (f"module {{ call @{symbol} }}", b"x86", "compiler", "toolchain"),
        )
    else:
        monkeypatch.setattr(
            rocm_native, "_compile_reduction_tile_ir",
            lambda tile_ir: ("target", "backend", b"hsaco", "compiler", "toolchain", (), "cold"),
        )
    bundle = compile_graph_module(
        module, source_origin="test", target=target,
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.request.graph_ir == artifact.graph_ir
    assert bundle.schedule is not None and bundle.schedule.text == artifact.schedule_ir
    assert bundle.tile is not None and bundle.tile.text == artifact.tile_ir
    assert bundle.lineage_complete
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-5"


@pytest.mark.skipif(
    not x86_native.tools_available() or find_tessera_opt() is None,
    reason="x86 compiler/image unavailable",
)
@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_x86_scheduled_semantic_kernel_executes_exact_artifact(family) -> None:
    module = _module(family=family, target="x86")
    bundle = compile_graph_module(
        module, source_origin="e2e-real-5-exact-device", target="x86",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "x86"}, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor, tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(5105)
    x = np.ascontiguousarray(rng.standard_normal((2, 3, 5)), dtype=np.float32)
    if family == "softmax":
        output = np.zeros_like(x)
        args = {"x": x, "o": output, "Rows": 6, "K": 5}
        shifted = x - np.max(x, axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    else:
        output = np.zeros((2, 3), dtype=np.float32)
        args = {"x": x, "o": output, "Outer": 6, "AxisExtent": 5, "Inner": 1}
        expected = np.mean(x, axis=-1)
    result = rt.launch(runtime_artifact, args)
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.hardware_rocm
@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)
@pytest.mark.parametrize("family", ["softmax", "reduce"])
def test_gfx1151_scheduled_semantic_kernel_executes_exact_artifact(family) -> None:
    module = _module(family=family, target="rocm")
    bundle = compile_graph_module(
        module, source_origin="e2e-real-5-exact-device", target="rocm_gfx1151",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "rocm_gfx1151"}, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor, tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(1151)
    x = np.ascontiguousarray(rng.standard_normal((2, 3, 5)), dtype=np.float32)
    if family == "softmax":
        output = np.zeros_like(x)
        args = {"x": x, "o": output, "Rows": 6, "K": 5}
        shifted = x - np.max(x, axis=-1, keepdims=True)
        expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    else:
        output = np.zeros((2, 5), dtype=np.float32)
        args = {"x": x, "o": output, "Outer": 2, "AxisExtent": 3, "Inner": 5}
        expected = np.mean(x, axis=1)
    result = rt.launch(runtime_artifact, args)
    assert result["ok"] is True, result.get("reason")
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)
