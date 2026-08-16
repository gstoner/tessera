from __future__ import annotations

import dataclasses
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
    compiler_target, architecture, workgroup = {
        "x86": ("x86", "zen5-avx512", 1),
        "rocm": ("rocm", "gfx1151", 256),
        "apple_gpu": ("apple_gpu", "apple7", 1),
    }[target]
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
        # Apple's synthesized reduce is last-axis only (inner == 1); gfx1151
        # owns the arbitrary-axis kernel, so its fixture keeps an interior axis.
        apple = target == "apple_gpu"
        reduce_axis = 2 if apple else 1
        tile_op = f'''tile.reduce_kernel %x, %o, %outer, %extent, %inner {{
      storage = "f32", accum = "f32", kind = "mean", axis = {reduce_axis} : i64,
      keepdims = false, schedule = "serial", nan_mode = "propagate",
      inner_is_one = {"true" if apple else "false"}, tessera.schedule_hash = "{digest}",
      tessera.workgroup_size = {workgroup} : i64
    }} : !llvm.ptr, !llvm.ptr, i64, i64, i64'''
        input_shape = (2, 3, 5)
        output_shape = (2, 3) if apple else (2, 5)
        kind, rows, columns = "mean", 1, 1
        axis = reduce_axis
        outer, extent, inner = (6, 5, 1) if apple else (2, 3, 5)
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
        lambda tile_ir, symbol, family: (
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


def test_apple_gpu_packages_exact_scheduled_softmax(monkeypatch, tmp_path) -> None:
    from tessera.compiler import apple_native

    artifact = _artifact(family="softmax", target="apple_gpu")
    fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    fake_dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)

    package = apple_native.package_scheduled_kernel(
        artifact, pipeline_name="tessera-lower-to-apple_gpu"
    )
    assert package.tile_ir == artifact.tile_ir
    assert package.descriptor.entry_symbol == "tessera_apple_gpu_softmax_f32"
    assert package.descriptor.provenance["work_item"] == "E2E-REAL-5"
    assert package.descriptor.provenance["route"] == "apple_softmax_native_library"
    assert package.descriptor.provenance["rows"] == 6
    assert package.descriptor.provenance["columns"] == 5
    assert package.descriptor.provenance["schedule_digest"] == artifact.schedule_digest
    assert package.descriptor.provenance["tile_ir_digest"] == artifact.tile_digest


def _reduce_module(shape, out_shape, *, axis=-1, keepdims=False, kind="mean"):
    x = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>",
               tuple(map(str, shape)), "fp32")
    o = IRType("tensor<" + "x".join(map(str, out_shape)) + "xf32>",
               tuple(map(str, out_shape)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_reduce", args=[IRArg("x", x)], result_types=[o],
        body=[IROp(result="o", op_name="tessera.reduce", operands=["%x"],
                   operand_types=[str(x)], result_type=str(o),
                   kwargs={"kind": kind, "axis": axis, "keepdims": keepdims})],
        return_values=["%o"])])


def test_apple_gpu_scheduled_reduce_envelope_fails_closed() -> None:
    """Apple's synthesized reduce folds the trailing extent, last axis only.

    An interior axis would have to be reordered into something the program did
    not ask for, so it fails closed — while gfx1151, which owns an arbitrary-axis
    kernel, must keep accepting it (the Apple bound must not leak to siblings).
    """
    supports = scheduled_kernel.supports_scheduled_kernel
    assert supports(_reduce_module((2, 3, 5), (2, 3)), target="apple_gpu")
    assert not supports(_reduce_module((2, 3, 5), (2, 5), axis=1), target="apple_gpu")
    assert not supports(_reduce_module((2, 3, 5), (3, 5), axis=0), target="apple_gpu")
    assert not supports(
        _reduce_module((2, 3, 5), (2, 3, 1), keepdims=True), target="apple_gpu"
    )
    # The sibling's arbitrary-axis support is untouched.
    assert supports(_reduce_module((2, 3, 5), (2, 5), axis=1), target="rocm_gfx1151")


def test_apple_gpu_scheduled_reduce_is_compiler_synthesized(monkeypatch, tmp_path) -> None:
    """The reduce family is emitted by the compiler, not delegated to a vendor.

    This is the distinguishing property of the family: matmul f32 delegates to
    MPS and softmax binds a hand-written kernel, but the reduce kernel is MSL the
    Decision #28 tier-1 synthesizer produced, carried in the Target IR and bound
    by a recorded source digest.
    """
    from tessera.compiler import apple_native

    fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    fake_dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)
    artifact = _artifact(family="reduce", target="apple_gpu")
    package = apple_native.package_scheduled_kernel(
        artifact, pipeline_name="tessera-lower-to-apple_gpu"
    )
    provenance = package.descriptor.provenance
    assert provenance["synthesized"] is True
    assert provenance["route"] == "apple_gpu_synth_pointwise_reduce_f32"
    assert provenance["work_item"] == "E2E-REAL-5"
    # The emitted kernel is the Target IR for this route, not a call stub.
    assert "kernel void synth_pw_reduce" in package.target_ir
    assert (
        hashlib.sha256(
            apple_native.synthesized_reduce_source(artifact.kind).encode()
        ).hexdigest()
        == provenance["source_sha256"]
    )
    # Consumed the shared launch tile verbatim.
    assert package.tile_ir == artifact.tile_ir


def test_apple_gpu_scheduled_reduce_rejects_unmappable_contracts(
    monkeypatch, tmp_path
) -> None:
    from tessera.compiler import apple_native

    fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    fake_dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)
    # A gfx1151 artifact is structurally valid but not the apple7 contract.
    with pytest.raises(ValueError, match="apple7 contract"):
        apple_native.package_scheduled_kernel(
            _artifact(family="reduce", target="rocm"),
            pipeline_name="tessera-lower-to-apple_gpu",
        )
    # An interior-axis artifact (inner != 1) has no synthesized kernel.
    interior = dataclasses.replace(
        _artifact(family="reduce", target="apple_gpu"), inner=5, axis=1
    )
    with pytest.raises(ValueError, match="last-axis contract"):
        apple_native.package_scheduled_kernel(
            interior, pipeline_name="tessera-lower-to-apple_gpu"
        )
    # A reduce kind with no synthesizer mapping must not be invented.
    with pytest.raises(ValueError, match="no kernel for"):
        apple_native.synthesized_reduce_source("prod")


@pytest.mark.parametrize(
    "target,family",
    [("x86", "softmax"), ("rocm_gfx1151", "reduce"), ("apple_gpu", "softmax")],
)
def test_driver_records_adjacent_semantic_kernel_lineage(monkeypatch, tmp_path, target, family) -> None:
    artifact_target = {"x86": "x86", "rocm_gfx1151": "rocm", "apple_gpu": "apple_gpu"}[target]
    artifact = _artifact(family=family, target=artifact_target)
    module = _module(family=family, target=artifact_target)
    monkeypatch.setattr(
        scheduled_kernel, "lower_scheduled_kernel", lambda module, *, target: artifact
    )
    # Host-free lineage proof: the lowering is stubbed, so pin the Apple
    # scheduled-boundary availability rather than depending on whether the
    # runner happens to have a built tessera-opt.
    from tessera.compiler import driver as _driver

    monkeypatch.setattr(_driver, "_apple_scheduled_boundary_available", lambda: True)
    if target == "x86":
        monkeypatch.setattr(
            x86_native, "_lower",
            lambda tile_ir, symbol, family: (f"module {{ call @{symbol} }}", b"x86", "compiler", "toolchain"),
        )
    elif target == "rocm_gfx1151":
        monkeypatch.setattr(
            rocm_native, "_compile_reduction_tile_ir",
            lambda tile_ir: ("target", "backend", b"hsaco", "compiler", "toolchain", (), "cold"),
        )
    else:
        from tessera.compiler import apple_native

        fake_dylib = tmp_path / "libTesseraAppleRuntime.dylib"
        fake_dylib.write_bytes(b"apple-runtime-image")
        monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: fake_dylib)
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


def test_apple_scheduled_boundary_falls_back_without_tessera_opt(monkeypatch, tmp_path) -> None:
    """A runtime-only Apple install must not hard-fail.

    The shared scheduled artifact is produced by running production
    ``tessera-opt``. When that tool is absent the driver must leave Apple on its
    independently proven descriptor route rather than entering a lane it cannot
    produce — the regression this locks is a `RuntimeError: scheduled
    softmax/reduction lowering requires production tessera-opt` raised from an
    ordinary rank-2 f32 softmax compile.
    """
    from tessera.compiler import apple_native
    from tessera.compiler import driver as _driver

    dylib = tmp_path / "libTesseraAppleRuntime.dylib"
    dylib.write_bytes(b"apple-runtime-image")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: dylib)
    monkeypatch.setattr(_driver, "_apple_scheduled_boundary_available", lambda: False)

    bundle = compile_graph_module(
        _softmax_module((3, 17)), source_origin="no-tessera-opt",
        target="apple_gpu", options={"package_native": True},
        enable_tool_validation=False,
    )
    event = next(e for e in bundle.trace_events
                 if e.pass_name == "apple-gpu-native-package")
    assert event.metadata["op_family"] == "softmax"
    assert event.metadata["work_item"] == "APPLE-E2E-1"


def _softmax_module(shape: tuple[int, ...]) -> GraphIRModule:
    x = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>",
               tuple(map(str, shape)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_softmax", args=[IRArg("x", x)], result_types=[x],
        body=[IROp(result="o", op_name="tessera.softmax", operands=["%x"],
                   operand_types=[str(x)], result_type=str(x),
                   kwargs={"axis": -1})],
        return_values=["%o"],
    )])


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(
    find_tessera_opt() is None,
    reason="Apple GPU runtime dylib / tessera-opt unavailable",
)
@pytest.mark.parametrize("kind,op_name", [("sum", "tessera.sum"),
                                          ("mean", "tessera.mean"),
                                          ("max", "tessera.max")])
@pytest.mark.parametrize("shape", [(64, 128), (8, 16, 32), (128,)])
def test_apple_gpu_scheduled_reduce_executes_synthesized_kernel(shape, kind, op_name) -> None:
    """The synthesized reduce must execute on Metal and match NumPy.

    Launching with the module's own logical shape, and asserting `native_gpu`,
    matters here for a specific reason: the underlying helper silently falls
    back to a NumPy reference on any dispatch failure, so a correct answer alone
    would not distinguish GPU execution from the CPU path.
    """
    from tessera.compiler import apple_native

    if not apple_native.tools_available():
        pytest.skip("Apple GPU runtime dylib unavailable")
    out_shape = shape[:-1]
    x_type = IRType("tensor<" + "x".join(map(str, shape)) + "xf32>",
                    tuple(map(str, shape)), "fp32")
    # A rank-1 reduction produces a *scalar*: `tensor<f32>`, not `tensor<xf32>`.
    # That path exercises the rank-0 output binding, which is why it is covered.
    o_type = IRType(
        "tensor<" + "x".join(map(str, out_shape)) + "xf32>" if out_shape else "tensor<f32>",
        tuple(map(str, out_shape)), "fp32",
    )
    module = GraphIRModule(functions=[GraphIRFunction(
        name="apple_reduce", args=[IRArg("x", x_type)], result_types=[o_type],
        body=[IROp(result="o", op_name=op_name, operands=["%x"],
                   operand_types=[str(x_type)], result_type=str(o_type),
                   kwargs={"axis": -1, "keepdims": False})],
        return_values=["%o"])])
    bundle = compile_graph_module(
        module, source_origin="e2e-real-5-exact-device", target="apple_gpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    provenance = bundle.launch_descriptor.provenance
    assert provenance["route"] == "apple_gpu_synth_pointwise_reduce_f32"
    assert provenance["synthesized"] is True
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "apple_gpu"}, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor,
        tile_ir=bundle.tile.text if bundle.tile else "",
        target_ir=bundle.target_ir.text if bundle.target_ir else "",
    )
    rng = np.random.default_rng(6021)
    x = np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
    output = np.zeros(out_shape, dtype=np.float32)
    result = rt.launch(runtime_artifact, {"x": x, "o": output})
    assert result["ok"] is True, result.get("reason")
    assert result.get("execution_kind") == "native_gpu", result
    expected = {"sum": x.sum(-1), "mean": x.mean(-1), "max": x.max(-1)}[kind]
    np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.hardware_apple_gpu
@pytest.mark.skipif(
    find_tessera_opt() is None,
    reason="Apple GPU runtime dylib / tessera-opt unavailable",
)
@pytest.mark.parametrize("shape", [(6, 5), (2, 3, 5), (2, 3, 4, 7)])
def test_apple_gpu_scheduled_softmax_executes_exact_artifact(shape) -> None:
    """Launch with the module's own LOGICAL shape, not a pre-flattened one.

    The descriptor binds the logical rank, so a rank-N graph must launch with
    its declared shape; the flatten to (rows, columns) happens at submission and
    is exact because softmax rows are independent.  Passing an already-flattened
    buffer here would hide a rank mismatch in the binding contract.
    """
    from tessera.compiler import apple_native

    if not apple_native.tools_available():
        pytest.skip("Apple GPU runtime dylib unavailable")
    bundle = compile_graph_module(
        _softmax_module(shape), source_origin="e2e-real-5-exact-device",
        target="apple_gpu", options={"package_native": True},
        enable_tool_validation=False,
    )
    assert bundle.native_image is not None and bundle.launch_descriptor is not None
    assert bundle.tile is not None and bundle.target_ir is not None
    assert bundle.launch_descriptor.provenance["route"] == "apple_softmax_native_library"
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-5"
    # The binding must carry the logical rank, not a flattened rank-2.
    assert [item.rank for item in bundle.launch_descriptor.buffers] == [len(shape)] * 2
    runtime_artifact = rt.RuntimeArtifact(
        metadata={"target": "apple_gpu"}, native_image=bundle.native_image,
        launch_descriptor=bundle.launch_descriptor, tile_ir=bundle.tile.text,
        target_ir=bundle.target_ir.text,
    )
    rng = np.random.default_rng(5115)
    x = np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)
    output = np.zeros_like(x)
    result = rt.launch(runtime_artifact, {"x": x, "o": output})
    assert result["ok"] is True, result.get("reason")
    assert result.get("execution_kind") == "native_gpu", result
    shifted = x - np.max(x, axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    np.testing.assert_allclose(output, expected, rtol=3e-5, atol=3e-5)
