"""Compare scheduled native kernels with retained Graph packages on SM120."""
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import nvidia_native
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import tensor_ir_type
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support.nvidia import nvidia_cuda_host_ready
from tests._support.nvidia_unary_baseline import baseline_reduction, baseline_softmax
from tests.unit.test_scheduled_kernel_consumers import _module

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.skipif(
    not nvidia_cuda_host_ready() or not nvidia_native.tools_available() or find_tessera_opt() is None,
    reason="requires SM120 GPU, compiler and native scheduling tool",
)]


@pytest.mark.parametrize("kind", ["softmax", "sum", "mean", "max"])
@pytest.mark.parametrize("shape", [(2, 3, 5), (7, 19, 257), (2, 3, 1)])
def test_scheduled_unary_matches_legacy_and_oracle(kind, shape):
    module = _module(family="softmax" if kind == "softmax" else "reduce", target="nvidia_sm120")
    if kind != "softmax":
        module.functions[0].body[0].op_name = "tessera.reduce"
        module.functions[0].body[0].kwargs["kind"] = kind
    fn = module.functions[0]
    fn.args[0].ir_type = tensor_ir_type(shape, "fp32")
    fn.result_types[0] = tensor_ir_type(shape if kind == "softmax" else (shape[0], shape[2]), "fp32")
    fn.body[0].operand_types = [str(fn.args[0].ir_type)]
    fn.body[0].result_type = str(fn.result_types[0])
    fn.body[0].inferred_type = fn.result_types[0]
    baseline = baseline_softmax if kind == "softmax" else baseline_reduction
    legacy = baseline(module, pipeline_name="tessera-nvidia-pipeline-sm120")
    direct = nvidia_native.package_native(module, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert direct.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    bundle = compile_graph_module(module, source_origin="F2-native-unary", target="nvidia_sm120",
                                  options={"package_native": True}, enable_tool_validation=False)
    assert bundle.schedule is not None and bundle.tile is not None
    assert bundle.tile.input_digest == bundle.schedule.output_digest
    x = np.random.default_rng(725).normal(size=shape).astype(np.float32)
    if kind == "softmax":
        exp = np.exp(x - x.max(axis=-1, keepdims=True))
        expected = exp / exp.sum(axis=-1, keepdims=True)
        scalars = {"Rows": shape[0] * shape[1], "K": shape[2]}
    else:
        expected = getattr(np, kind)(x, axis=1)
        scalars = {"Outer": shape[0], "AxisExtent": shape[1], "Inner": shape[2]}
    results = []
    for image, descriptor, tile, target in (
        (legacy.image, legacy.descriptor, legacy.tile_ir, legacy.target_ir),
        (direct.image, direct.descriptor, direct.tile_ir, direct.target_ir),
        (bundle.native_image, bundle.launch_descriptor, bundle.tile.text, bundle.target_ir.text),
    ):
        output = np.empty_like(expected)
        result = rt.launch(rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=image,
                                              launch_descriptor=descriptor, tile_ir=tile, target_ir=target), {"x": x, "o": output, **scalars})
        assert result["ok"], result
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
        results.append(output)
    for output in results[1:]:
        np.testing.assert_array_equal(results[0], output)
    assert legacy.descriptor.abi_id == bundle.launch_descriptor.abi_id


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", [(3, 17), (129, 257)])
def test_narrow_softmax_native_parity(dtype, shape):
    storage = np.float16 if dtype == "fp16" else pytest.importorskip("ml_dtypes").bfloat16
    module = _module(family="softmax", target="nvidia_sm120")
    fn = module.functions[0]
    fn.args[0].ir_type = tensor_ir_type(shape, dtype)
    fn.result_types[0] = fn.args[0].ir_type
    fn.body[0].operand_types = [str(fn.args[0].ir_type)]
    fn.body[0].result_type = str(fn.result_types[0])
    fn.body[0].inferred_type = fn.result_types[0]
    legacy = baseline_softmax(module, pipeline_name="tessera-nvidia-pipeline-sm120")
    direct = nvidia_native.package_native(module, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert direct.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    x = np.random.default_rng(726).normal(size=shape).astype(storage)
    ex = np.exp(x.astype(np.float32) - x.astype(np.float32).max(axis=-1, keepdims=True))
    expected = ex / ex.sum(axis=-1, keepdims=True)
    outputs = []
    for package in (legacy, direct):
        output = np.empty_like(x)
        result = rt.launch(rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=package.image,
            launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir),
            {"x": x, "o": output, "Rows": shape[0], "K": shape[1]})
        assert result["ok"], result
        np.testing.assert_allclose(output.astype(np.float32), expected, rtol=0.01, atol=2e-4)
        outputs.append(output.astype(np.float32))
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("kind", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("mode", ["serial", "cooperative_128"])
def test_reduction_breadth_native_parity(dtype, kind, axis, keepdims, mode, shape=(2, 3, 5)):
    storage = {"fp32": np.float32, "fp16": np.float16}.get(dtype)
    if storage is None:
        storage = pytest.importorskip("ml_dtypes").bfloat16
    output_shape = shape[:axis] + ((1,) if keepdims else ()) + shape[axis + 1:]
    module = _module(family="reduce", target="nvidia_sm120")
    fn = module.functions[0]
    fn.args[0].ir_type = tensor_ir_type(shape, dtype)
    fn.result_types[0] = tensor_ir_type(output_shape, "fp32")
    op = fn.body[0]
    op.op_name = "tessera.reduce"
    op.kwargs = {"kind": kind, "axis": axis, "keepdims": keepdims}
    op.operand_types = [str(fn.args[0].ir_type)]
    op.result_type = str(fn.result_types[0])
    op.inferred_type = fn.result_types[0]
    old = baseline_reduction(module, pipeline_name="tessera-nvidia-pipeline-sm120", schedule=mode)
    new = nvidia_native.package_native(module, pipeline_name="tessera-nvidia-pipeline-sm120", options={"nvidia_reduction_schedule": mode})
    assert new.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert new.descriptor.provenance["schedule"] == mode
    x = np.random.default_rng(727).normal(size=shape).astype(storage)
    expected = getattr(np, kind)(x.astype(np.float32), axis=axis, keepdims=keepdims)
    scalars = {"Outer": int(np.prod(shape[:axis])), "AxisExtent": shape[axis], "Inner": int(np.prod(shape[axis+1:]))}
    outputs = []
    for package in (old, new):
        output = np.empty(output_shape, dtype=np.float32)
        result = rt.launch(rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=package.image,
            launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir),
            {"x": x, "o": output, **scalars})
        assert result["ok"], result
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
        outputs.append(output)
    np.testing.assert_array_equal(*outputs)
    assert old.descriptor.abi_id == new.descriptor.abi_id


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("kind", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("shape,axis", [((2, 257), 1), ((2, 257, 3), 1)])
def test_cooperative_reduction_multi_iteration_parity(dtype, kind, shape, axis):
    test_reduction_breadth_native_parity(dtype, kind, axis, True, "cooperative_128", shape)
