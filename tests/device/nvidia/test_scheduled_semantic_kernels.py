"""Compare scheduled native kernels with retained Graph packages on SM120."""
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import nvidia_native
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.graph_ir import tensor_ir_type
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support.nvidia import nvidia_cuda_host_ready
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
    legacy = nvidia_native.package_native(module, pipeline_name="tessera-nvidia-pipeline-sm120")
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
        (bundle.native_image, bundle.launch_descriptor, bundle.tile.text, bundle.target_ir.text),
    ):
        output = np.empty_like(expected)
        result = rt.launch(rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=image,
                                              launch_descriptor=descriptor, tile_ir=tile, target_ir=target), {"x": x, "o": output, **scalars})
        assert result["ok"], result
        np.testing.assert_allclose(output, expected, rtol=2e-5, atol=2e-6)
        results.append(output)
    np.testing.assert_array_equal(results[0], results[1])
    assert legacy.descriptor.abi_id == bundle.launch_descriptor.abi_id
