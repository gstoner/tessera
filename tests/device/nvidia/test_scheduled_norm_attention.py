"""Owning-device comparisons for native norm and forward attention contracts."""
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import nvidia_native
from tessera.compiler.driver import compile_graph_module
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support.nvidia import nvidia_cuda_host_ready
from tests._support.nvidia_attention_baseline import baseline_attention
from tests._support.nvidia_norm_baseline import baseline_norm
from tests.device.nvidia.test_flash_attention import _reference
from tests.unit.test_nvidia_norm_attention_schedule import attn_module, norm_module

pytestmark = [pytest.mark.hardware_nvidia, pytest.mark.skipif(
    not nvidia_cuda_host_ready() or not nvidia_native.tools_available() or find_tessera_opt() is None,
    reason="requires SM120 GPU and native compilers",
)]
PIPELINE = "tessera-nvidia-pipeline-sm120"


def storage_type(dtype):
    return pytest.importorskip("ml_dtypes").bfloat16 if dtype == "bf16" else getattr(np, dtype.replace("fp", "float"))


def execute(package, bindings):
    result = rt.launch(rt.RuntimeArtifact(metadata={"target": "nvidia_sm120"}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=package.tile_ir, target_ir=package.target_ir), bindings)
    assert result["ok"], result


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("op", ["rmsnorm", "rmsnorm_safe", "layer_norm"])
@pytest.mark.parametrize("shape", [(3, 17), (129, 257)])
def test_scheduled_norm_parity(dtype, op, shape):
    module = norm_module(dtype, op, shape)
    old = baseline_norm(module, pipeline_name=PIPELINE)
    new = nvidia_native.package_native(module, pipeline_name=PIPELINE)
    bundle = compile_graph_module(module, source_origin="native-norm", target="nvidia_sm120",
                                  options={"package_native": True}, enable_tool_validation=False)
    assert bundle.tile.input_digest == bundle.schedule.output_digest
    assert bundle.tile.text == new.tile_ir
    x = np.random.default_rng(728).normal(size=shape).astype(storage_type(dtype))
    values = x.astype(np.float32)
    centered = values - values.mean(-1, keepdims=True) if op == "layer_norm" else values
    expected = centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + 1e-5)
    outputs = []
    for package in (old, new):
        out = np.empty_like(x)
        execute(package, {"x": x, "o": out, "Rows": shape[0], "Columns": shape[1]})
        np.testing.assert_allclose(out.astype(np.float32), expected, rtol=8e-3 if dtype == "bf16" else 1e-3 if dtype == "fp16" else 2e-5, atol=2e-6)
        outputs.append(out)
    np.testing.assert_array_equal(*outputs)
    assert old.descriptor.abi_id == new.descriptor.abi_id


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_scheduled_attention_parity(dtype, bias, causal):
    module = attn_module(dtype, bias)
    module.functions[0].body[0].kwargs["causal"] = causal
    old = baseline_attention(module, pipeline_name=PIPELINE)
    new = nvidia_native.package_native(module, pipeline_name=PIPELINE)
    bundle = compile_graph_module(module, source_origin="native-attention", target="nvidia_sm120",
                                  options={"package_native": True}, enable_tool_validation=False)
    assert bundle.tile.input_digest == bundle.schedule.output_digest
    assert bundle.tile.text == new.tile_ir
    rng = np.random.default_rng(729)
    arrays = {arg.name: rng.normal(size=tuple(map(int, arg.ir_type.shape))).astype(
        np.float32 if arg.name == "bias" else storage_type(dtype)) for arg in module.functions[0].args}
    q, k, v = (arrays[name].astype(np.float32) for name in ("q", "k", "v"))
    expected = _reference(q, k, v, scale=0.5, causal=causal, bias=arrays.get("bias"),
                          window=3 if bias else None, softcap=4.0 if bias else None)
    # Preserve the existing all-masked-row NaNs in both the oracle and kernel.
    b, hq, sq, d = q.shape
    _, hkv, sk, dv = v.shape
    outputs = []
    for package in (old, new):
        out = np.empty_like(expected)
        execute(package, {**arrays, "o": out, "B": b, "Hq": hq, "Hkv": hkv, "Sq": sq, "Sk": sk, "D": d, "Dv": dv})
        np.testing.assert_allclose(out, expected, atol=2e-5, rtol=2e-5)
        outputs.append(out)
    np.testing.assert_array_equal(*outputs)
    assert old.descriptor.abi_id == new.descriptor.abi_id


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("dropout_p", [0.0, 0.25])
def test_scheduled_gqa_dropout_parity(dtype, dropout_p):
    from tessera.compiler.attention_contract import reference_streaming_attention
    from tessera.compiler.graph_ir import tensor_ir_type

    module = attn_module(dtype)
    fn = module.functions[0]
    for arg in fn.args[1:3]:
        shape = tuple(map(int, arg.ir_type.shape))
        arg.ir_type = tensor_ir_type((shape[0], 1, shape[2], shape[3]), dtype)
    fn.body[0].operand_types = [str(arg.ir_type) for arg in fn.args]
    fn.body[0].kwargs.update(dropout_p=dropout_p, dropout_seed=123456789)
    rng = np.random.default_rng(730)
    arrays = {arg.name: rng.normal(size=tuple(map(int, arg.ir_type.shape))).astype(storage_type(dtype)) for arg in fn.args}
    expected = reference_streaming_attention(arrays["q"], arrays["k"], arrays["v"], block_size=4,
        scale=0.5, dropout_p=dropout_p, dropout_seed=123456789)
    outputs = []
    for package in (baseline_attention(module, pipeline_name=PIPELINE), nvidia_native.package_native(module, pipeline_name=PIPELINE)):
        out = np.empty_like(expected)
        execute(package, {**arrays, "o": out, "B": 1, "Hq": 2, "Hkv": 1, "Sq": 17, "Sk": 7, "D": 4, "Dv": 3})
        np.testing.assert_allclose(out, expected, atol=2e-5, rtol=2e-5)
        outputs.append(out)
    np.testing.assert_array_equal(*outputs)
