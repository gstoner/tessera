"""Native norm/attention boundaries and rejection contracts."""
from dataclasses import replace

import pytest

from tessera.compiler import nvidia_native, scheduled_attention, scheduled_kernel
from tessera.compiler.graph_ir import tensor_ir_type
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests.unit.test_scheduled_attention_consumers import _module as attention_module
from tests.unit.test_scheduled_kernel_consumers import _module as unary_module


def norm_module(dtype="fp32", op="rmsnorm", shape=(3, 17), epsilon=1e-5):
    module = unary_module(family="softmax", target="nvidia_sm120")
    fn = module.functions[0]
    fn.args[0].ir_type = tensor_ir_type(shape, dtype)
    fn.result_types[0] = fn.args[0].ir_type
    fn.body[0].op_name = "tessera." + op
    fn.body[0].kwargs = {"eps": epsilon}
    fn.body[0].operand_types = [str(fn.args[0].ir_type)]
    fn.body[0].result_type = str(fn.result_types[0])
    fn.body[0].inferred_type = fn.result_types[0]
    return module


def attn_module(dtype="fp32", bias=False):
    module = attention_module(target="nvidia_sm120", bias=bias)
    fn = module.functions[0]
    for arg in fn.args[:3]:
        arg.ir_type = tensor_ir_type(tuple(map(int, arg.ir_type.shape)), dtype)
    fn.body[0].operand_types = [str(arg.ir_type) for arg in fn.args]
    return module


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("op", ["rmsnorm", "rmsnorm_safe", "layer_norm"])
def test_norm_native_boundary(dtype, op, monkeypatch):
    module = norm_module(dtype, op)
    artifact = scheduled_kernel.lower_scheduled_kernel(module, target="nvidia_sm120")
    assert artifact.family == "norm"
    assert run_tessera_opt(find_tessera_opt(), artifact.schedule_ir, "--tessera-schedule-to-tile") == artifact.tile_ir
    calls = []
    monkeypatch.setattr(nvidia_native, "_compile_tile_ir", lambda text, entry: (
        calls.append(text) or text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    monkeypatch.setattr(nvidia_native, "emit_norm_tile_ir", lambda **kw: pytest.fail("Graph emitter called"))
    package = nvidia_native.package_scheduled_kernel(replace(artifact, graph_ir="discarded"), pipeline_name="tessera-nvidia-pipeline-sm120")
    assert calls == [artifact.tile_ir]
    assert package.descriptor.buffers[1].dtype == dtype
    with pytest.raises(ValueError, match="epsilon"):
        nvidia_native.package_scheduled_kernel(replace(artifact, epsilon=0.5), pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.parametrize("epsilon", [True, 0.0, -1.0, float("inf"), 1e-100, 1e100])
def test_norm_rejects_unrepresentable_epsilon(epsilon):
    module = norm_module(epsilon=epsilon)
    assert not nvidia_native.supports_norm(module)
    assert not scheduled_kernel.supports_scheduled_kernel(module, target="nvidia_sm120")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("bias", [False, True])
def test_attention_native_boundary(dtype, bias, monkeypatch):
    module = attn_module(dtype, bias)
    artifact = scheduled_attention.lower_scheduled_attention(module, target="nvidia_sm120")
    assert "func.func" not in artifact.tile_ir
    calls = []
    monkeypatch.setattr(nvidia_native, "_compile_tile_ir", lambda text, entry: (
        calls.append(text) or text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    monkeypatch.setattr(nvidia_native, "emit_attention_tile_ir", lambda **kw: pytest.fail("Graph emitter called"))
    package = nvidia_native.package_scheduled_attention(replace(artifact, graph_ir="discarded"), pipeline_name="tessera-nvidia-pipeline-sm120")
    assert calls == [artifact.tile_ir]
    assert package.descriptor.buffers[-1].dtype == "fp32"
    with pytest.raises(ValueError, match="scale"):
        nvidia_native.package_scheduled_attention(replace(artifact, scale=0.25), pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_attention_hash_distinguishes_adjacent_f32_policy_values():
    module = attn_module()
    first = scheduled_attention.lower_scheduled_attention(module, target="nvidia_sm120")
    module.functions[0].body[0].kwargs["scale"] = 0.5 + 2 ** -24
    second = scheduled_attention.lower_scheduled_attention(module, target="nvidia_sm120")
    assert first.schedule_digest != second.schedule_digest


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_norm_refuses_edited_epsilon_operand():
    artifact = scheduled_kernel.lower_scheduled_kernel(norm_module(), target="nvidia_sm120")
    changed = artifact.tile_ir
    import re
    changed = re.sub(r"(arith.constant )[^ ]+( : f32)", r"\g<1>5.000000e-01\g<2>", changed, count=1)
    assert changed != artifact.tile_ir
    with pytest.raises(ValueError, match="native Schedule replay"):
        nvidia_native.package_scheduled_kernel(replace(artifact, tile_ir=changed), pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_attention_refuses_edited_binding_names():
    artifact = scheduled_attention.lower_scheduled_attention(attn_module(), target="nvidia_sm120")
    with pytest.raises(ValueError, match="bindings"):
        nvidia_native.package_scheduled_attention(replace(artifact, q_name="key"), pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.parametrize("kwargs", [{"causal": True}, {"window_left": 3, "window_right": 0}])
def test_attention_accepts_short_query_mask_alignment(kwargs):
    module = attention_module(target="nvidia_sm120", query_rows=3)
    module.functions[0].body[0].kwargs.update(kwargs)
    assert scheduled_attention.supports_scheduled_attention(module, target="nvidia_sm120")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_native_pass_accepts_short_query_mask_alignment():
    module = attention_module(target="nvidia_sm120", query_rows=3)
    artifact = scheduled_attention.lower_scheduled_attention(module, target="nvidia_sm120")
    changed = artifact.graph_ir.replace("causal = false", "causal = true")
    assert changed != artifact.graph_ir
    result = run_tessera_opt(find_tessera_opt(), changed, "--tessera-graph-to-schedule")
    assert "schedule.attention" in result


def test_norm_policy_overrides_are_not_silently_dropped():
    module = norm_module()
    module.functions[0].body[0].kwargs["numeric_policy"] = "custom"
    assert not scheduled_kernel.supports_scheduled_kernel(module, target="nvidia_sm120")
    assert not nvidia_native.supports_norm(module)
