"""Native recompute ancestry, policy and binding certificates."""
import copy
from dataclasses import replace

import pytest

from tessera.compiler.graph_ir import tensor_ir_type
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tessera.compiler.scheduled_attention_recompute import lower_native_attention_recompute
from tests.device.nvidia.test_lse_checkpoint_native import _backward_module

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="requires production scheduling compiler")


def recompute_module(dtype="fp32", bias=False, permuted=False):
    module = _backward_module(saved=False, bias=bias)
    fn = module.functions[0]
    for arg in fn.args[:4]:
        arg.ir_type = tensor_ir_type(tuple(map(int, arg.ir_type.shape)), dtype)
    fn.result_types = [arg.ir_type for arg in fn.args[1:4]]
    op = fn.body[0]
    op.operand_types = [str(arg.ir_type) for arg in fn.args]
    op.result_type = "(" + ", ".join(map(str, fn.result_types)) + ")"
    if permuted:
        fn.args.reverse()
    return module


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_recompute_native_ancestry_and_original_graph(dtype, bias, permuted, monkeypatch):
    from tessera.compiler import nvidia_native as native
    module = recompute_module(dtype, bias, permuted)
    before = copy.deepcopy(module)
    monkeypatch.setattr(native, "emit_attention_backward_tile_ir",
                        lambda **kwargs: pytest.fail("legacy Python Tile constructor"))
    monkeypatch.setattr(native, "_compile_tile_ir",
                        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    package = native.package_attention_backward(module, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert module == before
    assert [buffer.name for buffer in package.descriptor.buffers] == (
        ["do", "q", "k", "v"] + (["bias"] if bias else []) + ["dq", "dk", "dv"])
    assert package.descriptor.provenance["compiler_route"] == "canonical_scheduled_tile_consumer"
    for key in ("graph_ir_digest", "schedule_digest", "schedule_ir_digest", "tile_ir_digest"):
        assert len(package.descriptor.provenance[key]) == 64
    assert "attention_backward_recompute" in package.tile_ir
    storage = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    assert package.descriptor.entry_symbol.startswith(
        f"tessera_tile_attention_backward_{storage}_recompute_")


@pytest.mark.parametrize("policy", [
    {"window": (2, 0), "softcap": .7, "dropout": .2, "seed": 17},
    {"scale": 1, "window": (-1, -1), "dropout_p": 0},
    {"scale": .3, "head_dim": 4},
])
def test_recompute_preserves_native_policy(policy):
    module = recompute_module()
    module.functions[0].body[0].kwargs.update(policy)
    artifact = lower_native_attention_recompute(module)
    assert artifact.dropout_seed == policy.get("seed", 0)
    assert artifact.window_left == policy.get("window", (-1, -1))[0]
    assert abs(artifact.scale - policy.get("scale", .5)) < 1e-6
    assert abs(artifact.softcap - policy.get("softcap", 0)) < 1e-6


@pytest.mark.parametrize("policy", [
    {"window": (2, 0), "window_left": 3},
    {"dropout": .2, "dropout_p": .3},
    {"softcap": .5, "logit_softcap": .7},
    {"seed": 1, "dropout_seed": 2},
    {"causal": 1}, {"deterministic": False}, {"unknown_policy": "approximate"},
    {"scale": 1e100}, {"scale": 1e-100}, {"head_dim": 5},
])
def test_recompute_diagnoses_policy_conflicts(policy):
    module = recompute_module()
    module.functions[0].body[0].kwargs.update(policy)
    with pytest.raises(RuntimeError):
        lower_native_attention_recompute(module)


@pytest.mark.parametrize("field", ["graph_ir", "schedule_ir", "tile_ir", "scale", "dims"])
def test_recompute_certificate_rejects_tampering(field):
    artifact = lower_native_attention_recompute(recompute_module())
    value = getattr(artifact, field)
    if field == "graph_ir":
        value = value.replace("scale = 0.5", "scale = 0.25")
    elif field == "schedule_ir":
        value = value.replace("scale = 5.000000e-01", "scale = 2.500000e-01")
        if value == artifact.schedule_ir:
            value = value.replace("dropout_seed = 0", "dropout_seed = 1")
    elif field == "tile_ir":
        value = value.replace("dropout_seed = 0", "dropout_seed = 1")
    elif field == "scale":
        value = .25
    else:
        value = (2,) + value[1:]
    assert value != getattr(artifact, field)
    with pytest.raises(ValueError):
        replace(artifact, **{field: value}).validate()


def test_native_schedule_rejects_changed_hash_before_tile():
    artifact = lower_native_attention_recompute(recompute_module())
    changed = artifact.schedule_ir.replace(artifact.schedule_digest, "0" * 64)
    with pytest.raises(RuntimeError, match="changed after hashing"):
        run_tessera_opt(find_tessera_opt(), changed, "--tessera-schedule-to-tile")
