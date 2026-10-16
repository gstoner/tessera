"""Typed native reverse packaging and compiler-free physical role validation."""
from dataclasses import replace
import copy

import pytest

pytestmark = pytest.mark.compiler_route

from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.typed_attention_backward import checkpoint_pair_shapes, validate_typed_backward_descriptor
from tests.unit.test_saved_lse_half_backward import source


@pytest.fixture
def compiler_package(monkeypatch):
    if find_tessera_opt() is None:
        pytest.skip("production compiler required")
    monkeypatch.setattr(native, "_compile_tile_ir",
                        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    def create(storage="f16", bias=True, saved=True, wide=False):
        return native.package_generated_attention_checkpoint_pair(
            source(storage, bias=bias, saved=saved, output="f32" if wide else storage),
            pipeline_name="tessera-nvidia-pipeline-sm120", prune_inactive=True, compact_gradients=True)
    return create


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("wide", [False, True])
def test_typed_native_pair_preserves_roles_and_hashes_without_replay_tools(compiler_package, storage, bias, saved, wide, monkeypatch):
    pair = compiler_package(storage, bias, saved, wide)
    assert validate_typed_backward_descriptor(pair.backward.descriptor) == (storage, "f32" if wide else storage)
    def forbidden(*args, **kwargs):
        pytest.fail("typed portable validation invoked a compiler")
    monkeypatch.setattr("subprocess.run", forbidden)
    dims, shapes = checkpoint_pair_shapes(pair)
    assert dims == (1, 2, 1, 3, 5, 4, 3) and shapes[-1] == (1, 2, 3)
    desc = pair.backward.descriptor
    inputs = [x for x in desc.buffers if x.direction == "input"]
    outputs = [x for x in desc.buffers if x.direction == "output"]
    assert inputs[0].dtype == ("fp32" if wide else "fp16" if storage == "f16" else "bf16")
    assert inputs[-1].dtype == "fp32"
    if bias:
        assert outputs[-1].dtype == "fp32"


@pytest.mark.parametrize("field,value", [
    ("storage", "f32"), ("output", "bf16"), ("gradient_storage", "f32"),
    ("row_correction", "rounded_output"), ("accum", "f16"),
    ("lse_cotangent", False), ("bias_gradient", False), ("causal", False),
    ("physical_gradient_roles", [0, 1, 2]), ("gradient_launch", "logical_v1"),
    ("schedule_digest", "0"*64), ("checkpoint_contract", "0"*64),
])
def test_typed_backward_rejects_policy_mutation(compiler_package, field, value):
    pair = compiler_package()
    desc = replace(pair.backward.descriptor, provenance={**pair.backward.descriptor.provenance, field: value})
    with pytest.raises(ValueError):
        validate_typed_backward_descriptor(desc)


@pytest.mark.parametrize("ordinal", range(12))
def test_typed_backward_rejects_per_role_wrong_byte_width(compiler_package, ordinal):
    pair = compiler_package()
    desc = pair.backward.descriptor
    assert len(desc.buffers) == 12
    buffers = list(desc.buffers)
    binding = buffers[ordinal]
    buffers[ordinal] = replace(binding, dtype="fp32" if binding.dtype != "fp32" else "fp16")
    with pytest.raises(ValueError, match="storage"):
        validate_typed_backward_descriptor(replace(desc, buffers=tuple(buffers)))


def test_typed_pair_checks_retained_native_dictionary_after_repinning_tile(compiler_package):
    import hashlib
    pair = compiler_package()
    changed = pair.backward.tile_ir.replace('row_correction = "probability_dot_v_f32_v1"',
                                           'row_correction = "rounded_output"')
    desc = replace(pair.backward.descriptor, provenance={**pair.backward.descriptor.provenance,
                   "tile_ir_digest": hashlib.sha256(changed.encode()).hexdigest()})
    broken = replace(pair, backward=replace(pair.backward, tile_ir=changed, descriptor=desc))
    with pytest.raises(ValueError, match="native contract"):
        checkpoint_pair_shapes(broken)


def test_typed_pair_checks_compact_native_activity_after_repinning_descriptor(compiler_package):
    pair = compiler_package()
    policy = copy.deepcopy(pair.backward.descriptor.provenance)
    policy["gradient_activity"] = [1, 0, 1, 1]
    policy["physical_gradient_roles"] = [0, 2, 3]
    with pytest.raises(ValueError):
        checkpoint_pair_shapes(replace(pair, backward=replace(pair.backward,
            descriptor=replace(pair.backward.descriptor, provenance=policy))))


@pytest.mark.parametrize("storage", ["f16", "bf16"])
@pytest.mark.parametrize("saved", [False, True])
def test_typed_vjp_json_has_distinct_schema_and_needs_no_compiler(compiler_package, storage, saved, monkeypatch):
    import json
    from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
    pair = compiler_package(storage=storage, saved=saved)
    mapping = tuple(pair.forward.descriptor.provenance["frontend_argument_indices"])
    program = NativeAttentionVJPProgram(pair, (0, 1, 2, 3), mapping)
    text = program.to_json()
    assert json.loads(text)["program"]["schema"] == "tessera.native_attention_vjp_program.v2"
    def forbidden(*args, **kwargs):
        pytest.fail("typed VJP JSON replay called a compiler")
    monkeypatch.setattr("subprocess.run", forbidden)
    replay = NativeAttentionVJPProgram.from_json(text, expected_digest=program.program_digest)
    assert replay.to_json() == text


def test_typed_vjp_cannot_be_repinned_as_legacy_schema(compiler_package):
    import json
    import hashlib
    from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
    from tessera.compiler.native_attention_vjp_artifact import canonical
    pair = compiler_package()
    program = NativeAttentionVJPProgram(pair, (0, 1, 2, 3), (0, 1, 2, 3))
    raw = json.loads(program.to_json())
    raw["program"]["schema"] = "tessera.native_attention_vjp_program.v1"
    raw["program_digest"] = hashlib.sha256(canonical(raw["program"]).encode()).hexdigest()
    with pytest.raises(ValueError, match="schema"):
        NativeAttentionVJPProgram.from_json(json.dumps(raw), expected_digest=raw["program_digest"])


def test_saved_fp32_vjp_keeps_legacy_schema(compiler_package):
    import json
    from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
    pair = compiler_package(storage="f32", saved=True)
    program = NativeAttentionVJPProgram(pair, (0, 1, 2, 3), (0, 1, 2, 3))
    text = program.to_json()
    assert json.loads(text)["program"]["schema"] == "tessera.native_attention_vjp_program.v1"
    assert NativeAttentionVJPProgram.from_json(text, expected_digest=program.program_digest).to_json() == text


@pytest.mark.parametrize("ordinal", range(6))
def test_typed_public_binding_checks_each_root_and_seed_dtype_before_native_prepare(compiler_package, ordinal, monkeypatch):
    import numpy as np
    from tessera.compiler.native_attention_program import NativeAttentionVJPProgram
    from tessera.compiler.prepared_attention_vjp import PreparedAttentionVJP
    pair = compiler_package()
    program = NativeAttentionVJPProgram(pair, (0, 1, 2, 3), (0, 1, 2, 3))
    metadata = dict(program_json=program.to_json(), program_digest=program.program_digest,
                    arg_names=["primal_0", "primal_1", "primal_2", "primal_3", "cotangent", "lse_cotangent"])
    owner = PreparedAttentionVJP(metadata)
    values = [np.zeros(shape, dtype) for shape, dtype in zip(owner.shapes, owner.dtypes, strict=True)]
    values[ordinal] = values[ordinal].astype(np.float16 if values[ordinal].dtype == np.float32 else np.float32)
    def forbidden(*args, **kwargs):
        pytest.fail("native prepare called for mismatched host storage")
    monkeypatch.setattr(owner, "_prepare", forbidden)
    with pytest.raises(ValueError, match="storage"):
        owner.invoke(metadata, values)
    assert not owner.handle
