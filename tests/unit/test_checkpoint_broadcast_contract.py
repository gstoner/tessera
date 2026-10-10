"""Broadcast checkpoint shapes and native replay identity."""
from dataclasses import replace
import numpy as np
import pytest
from tessera.compiler import scheduled_checkpoint as checkpoint
from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

pytestmark=pytest.mark.skipif(find_tessera_opt() is None,reason="requires native compiler")
DIMS=(2,4,2,5,7,4,3)


def artifact(physical, backward):
    names=(("do","q","k","v","output","bias","lse","dq","dk","dv","dbias") if backward else
           ("q","k","v","bias","output","lse"))
    return checkpoint.lower_scheduled_checkpoint(names,DIMS,.5,True,bias=True,
        backward=backward,bias_gradient=backward,bias_shape=physical)


@pytest.mark.parametrize("backward",[False,True])
@pytest.mark.parametrize("physical",[(1,4,5,7),(2,1,5,7),(2,4,1,7),(2,4,5,1),(1,1,1,7),(1,1,1,1)])
def test_native_checkpoint_preserves_each_physical_broadcast_axis(physical,backward):
    result=artifact(physical,backward)
    assert result.bias_shape==physical
    text="bias_shape = array<i64: "+", ".join(map(str,physical))+">"
    assert text in result.schedule_ir and text in result.tile_ir
    assert "physical_owner_lexicographic_bhqk_v1" in result.tile_ir
    result.validate()


def test_replay_rejects_bias_shape_mutation():
    result=artifact((1,4,5,7),True)
    changed=result.schedule_ir.replace("bias_shape = array<i64: 1, 4, 5, 7>",
                                       "bias_shape = array<i64: 2, 4, 5, 7>")
    assert changed != result.schedule_ir
    with pytest.raises(RuntimeError,match="contract changed"):
        run_tessera_opt(find_tessera_opt(),changed,"--tessera-schedule-to-tile")
    with pytest.raises(ValueError,match="bias shape|metadata"):
        replace(result,bias_shape=(2,4,5,7)).validate()


def test_native_graph_rejects_nonbroadcast_physical_shape():
    result=artifact((1,4,5,7),True)
    changed=result.graph_ir.replace("tensor<1x4x5x7xf32>","tensor<3x4x5x7xf32>")
    assert changed != result.graph_ir
    with pytest.raises(RuntimeError,match="bias axes"):
        run_tessera_opt(find_tessera_opt(),changed,"--tessera-graph-to-schedule")


def test_broadcast_package_uses_physical_guards_and_separate_abi(monkeypatch):
    physical = (1, 1, 1, 7)
    result = artifact(physical, True)
    monkeypatch.setattr(native, "_compile_tile_ir",
                        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    package = native.package_scheduled_checkpoint(result, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert package.descriptor.abi_id == native.SM120_ATTN_BWD_LSE_BCAST_GRAD_F32_ABI
    assert [s.name for s in package.descriptor.scalars][-4:] == ["BiasB", "BiasH", "BiasQ", "BiasK"]
    for name in ("bias", "dbias"):
        assert [g.value for g in package.descriptor.shape_guards if g.binding == name] == list(physical)


def test_resident_pair_accepts_physical_storage_contract(monkeypatch):
    from tessera.compiler.resident_attention import checkpoint_shapes
    physical = (1, 4, 1, 7)
    forward, backward = artifact(physical, False), artifact(physical, True)
    monkeypatch.setattr(native, "_compile_tile_ir",
                        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    identity = native._checkpoint_identity(DIMS, .5, True, bias=True, bias_shape=physical)
    pair = native.AttentionCheckpointPair(
        native.package_scheduled_checkpoint(forward, pipeline_name="tessera-nvidia-pipeline-sm120"),
        native.package_scheduled_checkpoint(backward, pipeline_name="tessera-nvidia-pipeline-sm120"), identity)
    assert checkpoint_shapes(pair)[0] == DIMS



@pytest.mark.parametrize("backward",[False,True])
def test_public_ad_export_retains_physical_bias_gradient(monkeypatch,backward):
    import tessera.compiler.native_attention_program as program
    from benchmarks.nvidia.benchmark_jit_attention_bias_vjp import function
    monkeypatch.setattr(program,"compile_attention_vjp_program",
                        lambda source,active,**kwargs:source)
    values=[np.ones(shape,np.float32) for shape in
        ((1,2,3,4),(1,1,5,4),(1,1,5,3),(1,2,1,5))]
    source=function(("bias","q"),True).compile_native_attention_vjp(*values,compiler="unused")
    result=checkpoint.lower_generated_checkpoint(source,backward=backward)
    assert result.bias_shape==(1,2,1,5)
    assert result.frontend_argument_indices==(0,1,2,3)
    if backward:
        assert "tensor<1x2x1x5xf32>" in result.graph_ir
        assert result.bias_gradient



def test_pair_identity_binds_physical_bias_storage():
    full = (2, 4, 5, 7)
    legacy = native._checkpoint_identity(DIMS, .5, True, bias=True)
    assert native._checkpoint_identity(DIMS, .5, True, bias=True, bias_shape=full) == legacy
    physical = [(1, 4, 5, 7), (2, 1, 5, 7), (2, 4, 1, 7),
                (2, 4, 5, 1), (1, 1, 1, 7), (1, 1, 1, 1)]
    identities = {native._checkpoint_identity(DIMS, .5, True, bias=True, bias_shape=shape)
                  for shape in physical}
    assert len(identities) == len(physical)
    assert legacy not in identities
    assert native._checkpoint_identity(DIMS, .5, False, bias=True,
                                       bias_shape=physical[0]) not in identities


@pytest.mark.parametrize("physical,bias", [((3, 4, 5, 7), True),
                                          ((1, 4, 5), True),
                                          ((True, 4, 5, 7), True),
                                          ((1, 4, 5, 7), False)])
def test_pair_identity_rejects_invalid_physical_bias(physical, bias):
    with pytest.raises(ValueError, match="physical bias shape"):
        native._checkpoint_identity(DIMS, .5, True, bias=bias, bias_shape=physical)


def test_generated_pair_rejects_physical_mismatch_before_compilation(monkeypatch):
    forward = artifact((1, 4, 5, 7), False)
    backward = artifact((2, 1, 5, 7), True)
    monkeypatch.setattr(checkpoint, "lower_generated_checkpoint",
                        lambda source, backward=False, prune_inactive=False, compact_gradients=False, compact_launch="packed_v1", compact_threads=128: products[backward])
    products = {False: forward, True: backward}
    monkeypatch.setattr(native, "package_scheduled_checkpoint",
                        lambda *args, **kwargs: pytest.fail("mismatched pair compiled"))
    with pytest.raises(ValueError, match="policies disagree"):
        native.package_generated_attention_checkpoint_pair(
            "generated source", pipeline_name="tessera-nvidia-pipeline-sm120")



@pytest.mark.parametrize("mutation", ["physical_shape", "reduction", "scalar", "guard", "abi"])
def test_broadcast_pair_rejects_mutated_saved_storage_before_driver(monkeypatch, mutation):
    import ctypes
    physical = (1, 4, 1, 7)
    forward, backward = artifact(physical, False), artifact(physical, True)
    monkeypatch.setattr(native, "_compile_tile_ir",
                        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    identity = native._checkpoint_identity(DIMS, .5, True, bias=True, bias_shape=physical)
    pair = native.AttentionCheckpointPair(
        native.package_scheduled_checkpoint(forward, pipeline_name="tessera-nvidia-pipeline-sm120"),
        native.package_scheduled_checkpoint(backward, pipeline_name="tessera-nvidia-pipeline-sm120"), identity)
    desc = pair.backward.descriptor
    if mutation in ("physical_shape", "reduction"):
        p = dict(desc.provenance)
        if mutation == "physical_shape":
            p["bias_shape"] = [2, 1, 1, 7]
        else:
            p["bias_gradient_reduction"] = "unordered"
        desc = replace(desc, provenance=p)
    elif mutation == "scalar":
        desc = replace(desc, scalars=desc.scalars[:-1])
    elif mutation == "guard":
        desc = replace(desc, shape_guards=tuple(g for g in desc.shape_guards if g.binding != "dbias"))
    else:
        desc = replace(desc, abi_id=native.SM120_ATTN_BWD_LSE_BIAS_GRAD_F32_ABI)
    pair = replace(pair, backward=replace(pair.backward, descriptor=desc))
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: pytest.fail("CUDA loaded before storage validation"))
    with pytest.raises((ValueError, RuntimeError)):
        pair.capture(None, None, None, bias=None)



def test_explicit_graph_pair_routes_broadcast_without_reconstruction_loss(monkeypatch):
    from tests.device.nvidia.test_lse_checkpoint_native import _forward_module, _backward_module
    from tessera.compiler.graph_ir import tensor_ir_type
    from tessera.compiler.resident_attention import checkpoint_shapes
    forward = _forward_module(saved=True, bias=True)
    backward = _backward_module(saved=True, bias=True)
    backward.functions[0].args[4].name = "o"
    backward.functions[0].body[0].operands[4] = "%o"
    logical = native._attention_lse_contract(forward)[1]
    physical = (1, 1, 1, logical[4])
    for graph in (forward, backward):
        fn = graph.functions[0]
        index = next(i for i, arg in enumerate(fn.args) if arg.name == "bias")
        fn.args[index].ir_type = tensor_ir_type(physical, "fp32")
        fn.body[0].operand_types[index] = str(fn.args[index].ir_type)
    monkeypatch.setattr(native, "_compile_tile_ir",
                        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    pair = native.package_attention_checkpoint_pair(
        forward, backward, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert pair.forward.descriptor.abi_id == native.SM120_ATTN_LSE_BCAST_F32_ABI
    assert pair.backward.descriptor.abi_id == native.SM120_ATTN_BWD_LSE_BCAST_F32_ABI
    assert checkpoint_shapes(pair)[0] == logical



@pytest.mark.parametrize("backward", [False, True])
def test_checked_submit_rejects_physical_scalars_different_from_native_shape(monkeypatch, backward):
    from types import SimpleNamespace
    from tessera import runtime as rt
    result = artifact((1, 1, 1, 7), backward)
    monkeypatch.setattr(native, "_compile_tile_ir",
                        lambda text, entry: (text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    package = native.package_scheduled_checkpoint(result, pipeline_name="tessera-nvidia-pipeline-sm120")
    buffers = {}
    for binding in package.descriptor.buffers:
        extents = sorted((g.dimension, g.value) for g in package.descriptor.shape_guards if g.binding == binding.name)
        buffers[binding.name] = np.zeros(tuple(value for _, value in extents), np.float32)
    scalars = dict(zip(("B","Hq","Hkv","Sq","Sk","D","Dv","BiasB","BiasH","BiasQ","BiasK"),
                       (*DIMS,1,1,5,7),strict=True))
    library = SimpleNamespace(tessera_nvidia_ptx_invoke=lambda *a: pytest.fail("copied mismatched physical bias"))
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch", lambda: library)
    monkeypatch.setattr(rt, "_register_nvidia_ptx", lambda *a: 0)
    with pytest.raises(RuntimeError, match="physical bias scalars disagree"):
        rt._submit_nvidia_sm120_native(package.image, package.descriptor, buffers, scalars, None)
