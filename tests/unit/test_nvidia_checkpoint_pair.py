"""Saved-LSE policy agreement before either half reaches target compilation."""
import pytest

from tessera.compiler import nvidia_native as native
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type


def checkpoint_modules(*, sq=3, sk=4, causal=True):
    q = tensor_ir_type((1, 2, sq, 4), "fp32")
    k = tensor_ir_type((1, 1, sk, 4), "fp32")
    v = tensor_ir_type((1, 1, sk, 3), "fp32")
    o = tensor_ir_type((1, 2, sq, 3), "fp32")
    lse = tensor_ir_type((1, 2, sq), "fp32")
    policy = {"scale": 0.5, "causal": causal, "lse_checkpoint": "saved"}
    def module(name, names, types, results, result_types, op_name):
        return GraphIRModule(functions=[GraphIRFunction(
            name=name, args=[IRArg(n, t) for n, t in zip(names, types)],
            result_types=result_types, return_values=["%" + n for n in results],
            body=[IROp(result=",".join(results), op_name=op_name,
                       operands=["%" + n for n in names], operand_types=list(map(str, types)),
                       kwargs=dict(policy), inferred_types=tuple(result_types))])])
    return (module("forward", ["q", "k", "v"], [q, k, v], ["o", "row_lse"], [o, lse], "tessera.flash_attn"),
            module("backward", ["do", "q", "k", "v", "o", "row_lse"], [o, q, k, v, o, lse],
                   ["dq", "dk", "dv"], [q, k, v], "tessera.flash_attn_bwd"))


@pytest.mark.parametrize("field,value", [("window", 2), ("logit_softcap", 2.0),
    ("dropout", 0.1), ("causal", 1), ("scale", True), ("scale", 1e100)])
@pytest.mark.parametrize("side", [0, 1])
def test_saved_lse_rejects_unsupported_policy(field, value, side):
    modules = checkpoint_modules()
    modules[side].functions[0].body[0].kwargs[field] = value
    supports = (native.supports_attention_lse, native.supports_attention_backward_lse)
    assert not supports[side](modules[side])


@pytest.mark.parametrize("change", ["scale", "causal", "binding", "return_order"])
def test_checkpoint_pair_rejects_mismatch_before_compilation(monkeypatch, change):
    forward, backward = checkpoint_modules()
    fn = backward.functions[0]
    if change == "binding":
        fn.args[-1].name = "other_lse"
        fn.body[0].operands[-1] = "%other_lse"
    elif change == "return_order":
        fn.return_values.reverse()
    else:
        fn.body[0].kwargs[change] = 0.25 if change == "scale" else False
    monkeypatch.setattr(native, "_compile_tile_ir", lambda *args: pytest.fail("compiled mismatched pair"))
    with pytest.raises(ValueError, match="saved-LSE"):
        native.package_attention_checkpoint_pair(forward, backward, pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_checkpoint_identity_matches_physical_float_policy(monkeypatch):
    forward, backward = checkpoint_modules()
    backward.functions[0].body[0].kwargs["scale"] += 1e-10
    monkeypatch.setattr(native, "_compile_tile_ir", lambda text, entry: (
        text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    pair = native.package_attention_checkpoint_pair(forward, backward, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert pair.forward.descriptor.provenance["checkpoint_contract"] == pair.contract_digest
    assert pair.backward.descriptor.provenance["checkpoint_contract"] == pair.contract_digest
    assert pair.forward.descriptor.provenance["scale"] == pair.backward.descriptor.provenance["scale"]


def test_bias_and_plain_checkpoint_pairs_have_distinct_semantic_identity(monkeypatch):
    from tests.device.nvidia.test_lse_checkpoint_native import _forward_module, _backward_module
    monkeypatch.setattr(native, "_compile_tile_ir",
                        lambda *args: pytest.fail("compiled mismatched checkpoint pair"))
    forward = _forward_module(saved=True, bias=True)
    forward.functions[0].body[0].result = "output,row_lse"
    forward.functions[0].return_values = ["%output", "%row_lse"]
    with pytest.raises(ValueError, match="policies"):
        native.package_attention_checkpoint_pair(
            forward, _backward_module(saved=True),
            pipeline_name="tessera-nvidia-pipeline-sm120")


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("broken", [None, "output", "lse"])
def test_generated_pair_matches_both_saved_output_and_lse_slots(monkeypatch, broken, bias):
    from types import SimpleNamespace
    import tessera.compiler.scheduled_checkpoint as scheduled
    names = ["dO", "q", "k", "v", "output", "lse", "dq", "dk", "dv"]
    if bias:
        names.insert(5, "bias")
        names.append("dbias")
    if broken is not None:
        names[4 if broken == "output" else 5 + int(bias)] = "other_generation"
    def lower(source, *, backward=False, prune_inactive=False, compact_gradients=False,
              compact_launch="packed_v1", compact_threads=128):
        assert compact_gradients is False
        assert compact_launch == "packed_v1" and compact_threads == 128
        return SimpleNamespace(dims=(1,2,1,3,5,4,3), scale=.5, causal=True, shape_bounds=(),
            bias_shape=(), names=tuple(names) if backward else (("q","k","v","bias","output","lse") if bias else ("q","k","v","output","lse")), bias=bias, frontend_argument_indices=tuple(range(3+int(bias))))
    monkeypatch.setattr(scheduled, "lower_generated_checkpoint", lower)
    monkeypatch.setattr(native, "package_scheduled_checkpoint", lambda artifact, **kwargs: artifact)
    if broken is not None:
        with pytest.raises(ValueError, match="bindings disagree"):
            native.package_generated_attention_checkpoint_pair("source", pipeline_name="test")
    else:
        pair=native.package_generated_attention_checkpoint_pair("source", pipeline_name="test")
        assert pair.forward.names[3+int(bias):5+int(bias)] == (pair.backward.names[4],pair.backward.names[5+int(bias)])


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_original_saved_graph_native_projection_is_immutable(monkeypatch, backward):
    import copy
    from tessera.compiler import scheduled_checkpoint as scheduled
    module = checkpoint_modules()[int(backward)]
    fn = module.functions[0]
    fn.args.reverse()
    original = copy.deepcopy(module)
    monkeypatch.setattr(scheduled, "_graph_text",
                        lambda *args, **kwargs: pytest.fail("reconstructed checkpoint Graph"))
    artifact = scheduled.lower_checkpoint_graph(module, backward=backward)
    assert module == original
    assert "tessera.flash_attn" in artifact.graph_ir
    assert artifact.names == (("do", "q", "k", "v", "o", "row_lse", "dq", "dk", "dv")
                              if backward else ("q", "k", "v", "o", "row_lse"))
    assert "tessera_attn.checkpoint_" in artifact.schedule_ir


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_original_saved_graph_does_not_discard_unknown_policy(backward):
    from tessera.compiler.scheduled_checkpoint import lower_checkpoint_graph
    module = checkpoint_modules()[int(backward)]
    module.functions[0].body[0].kwargs["unhandled_policy"] = "approximate"
    with pytest.raises(RuntimeError, match="unsupported policy attribute"):
        lower_checkpoint_graph(module, backward=backward)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("field,value", [
    ("scale", 1), ("softcap", 0), ("logit_softcap", 0),
    ("dropout", 0), ("dropout_p", 0), ("bias", False), ("window", (-1, -1)),
    ("softcap", None), ("logit_softcap", None), ("dropout", None),
    ("dropout_p", None), ("bias", None), ("window", None)])
@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_original_saved_graph_preserves_neutral_policy(field, value, backward):
    from tessera.compiler.scheduled_checkpoint import lower_checkpoint_graph
    module = checkpoint_modules()[int(backward)]
    module.functions[0].body[0].kwargs[field] = value
    artifact = lower_checkpoint_graph(module, backward=backward)
    assert artifact.scale == (1.0 if field == "scale" else 0.5)
    assert artifact.causal is True


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.skipif(find_tessera_opt() is None, reason="requires native scheduling compiler")
def test_recompute_backward_graph_still_verifies(bias):
    from tests.device.nvidia.test_lse_checkpoint_native import _backward_module
    from tessera.compiler.scheduled_matmul import run_tessera_opt
    module = _backward_module(saved=False, bias=bias)
    source = module.to_mlir(canonical=True)
    assert "tessera.flash_attn_bwd" in run_tessera_opt(find_tessera_opt(), source, "--verify-each")
