"""Saved-LSE policy agreement before either half reaches target compilation."""
import pytest

from tessera.compiler import nvidia_native as native
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
            module("backward", ["do", "q", "k", "v", "row_lse"], [o, q, k, v, lse],
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


def test_checkpoint_identity_matches_physical_float_policy(monkeypatch):
    forward, backward = checkpoint_modules()
    backward.functions[0].body[0].kwargs["scale"] += 1e-10
    monkeypatch.setattr(native, "_compile_tile_ir", lambda text, entry: (
        text, "// PTX", {}, "compiler", "toolchain", (), "cold"))
    pair = native.package_attention_checkpoint_pair(forward, backward, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert pair.forward.descriptor.provenance["checkpoint_contract"] == pair.contract_digest
    assert pair.backward.descriptor.provenance["checkpoint_contract"] == pair.contract_digest
    assert pair.forward.descriptor.provenance["scale"] == pair.backward.descriptor.provenance["scale"]
