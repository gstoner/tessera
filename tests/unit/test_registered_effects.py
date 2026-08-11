from tessera.compiler.effects import (
    Effect,
    EffectLattice,
    infer_graph_effects,
    registered_op_effect,
)
from tessera.compiler.graph_ir import IROp
from tessera.compiler.op_catalog import get_op_spec


def test_unknown_graph_operation_fails_closed():
    assert registered_op_effect("tessera.not_registered") is Effect.top


def test_graph_records_are_the_effect_authority():
    body = [
        IROp("sample", "tessera.rng_uniform", [], [], kwargs={"shape": [4]}),
        IROp("out", "tessera.add", ["%sample", "%x"], ["tensor<4xf32>"] * 2),
    ]
    effect, offenders = infer_graph_effects(body)
    assert effect is Effect.random
    assert offenders == ["tessera.rng_uniform"]


def test_legacy_ast_name_visitor_is_retired():
    import tessera.compiler.effects as effects

    assert not hasattr(effects, "_EffectVisitor")


def test_alias_and_stochastic_identity_are_registered():
    assert get_op_spec("tessera.view").aliasing == "operand_0"
    assert get_op_spec("tessera.rng_uniform").stochastic_identity == "implicit_stream"
    assert get_op_spec("tessera.rng_philox_normal").stochastic_identity == "key_counter"


def test_graph_printer_emits_pure_and_alias_contracts_explicitly():
    op = IROp(
        "v", "tessera.view", ["%x"], ["tensor<4xf32>"],
        result_type="tensor<4xf32>",
    )
    text = op.to_mlir()
    assert 'tessera.effect_kind = "pure"' in text
    assert 'tessera.aliasing = "operand_0"' in text
