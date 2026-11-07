"""E2E-REAL-6F -- decoration-time abstract tracing over symbolic dimensions.

A symbolic signature (``tessera.f32["B", "D"]``) used to drop straight to the
AST ``_OpExtractor`` because ``GraphIRBuilder.lower`` only seeded the tracer
from all-digit shapes. The tracer now carries a symbol name per extent: equal
names unify, a static ``1`` broadcasts against a name, and anything it cannot
prove (two different names, a name against another static extent, a matmul
contraction over unequal names) raises ``TesseraTraceError`` so the program
stays with the AST candidate.

The contract is no behaviour change. Every accepted program must produce the
Graph IR the AST frontend produces for it -- the same op sequence, operand and
result types, attributes, arguments (incl. ``tessera.dim_names``) and result
types -- which ``test_symbolic_trace_matches_ast_frontend`` checks
differentially.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera
from tessera import ops
from tessera.compiler import trace as trace_module
from tessera.compiler.frontend_authority import graph_signature
from tessera.compiler.graph_ir import GraphIRBuilder, GraphIRModule
from tessera.compiler.trace import TesseraTraceError

f32 = tessera.f32
bf16 = tessera.bf16


def mlp(x: bf16["B", "D"], w1: bf16["D", "H"], w2: bf16["H", "D"]):  # noqa: F821
    return ops.matmul(ops.gelu(ops.matmul(x, w1)), w2)


def bias_add(x: f32["B", "S", "D"], b: f32["D"]):  # noqa: F821
    return ops.add(x, b)


def broadcast_one(x: f32["B", "D"], b: f32[1, "D"]):  # noqa: F821
    return ops.sub(x, b)


def mixed_static(x: f32["B", 16], w: f32[16, "N"]):  # noqa: F821
    return ops.matmul(x, w)


def residual(x: f32["B", "D"], w: f32["D", "D"], g: f32["D"]):  # noqa: F821
    return ops.rmsnorm(ops.add(ops.matmul(x, w), x), g)


def gate(x: f32["B", "D"], y: f32["B", "D"]):  # noqa: F821
    return ops.mul(ops.sigmoid(x), ops.tanh(y))


def transpose_contract(x: f32["M", "K"], y: f32["M", "N"]):  # noqa: F821
    return ops.matmul(ops.transpose(x), y)


def softmax_scores(q: f32["S", "D"], k: f32["D", "T"]):  # noqa: F821
    return ops.softmax(ops.matmul(q, k))


def layer_norm_silu(x: f32["B", "D"]):  # noqa: F821
    return ops.silu(ops.layer_norm(x))


def row_sum(x: f32["B", "D"]):  # noqa: F821
    return ops.sum(ops.relu(x), axis=-1)


def cast_bf16(x: f32["B", "D"]):  # noqa: F821
    return ops.cast(x, "bf16")


def attention(q: f32["B", "H", "S", "D"], k: f32["B", "H", "S", "D"],  # noqa: F821
              v: f32["B", "H", "S", "D"]):  # noqa: F821
    return ops.flash_attn(q, k, v)


SYMBOLIC_PROGRAMS = [
    mlp, bias_add, broadcast_one, mixed_static, residual, gate,
    transpose_contract, softmax_scores, layer_norm_silu, row_sum, cast_bf16,
    attention,
]


def _lower(fn, *, prefer_abstract_trace: bool) -> GraphIRModule:
    builder = GraphIRBuilder()
    builder.lower(fn, effect_tag="pure", prefer_abstract_trace=prefer_abstract_trace)
    return builder.module()


def _typed_signature(module: GraphIRModule):
    """Everything a downstream consumer reads off one function, SSA-independent."""
    function = module.functions[0]
    args = [
        (arg.name, str(arg.ir_type), tuple(arg.ir_type.shape), arg.ir_type.dtype,
         arg.dim_names, arg.effect, arg.layout, arg.dtype_status)
        for arg in function.args
    ]
    types = [
        (op.op_name, list(op.operand_types), op.result_type,
         [str(t) for t in op.inferred_types])
        for op in function.body
    ]
    return (args, types, [str(t) for t in function.result_types],
            len(function.return_values), function.fn_attrs.get("tessera.effect"))


@pytest.mark.parametrize("fn", SYMBOLIC_PROGRAMS, ids=lambda fn: fn.__name__)
def test_symbolic_trace_matches_ast_frontend(fn) -> None:
    traced = _lower(fn, prefer_abstract_trace=True)
    legacy = _lower(fn, prefer_abstract_trace=False)

    function = traced.functions[0]
    assert function.fn_attrs["tessera.frontend.authority"] == '"tracer"'
    assert traced.module_attrs["tessera.frontend.authority"] == '"tracer"'
    assert "tessera.frontend.authority" not in legacy.module_attrs
    # The program is genuinely symbolic: at least one `?` extent named by
    # the annotation, which the trace must have kept on the argument.
    assert any("?" in arg.ir_type.shape and arg.dim_names for arg in function.args)

    # Op sequence, operand topology and attributes (the certify machinery's
    # SSA-independent structural signature) ...
    assert graph_signature(traced) == graph_signature(legacy)
    # ... and every type and argument fact the AST spelling carries.
    assert _typed_signature(traced) == _typed_signature(legacy)
    assert traced.verify().ok


def _refused(fn) -> str:
    """Lower ``fn`` and return the trace refusal that sent it to the AST."""
    errors: list[str] = []
    original = trace_module.trace

    def recording(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except TesseraTraceError as exc:
            errors.append(str(exc))
            raise

    trace_module.trace = recording
    try:
        module = _lower(fn, prefer_abstract_trace=True)
    finally:
        trace_module.trace = original
    assert "tessera.frontend.authority" not in module.module_attrs
    assert _typed_signature(module) == _typed_signature(
        _lower(fn, prefer_abstract_trace=False))
    assert len(errors) == 1
    return errors[0]


def test_unequal_contraction_names_fall_back_to_ast() -> None:
    def contract(a: f32["M", "K"], b: f32["J", "N"]):  # noqa: F821
        return ops.matmul(a, b)

    assert "inner-dim mismatch" in _refused(contract)


def test_symbol_against_static_extent_falls_back_to_ast() -> None:
    def add(a: f32["M", "K"], b: f32["M", 16]):  # noqa: F821
        return ops.add(a, b)

    assert "cannot prove broadcast" in _refused(add)


def test_two_symbols_in_one_broadcast_fall_back_to_ast() -> None:
    def add(a: f32["M", "K"], b: f32["M", "N"]):  # noqa: F821
        return ops.add(a, b)

    assert "cannot prove broadcast" in _refused(add)


def test_catalog_disagreement_falls_back_to_ast() -> None:
    # The tracer broadcasts to rank 2; the AST types `add` as its first
    # operand (rank 1). Neither spelling is chosen -- the program stays AST.
    def add(b: f32["D"], x: f32["B", "D"]):  # noqa: F821
        return ops.add(b, x)

    assert "disagrees with the catalog inference" in _refused(add)


def test_less_precise_catalog_type_falls_back_to_ast() -> None:
    # The catalog types a rank-3 symbolic matmul as unranked; the trace's
    # rank-3 result would be a different spelling, so the AST keeps it.
    def batched(q: f32["B", "S", "D"], k: f32["B", "D", "T"]):  # noqa: F821
        return ops.matmul(q, k)

    assert "disagrees with the catalog inference" in _refused(batched)


def test_symbolic_control_flow_falls_back_to_ast() -> None:
    def loop(x: f32["B", "D"]):  # noqa: F821
        return tessera.control.fori_loop(0, 2, lambda _i, c: ops.relu(c), x)

    assert "control flow" in _refused(loop)


def test_unnamed_dynamic_extents_do_not_unify() -> None:
    # `tensor<?x...>` string annotations carry no dim names. Each `?` gets a
    # name of its own, so two of them never prove equal.
    def contract(a: "tensor<?x?xf32>", b: "tensor<?x?xf32>"):
        return ops.matmul(a, b)

    assert "inner-dim mismatch" in _refused(contract)


def test_symbolic_broadcast_rules() -> None:
    broadcast = trace_module._broadcast_shape
    assert broadcast([("B", "D"), ("B", "D")], {}) == ("B", "D")
    assert broadcast([("B", "D"), (1, "D")], {}) == ("B", "D")
    assert broadcast([("B", 1), (1, "D")], {}) == ("B", "D")
    assert broadcast([("B", "S", "D"), ("D",)], {}) == ("B", "S", "D")
    assert broadcast([(1, 1), (1,)], {}) == (1, 1)
    for bad in ([("B",), ("D",)], [("B",), (16,)]):
        with pytest.raises(TesseraTraceError):
            broadcast(bad, {})
    # The static path is still NumPy's, including its own error.
    assert broadcast([(4, 1), (3,)], {}) == (4, 3)
    with pytest.raises(ValueError):
        broadcast([(4,), (3,)], {})


def test_symbolic_matmul_rule() -> None:
    matmul = trace_module._matmul_shape
    assert matmul([("M", "K"), ("K", "N")], {}) == ("M", "N")
    assert matmul([("B", "M", 8), (8, "N")], {}) == ("B", "M", "N")
    for bad in ([("M", "K"), (8, "N")], [("M", "K"), ("J", "N")], [("M", 1), ("K", "N")]):
        with pytest.raises(TesseraTraceError):
            matmul(bad, {})


def test_anonymous_spec_extents_are_refused() -> None:
    for dim in ("?", "*", "", "?0"):
        with pytest.raises(TesseraTraceError):
            trace_module.trace(lambda x: ops.relu(x), ((dim, 4), "f32"))


def test_concrete_tracing_keeps_static_shapes() -> None:
    traced = trace_module.trace(
        lambda x, w: ops.matmul(x, w),
        np.ones((2, 3), np.float32), np.ones((3, 4), np.float32))
    assert traced.args[0][1] == (2, 3)
    assert traced.output_specs[0][0] == (2, 4)
    assert traced.output_values[0].shape == (2, 4)


def test_symbolic_jit_is_tracer_authored_and_still_checks_bindings() -> None:
    from tessera.compiler import graph_ir_cache
    from tessera.compiler.constraints import TesseraConstraintError

    graph_ir_cache.clear_graph_ir_cache()

    @tessera.jit
    def project(x: f32["B", "D"], w: f32["D", "N"]):  # noqa: F821
        return ops.relu(ops.matmul(x, w))

    assert project.graph_ir.module_attrs["tessera.frontend.authority"] == '"tracer"'
    assert [arg.dim_names for arg in project.graph_ir.functions[0].args] == [
        ("B", "D"), ("D", "N")]
    x = np.arange(6, dtype=np.float32).reshape(2, 3) - 2.0
    w = np.ones((3, 4), np.float32)
    np.testing.assert_allclose(project(x, w), np.maximum(x @ w, 0.0))
    # One name binds one extent across arguments at call time; that check is
    # what licenses the trace to unify `D` between `x` and `w`.
    with pytest.raises(TesseraConstraintError):
        project(x, np.ones((5, 4), np.float32))
