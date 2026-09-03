"""GRAPH_IR_UNRESOLVED_ELEMENT_TYPE -- Decision #21a at the parser boundary.

A tensor whose element slot is ``?`` (``tensor<*x?>``, ``tensor<?x?x?>``) is what
``tensor_ir_type`` renders for an unresolved dtype. MLIR rejects every such
type, so a parser-bound consumer must fail closed with a NAMED reason before
rendering -- not surface the parser's symptom one level down. The renders
themselves are untouched: ``?`` remains the symbolic module's placeholder.
"""

from tessera.compiler.graph_ir import (
    BOOL,
    INDEX,
    TENSOR_OPAQUE,
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    IRType,
    SourceSpan,
    handle_ir_type,
    tensor_ir_type,
    unresolved_element_type_diagnostics,
)

CODE = "GRAPH_IR_UNRESOLVED_ELEMENT_TYPE"


def _module(*fns):
    return GraphIRModule(functions=list(fns))


def test_opaque_argument_is_flagged_by_name():
    fn = GraphIRFunction(name="f", args=[IRArg("x", TENSOR_OPAQUE)])
    diags = unresolved_element_type_diagnostics(_module(fn))
    assert len(diags) == 1
    d = diags[0]
    assert d.code == CODE and d.severity == "error"
    assert "%x" in d.message and "tensor<*x?>" in d.message
    assert "Decision #21a" in d.message
    assert CODE in d.format()


def test_ranked_unresolved_dtype_is_flagged():
    t = tensor_ir_type(("?", "?"), None)  # what Tensor['M','K'] renders as
    assert str(t) == "tensor<?x?x?>"
    fn = GraphIRFunction(name="f", args=[IRArg("a", t)])
    assert [d.code for d in unresolved_element_type_diagnostics(_module(fn))] == [CODE]


def test_concrete_types_are_not_flagged():
    t = IRType("tensor<8x8xf32>", ("8", "8"), "fp32")
    fn = GraphIRFunction(
        name="f", args=[IRArg("a", t)], result_types=[t],
        body=[IROp(result="c", op_name="tessera.cholesky", operands=["%a"],
                   operand_types=["tensor<8x8xf32>"], result_type="tensor<8x8xf32>")],
        return_values=["%c"],
    )
    assert unresolved_element_type_diagnostics(_module(fn)) == ()


def test_scalars_and_handles_with_no_dtype_are_valid_mlir_not_flagged():
    fn = GraphIRFunction(name="f", args=[
        IRArg("i", INDEX), IRArg("b", BOOL), IRArg("kv", handle_ir_type("kv_cache")),
    ])
    assert unresolved_element_type_diagnostics(_module(fn)) == ()


def test_op_result_is_flagged_and_carries_the_op_span():
    span = SourceSpan(line=7, col=3, source_name="user.py")
    op = IROp(result="r", op_name="tessera.matmul", operands=[], operand_types=[],
              result_type="tensor<*x?>", source_span=span)
    fn = GraphIRFunction(name="f", body=[op])
    (d,) = unresolved_element_type_diagnostics(_module(fn))
    assert d.code == CODE and d.span is span
    assert "tessera.matmul" in d.message and "%r" in d.message
    assert "user.py:7:3" in d.format()


def test_multi_result_aggregate_flags_an_unresolved_component():
    """A multi-result op renders one parenthesized aggregate. Checking only the
    whole string missed an unresolved component -- the preflight passed and the
    generic parser failure came back instead (PR #706 review, P2)."""
    agg = f"({TENSOR_OPAQUE}, tensor<4xf32>)"
    op = IROp(result="a,b", op_name="tessera.qr", operands=[], operand_types=[],
              result_type=agg)
    (d,) = unresolved_element_type_diagnostics(_module(GraphIRFunction(name="f", body=[op])))
    assert d.code == CODE and "tessera.qr" in d.message and "%a,b" in d.message


def test_multi_result_uses_the_structured_contract_when_declared():
    op = IROp(result="a,b", op_name="tessera.qr", operands=[], operand_types=[],
              result_type="(tensor<4xf32>, tensor<4xf32>)",   # rendered form disagrees
              inferred_types=(TENSOR_OPAQUE, tensor_ir_type(("4",), "fp32")))
    (d,) = unresolved_element_type_diagnostics(_module(GraphIRFunction(name="f", body=[op])))
    assert d.code == CODE and "tensor<*x?>" in d.message


def test_multi_result_fully_resolved_is_not_flagged():
    op = IROp(result="a,b", op_name="tessera.qr", operands=[], operand_types=[],
              result_type="(tensor<4xf32>, tensor<2xcomplex<f32>>)")
    assert unresolved_element_type_diagnostics(_module(GraphIRFunction(name="f", body=[op]))) == ()


def test_component_split_respects_commas_inside_angle_brackets():
    from tessera.compiler.graph_ir import _type_components

    assert _type_components("tensor<4xf32>") == ["tensor<4xf32>"]
    assert _type_components("(tensor<*x?>, tensor<4xf32>)") == ["tensor<*x?>", "tensor<4xf32>"]
    # A parameterized element type is one component, not two.
    assert _type_components("(tensor<2xcomplex<f32>>, tensor<*x?>)") == [
        "tensor<2xcomplex<f32>>", "tensor<*x?>"]


def test_declared_result_type_is_flagged():
    fn = GraphIRFunction(name="f", result_types=[TENSOR_OPAQUE])
    (d,) = unresolved_element_type_diagnostics(_module(fn))
    assert d.code == CODE and "result #0" in d.message


def test_renders_are_untouched_placeholder_survives():
    fn = GraphIRFunction(name="f", args=[IRArg("x", TENSOR_OPAQUE)])
    assert "tensor<*x?>" in fn.to_mlir()
    assert "tensor<*x?>" in fn.to_mlir(canonical=True)
