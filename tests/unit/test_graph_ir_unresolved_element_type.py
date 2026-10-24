"""GRAPH_IR_UNRESOLVED_ELEMENT_TYPE -- Decision #21a at the parser boundary.

A tensor whose element slot is ``?`` (``tensor<*x?>``, ``tensor<?x?x?>``) is what
``tensor_ir_type`` renders for an unresolved dtype. MLIR rejects every such
type, so a parser-bound consumer must fail closed with a NAMED reason before
rendering -- not surface the parser's symptom one level down. ``?`` remains the
symbolic module's placeholder in the inspection render and in function-level
renders; since 2026-10-24 the module's canonical (parser-bound) render refuses
it itself, so no parser-bound caller can forget the check.
"""

import pytest

from tessera.compiler.graph_ir import GraphIRVerificationError

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


# ── The canonical module render is the boundary (2026-10-24) ───────────────


def test_canonical_module_render_refuses_an_unresolved_element_type():
    fn = GraphIRFunction(name="f", args=[IRArg("x", TENSOR_OPAQUE)])
    with pytest.raises(GraphIRVerificationError, match=CODE) as info:
        _module(fn).to_mlir(canonical=True, verify=False)
    assert "%x" in str(info.value)


def test_inspection_render_of_a_symbolic_module_is_unchanged():
    fn = GraphIRFunction(name="f", args=[IRArg("x", TENSOR_OPAQUE)])
    assert "tensor<*x?>" in _module(fn).to_mlir(verify=False)


def test_allow_unresolved_is_an_explicit_opt_out_for_hashing_callers():
    fn = GraphIRFunction(name="f", args=[IRArg("x", TENSOR_OPAQUE)])
    text = _module(fn).to_mlir(canonical=True, verify=False, allow_unresolved=True)
    assert "tensor<*x?>" in text


def test_canonical_render_of_a_specialized_module_is_not_refused():
    fn = GraphIRFunction(name="f", args=[IRArg("x", tensor_ir_type((4, 8), "fp32"))])
    assert "tensor<4x8xf32>" in _module(fn).to_mlir(canonical=True, verify=False)


def test_every_canonical_module_render_that_opts_out_says_why():
    """An opt-out must be a decision, not a workaround: each call site carries
    a comment naming why that render never reaches the parser."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "python" / "tessera"
    sites = []
    for path in root.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            call = "to_mlir(" in line or (i > 0 and "to_mlir(" in lines[i - 1])
            if "allow_unresolved=True" in line and call:
                window = "\n".join(lines[max(0, i - 4):i + 1])
                sites.append((path.name, i + 1, bool(re.search(r"#.*(hash|record|never|not parsed)", window))))
    assert sites, "expected the documented opt-out call sites"
    undocumented = [(name, line) for name, line, ok in sites if not ok]
    assert not undocumented, f"allow_unresolved=True without a stated reason: {undocumented}"



# ── Review on #1007: both bounded native AD entry points hash symbolically ──


def _sentinel(*_args, **_kwargs):
    raise LookupError("past the prepared-call key")


def _symbolic_attention(q, k, v):
    import tessera as ts

    return ts.ops.flash_attn(q, k, v, lse_checkpoint="saved")


def _roots():
    import numpy as np

    return tuple(np.zeros(shape, np.float16) for shape in
                 ((1, 2, 3, 16), (1, 1, 4, 16), (1, 1, 4, 16)))


def test_bounded_source_reverse_keys_a_symbolic_legacy_module(monkeypatch):
    """`execute_bounded` hashes the unannotated legacy module for its owner key;
    a cache miss then specializes. Reaching the specialization proves the key
    render did not refuse the symbolic module."""
    import tessera as ts
    from tessera.compiler.source_attention_reverse_frontend import execute_bounded

    fn = ts.jit(target="nvidia_sm120", autodiff="reverse", wrt=("q", "k", "v"))(
        _symbolic_attention)
    assert any("?" in str(arg.ir_type) for arg in fn._ensure_legacy_graph_ir().functions[0].args)
    monkeypatch.setattr(fn, "_specialized_autodiff_module", _sentinel)
    with pytest.raises(LookupError, match="past the prepared-call key"):
        execute_bounded(fn, _roots(), {}, (), None, (7, 9))
