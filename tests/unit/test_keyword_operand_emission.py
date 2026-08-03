"""W1.3 — a tensor passed by KEYWORD must emit as an operand, not an attribute.

`_POSITIONAL_ATTR_PARAMS` fixed one direction of the operand/attribute split: a
positional arg that is really an attribute. The keyword side had the mirror
defect, and it was worse because it was silent. Every ``call.keywords`` entry
became an attribute regardless of what it bound, so a tensor passed by keyword
emitted as a string attribute holding an SSA name::

    tessera.attn_top_k_blocks(%Q, %K, %V) {scores = "%s", top_k = 2}

``%s`` appears nowhere in the operand list. Use-def walks, liveness, and DCE all
see it as unused, and the graph verifier could not catch it because it only
checked ``op.operands`` — an edge parked in an attribute is invisible to the one
check that would have found it.

Three things are pinned here:

1. keyword tensors become operands, in DECLARED order (not call-site order);
2. keyword scalars stay attributes;
3. the verifier now rejects an SSA name in an attribute, so this cannot come
   back by a different route.

Point 3 is the one that matters most. The first two describe today's emitter;
the third makes the whole class of defect fail loudly.
"""

from __future__ import annotations

from tessera import ops
from tessera.compiler.graph_ir import (
    GraphIRBuilder,
    GraphIRFunction,
    GraphIRModule,
    IRArg,
    IROp,
    TENSOR_OPAQUE,
)


# Module-level so inspect.getsource() can retrieve the body for AST lowering.
def _kw_tensor_fn(Q, K, V, s):
    return ops.attn_top_k_blocks(Q, K, V, scores=s, top_k=2, block_size=4)


def _kw_tensor_reordered_fn(Q, K, V, s):
    # Same program, keywords written in a different order.
    return ops.attn_top_k_blocks(Q, K, V, block_size=4, top_k=2, scores=s)


def _kw_varlen_fn(Q, K, V, cq, ck):
    return ops.varlen_sdpa(Q, K, V, cu_seqlens_q=cq, cu_seqlens_k=ck)


def _kw_scalar_fn(x):
    return ops.softcap(x, cap=30.0)


def _kw_cast_fn(x):
    return ops.cast(x, dtype="bf16")


def _emit(fn) -> str:
    b = GraphIRBuilder()
    b.lower(fn)
    module = b.module()
    assert module.verify().ok, "emitted keyword-operand IR failed verification"
    return module.to_mlir(verify=False)


def _op_line(mlir: str, op_name: str) -> str:
    for line in mlir.splitlines():
        if op_name in line:
            return line.strip()
    raise AssertionError(f"{op_name} not emitted:\n{mlir}")


def test_keyword_tensor_emits_as_operand():
    line = _op_line(_emit(_kw_tensor_fn), "tessera.attn_top_k_blocks")
    assert "%s" in line.split(")")[0], f"scores not in the operand list: {line}"
    assert 'scores = ' not in line, f"scores still emitted as an attribute: {line}"
    # The scalars stay attributes.
    assert "top_k = 2" in line and "block_size = 4" in line, line


def test_keyword_operand_order_is_declared_not_call_site():
    """Reordering the keywords must not reorder the operands.

    This is why `_KEYWORD_OPERANDS` is a declaration rather than pure
    derivation: "a kwarg binding an SSA value is an operand" alone would make
    the operand list depend on the order the caller happened to type.
    """
    a = _op_line(_emit(_kw_tensor_fn), "tessera.attn_top_k_blocks")
    b = _op_line(_emit(_kw_tensor_reordered_fn), "tessera.attn_top_k_blocks")
    # Compare the OPERAND LIST only — the parenthesized head. The attribute
    # dict after it does follow call-site order, which is fine: an MLIR
    # attribute dictionary is unordered, so `{a, b}` and `{b, a}` are the same
    # attributes. Operand POSITION is semantic and must not vary; attribute
    # order is not. Asserting on the whole line conflates the two.
    assert a.split(")")[0] == b.split(")")[0], (
        f"operand list depends on keyword order:\n  {a}\n  {b}"
    )


def test_varlen_cu_seqlens_reach_the_operand_list():
    """The catalog always said 5 operands; only the frontend disagreed."""
    line = _op_line(_emit(_kw_varlen_fn), "tessera.varlen_sdpa")
    head = line.split(")")[0]
    assert "%cq" in head and "%ck" in head, line
    assert line.count("%") == 6, f"expected result + 5 operands: {line}"


def test_keyword_scalar_stays_an_attribute():
    line = _op_line(_emit(_kw_scalar_fn), "tessera.softcap")
    assert "cap = 30.0" in line, line
    assert line.count("%") == 2, f"expected result + 1 operand: {line}"


def test_attribute_rule_sees_keyword_bound_attributes():
    """Keywords are bound BEFORE the result type is inferred.

    They used to be bound after, which quietly disabled every attribute-reading
    shape rule for the normal calling convention: `cast(x, dtype="bf16")`
    inferred from an empty attribute set and returned the opaque type. The rule
    was correct and simply consulted too early — the kind of defect that leaves
    no trace because the fallback result is well-formed.
    """
    line = _op_line(_emit(_kw_cast_fn), "tessera.cast")
    assert "-> tensor<*xbf16>" in line, (
        f"cast rule did not see the keyword dtype: {line}"
    )


def test_verifier_rejects_an_ssa_name_hidden_in_an_attribute():
    """The negative case — without it, the two positive tests above only prove
    that today's emitter happens to be right, not that a regression is caught."""
    # `%s` is a real function argument, so it is a well-defined value — it is
    # simply not among the op's operands. That is precisely the defect: the
    # value exists and the op consumes it, but no edge records that.
    fn = GraphIRFunction(name="hidden_edge",
                         args=[IRArg("x", TENSOR_OPAQUE),
                               IRArg("s", TENSOR_OPAQUE)])
    fn.body.append(IROp(
        result="v0",
        op_name="tessera.attn_top_k_blocks",
        operands=["%x"],
        operand_types=[str(TENSOR_OPAQUE)],
        result_type=str(TENSOR_OPAQUE),
        kwargs={"scores": "%s"},
    ))
    module = GraphIRModule(functions=[fn])

    result = module.verify()
    assert not result.ok, "an SSA name in an attribute must not verify"
    codes = {d.code for d in result.diagnostics}
    assert "GRAPH_IR_SSA_VALUE_IN_ATTRIBUTE" in codes, codes


def test_verifier_allows_an_attribute_that_LABELS_an_existing_operand():
    """`{name = "%A"}` next to `%A` in the operand list is a label, not a
    hidden edge — and must not be flagged.

    `tessera.graph.debug_value` does exactly this: it captures a value and
    records which one for the debug artifact. A first cut of the check keyed on
    the `%` sigil alone and rejected it, which would have made the diagnostic
    mean "an attribute mentions a value". That is not a defect, and a gate that
    fires on non-defects is one the next person weakens rather than fixes.
    """
    fn = GraphIRFunction(name="labelled",
                         args=[IRArg("A", TENSOR_OPAQUE)])
    fn.body.append(IROp(
        result="A_dbg",
        op_name="tessera.graph.debug_value",
        operands=["%A"],
        operand_types=[str(TENSOR_OPAQUE)],
        result_type=str(TENSOR_OPAQUE),
        kwargs={"name": "%A"},
    ))
    module = GraphIRModule(functions=[fn])

    codes = {d.code for d in module.verify().diagnostics}
    assert "GRAPH_IR_SSA_VALUE_IN_ATTRIBUTE" not in codes, codes


def test_verifier_accepts_ordinary_string_attributes():
    """A gate that rejected every string attribute would be useless."""
    fn = GraphIRFunction(name="ok", args=[IRArg("x", TENSOR_OPAQUE)])
    fn.body.append(IROp(
        result="v0",
        op_name="tessera.cast",
        operands=["%x"],
        operand_types=[str(TENSOR_OPAQUE)],
        result_type=str(TENSOR_OPAQUE),
        kwargs={"dtype": "bf16"},
    ))
    module = GraphIRModule(functions=[fn])

    codes = {d.code for d in module.verify().diagnostics}
    assert "GRAPH_IR_SSA_VALUE_IN_ATTRIBUTE" not in codes, codes
