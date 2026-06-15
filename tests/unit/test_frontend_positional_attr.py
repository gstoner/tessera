"""@jit AST frontend — positional scalar args lower to attributes, not operands.

Many ops take a tensor operand plus a scalar parameter positionally
(``top_k(x, k)``, ...).  Before this, a literal positional scalar was emitted as
a bogus ``%?`` operand, which broke the op's arity and dropped the whole op (and
graph).  The frontend now binds such literals to the op's named attribute — so
the positional call lowers identically to the keyword call.
"""

from __future__ import annotations

import tessera as ts


def _graph(fn):
    return ts.jit(target="apple_gpu")(fn).runtime_artifact().graph_ir or ""


def test_top_k_positional_scalar_lowers_to_attribute():
    def f(x):
        return ts.ops.top_k(x, 5)

    g = _graph(f)
    # one operand (%x), k as an attribute — not a "%?" operand
    assert "tessera.top_k(%x) {k = 5}" in g
    assert "%?" not in g


def test_top_k_positional_matches_keyword_form():
    def pos(x):
        return ts.ops.top_k(x, 5)

    def kw(x):
        return ts.ops.top_k(x, k=5)

    gp, gk = _graph(pos), _graph(kw)
    assert "tessera.top_k(%x) {k = 5}" in gp
    assert "tessera.top_k(%x) {k = 5}" in gk


def test_tensor_arg_ops_still_emit_operands():
    # regression: ops whose positional args are tensors are unaffected
    def mm(a, b):
        return ts.ops.matmul(a, b)

    def qkv(x, W):
        return ts.ops.qkv_projection(x, W)

    assert "tessera.matmul(%a, %b)" in _graph(mm)
    assert "tessera.qkv_projection(%x, %W)" in _graph(qkv)


def test_positional_attr_table_only_affects_listed_ops():
    from tessera.compiler.graph_ir import _POSITIONAL_ATTR_PARAMS

    assert "tessera.top_k" in _POSITIONAL_ATTR_PARAMS
    assert _POSITIONAL_ATTR_PARAMS["tessera.top_k"][0] == "k"
