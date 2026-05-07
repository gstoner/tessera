from __future__ import annotations

import tessera as ts
from tessera.compiler.graph_ir import GraphIRBuilder


def test_static_range_loop_emits_structured_control_markers():
    def looped(x):
        for _ in range(2):
            y = ts.ops.relu(x)
        return y

    builder = GraphIRBuilder()
    fn = builder.lower(looped)
    text = fn.to_mlir()

    assert "tessera.scf.for.begin" in text
    assert "trip_count = 2" in text
    assert "tessera.relu" in text
    assert not builder.diagnostics


def test_static_if_emits_structured_control_markers():
    def branch(x):
        if 2 > 1:
            return ts.ops.gelu(x)
        else:
            return ts.ops.relu(x)

    builder = GraphIRBuilder()
    fn = builder.lower(branch)
    text = fn.to_mlir()

    assert "tessera.scf.if.begin" in text
    assert "condition = true" in text
    assert "tessera.scf.else" in text
    assert not builder.diagnostics


def test_dynamic_if_keeps_precise_unsupported_diagnostic():
    @ts.jit
    def dynamic_branch(x):
        if x.sum() > 0:
            return ts.ops.relu(x)
        return x

    explanation = dynamic_branch.explain_lowering()
    assert "PY_FRONTEND_UNSUPPORTED" in explanation
    assert "if/else control flow" in explanation
