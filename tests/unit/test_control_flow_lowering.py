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


def test_dynamic_if_lowers_structurally_with_named_diagnostic():
    """D.1 contract change: dynamic if/else is no longer rejected with a
    blanket ``PY_FRONTEND_UNSUPPORTED`` warning. The audit's framing was
    that "Python if/else control flow is not lowered" is a real gap; D.1
    closes it by emitting ``tessera.scf.if.*`` markers regardless of
    whether the test expression itself is yet emittable as SSA.

    When the test expression isn't lowerable (here ``x.sum() > 0`` —
    ``ast.Compare`` plus an attribute call), the markers carry the
    Python source text under ``condition_text=`` and an informational
    diagnostic names the specific unlowered axis. Downstream CPU
    compile then honestly reports ``tessera.scf.if.begin`` as an
    unsupported op (separate gap, separate fix)."""
    @ts.jit
    def dynamic_branch(x):
        if x.sum() > 0:
            return ts.ops.relu(x)
        return x

    explanation = dynamic_branch.explain_lowering()
    # The new diagnostic surface — structural lowering succeeded.
    assert "PY_FRONTEND_DYNAMIC_IF_UNLOWERED_CONDITION" in explanation
    assert "if/else lowered structurally" in explanation
    # The old "unsupported" warning must NOT fire — that would mean we
    # regressed to the pre-D.1 behavior of emitting no IR at all.
    assert "PY_FRONTEND_UNSUPPORTED" not in explanation
    # The IR must contain the marker shape so downstream passes see the
    # if/else boundary even if they can't yet act on it.
    text = dynamic_branch.graph_ir.to_mlir(verify=False)
    assert "tessera.scf.if.begin" in text
    assert "tessera.scf.if.end" in text
