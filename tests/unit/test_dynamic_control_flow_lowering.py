"""D.1 / D.2 / D.3 — dynamic control flow lowering in the Graph IR frontend.

Before D.1 the frontend rejected every dynamic ``if`` / dynamic ``for`` /
``while`` / aug-assign with a blanket ``PY_FRONTEND_UNSUPPORTED`` warning.
The audit's framing was that this is a real gap for "real model code"
usefulness. D's job: lower these constructs to the same
``tessera.scf.*`` marker shape the static path already produces, so
downstream consumers see a structural boundary even when the specific
test/iterable expression isn't yet emittable as SSA.

The contract these tests pin per construct:

* **Lowerable expression** — when the test / trip-count expression can
  be lowered (``_emit_expr`` returns an SSA value), the marker carries
  it as an operand AND ``kind="dynamic"``; the diagnostic is an
  **info note** (``PY_FRONTEND_DYNAMIC_*_LOWERED``) confirming the
  dynamic path was taken.
* **Unlowerable expression** — when ``_emit_expr`` returns None
  (e.g. ``ast.Compare``, calls into objects), the marker still emits
  with ``condition_text="..."`` (or ``iter_text=...``) and a more
  specific info diagnostic (``..._UNLOWERED_CONDITION`` etc.). The
  downstream consumer sees the structural shape regardless.
* **Aug-assign** — desugars to ``x = x op y``; goes through the normal
  Assign / BinOp path. Records an info note.

Never re-emit the pre-D.1 ``PY_FRONTEND_UNSUPPORTED`` warning for these
constructs — that's the audit gap, and these tests guard against
regressing to it.
"""

from __future__ import annotations

from tessera.compiler.graph_ir import GraphIRBuilder


# ---- dynamic if/else (D.1) ----

def test_dynamic_if_unlowered_condition_still_emits_structural_markers():
    """A dynamic if whose test isn't emittable (Compare with attr-call)
    must still produce the structural marker shape and an *info* note —
    never the pre-D.1 ``PY_FRONTEND_UNSUPPORTED`` warning."""
    def f(x):
        if x.sum() > 0:
            y = x
        else:
            y = x
        return y

    builder = GraphIRBuilder()
    fn = builder.lower(f)
    text = fn.to_mlir(verify=False)

    assert "tessera.scf.if.begin" in text
    assert "tessera.scf.else" in text
    assert "tessera.scf.if.end" in text
    # condition_text is recorded so downstream consumers see the source.
    assert "condition_text" in text
    assert "kind = \"dynamic\"" in text
    codes = [d.code for d in builder.diagnostics]
    assert "PY_FRONTEND_DYNAMIC_IF_UNLOWERED_CONDITION" in codes
    assert "PY_FRONTEND_UNSUPPORTED" not in codes


def test_static_if_path_still_works():
    """The static path is unchanged — D.1 is additive."""
    import tessera as ts

    def f(x):
        if 2 > 1:
            return ts.ops.relu(x)
        return ts.ops.gelu(x)

    builder = GraphIRBuilder()
    fn = builder.lower(f)
    text = fn.to_mlir()
    assert "tessera.scf.if.begin" in text
    assert "condition = true" in text
    assert not builder.diagnostics  # static path emits no diagnostics


# ---- dynamic for (D.2) ----

def test_dynamic_for_unlowered_iterable_still_emits_structural_markers():
    """``for i in some_dynamic_iterable`` — the iterable isn't a static
    ``range(N)`` so we can't compute trip_count at decoration time."""
    def f(xs):
        total = 0
        for x in xs:
            total = total + x
        return total

    builder = GraphIRBuilder()
    fn = builder.lower(f)
    text = fn.to_mlir(verify=False)

    assert "tessera.scf.for.begin" in text
    assert "tessera.scf.for.end" in text
    assert "iter_text" in text
    codes = [d.code for d in builder.diagnostics]
    assert "PY_FRONTEND_DYNAMIC_FOR_UNLOWERED_ITERABLE" in codes
    assert "PY_FRONTEND_UNSUPPORTED" not in codes


def test_static_for_path_still_works():
    import tessera as ts

    def f(x):
        for _ in range(3):
            y = ts.ops.relu(x)
        return y

    builder = GraphIRBuilder()
    fn = builder.lower(f)
    text = fn.to_mlir()
    assert "tessera.scf.for.begin" in text
    assert "trip_count = 3" in text
    assert not builder.diagnostics


# ---- while (D.3) ----

def test_while_loop_with_unlowered_condition_emits_structural_markers():
    """Plain ``while x > 0:`` — Compare not yet emittable as SSA."""
    def f(x):
        while x > 0:
            y = x
        return y

    builder = GraphIRBuilder()
    fn = builder.lower(f)
    text = fn.to_mlir(verify=False)

    assert "tessera.scf.while.begin" in text
    assert "tessera.scf.while.end" in text
    assert "condition_text" in text
    codes = [d.code for d in builder.diagnostics]
    assert "PY_FRONTEND_WHILE_UNLOWERED_CONDITION" in codes
    assert "PY_FRONTEND_UNSUPPORTED" not in codes


# ---- aug-assign (D bonus) ----

def test_augassign_desugars_to_assign():
    """``x += y`` becomes ``x = x + y`` — re-uses BinOp lowering."""
    import tessera as ts

    def f(a, b):
        c = ts.ops.relu(a)
        c += b
        return c

    builder = GraphIRBuilder()
    builder.lower(f)
    codes = [d.code for d in builder.diagnostics]
    # The desugaring fired (info note).
    assert "PY_FRONTEND_AUGASSIGN_DESUGARED" in codes
    # No "augmented assignment is not lowered" warning.
    msgs = [d.message for d in builder.diagnostics]
    assert not any("augmented assignment is not lowered" in m for m in msgs)


def test_augassign_with_non_name_target_still_unsupported():
    """``a[i] += b`` — the Name-target requirement is honest. The
    desugarer doesn't try to handle subscript stores in v1."""
    def f(a, b):
        a[0] += b
        return a

    builder = GraphIRBuilder()
    builder.lower(f)
    codes = [d.code for d in builder.diagnostics]
    # Honest: this case stays unsupported (subscript target).
    assert "PY_FRONTEND_UNSUPPORTED" in codes


# ---- pre-D.1 regression guard ----

def test_pre_d1_unsupported_warning_no_longer_fires_for_dynamic_if():
    """Regression guard against accidentally reverting D.1: the literal
    pre-D.1 warning message must not appear for any dynamic if/for/while
    body. Catches a future change that silently reintroduces the old
    blanket-unsupported path."""
    def f(x):
        if x.sum() > 0:
            y = x
        else:
            y = x
        return y

    builder = GraphIRBuilder()
    builder.lower(f)
    for d in builder.diagnostics:
        assert "Python if/else control flow is not lowered" not in d.message
        assert "Python for-loops are not lowered" not in d.message
        assert "Python while-loops are not lowered" not in d.message
