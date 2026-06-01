"""Followup 3 — SSA rebinding for subscript assignments.

The audit (and an earlier summary of mine) flagged this as the "same
pattern as P1, smaller scope": ``a[i] = ...`` goes through the
``tessera.copy`` op path, and a rebinding might trigger the verifier.

**Empirical answer (2026-05-31):** the verifier doesn't reject. The
``DUP_VALUE`` check only fires on an op's *result*, and ``tessera.copy``
emits ``result=None`` (it's an in-place mutation, not an SSA-producing
op). Two ``a[:] = ...`` writes are two copy ops referencing ``%a`` —
both reference an already-defined SSA, neither produces a new one, no
duplicate. ``@tessera.jit`` decoration succeeds.

These tests lock that fact as a regression guard: a future verifier
change that started rejecting in-place writes would surface here.

They also document the **current value semantics**: subsequent reads
of ``a`` resolve to the pre-write SSA. The IR captures the writes
correctly but doesn't snapshot them into a versioned read-side SSA.
That's a separate semantic surface (would need a new op or a paired
``identity`` capture); the audit's specific verifier-rejection
prediction does not apply.
"""

from __future__ import annotations

import textwrap

import pytest


def _decorate(tmp_path, source: str, fn_name: str = "f"):
    (tmp_path / "sub_mod.py").write_text(source)
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        if "sub_mod" in sys.modules:
            del sys.modules["sub_mod"]
        import sub_mod
        return getattr(sub_mod, fn_name)
    finally:
        sys.path.remove(str(tmp_path))


# ---- Verifier safety guards ---------------------------------------------

def test_single_subscript_write_decorates_clean(tmp_path):
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            a[:] = tessera.ops.relu(b)
            return a
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    assert fn.compile_result is not None
    mlir = fn.graph_ir.to_mlir()
    assert "tessera.copy" in mlir


def test_repeated_subscript_write_to_same_target_decorates_clean(tmp_path):
    """The verifier-safety bet: two ``a[:] = ...`` writes don't trip
    ``GRAPH_IR_DUP_VALUE`` because ``tessera.copy`` has no result. A
    future verifier change that started checking operand-mutation
    would surface here."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b, c):
            a[:] = tessera.ops.relu(b)
            a[:] = tessera.ops.relu(c)
            return a
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    assert fn.compile_result is not None
    mlir = fn.graph_ir.to_mlir()
    # Both copies are present in the IR.
    assert mlir.count("tessera.copy") == 2


def test_scalar_then_subscript_rebind_decorates_clean(tmp_path):
    """``a = relu(a); a[:] = relu(b)`` — the scalar rebind versions
    ``a`` via P1's SSA fix (``a__1``), and the subscript write then
    references ``%a__1`` via the alias resolver. No verifier crash."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            a = tessera.ops.relu(a)
            a[:] = tessera.ops.relu(b)
            return a
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    assert fn.compile_result is not None
    mlir = fn.graph_ir.to_mlir()
    # The scalar rebind minted %a__1; the subscript write copies into it.
    assert "%a__1" in mlir
    assert "tessera.copy" in mlir


def test_subscript_write_followed_by_read_does_not_crash(tmp_path):
    """The semantics here are honest mutation: after ``a[:] = X``, the
    subsequent ``c = a + b`` reads ``%a`` — the SAME SSA name. Backend
    lowerings that respect copy's mutation semantics see the post-write
    value; pure-SSA optimizers would need additional info to know ``a``
    changed. v1 ships in-place semantics; this test just guards that
    the read-after-write pattern doesn't crash the verifier."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            a[:] = tessera.ops.relu(b)
            c = a + b
            return c
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    assert fn.compile_result is not None


# ---- Document the augassign-subscript boundary --------------------------

def test_subscript_aug_assign_still_unsupported(tmp_path):
    """``a[:] += b`` is intentionally not lowered today — the
    desugarer rejects non-Name targets with the precise warning. This
    test guards against a regression that would silently desugar it
    into a malformed IR shape."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            a[:] += b
            return a
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    explanation = fn.explain_lowering()
    assert "PY_FRONTEND_UNSUPPORTED" in explanation
    assert "augmented assignment" in explanation
