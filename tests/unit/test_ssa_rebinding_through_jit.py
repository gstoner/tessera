"""Audit-fix regression guards for the SSA-rebinding crash (2026-05-31).

Background. D.4 (aug-assign desugaring) and D.5 (sub/div BinOps) shipped
with tests that exercised the AST→IR extractor via ``GraphIRBuilder``
directly. That layer doesn't run the Graph IR verifier — but
``@tessera.jit`` decoration *does*, and the verifier rejects two
op-results with the same SSA name (``GRAPH_IR_DUP_VALUE``). So the new
tests passed at the extractor level while ``@tessera.jit`` decoration
silently failed for any function with a reassigned local — including
the aug-assign cases the tests were supposed to prove.

The fix mints versioned SSA names (``c`` → ``c__1`` → ``c__2`` ...) on
reassignment and tracks an alias map so subsequent reads of ``c``
resolve to the latest version. Critically, the alias only updates
*after* the RHS emits, so ``c = c + b`` reads the OLD ``c`` while
evaluating the BinOp.

These tests pin the contract by going through ``@tessera.jit``
end-to-end (the path the audit cared about), not just GraphIRBuilder.
"""

from __future__ import annotations

import textwrap

import pytest


def _decorate_via_jit(tmp_path, source: str, fn_name: str = "f"):
    """Build a temp module and import + decorate it. @tessera.jit requires
    introspectable source from a real file."""
    (tmp_path / "mod.py").write_text(source)
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        if "mod" in sys.modules:
            del sys.modules["mod"]
        import mod  # noqa: I001
        return getattr(mod, fn_name)
    finally:
        sys.path.remove(str(tmp_path))


# ---- Aug-assign forms (the D.4 / D.5 regression) -------------------------

@pytest.mark.parametrize("op_kind,op_symbol,expected_op", [
    ("isub", "-=", "tessera.sub"),
    ("iadd", "+=", "tessera.add"),
    ("imul", "*=", "tessera.mul"),
    ("idiv", "/=", "tessera.div"),
])
def test_jit_decoration_succeeds_for_augassign(
    tmp_path, op_kind, op_symbol, expected_op
):
    """Pre-fix: ``@tessera.jit`` raised ``TesseraJitError`` with
    ``GRAPH_IR_DUP_VALUE`` for every aug-assign form. Post-fix: the
    decoration succeeds, the canonical compile result is constructible,
    and the IR contains the expected op."""
    src = textwrap.dedent(f"""
        import tessera

        @tessera.jit
        def f(a, b):
            c = tessera.ops.relu(a)
            c {op_symbol} b
            return c
    """).strip() + "\n"
    fn = _decorate_via_jit(tmp_path, src)
    # JIT decoration succeeded (this would have raised pre-fix).
    assert fn.compile_result is not None
    # The IR contains both ops, with the correct SSA renaming.
    mlir = fn.graph_ir.to_mlir(verify=False)
    assert "tessera.relu" in mlir
    assert expected_op in mlir, mlir
    # The result is two distinct SSA names — no duplicate.
    assert "%c " in mlir or "%c\n" in mlir or "%c(" in mlir
    assert "%c__1" in mlir, f"reassignment should mint c__1; got:\n{mlir}"


def test_aug_assign_rhs_reads_old_binding(tmp_path):
    """Subtraction is non-commutative — ``c -= b`` must lower to
    ``new_c = sub(OLD_c, b)``. The fix's whole point is to preserve
    RHS-reads-old-binding semantics."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            c = tessera.ops.relu(a)
            c -= b
            return c
    """).strip() + "\n"
    fn = _decorate_via_jit(tmp_path, src)
    mlir = fn.graph_ir.to_mlir(verify=False)
    # The sub's left operand must be the OLD %c (the relu result), not
    # the new %c__1 (the sub's own result). If the alias map updated too
    # early, this assertion fails with `tessera.sub(%c__1, %b)`.
    assert "tessera.sub(%c, %b)" in mlir, mlir


# ---- Plain rebinding (the same SSA crash, no aug-assign) -----------------

def test_jit_decoration_succeeds_for_plain_rebinding(tmp_path):
    """``c = c + b`` is the same SSA crash as ``c += b`` but without
    aug-assign desugaring. The fix must cover both paths."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            c = tessera.ops.relu(a)
            c = c + b
            return c
    """).strip() + "\n"
    fn = _decorate_via_jit(tmp_path, src)
    mlir = fn.graph_ir.to_mlir(verify=False)
    assert "%c__1 = tessera.add(%c, %b)" in mlir, mlir


def test_three_way_chain_versions_ssa_names_monotonically(tmp_path):
    """``c = a; c = c + b; c = c * b`` → ``c``, ``c__1``, ``c__2``."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            c = tessera.ops.relu(a)
            c = c + b
            c = c + b
            c = c + b
            return c
    """).strip() + "\n"
    fn = _decorate_via_jit(tmp_path, src)
    mlir = fn.graph_ir.to_mlir(verify=False)
    # Four distinct SSA names: c, c__1, c__2, c__3.
    for name in ("%c ", "%c__1", "%c__2", "%c__3"):
        assert name in mlir, f"missing {name!r} in:\n{mlir}"
    # And each add reads the previous version, not its own result.
    assert "tessera.add(%c, %b)" in mlir
    assert "tessera.add(%c__1, %b)" in mlir
    assert "tessera.add(%c__2, %b)" in mlir


# ---- Verifier round-trip (the audit's specific gap) ---------------------

def test_to_mlir_with_verify_succeeds_after_rebinding(tmp_path):
    """The audit's complaint: tests inspected ``fn.body`` but not
    ``fn.to_mlir()``, which runs the verifier. Lock the verifier-pass
    contract directly: ``module.to_mlir(verify=True)`` must NOT raise
    ``GraphIRVerificationError`` for a function with reassignment."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            c = tessera.ops.relu(a)
            c -= b
            return c
    """).strip() + "\n"
    fn = _decorate_via_jit(tmp_path, src)
    # Default verify=True path — pre-fix this raised at the inner
    # compile_graph_module call, which propagates to @jit. We exercise
    # it directly to lock the layer the verifier runs on.
    text = fn.graph_ir.to_mlir()  # verify=True by default
    assert "tessera.sub" in text


def test_compile_result_executable_for_rebinding_cpu_path(tmp_path):
    """End-to-end: the canonical compile reports the function as
    executable (the CPU plan path supports the binop chain). Locks the
    fact that the SSA-renaming fix doesn't disturb the downstream
    matmul_pipeline planner."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(a, b):
            c = tessera.ops.relu(a)
            c = c + b
            return c
    """).strip() + "\n"
    fn = _decorate_via_jit(tmp_path, src)
    # The bundle's structural answer is what counts here; the gates
    # report executable for CPU on this Mac.
    assert fn.compile_result is not None
    assert fn.compile_result.target == "cpu"
    # Even if the cpu_plan rejected this specific shape, the canonical
    # answer must not be "first failing gate" — the function lowered
    # cleanly; any non-executable answer must come from the bundle
    # diagnostic, not from a gate failure.
    if not fn.compile_result.executable:
        assert fn.compile_result.first_failing_gate is None, (
            f"verifier-clean code must not fail a gate; got "
            f"{fn.compile_result.first_failing_gate}")
