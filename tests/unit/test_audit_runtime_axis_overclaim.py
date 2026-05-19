"""Regression: the audit runtime walker does not promote
``target_cap.default_runtime_status`` to ``ready`` when the op has
no explicit per-target registration.

Findings audit (2026-05-19) flagged that the old walker fell back
to ``target_cap.default_runtime_status`` whenever ``op_caps`` was
``None``.  Because CPU's default is ``"ready"``, every primitive
outside ``OP_SPECS`` showed ``runtime=ready`` in the support
table — a real overclaim that undercuts M0's "machine-checkable
claims" goal.

The fix is in :func:`tessera.compiler.audit._axis_runtime`:

* explicit per-op entry  → use its ``runtime_status`` verbatim
* missing + target advertises ``reference_execution``  → ``reference``
* otherwise               → ``unknown``

We never silently promote a missing entry to ``ready``.
"""

from __future__ import annotations

import pytest

from tessera.compiler import audit, capabilities as cap


def test_runtime_axis_never_promotes_missing_to_ready() -> None:
    """An op that has no explicit OpCapability anywhere must not
    surface as ``ready`` in the runtime axis."""
    # Drive a synthetic op name that's guaranteed to be absent from
    # every target's `supported_ops`.
    cell = audit._axis_runtime("__synthetic_op_not_in_any_target__")
    assert cell.status != "ready", (
        f"missing OpCapability promoted to {cell.status!r}; "
        f"this is the M0 overclaim the audit walker was supposed to "
        f"prevent.  source={cell.source!r}"
    )
    # The op also isn't in OP_SPECS, so the reference-execution
    # fallback shouldn't fire — we expect ``unknown``.
    assert cell.status == "unknown", cell


def test_runtime_axis_uses_reference_for_op_specs_without_explicit_entry() -> None:
    """An op present in OP_SPECS but lacking a per-target
    ``OpCapability`` should report ``reference`` (CPU's numpy
    fallback covers it) — not ``ready``."""
    # Pick an op_spec name that exists in OP_SPECS but isn't in
    # most targets' supported_ops.
    from tessera.compiler.op_catalog import OP_SPECS
    candidate = None
    for name in OP_SPECS:
        spec = OP_SPECS[name]
        if not any(
            spec.graph_name in tc.supported_ops or name in tc.supported_ops
            for tc in cap.TARGET_CAPABILITIES.values()
        ):
            candidate = name
            break
    if candidate is None:
        pytest.skip("every OP_SPECS entry has at least one explicit target — nothing to test")
    cell = audit._axis_runtime(candidate)
    assert cell.status == "reference", (
        f"OP_SPECS entry {candidate!r} without explicit per-target "
        f"OpCapability should reach 'reference' via CPU numpy "
        f"fallback; got {cell.status!r}"
    )


def test_runtime_axis_honors_explicit_op_capability() -> None:
    """When a target registers an explicit ``OpCapability``, its
    ``runtime_status`` shows through directly — the walker
    doesn't second-guess it."""
    # `matmul` is one of the most explicitly-registered ops; CPU
    # has it as `ready`.
    cell = audit._axis_runtime("matmul")
    # Exact value depends on which target wins, but it must be at
    # least as concrete as the most-favored explicit entry.  We
    # assert it's NOT "unknown" (the new walker's no-opinion
    # sentinel).
    assert cell.status != "unknown", cell


def test_unknown_status_is_in_glyph_table() -> None:
    """The runtime walker can now emit ``unknown``; the glyph
    table must accept it (otherwise the support-table renderer
    will print ``?`` for an unmapped status — same outcome but
    documented)."""
    assert "unknown" in audit.AXIS_VALUE_GLYPHS
