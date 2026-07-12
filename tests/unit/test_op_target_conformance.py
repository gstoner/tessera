"""Drift gate + structural guards for the op×target conformance matrix.

The dashboard at ``docs/audit/op_target_conformance.md`` is generated from
``python/tessera/compiler/conformance_matrix.py``. This test suite locks four
properties so the dashboard stays a real audit surface:

1. **Drift gate** — the on-disk dashboard must match the freshly-rendered one.
   If you change the matrix logic or any upstream truth source, regenerate via
   ``python -m tessera.cli.conformance_matrix --render``.
2. **Pure aggregator** — the matrix module must not import from anything
   beyond ``tessera.compiler.{primitive_coverage, backend_manifest,
   execution_matrix, driver}`` and the stdlib. New truth sources should
   first land in those upstream modules, then be aggregated here.
3. **Coverage shape** — every (op, target) pair has a cell; cell statuses are
   from the documented enum; the "weakest column wins" overall is consistent.
4. **Honest signal** — at least one cell is ``complete`` (proves the matrix
   isn't always-pessimistic) and at least one is ``missing`` (proves it isn't
   always-optimistic).
"""

from __future__ import annotations

import importlib
import re
import textwrap
from pathlib import Path

from tessera.compiler import conformance_matrix as cm


_DASHBOARD = (Path(__file__).resolve().parents[2]
              / "docs" / "audit" / "op_target_conformance.md")


def test_dashboard_in_sync_with_generator():
    """The on-disk dashboard must match the freshly-rendered output."""
    expected = cm.render_markdown()
    assert _DASHBOARD.is_file(), (
        f"dashboard missing — regenerate via "
        f"`python -m tessera.cli.conformance_matrix --render`")
    actual = _DASHBOARD.read_text()
    if actual != expected:
        # Print a focused diff so the failure is actionable.
        import difflib
        diff = "\n".join(difflib.unified_diff(
            actual.splitlines()[:80],
            expected.splitlines()[:80],
            fromfile="on disk", tofile="regenerated", lineterm=""))
        raise AssertionError(
            "op_target_conformance.md is stale — regenerate via "
            "`python -m tessera.cli.conformance_matrix --render`\n\n" + diff
        )


def test_matrix_is_pure_aggregator():
    """The module must not import from arbitrary parts of the codebase —
    it can only aggregate from the four declared truth sources + stdlib.
    Catches a regression where the matrix grows its own private knowledge.
    """
    src = Path(cm.__file__).read_text()
    # Strip docstrings / comments so the regex match is on real imports only.
    src_no_strings = re.sub(r'"""[\s\S]*?"""', '', src)
    src_no_strings = re.sub(r"#[^\n]*", "", src_no_strings)
    # Collect both forms: `import X` and `from X import a, b`. For the second,
    # resolve each imported name to ``X.a`` / ``X.b`` so we check the actual
    # leaf module the matrix depends on, not just the package.
    bare_imports = re.findall(
        r"^import\s+([\w\.]+)", src_no_strings, flags=re.M)
    from_imports = re.findall(
        r"^from\s+([\w\.]+)\s+import\s+([\w\., ]+)",
        src_no_strings, flags=re.M)
    resolved: list[str] = list(bare_imports)
    for pkg, names in from_imports:
        # Drop ``as alias`` clauses + whitespace.
        for raw in names.split(","):
            leaf = raw.strip().split(" as ")[0].strip()
            if leaf:
                resolved.append(f"{pkg}.{leaf}" if pkg != "__future__"
                                else pkg)
    allowed_prefixes = (
        "tessera.compiler.primitive_coverage",
        "tessera.compiler.backend_manifest",
        "tessera.compiler.execution_matrix",
        "tessera.compiler.driver",
        # Audit recommendation B — named pipeline capability gates.
        # The conformance matrix consumes the gate evaluator to surface the
        # *first failing gate* per cell, but the gates module is itself a
        # pure aggregator (its own allowlist test enforces that).
        "tessera.compiler.pipeline_gates",
        # stdlib + typing
        "__future__",
        "dataclasses",
        "pathlib",
        "typing",
    )
    for mod in resolved:
        assert any(mod.startswith(p) for p in allowed_prefixes), (
            f"conformance_matrix.py is supposed to be a pure aggregator; "
            f"import {mod!r} not in allowed truth-source set")


def test_every_pair_has_a_cell():
    cells = cm.build_matrix()
    pairs = {(c.op, c.target) for c in cells}
    expected = {(o.name, t) for o in cm.CONFORMANCE_OPS
                for t in cm.CONFORMANCE_TARGETS}
    assert pairs == expected
    assert len(cells) == len(expected) == len(cm.CONFORMANCE_OPS) * len(
        cm.CONFORMANCE_TARGETS
    )


def test_cell_statuses_are_in_enum():
    enum = {
        cm.PROOF_COMPLETE, cm.PROOF_PARTIAL, cm.PROOF_ARTIFACT_ONLY,
        cm.PROOF_PLANNED, cm.PROOF_MISSING, cm.PROOF_NA,
    }
    for cell in cm.build_matrix():
        for axis in (cell.graph_emitted, cell.schedule_legal, cell.tile_legal,
                     cell.target_legal, cell.backend_compile,
                     cell.runtime_execute, cell.numerical_check):
            assert axis in enum, f"{cell.op}/{cell.target}: bad status {axis!r}"
        assert cell.overall in enum


def test_weakest_column_consistency():
    """Overall = weakest non-NA column. Spot-check a few cells."""
    for cell in cm.build_matrix():
        axes = [cell.graph_emitted, cell.schedule_legal, cell.tile_legal,
                cell.target_legal, cell.backend_compile,
                cell.runtime_execute, cell.numerical_check]
        non_na = [a for a in axes if a != cm.PROOF_NA]
        if not non_na:
            assert cell.overall == cm.PROOF_NA
        else:
            assert cell.overall in non_na


def test_matrix_produces_honest_signal():
    """Sanity: at least one ``complete`` cell (so the matrix isn't always
    pessimistic) AND at least one ``missing`` cell (so it isn't always
    optimistic). This is a regression guard against the matrix collapsing
    to a single uniform status."""
    cells = cm.build_matrix()
    statuses = {c.overall for c in cells}
    assert cm.PROOF_COMPLETE in statuses, (
        "no cell reports `complete` — matrix may be always-pessimistic"
    )
    assert cm.PROOF_MISSING in statuses, (
        "no cell reports `missing` — matrix may be always-optimistic"
    )


def test_apple_gpu_fused_chain_is_marked_fused():
    """matmul_softmax on apple_gpu should be recognized as a real fused
    single-kernel cell (not just composes), since the MSL fusion pass
    actually lives in the apple_gpu pipeline. Regression guard against
    losing the fused-chain modeling."""
    cell = next(c for c in cm.build_matrix()
                if c.op == "matmul_softmax" and c.target == "apple_gpu")
    assert any("fused" in n for n in cell.notes), cell.notes
    assert cell.backend_compile == cm.PROOF_COMPLETE
    assert cell.runtime_execute == cm.PROOF_COMPLETE


def test_compose_only_chain_is_marked_compose():
    """matmul_relu on every target should be marked as a compose, never a
    fuse — no backend ships a matmul+relu fusion pass today."""
    for cell in cm.build_matrix():
        if cell.op != "matmul_relu":
            continue
        assert any("composes" in n for n in cell.notes), (
            f"{cell.target}: matmul_relu should be marked compose-only, "
            f"got notes={cell.notes}"
        )


def test_host_x86_cpu_and_rocm_rows_are_closed():
    """Every in-scope host CPU and ROCm program has full proof.

    Multi-op rows may be sequential compositions; fusion is a performance
    property and must not demote end-to-end conformance when every component
    compiles, executes, and has a declared numerical fixture.
    """
    open_cells = [
        cell for cell in cm.build_matrix()
        if cell.target in {"cpu", "rocm"} and cell.overall != cm.PROOF_COMPLETE
    ]
    assert open_cells == []


def test_host_and_apple_rows_are_closed_without_a_failing_gate():
    """Completed target proofs cannot retain a host-environment failure."""
    in_scope = {"cpu", "apple_cpu", "apple_gpu", "rocm"}
    open_cells = [
        cell for cell in cm.build_matrix()
        if cell.target in in_scope and cell.overall != cm.PROOF_COMPLETE
    ]
    contradictory = [
        cell for cell in cm.build_matrix()
        if cell.target in in_scope and cell.first_failing_gate is not None
    ]
    assert open_cells == []
    assert contradictory == []
