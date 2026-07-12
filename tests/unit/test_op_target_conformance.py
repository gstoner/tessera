"""Drift gate + structural guards for the op×target conformance matrix.

The dashboard at ``docs/audit/op_target_conformance.md`` is generated from
``python/tessera/compiler/conformance_matrix.py``. This test suite locks four
properties so the dashboard stays a real audit surface:

1. **Drift gate** — the on-disk dashboard must match the freshly-rendered one.
   If you change the matrix logic or any upstream truth source, regenerate via
   ``python -m tessera.cli.conformance_matrix --render``.
2. **Evidence sources** — IR rungs come from actual typed lowering/verifiers;
   backend/runtime/numerical rungs come from their canonical registries.
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
        "dashboard missing — regenerate via "
        "`python -m tessera.cli.conformance_matrix --render`")
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


def test_matrix_uses_only_declared_evidence_sources():
    """Keep every proof rung tied to an inspectable compiler authority."""
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
        "tessera.compiler.conformance_evaluator",
        "tessera.compiler.schedule_ir",
        "tessera.compiler.tile_ir",
        "tessera.compiler.target_ir",
        "functools",
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
        cm.PROOF_COMPLETE, cm.PROOF_REFERENCE, cm.PROOF_COMPILEABLE,
        cm.PROOF_PARTIAL, cm.PROOF_ARTIFACT_ONLY,
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


def test_rows_use_exact_target_grain():
    assert "nvidia" not in cm.CONFORMANCE_TARGETS
    assert {"nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm120"} \
        <= set(cm.CONFORMANCE_TARGETS)
    assert {"cpu", "x86"} <= set(cm.CONFORMANCE_TARGETS)


def test_ir_columns_are_derived_from_emitted_verified_ir():
    matmul = cm._ir_proof("matmul", "cpu")
    assert (matmul.graph_emitted, matmul.schedule_legal,
            matmul.tile_legal, matmul.target_legal) == (
        cm.PROOF_COMPLETE,
        cm.PROOF_COMPLETE,
        cm.PROOF_COMPLETE,
        cm.PROOF_COMPLETE,
    )
    stateful = cm._ir_proof("kv_cache_read", "apple_gpu")
    assert (stateful.graph_emitted, stateful.schedule_legal,
            stateful.tile_legal, stateful.target_legal) == (
        cm.PROOF_COMPLETE,
        cm.PROOF_COMPLETE,
        cm.PROOF_COMPLETE,
        cm.PROOF_COMPLETE,
    )


def test_backend_compile_keeps_reference_and_compileable_distinct():
    assert cm._proof_status_from_backend_compile(
        ["reference"], "softmax", "apple_cpu"
    ) == cm.PROOF_REFERENCE
    assert cm._proof_status_from_backend_compile(
        ["compileable"], "softmax", "nvidia_sm90"
    ) == cm.PROOF_COMPILEABLE


def test_runtime_and_numerical_proof_are_exact_target_specific():
    rows = {(c.op, c.target): c for c in cm.build_matrix()}
    assert rows[("matmul", "nvidia_sm120")].runtime_execute == cm.PROOF_COMPLETE
    assert rows[("matmul", "nvidia_sm120")].numerical_check == cm.PROOF_COMPLETE
    assert rows[("matmul", "nvidia_sm90")].runtime_execute == cm.PROOF_MISSING
    assert rows[("matmul", "nvidia_sm90")].numerical_check == cm.PROOF_MISSING


def test_composite_first_failure_checks_every_component():
    cell = next(c for c in cm.build_matrix()
                if c.op == "matmul_relu" and c.target == "nvidia_sm120")
    assert cell.first_failing_gate == "backend_compile"
    assert cell.backend_compile == cm.PROOF_MISSING
    assert "matmul,relu" in cell.first_failing_gate_detail


def test_complete_cells_have_no_failure_and_open_cells_name_first_rung():
    for cell in cm.build_matrix():
        if cell.overall == cm.PROOF_COMPLETE:
            assert cell.first_failing_gate is None
        else:
            assert cell.first_failing_gate in {
                "graph_emitted", "schedule_legal", "tile_legal",
                "target_legal", "backend_compile", "runtime_execute",
                "numerical_check",
            }
