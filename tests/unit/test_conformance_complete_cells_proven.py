"""P0 gate — "claimed-complete must be proven" (fixture-driven proof).

The audit's P0: every op×target conformance cell that reaches
``overall == "complete"`` must have its ``numerical_check`` backed by a
**manifest-declared ``execute_compare_fixture``** — a real execute-and-compare
test on disk — and NOT the legacy filename/keyword heuristic. The heuristic is a
soft signal that keeps circumstantially-covered cells from regressing to
``missing``; it may grant at most ``partial`` and can never, on its own, justify a
``complete`` claim.

This module is the structural gate (runtime-free, runs in CI everywhere):

  1. Every ``complete`` cell's numerical proof source is ``"fixture"``.
  2. The declared fixture file exists on disk.
  3. The fixture genuinely execute-compares: it references a component op AND a
     numerical comparison (``assert_allclose`` / ``allclose``).
4. Generator invariant: keyword/filename heuristics are not numerical proof.

The matching **execution** proof — actually running each fixture — is
``python -m tessera.cli.conformance_matrix --verify-fixtures`` (GPU-dependent;
run locally / on Apple-silicon CI). Here we prove the *claim structure*; that
command proves the *numbers*.
"""

from pathlib import Path

import pytest

from tessera.compiler import conformance_matrix as C

_REPO = Path(__file__).resolve().parents[2]
_CELLS = C.build_matrix()
_COMPLETE = [c for c in _CELLS if c.overall == "complete"]
_COMPONENTS = {op.name: op.component_ops for op in C.CONFORMANCE_OPS}


def test_there_is_at_least_one_complete_cell():
    """Sanity: the gate is vacuous if nothing is complete. (Apple matmul /
    softmax / matmul_softmax / flash_attn cells are complete on this tree.)"""
    assert _COMPLETE, "no complete conformance cells — gate would be vacuous"


@pytest.mark.parametrize("cell", _COMPLETE, ids=lambda c: f"{c.op}-{c.target}")
def test_complete_cell_is_fixture_proven_not_heuristic(cell):
    """(1) Every complete cell rests on a real fixture, never the heuristic."""
    source = C._numerical_proof_source(cell.op, cell.target)
    assert source == "fixture", (
        f"{cell.op}/{cell.target} is overall=complete but its numerical_check "
        f"proof source is {source!r}, not a declared execute_compare_fixture. "
        f"A complete claim must be backed by a real execute-and-compare test "
        f"(add one to backend_manifest._NUMERICAL_FIXTURES), not the keyword "
        f"heuristic.")


@pytest.mark.parametrize("cell", _COMPLETE, ids=lambda c: f"{c.op}-{c.target}")
def test_complete_cell_fixture_exists_and_execute_compares(cell):
    """(2)+(3) The declared fixture exists and genuinely execute-compares."""
    from tessera.compiler import backend_manifest as bm

    rel = bm._NUMERICAL_FIXTURES.get((cell.op, cell.target))
    # Fall back to a manifest-entry-carried fixture (per-arch rows).
    if rel is None:
        for e in C._manifest_for_target(cell.op, cell.target):
            if e.execute_compare_fixture:
                rel = e.execute_compare_fixture
                break
    assert rel, f"{cell.op}/{cell.target}: complete but no declared fixture path"

    path = _REPO / rel
    assert path.is_file(), f"declared fixture does not exist: {rel}"

    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    assert any(token in text for token in (
        "assert_allclose", "allclose", "maxerr", "max_abs_err",
    )), (
        f"fixture {rel} for {cell.op}/{cell.target} has no numerical "
        f"comparison (allclose or explicit error bound) — it is not an "
        f"execute-compare")

    # The fixture must mention at least one component op of the row, so a
    # mis-declared path (points at an unrelated test) is caught.
    comps = _COMPONENTS.get(cell.op, (cell.op,))
    op_tokens = set(comps) | {cell.op} | {t for c in comps for t in c.split("_")}
    assert any(tok in text for tok in op_tokens), (
        f"fixture {rel} for {cell.op}/{cell.target} mentions none of "
        f"{sorted(op_tokens)} — likely a mis-declared fixture")


def test_keyword_heuristics_are_not_numerical_proof():
    """(4) Only an exact-target declared fixture can satisfy the column."""
    assert C._numerical_proof_source("softmax", "nvidia_sm90") is None


def test_complete_cells_match_dashboard_csv():
    """The in-memory complete set matches the checked-in dashboard CSV, so the
    gate and the published dashboard can't silently diverge."""
    import csv

    csv_path = _REPO / "docs" / "audit" / "op_target_conformance.csv"
    rows = list(csv.DictReader(csv_path.open()))
    csv_complete = {(r["op"], r["target"]) for r in rows if r["overall"] == "complete"}
    mem_complete = {(c.op, c.target) for c in _COMPLETE}
    assert csv_complete == mem_complete, (
        f"dashboard CSV complete set {csv_complete} != in-memory {mem_complete} "
        f"— regenerate with `python -m tessera.cli.conformance_matrix --render`")
