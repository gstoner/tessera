"""Audit follow-up A.3 — numerical_check column via
``BackendKernelEntry.execute_compare_fixture``.

The conformance matrix used to rely on a filename/content scan of
``tests/unit/`` to populate the ``numerical_check`` column — brittle (a
file could mention an op + target tokens without actually testing
them) and ad-hoc. A.3 replaces that with first-class manifest data:

* ``backend_manifest._NUMERICAL_FIXTURES`` maps ``(op, target)`` to a
  repo-relative test file path.
* ``manifest_for(op)`` post-processes entries to attach
  ``execute_compare_fixture`` from this map.
* ``conformance_matrix._numerical_check_present`` accepts only exact-target
  manifest evidence; the legacy keyword scan is not proof.

These tests pin the contract:

1. Every fixture path in ``_NUMERICAL_FIXTURES`` must actually exist
   (drift gate — a stale fixture path is worse than no fixture).
2. ``manifest_for(op)`` attaches the right fixture to the right
   target — not all rows.
3. ``conformance_matrix._numerical_check_present`` resolves to True only when
   the manifest declares a fixture for that exact architecture.
4. Evidence from one architecture cannot leak into a sibling target row.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera.compiler import backend_manifest as bm
from tessera.compiler import conformance_matrix as cm


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---- fixture map integrity ----

def test_every_declared_fixture_actually_exists():
    """Each fixture path in ``_NUMERICAL_FIXTURES`` must resolve to a
    real file under tests/unit/. Catches a regression where a fixture
    is renamed/deleted but the manifest still claims it."""
    missing: list[tuple[tuple[str, str], str]] = []
    for key, rel in bm._NUMERICAL_FIXTURES.items():
        if not (_REPO_ROOT / rel).is_file():
            missing.append((key, rel))
    assert not missing, (
        f"manifest declares fixtures that don't exist on disk: {missing}")


def test_every_declared_fixture_is_under_tests_unit():
    """Sanity: fixtures should live in tests/unit/, not somewhere
    else where they wouldn't be exercised by the regular test sweep."""
    for key, rel in bm._NUMERICAL_FIXTURES.items():
        assert rel.startswith("tests/unit/"), (
            f"fixture {key} → {rel!r} should be under tests/unit/")


# ---- manifest_for attaches the fixture ----

def test_manifest_for_attaches_fixture_to_apple_gpu_matmul():
    rows = [r for r in bm.manifest_for("matmul") if r.target == "apple_gpu"]
    assert rows, "matmul should have an apple_gpu manifest row"
    assert rows[0].execute_compare_fixture is not None
    assert rows[0].execute_compare_fixture.endswith(".py")


def test_manifest_for_attaches_fixture_only_to_matching_target():
    """matmul has many manifest rows; a fixture must land on exactly the
    target the map declares — ``cpu`` and (consumer-Blackwell bring-up,
    2026-06-25) ``nvidia_sm120`` — and not bleed onto sibling arches."""
    rows = bm.manifest_for("matmul")
    by_target = {r.target: r for r in rows}
    assert by_target["cpu"].execute_compare_fixture == (
        "tests/unit/test_end_to_end_matmul_cpu_path.py")
    # nvidia_sm120 now ships a shipped runtime symbol + execute-compare fixture.
    assert by_target["nvidia_sm120"].execute_compare_fixture == (
        "tests/unit/test_nvidia_mma_runtime_symbol.py")
    # The other NVIDIA arches are still artifact_only — the fixture must NOT
    # falsely attach to them.
    for t in ("nvidia_sm80", "nvidia_sm90", "nvidia_sm100"):
        assert by_target[t].execute_compare_fixture is None, (
            f"{t} should not have a fixture; got "
            f"{by_target[t].execute_compare_fixture!r}")


def test_apple_gpu_relu_has_manifest_fixture_after_a1():
    """A.1 added the relu/conv2d/kv_cache_read apple_gpu rows; A.3
    attaches fixtures to ``relu`` and ``conv2d``. Lock the fixture is
    present so the dashboard cell flips from heuristic-driven to
    manifest-driven."""
    rows = [r for r in bm.manifest_for("relu") if r.target == "apple_gpu"]
    assert rows
    assert rows[0].execute_compare_fixture is not None


# ---- conformance_matrix prefers the manifest ----

def test_numerical_check_returns_true_when_manifest_declares_fixture():
    """``conformance_matrix._numerical_check_present`` must consult the
    manifest FIRST. Verify by checking a cell that has a manifest
    fixture: the result is True without any keyword fallback."""
    # matmul + cpu has a manifest fixture.
    assert cm._numerical_check_present("matmul", "cpu") is True


def test_numerical_check_returns_true_for_apple_gpu_cells_with_fixtures():
    """Every (op, target) in the fixture map for apple_gpu must report
    numerical_check=True via the manifest path."""
    apple_gpu_ops = [op for (op, t) in bm._NUMERICAL_FIXTURES if t == "apple_gpu"]
    for op in apple_gpu_ops:
        assert cm._numerical_check_present(op, "apple_gpu") is True, (
            f"manifest declares a fixture for ({op}, apple_gpu) — "
            f"numerical_check must report True")


def test_numerical_check_is_architecture_aligned_without_heuristics():
    """sm_120 proof must not leak into other NVIDIA architecture rows."""
    assert cm._numerical_check_present("matmul", "nvidia_sm120") is True
    assert cm._numerical_check_present("matmul", "nvidia_sm90") is False


# ---- the dashboard counts stay sensible after A.1 + A.3 ----

def test_status_summary_has_complete_and_missing_cells():
    """The dashboard must continue to surface both confirmed-passing
    and confirmed-missing cells — regression guard against A.3
    accidentally flipping everything to one side."""
    counts = cm.status_summary()
    assert counts["complete"] > 0
    assert counts["missing"] > 0


def test_surfaced_upstream_gaps_section_is_empty_after_a1():
    """A.1 closed the only known runtime-envelope-vs-manifest gap
    (relu / conv2d / kv_cache_read on apple_gpu). The dashboard's
    "Surfaced upstream gaps" generator must therefore find zero rows."""
    cells = cm.build_matrix()
    gaps = cm._surfaced_upstream_gaps(cells)
    assert gaps == [], (
        f"expected no surfaced upstream gaps after A.1; got {gaps}")
