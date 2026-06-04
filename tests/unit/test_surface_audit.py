"""Unit guards for the four per-surface manifests + their drift docs.

Same pattern as ``tests/unit/test_examples_audit.py`` but extended
to ``benchmarks`` / ``research`` / ``tools`` and locking the drift
gate for each generated dashboard.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from tessera.compiler.surface_manifest import (
    ALLOWED_STATUSES,
    REASON_REQUIRED_STATUSES,
    SurfaceEntry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = REPO_ROOT / "docs" / "audit" / "generated"


SURFACES = ("examples", "benchmarks", "research", "tools", "tests")


def _manifest(surface: str):
    return importlib.import_module(f"tessera.compiler.{surface}_manifest")


class TestSharedSurfaceManifest:
    def test_status_taxonomy_has_archived(self) -> None:
        assert "archived" in ALLOWED_STATUSES

    def test_executable_statuses_are_three(self) -> None:
        from tessera.compiler.surface_manifest import EXECUTABLE_STATUSES
        assert EXECUTABLE_STATUSES == frozenset(
            {"runnable", "runnable_optional", "compile_only"}
        )

    def test_reason_required_statuses(self) -> None:
        assert REASON_REQUIRED_STATUSES == frozenset(
            {"scaffold", "broken", "archived"}
        )


class TestPerSurfaceManifests:
    @pytest.mark.parametrize("surface", SURFACES)
    def test_manifest_loads(self, surface: str) -> None:
        mod = _manifest(surface)
        entries = mod.all_entries()
        assert isinstance(entries, tuple)
        assert all(isinstance(e, SurfaceEntry) for e in entries)
        assert len(entries) > 0, (
            f"{surface} manifest is empty — surfaces must declare at "
            f"least one row"
        )

    @pytest.mark.parametrize("surface", SURFACES)
    def test_filesystem_audit_clean(self, surface: str) -> None:
        mod = _manifest(surface)
        issues = mod.audit_filesystem()
        assert issues == [], "\n".join(issues)

    @pytest.mark.parametrize("surface", SURFACES)
    def test_status_counts_match_entries(self, surface: str) -> None:
        mod = _manifest(surface)
        counts = mod.status_counts()
        total = sum(counts.values())
        assert total == len(mod.all_entries())

    @pytest.mark.parametrize("surface", SURFACES)
    def test_no_duplicate_entry_points(self, surface: str) -> None:
        mod = _manifest(surface)
        seen: set[str] = set()
        for entry in mod.all_entries():
            assert entry.entry_point not in seen, (
                f"{surface}: duplicate manifest row for entry point "
                f"{entry.entry_point}"
            )
            seen.add(entry.entry_point)


class TestGeneratedDashboardDriftGate:
    """The five per-surface ``*_status.md`` docs + operator-benchmark
    coverage were consolidated (2026-06-04) into one ``surface_status``
    dashboard managed by the generated-doc registry.  Drift is gated
    there (CSV-canonical); here we only assert the consolidated doc is
    in sync, so the per-manifest render stays exercised end-to-end."""

    def test_surface_status_in_sync(self) -> None:
        from tessera.compiler import generated_docs as gd

        msg = gd.check(gd.get("surface_status"))
        assert msg is None, msg

    def test_surface_status_covers_every_surface(self) -> None:
        from tessera.compiler import generated_docs as gd

        csv = gd.get("surface_status").render_csv()
        for surface in SURFACES:
            assert f"\n{surface}," in csv, (
                f"surface {surface!r} has no rows in surface_status.csv"
            )


class TestHistoricalBreadcrumbStatusMd:
    """A ``runnable`` row may *optionally* ship a ``STATUS.md`` as a
    historical breadcrumb (e.g., documenting a prior
    ``broken`` → ``runnable`` transition).  The audit must tolerate
    these — they're useful narrative for anyone reading the directory
    standalone.

    Canonical example: ``tools/roofline_tools/STATUS.md``, kept as
    a breadcrumb after the 2026-05-19 repair pass closed the original
    ``ImportError`` for ``tprof_roofline.model.analyze``.
    """

    def test_runnable_with_status_md_does_not_trigger_audit_issue(self) -> None:
        """The shared ``audit_filesystem`` walker only *requires*
        STATUS.md for ``scaffold`` / ``broken`` (by default).  A
        runnable row with a STATUS.md is allowed and must not be
        flagged."""

        import tempfile
        import os
        from tessera.compiler.surface_manifest import (
            SurfaceEntry,
            audit_filesystem,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # Build a runnable row whose directory + entry point + a
            # STATUS.md breadcrumb all exist on disk.
            dir_path = Path(tmp) / "demo_runnable_with_breadcrumb"
            dir_path.mkdir()
            entry_path = dir_path / "entry.py"
            entry_path.write_text("# breadcrumb demo")
            (dir_path / "STATUS.md").write_text(
                "# Status: `runnable`\n\n"
                "## Historical note\n\n"
                "This row used to be broken; repaired YYYY-MM-DD.\n"
            )
            rel_dir = str(dir_path.relative_to(REPO_ROOT)) \
                if str(dir_path).startswith(str(REPO_ROOT)) \
                else os.path.relpath(dir_path, REPO_ROOT)
            # ``audit_filesystem`` resolves paths against the repo
            # root, so we feed an entry that does so deliberately.
            entry = SurfaceEntry(
                directory=rel_dir,
                entry_point=str(Path(rel_dir) / "entry.py"),
                status="runnable",
                command="python -c \"pass\"",
            )
            issues = audit_filesystem((entry,))
            assert issues == [], (
                "A runnable row with a STATUS.md breadcrumb must not "
                "trigger an audit issue:\n" + "\n".join(issues)
            )

    def test_roofline_tools_keeps_its_breadcrumb(self) -> None:
        """Lock the canonical historical-breadcrumb example: the
        roofline_tools STATUS.md must continue to exist + announce
        the prior failure mode so future audits have context."""

        status_md = REPO_ROOT / "tools" / "roofline_tools" / "STATUS.md"
        assert status_md.is_file(), (
            f"missing historical-breadcrumb STATUS.md at {status_md}"
        )
        text = status_md.read_text(encoding="utf-8")
        # The breadcrumb must say what the *current* status is AND
        # carry a "Historical" / "Previously" / "Before the" marker
        # so it can't be mistaken for stale documentation.
        assert "runnable" in text, (
            "roofline_tools STATUS.md must declare the current "
            "status (`runnable`)"
        )
        history_markers = ("Historical", "Previously", "Before the")
        assert any(m in text for m in history_markers), (
            f"roofline_tools STATUS.md must carry one of the "
            f"history markers {history_markers!r} so it's clearly a "
            f"breadcrumb, not stale current-state documentation"
        )


class TestSurfaceAuditCLI:
    def test_cli_module_importable(self) -> None:
        m = importlib.import_module("tessera.cli.surface_audit")
        assert hasattr(m, "main")
        assert hasattr(m, "build_parser")
        assert hasattr(m, "SURFACES")
        assert m.SURFACES == ("examples", "benchmarks", "research", "tools", "tests")

    def test_command_chain_splits_on_double_ampersand(self) -> None:
        from tessera.cli.surface_audit import _run_entry
        from tessera.compiler.surface_manifest import SurfaceEntry

        # A compile_only smoke that chains two `python -c` steps; both
        # must succeed for the audit to call the entry passing.
        entry = SurfaceEntry(
            directory="tools/CLI/Tessera_CLI_Starter_v0_1",
            entry_point="tools/CLI/Tessera_CLI_Starter_v0_1/CMakeLists.txt",
            status="compile_only",
            command=(
                "python -c \"print('step1')\" && "
                "python -c \"print('step2')\""
            ),
        )
        ok, tail = _run_entry(entry, timeout=30)
        assert ok, f"chained command failed: {tail}"
        assert "step2" in tail or "step1" in tail


class TestExtendedClaimLint:
    def test_runs_all_surfaces_by_default(self) -> None:
        from tessera.cli.claim_lint import find_violations
        violations = find_violations()
        if violations:
            descriptions = [
                f"[{v.surface}] {v.file.relative_to(REPO_ROOT)}:"
                f"{v.line_no} ({v.description})"
                for v in violations
            ]
            pytest.fail(
                "Multi-surface claim_lint regressed:\n"
                + "\n".join(descriptions)
            )

    @pytest.mark.parametrize("surface", SURFACES)
    def test_per_surface_subset_clean(self, surface: str) -> None:
        from tessera.cli.claim_lint import find_violations
        violations = find_violations(surface=surface)
        if violations:
            descriptions = [
                f"{v.file.relative_to(REPO_ROOT)}:"
                f"{v.line_no} ({v.description})"
                for v in violations
            ]
            pytest.fail(
                f"{surface} claim_lint regressed:\n"
                + "\n".join(descriptions)
            )
