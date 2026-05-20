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


SURFACES = ("examples", "benchmarks", "research", "tools")


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
    def test_no_duplicate_directories(self, surface: str) -> None:
        mod = _manifest(surface)
        seen: set[str] = set()
        for entry in mod.all_entries():
            assert entry.directory not in seen, (
                f"{surface}: duplicate manifest row for "
                f"{entry.directory}"
            )
            seen.add(entry.directory)


class TestGeneratedDashboardDriftGate:
    """Lock each generated ``<surface>_status.md`` against the manifest."""

    @pytest.mark.parametrize(
        "surface,doc_name",
        [
            ("examples", "examples_status.md"),
            ("benchmarks", "benchmarks_status.md"),
            ("research", "research_status.md"),
            ("tools", "tools_status.md"),
        ],
    )
    def test_generated_doc_matches_render(
        self, surface: str, doc_name: str
    ) -> None:
        doc_path = GENERATED_DIR / doc_name
        mod = _manifest(surface)
        if not doc_path.exists():
            pytest.fail(
                f"missing {doc_path.relative_to(REPO_ROOT)} — "
                f"regenerate via "
                f"`python -m tessera.cli.surface_audit "
                f"--surface={surface} --render`"
            )
        on_disk = doc_path.read_text(encoding="utf-8")
        rendered = mod.render_markdown()
        if on_disk != rendered:
            from difflib import unified_diff
            diff = "\n".join(
                list(
                    unified_diff(
                        on_disk.splitlines()[:80],
                        rendered.splitlines()[:80],
                        lineterm="",
                        fromfile="on_disk",
                        tofile="render_markdown()",
                    )
                )[:40]
            )
            pytest.fail(
                f"{doc_name} is out of date with "
                f"tessera.compiler.{surface}_manifest.  Regenerate "
                f"via `python -m tessera.cli.surface_audit "
                f"--surface={surface} --render`.\n\ndiff (truncated):\n"
                f"{diff}"
            )


class TestSurfaceAuditCLI:
    def test_cli_module_importable(self) -> None:
        m = importlib.import_module("tessera.cli.surface_audit")
        assert hasattr(m, "main")
        assert hasattr(m, "build_parser")
        assert hasattr(m, "SURFACES")
        assert m.SURFACES == ("examples", "benchmarks", "research", "tools")

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
