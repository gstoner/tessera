"""Unit guards for the examples surface audit.

These tests do **not** execute the example scripts (that's
``tessera.cli.examples_audit --check``'s job — a separate gate wired
into ``scripts/validate.sh``).  They lock the static contracts:

* The manifest is self-consistent: every entry resolves on disk,
  every ``scaffold``/``broken`` row ships a ``STATUS.md``.
* The generated dashboard ``docs/audit/generated/examples_status.md``
  matches ``render_markdown()`` — drift fails CI.
* The CLI surface (``tessera.cli.examples_audit``,
  ``tessera.cli.claim_lint``) parses and exposes the documented
  subcommands.
* ``claim_lint`` reports zero overclaim violations on the current
  README set.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from tessera.compiler.examples_manifest import (
    ALLOWED_STATUSES,
    ExampleEntry,
    all_entries,
    audit_filesystem,
    entries_by_status,
    render_markdown,
    status_counts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DOC = REPO_ROOT / "docs" / "audit" / "generated" / "examples_status.md"


class TestManifestShape:
    def test_status_taxonomy_is_5_elements(self) -> None:
        assert ALLOWED_STATUSES == (
            "runnable",
            "runnable_optional",
            "compile_only",
            "scaffold",
            "broken",
        )

    def test_every_entry_has_legal_status(self) -> None:
        for entry in all_entries():
            assert entry.status in ALLOWED_STATUSES, entry

    def test_runnable_entries_have_command(self) -> None:
        for entry in all_entries():
            if entry.status in ("runnable", "runnable_optional", "compile_only"):
                assert entry.command, (
                    f"{entry.directory} has status {entry.status!r} but no command"
                )

    def test_runnable_optional_has_extras(self) -> None:
        for entry in entries_by_status("runnable_optional"):
            assert entry.extras_required, (
                f"{entry.directory} is runnable_optional but declares no "
                f"extras_required"
            )

    def test_scaffold_and_broken_have_reasons(self) -> None:
        for entry in entries_by_status("scaffold") + entries_by_status("broken"):
            assert entry.reason, (
                f"{entry.directory} is {entry.status!r} but has no reason"
            )

    def test_no_duplicate_directories(self) -> None:
        seen: set[str] = set()
        for entry in all_entries():
            assert entry.directory not in seen, (
                f"duplicate manifest entry for {entry.directory}"
            )
            seen.add(entry.directory)


class TestManifestFilesystem:
    def test_audit_is_clean(self) -> None:
        issues = audit_filesystem()
        assert issues == [], "\n".join(issues)

    def test_every_directory_exists(self) -> None:
        for entry in all_entries():
            assert entry.directory_path.is_dir(), entry.directory

    def test_every_entry_point_exists(self) -> None:
        for entry in all_entries():
            assert entry.entry_point_path.exists(), entry.entry_point

    def test_scaffold_status_md_present(self) -> None:
        for entry in entries_by_status("scaffold"):
            status_md = entry.directory_path / "STATUS.md"
            assert status_md.exists(), (
                f"{entry.directory} is scaffold but missing STATUS.md"
            )

    def test_archive_not_in_manifest(self) -> None:
        """``examples/archive/**`` is intentionally out of scope."""

        for entry in all_entries():
            assert "examples/archive" not in entry.directory, entry


class TestGeneratedDashboardDriftGate:
    """Lock ``docs/audit/generated/examples_status.md`` against the manifest.

    Same pattern as ``test_standalone_compiler_roadmap``'s drift gate —
    the doc is regenerated from code, and CI fails if the
    checked-in copy diverges from the renderer.
    """

    def test_generated_doc_matches_render(self) -> None:
        if not GENERATED_DOC.exists():
            pytest.fail(
                f"missing {GENERATED_DOC.relative_to(REPO_ROOT)} — "
                "regenerate via "
                "`python -m tessera.cli.examples_audit --render`"
            )
        on_disk = GENERATED_DOC.read_text(encoding="utf-8")
        rendered = render_markdown()
        if on_disk != rendered:
            # Show a tight diff hint for the failure message.
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
                "examples_status.md is out of date with the manifest.\n"
                "Regenerate via "
                "`python -m tessera.cli.examples_audit --render`.\n\n"
                f"diff (truncated):\n{diff}"
            )

    def test_dashboard_lists_every_entry(self) -> None:
        text = GENERATED_DOC.read_text(encoding="utf-8")
        for entry in all_entries():
            assert f"``{entry.directory}``" in text, entry.directory

    def test_status_counts_appear_in_doc(self) -> None:
        text = GENERATED_DOC.read_text(encoding="utf-8")
        counts = status_counts()
        for status in ALLOWED_STATUSES:
            assert (
                f"``{status}`` | {counts[status]} |" in text
            ), f"missing count row for {status}"


class TestCLIs:
    def test_examples_audit_module_importable(self) -> None:
        m = importlib.import_module("tessera.cli.examples_audit")
        assert hasattr(m, "main")
        assert hasattr(m, "build_parser")

    def test_claim_lint_module_importable(self) -> None:
        m = importlib.import_module("tessera.cli.claim_lint")
        assert hasattr(m, "main")
        assert hasattr(m, "find_violations")

    def test_examples_audit_list_subcommand_runs(self) -> None:
        from tessera.cli.examples_audit import main
        rc = main(["list"])
        assert rc == 0

    def test_examples_audit_python_token_rewrite(self) -> None:
        from tessera.cli.examples_audit import _resolve_command
        import sys
        # The bare token gets rewritten...
        assert _resolve_command("python script.py").startswith(
            sys.executable
        )
        # ...but the inside of ``PYTHONPATH=python`` does not.
        rewritten = _resolve_command(
            "PYTHONPATH=python python script.py"
        )
        assert "PYTHONPATH=python " in rewritten
        assert sys.executable in rewritten

    def test_claim_lint_passes_today(self) -> None:
        from tessera.cli.claim_lint import find_violations
        violations = find_violations()
        if violations:
            descriptions = [
                f"{v.file.relative_to(REPO_ROOT)}:{v.line_no} "
                f"({v.description})"
                for v in violations
            ]
            pytest.fail(
                "claim_lint regressed; new overclaim language found:\n"
                + "\n".join(descriptions)
            )


class TestPlaceholderAssertTrueBan:
    """No ``assert True`` placeholder tests under examples/**/tests.

    The Jet-Nemotron incident (a literal ``assert True`` test masked a
    real import-time crash for weeks) is the motivating case.  Ban
    the pattern under example test directories so the next research
    scaffold can't pull the same trick.
    """

    def test_no_assert_true_in_examples_tests(self) -> None:
        examples_root = REPO_ROOT / "examples"
        offenders: list[str] = []
        # Restrict to examples/**/tests/**/test_*.py (and smoke_*.py)
        for path in examples_root.glob("**/tests/**/*.py"):
            if "archive" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(source.splitlines(), start=1):
                # Match only statements ``assert True`` (with optional
                # comment).  Documented references like
                # ``previous version was a literal ``assert True``
                # placeholder`` are inside a docstring/triple-quoted
                # string, so the line itself doesn't START with
                # ``assert True``.
                if re.match(r"^\s*assert\s+True\s*(#.*)?$", line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{line_no}")
        assert offenders == [], (
            "Placeholder `assert True` tests are banned under "
            "examples/**/tests/.  Replace with real coverage or a "
            "pytest.skip naming the missing piece.\n"
            + "\n".join(f"  - {o}" for o in offenders)
        )
