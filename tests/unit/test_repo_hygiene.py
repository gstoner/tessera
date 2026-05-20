"""Repo hygiene guards — no tracked build artifacts under tools/.

Catches the 2026-05-19 incident where
``tools/CLI/Tessera_CLI_Starter_v0_1/_build/`` shipped 131 tracked
files including absolute local paths in ``CMakeCache.txt`` and
``CTestTestfile.cmake``.

The .gitignore now covers ``**/_build/`` globally; this test makes
the rule load-bearing so a future re-introduction fails CI rather
than silently re-polluting the repo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _tracked_files() -> list[str]:
    """Return every file currently tracked in git, repo-relative."""

    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git not available or repo not initialized")
        return []  # unreachable but quiets mypy
    return [line for line in proc.stdout.splitlines() if line]


class TestNoTrackedBuildArtifacts:
    """No build / CMake staging directories may be tracked."""

    def test_no_tracked_build_under_tools(self) -> None:
        offenders = [
            f for f in _tracked_files()
            if "/_build/" in f and f.startswith("tools/")
        ]
        assert offenders == [], (
            "Build artifacts must not be tracked under tools/**/_build/. "
            "Untrack via `git rm -r --cached <path>` and ensure "
            "`.gitignore` carries `**/_build/`.\nOffenders:\n"
            + "\n".join(f"  - {f}" for f in offenders[:20])
            + (f"\n  ... and {len(offenders) - 20} more" if len(offenders) > 20 else "")
        )

    def test_no_tracked_cmake_cache_or_ctest_testfile(self) -> None:
        """Two specific files are always machine-local."""

        forbidden = ("CMakeCache.txt", "CTestTestfile.cmake")
        offenders = [
            f for f in _tracked_files()
            if Path(f).name in forbidden
        ]
        assert offenders == [], (
            "CMakeCache.txt / CTestTestfile.cmake carry absolute "
            "machine-local paths and must never be tracked.\n"
            "Offenders:\n" + "\n".join(f"  - {f}" for f in offenders)
        )

    def test_gitignore_covers_build_globs(self) -> None:
        gitignore = REPO_ROOT / ".gitignore"
        text = gitignore.read_text(encoding="utf-8")
        # The two patterns that keep the tools/ scaffolds and top-level
        # build trees out of git.
        required_patterns = ("**/_build/", "build/")
        missing = [p for p in required_patterns if p not in text]
        assert missing == [], (
            f".gitignore missing required pattern(s): {missing!r}. "
            f"These keep machine-local build trees out of the repo."
        )
