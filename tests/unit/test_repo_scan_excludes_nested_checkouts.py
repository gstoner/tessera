"""Repo-wide audit scans must not walk into a nested checkout.

A `git worktree` inside the tree is a complete second copy of the repo, so
a plain `ROOT.rglob("*")` finds every file twice under two paths. That made
`test_ssa_buffer_ref_retirement::test_deprecated_buffer_ref_is_parser_only`
fail on one machine and nowhere else, and skewed a generated-doc reference
count on another — both traced back here.

The pruning is STRUCTURAL: a nested checkout carries its own `.git` entry
(a file for a worktree, a directory for a clone). Keying on that catches
one created anywhere; keying on the `.claude/` path that happened to bite
first would fix the instance and leave the class.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests._support.repo_scan import iter_repo_files


def _make_fake_checkout(directory: Path, *, as_worktree: bool) -> None:
    """A directory that looks like a nested worktree (file) or clone (dir)."""
    directory.mkdir(parents=True)
    (directory / "TileDialect.cpp").write_text("marker\n", encoding="utf-8")
    if as_worktree:
        (directory / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    else:
        (directory / ".git").mkdir()


@pytest.mark.parametrize("as_worktree", [True, False])
def test_a_nested_checkout_is_not_walked(tmp_path, as_worktree):
    """Both spellings of a nested checkout are pruned."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "TileDialect.cpp").write_text("marker\n", encoding="utf-8")
    _make_fake_checkout(tmp_path / "nested", as_worktree=as_worktree)

    found = {p.relative_to(tmp_path).as_posix()
             for p in iter_repo_files(tmp_path, suffixes={".cpp"})}
    assert found == {"src/TileDialect.cpp"}


def test_the_plain_walk_really_does_double_count(tmp_path):
    """Guards the test above from being vacuous.

    If `rglob` did not find the nested copy, pruning it would prove nothing.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "TileDialect.cpp").write_text("marker\n", encoding="utf-8")
    _make_fake_checkout(tmp_path / "nested", as_worktree=True)

    naive = {p.relative_to(tmp_path).as_posix()
             for p in tmp_path.rglob("*") if p.suffix == ".cpp"}
    assert len(naive) == 2, naive


def test_build_trees_and_caches_are_pruned(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.cpp").write_text("x\n", encoding="utf-8")
    for noise in ("build", "build-nvidia-cuda", "__pycache__", ".venv"):
        (tmp_path / noise).mkdir()
        (tmp_path / noise / "b.cpp").write_text("x\n", encoding="utf-8")

    found = {p.relative_to(tmp_path).as_posix()
             for p in iter_repo_files(tmp_path, suffixes={".cpp"})}
    assert found == {"src/a.cpp"}


def test_ordinary_directories_are_still_walked(tmp_path):
    """The exclusion list must stay small.

    An over-broad prune silently shrinks what an audit test checks, which is
    the same defect pointing the other way — a green scan that examined less
    than it claimed.
    """
    for name in ("src", "python", "tests", "docs", "archive"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "f.cpp").write_text("x\n", encoding="utf-8")

    found = {p.relative_to(tmp_path).parts[0]
             for p in iter_repo_files(tmp_path, suffixes={".cpp"})}
    assert found == {"src", "python", "tests", "docs", "archive"}


def test_the_real_repo_has_no_duplicate_source_paths():
    """End to end, against this checkout — including any live worktree."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():          # pragma: no cover
        pytest.skip("not a git checkout")

    worktrees = subprocess.run(
        ["git", "worktree", "list"], cwd=root, capture_output=True, text=True
    ).stdout.splitlines()

    seen: dict[str, Path] = {}
    for path in iter_repo_files(root, suffixes={".cpp", ".h", ".td"}):
        rel = path.relative_to(root).as_posix()
        assert rel not in seen, f"scanned twice: {rel}"
        seen[rel] = path
    assert seen, "the scan found nothing, which cannot be right"
    # If a worktree is live, this assertion is doing real work rather than
    # passing because there was nothing to trip over.
    if len(worktrees) > 1:
        assert not any(".claude/worktrees" in p for p in seen)
