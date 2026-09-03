"""Walk repository sources without walking into things that are not sources.

Several audit tests scan from the repo root looking for a pattern across
`.cpp` / `.h` / `.td` / `.py` files. A plain `ROOT.rglob("*")` also
descends into a **nested git worktree** — a complete second copy of the
repo — so every file is found twice, under two paths.

That is not hypothetical. `git worktree add` under `.claude/worktrees/`
(what a background task does) made
`test_ssa_buffer_ref_retirement::test_deprecated_buffer_ref_is_parser_only`
fail on one machine and nowhere else, because it compares a set of PATHS
and a duplicate path is a new element. Tests that collect content into a
set never noticed, which is why this survived: the bug is invisible until
a scan's result happens to be path-valued.

**Worktrees are detected structurally, not by name.** A checkout nested
inside the tree carries its own `.git` entry — a file for a worktree, a
directory for a clone — so pruning on that catches one created anywhere,
not just under the `.claude/` path that happened to bite first. Hardcoding
that path would fix the instance and leave the class.

The remaining exclusions are directories that are by definition not source:
build trees, virtualenvs, caches, and generated indexes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

#: Directory names that never hold repository sources. Kept small on
#: purpose: an over-broad list silently shrinks what an audit test checks,
#: which is the same failure mode in the other direction.
_NON_SOURCE_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "node_modules", "graphify-out", ".codegraph",
})


def _is_nested_checkout(path: Path) -> bool:
    """True if `path` is a git worktree or clone nested inside the tree."""
    return (path / ".git").exists()


def iter_repo_files(
    root: Path,
    *,
    suffixes: Iterable[str] | None = None,
    skip_nested_checkouts: bool = True,
) -> Iterator[Path]:
    """Yield files under `root`, skipping non-source and nested-checkout trees.

    `suffixes` filters by extension (e.g. `{".cpp", ".h"}`); `None` yields
    every file.
    """
    wanted = frozenset(suffixes) if suffixes is not None else None
    stack = [Path(root)]
    while stack:
        directory = stack.pop()
        try:
            entries = list(directory.iterdir())
        except (PermissionError, FileNotFoundError):  # pragma: no cover
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name in _NON_SOURCE_DIRS or entry.name.startswith("build"):
                    continue
                if skip_nested_checkouts and _is_nested_checkout(entry):
                    continue
                stack.append(entry)
            elif wanted is None or entry.suffix in wanted:
                yield entry
