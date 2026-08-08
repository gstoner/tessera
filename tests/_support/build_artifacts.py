"""Shared discovery for artifacts produced by an out-of-tree CMake build.

Tests must not assume that the active build is named ``build``.  CI, backend
release gates, and local multi-toolchain work routinely use directories such
as ``build-rocm`` or a temporary clean-build directory.  ``TESSERA_BUILD_DIR``
is the canonical explicit selector; when it is absent, an explicitly selected
``tessera-opt`` can identify its owning CMake root.

An explicit selector is fail-closed.  If it points at the wrong directory we
do not silently fall back to another build and accidentally validate a stale
artifact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


def _selected_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _cmake_root(path: Path) -> Path | None:
    """Return the closest ancestor containing ``CMakeCache.txt``."""

    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / "CMakeCache.txt").is_file():
            return candidate
    return None


def build_roots(*, defaults: Iterable[str] = ("build",)) -> tuple[Path, ...]:
    """Candidate CMake roots, with explicit selection and stable precedence."""

    if selected := _selected_path("TESSERA_BUILD_DIR"):
        return (selected,)

    roots: list[Path] = []
    for name in ("TESSERA_OPT", "TESSERA_OPT_BIN"):
        selected_tool = _selected_path(name)
        if selected_tool is not None and (root := _cmake_root(selected_tool)):
            roots.append(root)
            break
    roots.extend((REPO_ROOT / name).resolve() for name in defaults)

    unique: dict[Path, None] = {}
    for root in roots:
        unique.setdefault(root, None)
    return tuple(unique)


def built_artifact(
    relative: str | Path,
    *,
    env: Iterable[str] = (),
    defaults: Iterable[str] = ("build",),
) -> Path | None:
    """Locate one built artifact without crossing an explicit bad selector."""

    for name in env:
        selected = _selected_path(name)
        if selected is not None:
            return selected if selected.is_file() else None
    suffix = Path(relative)
    return next(
        (candidate for root in build_roots(defaults=defaults)
         if (candidate := root / suffix).is_file()),
        None,
    )
