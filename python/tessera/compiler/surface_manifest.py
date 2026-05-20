"""Shared dataclass + helpers backing the per-surface audit manifests.

Each "surface" — examples, benchmarks, research, tools — declares its
entries via a thin module that consumes :class:`SurfaceEntry` and
:data:`ALLOWED_STATUSES`.  The drift-gate pattern, render contract,
and filesystem audit are unified here so every surface tells the same
story.

The 6-element status taxonomy:

======================  ===========================================================
Status                  Meaning
======================  ===========================================================
``runnable``            Runs on default venv + CPU-only CI.
``runnable_optional``   Runs when declared ``extras_required`` are importable.
``compile_only``        Emits IR/artifacts but does not execute the workload.
``scaffold``            Intentionally illustrative; not runnable today.
``broken``              Expected to run but currently fails — followup needed.
``archived``            Intentionally retired; kept in-tree for reference only.
======================  ===========================================================

``archived`` is the new status added when the per-surface manifest
infrastructure landed (2026-05-19).  It distinguishes "we used to ship
this and have moved on" from ``scaffold`` ("we plan to ship this but
haven't yet") and ``broken`` ("we said this works and it doesn't").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Repo root, derived without importing tessera.
_REPO_ROOT = Path(__file__).resolve().parents[3]


ALLOWED_STATUSES: tuple[str, ...] = (
    "runnable",
    "runnable_optional",
    "compile_only",
    "scaffold",
    "broken",
    "archived",
)


# Statuses whose rows declare an executable command.
EXECUTABLE_STATUSES: frozenset[str] = frozenset(
    {"runnable", "runnable_optional", "compile_only"}
)

# Statuses that require a non-empty ``reason`` field.
REASON_REQUIRED_STATUSES: frozenset[str] = frozenset(
    {"scaffold", "broken", "archived"}
)


@dataclass(frozen=True)
class SurfaceEntry:
    """One row of a per-surface audit manifest.

    Attributes
    ----------
    directory:
        Path to the entry's directory (relative to repo root).  Audit
        rows are indexed per directory; most directories have one
        canonical entry point.
    entry_point:
        Path to the executable file (relative to repo root).
    status:
        One of :data:`ALLOWED_STATUSES`.
    command:
        Exact shell command to run from the repo root.  Required for
        executable statuses; ``None`` for ``scaffold``/``broken``/
        ``archived``.
    extras_required:
        Optional list of importable module names that gate the run.
        Required when status is ``runnable_optional``.
    reason:
        Human-readable explanation for ``scaffold``/``broken``/
        ``archived`` rows.  Empty otherwise.
    notes:
        Free-text supplementary notes surfaced in the generated doc.
    """

    directory: str
    entry_point: str
    status: str
    command: str | None = None
    extras_required: tuple[str, ...] = ()
    reason: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(
                f"SurfaceEntry({self.directory!r}): "
                f"status={self.status!r} not in {ALLOWED_STATUSES!r}"
            )
        if self.status in EXECUTABLE_STATUSES and not self.command:
            raise ValueError(
                f"SurfaceEntry({self.directory!r}): "
                f"status={self.status!r} requires a 'command' field"
            )
        if self.status in REASON_REQUIRED_STATUSES and not self.reason:
            raise ValueError(
                f"SurfaceEntry({self.directory!r}): "
                f"status={self.status!r} requires a non-empty 'reason'"
            )
        if self.status == "runnable_optional" and not self.extras_required:
            raise ValueError(
                f"SurfaceEntry({self.directory!r}): "
                f"status='runnable_optional' requires at least one "
                f"entry in 'extras_required'"
            )

    @property
    def directory_path(self) -> Path:
        return _REPO_ROOT / self.directory

    @property
    def entry_point_path(self) -> Path:
        return _REPO_ROOT / self.entry_point

    def resolve_extras_available(self) -> bool:
        """Return True iff every ``extras_required`` module is importable."""

        import importlib.util

        for mod in self.extras_required:
            if importlib.util.find_spec(mod) is None:
                return False
        return True


def status_counts(entries: Iterable[SurfaceEntry]) -> dict[str, int]:
    out = dict.fromkeys(ALLOWED_STATUSES, 0)
    for e in entries:
        out[e.status] = out[e.status] + 1
    return out


def audit_filesystem(
    entries: Iterable[SurfaceEntry],
    *,
    require_status_md_for: tuple[str, ...] = ("scaffold", "broken"),
) -> list[str]:
    """Return a list of structural issues with the manifest.

    Catches stale rows (declared entry file missing on disk) without
    actually executing anything.  Empty list means clean.

    ``require_status_md_for`` controls which statuses require a
    ``STATUS.md`` next to the directory.  ``archived`` directories
    don't need one by default since the row's ``reason`` is usually
    enough — but a surface can opt them in.
    """

    issues: list[str] = []
    for entry in entries:
        if not entry.directory_path.is_dir():
            issues.append(
                f"{entry.directory}: directory does not exist on disk"
            )
            continue
        if not entry.entry_point_path.exists():
            issues.append(
                f"{entry.directory}: entry_point {entry.entry_point!r} "
                f"does not exist on disk"
            )
        if entry.status in require_status_md_for:
            status_md = entry.directory_path / "STATUS.md"
            if not status_md.exists():
                issues.append(
                    f"{entry.directory}: status={entry.status!r} "
                    f"requires a STATUS.md but none was found"
                )
    return issues


def render_markdown(
    *,
    surface_title: str,
    surface_intro: str,
    entries: Iterable[SurfaceEntry],
    regenerate_command: str,
) -> str:
    """Render a ``<surface>_status.md`` dashboard for the given entries.

    The header carries the AUTO-GENERATED warning + a regeneration
    command so the drift gate's failure message can quote it back to
    the contributor.
    """

    rows = tuple(entries)
    counts = status_counts(rows)
    lines: list[str] = [
        "<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->",
        f"<!-- Regenerate via: {regenerate_command} -->",
        "",
        f"# {surface_title}",
        "",
        surface_intro,
        "",
        "## Status taxonomy",
        "",
        "| Status              | Meaning                                                       |",
        "|---------------------|---------------------------------------------------------------|",
        "| ``runnable``          | Runs on default venv + CPU-only CI.                            |",
        "| ``runnable_optional`` | Runs when declared ``extras_required`` are importable.         |",
        "| ``compile_only``      | Emits IR/artifacts but does not execute the workload.        |",
        "| ``scaffold``          | Intentionally illustrative; not runnable today.              |",
        "| ``broken``            | Expected to run, currently fails — followup needed.          |",
        "| ``archived``          | Intentionally retired; in-tree for reference only.           |",
        "",
        "## Counts",
        "",
        "| Status | Count |",
        "|--------|------:|",
    ]
    for status in ALLOWED_STATUSES:
        lines.append(f"| ``{status}`` | {counts[status]} |")
    lines.append(f"| **total** | **{len(rows)}** |")
    lines.append("")
    lines.append("## Entries")
    lines.append("")
    lines.append(
        "| Directory | Status | Entry point | Command / Reason |"
    )
    lines.append(
        "|-----------|--------|-------------|------------------|"
    )
    for entry in rows:
        if entry.status in REASON_REQUIRED_STATUSES:
            cell = entry.reason
        elif entry.status == "runnable_optional":
            extras = ", ".join(f"``{m}``" for m in entry.extras_required)
            cell = f"``{entry.command}``<br/>extras: {extras}"
        else:
            cell = f"``{entry.command}``"
        lines.append(
            f"| ``{entry.directory}`` | ``{entry.status}`` | "
            f"``{entry.entry_point}`` | {cell} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "ALLOWED_STATUSES",
    "EXECUTABLE_STATUSES",
    "REASON_REQUIRED_STATUSES",
    "SurfaceEntry",
    "audit_filesystem",
    "render_markdown",
    "status_counts",
]
