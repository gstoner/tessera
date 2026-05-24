"""Audit-A (2026-05-22) — Documentation freshness manifest.

Catalogues every documentation file under the canonical docs tree
(``docs/spec/``, ``docs/guides/``, ``docs/programming_guide/``,
``docs/operations/``, ``docs/architecture/``, ``docs/reference/``)
and tags each with:

  * Last-updated date parsed from YAML frontmatter
    (``last_updated: YYYY-MM-DD``) or document body (e.g.,
    ``Last updated: 2026-05-22``).
  * Days-stale relative to a reference date.
  * Whether the doc declares a ``status`` field (Normative /
    Informative / Historical / etc.) and an ``authority`` line.
  * Optional ``reconciliation_target`` — the live data source whose
    values the doc claims to reflect (e.g., ``primitive_coverage``
    for ``Tessera_Standard_Operations.md``).

The dashboard at ``docs/audit/generated/docs_freshness.md`` is
generated from this manifest.  Drift gates at
``tests/unit/test_docs_freshness.py`` lock:

  * Every doc in the canonical tree has either real frontmatter or
    is explicitly listed as exempt.
  * No "Normative" doc is more than 180 days stale (the TSOL spec
    was 24 days stale and that surfaced bugs — 180 is the soft
    floor; halve to 90 once the system catches up).
  * Reconciliation claims (numeric counts in prose) match the live
    registry values where the registry is available.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


# ─────────────────────────────────────────────────────────────────────────
# Canonical doc tree roots
# ─────────────────────────────────────────────────────────────────────────


_DOC_ROOTS: tuple[str, ...] = (
    "docs/spec",
    "docs/guides",
    "docs/programming_guide",
    "docs/operations",
    "docs/architecture",
    "docs/reference",
)


# Docs intentionally exempt from the freshness gate (e.g., legacy
# pointer pages, draft-folders, or auto-generated rollups whose
# frontmatter doesn't belong on them).  Keep this set small; every
# addition needs a justification comment.
_EXEMPT_DOCS: frozenset[str] = frozenset({
    # Auto-generated dashboards live under docs/audit/generated/ and
    # are excluded from the tree above already.  Listed here as a
    # placeholder for future exemptions.
})


# ─────────────────────────────────────────────────────────────────────────
# Frontmatter + body parsing
# ─────────────────────────────────────────────────────────────────────────


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL
)
_LAST_UPDATED_FRONTMATTER_RE = re.compile(
    r"^last_updated\s*:\s*(\S+)\s*$", re.MULTILINE
)
_STATUS_FRONTMATTER_RE = re.compile(
    r"^status\s*:\s*(\S+)\s*$", re.MULTILINE
)
_RECONCILE_TARGET_FRONTMATTER_RE = re.compile(
    r"^reconciliation_target\s*:\s*(\S+)\s*$", re.MULTILINE
)
_GENERATED_DASHBOARD_FRONTMATTER_RE = re.compile(
    r"^generated_dashboard\s*:\s*(\S+)\s*$", re.MULTILINE
)
_AUTHORITY_FRONTMATTER_RE = re.compile(
    r"^authority\s*:\s*(.+?)\s*$", re.MULTILINE
)
# Body-form date markers (some docs use prose at the top instead of
# YAML frontmatter):  "Last updated: 2026-05-22"
_BODY_LAST_UPDATED_RE = re.compile(
    r"[Ll]ast\s+updated\s*:\s*(\d{4}-\d{2}-\d{2})"
)
_DATE_FMT = "%Y-%m-%d"


@dataclass(frozen=True)
class DocEntry:
    """One audited documentation file.

    Fields
    ------
    path
        Repo-relative path.
    last_updated
        ISO date string parsed from frontmatter / body, or ``None``
        if no marker was found.
    status
        Frontmatter ``status:`` value (``"Normative"`` / ``"Informative"`` /
        ``"Historical"`` / etc.) or ``None``.
    authority
        Frontmatter ``authority:`` line if present.  Documents the
        normative scope of the file.
    reconciliation_target
        Optional name of a Python module / registry whose live data
        the doc's claims should match (e.g.,
        ``"primitive_coverage"``).  Used by the cross-check gate.
    generated_dashboard
        Optional path to a generated dashboard that companions this
        spec.  When set, drift between the spec and the dashboard is
        an explicit failure mode.
    has_frontmatter
        True iff the file starts with a YAML frontmatter block.
    """

    path: str
    last_updated: str | None
    status: str | None
    authority: str | None
    reconciliation_target: str | None
    generated_dashboard: str | None
    has_frontmatter: bool


# ─────────────────────────────────────────────────────────────────────────
# Manifest construction
# ─────────────────────────────────────────────────────────────────────────


def _parse_doc(path: Path) -> DocEntry:
    """Read a single .md file and extract frontmatter + last-updated."""
    rel = path.relative_to(_REPO_ROOT).as_posix()
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return DocEntry(
            path=rel, last_updated=None, status=None,
            authority=None, reconciliation_target=None,
            generated_dashboard=None, has_frontmatter=False,
        )

    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        fm_block = fm_match.group(1)
        last_updated_match = _LAST_UPDATED_FRONTMATTER_RE.search(fm_block)
        status_match = _STATUS_FRONTMATTER_RE.search(fm_block)
        authority_match = _AUTHORITY_FRONTMATTER_RE.search(fm_block)
        recon_match = _RECONCILE_TARGET_FRONTMATTER_RE.search(fm_block)
        dashboard_match = _GENERATED_DASHBOARD_FRONTMATTER_RE.search(fm_block)
        return DocEntry(
            path=rel,
            last_updated=(
                last_updated_match.group(1) if last_updated_match else None
            ),
            status=status_match.group(1) if status_match else None,
            authority=(
                authority_match.group(1) if authority_match else None
            ),
            reconciliation_target=(
                recon_match.group(1) if recon_match else None
            ),
            generated_dashboard=(
                dashboard_match.group(1) if dashboard_match else None
            ),
            has_frontmatter=True,
        )

    # No frontmatter — try body-form markers.
    body_match = _BODY_LAST_UPDATED_RE.search(text[:4000])
    return DocEntry(
        path=rel,
        last_updated=body_match.group(1) if body_match else None,
        status=None,
        authority=None,
        reconciliation_target=None,
        generated_dashboard=None,
        has_frontmatter=False,
    )


def collect_doc_manifest() -> tuple[DocEntry, ...]:
    """Walk the canonical doc roots + return entries in path order."""
    entries: list[DocEntry] = []
    for root in _DOC_ROOTS:
        root_path = _REPO_ROOT / root
        if not root_path.is_dir():
            continue
        for path in sorted(root_path.rglob("*.md")):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel in _EXEMPT_DOCS:
                continue
            entries.append(_parse_doc(path))
    return tuple(entries)


# ─────────────────────────────────────────────────────────────────────────
# Freshness analysis
# ─────────────────────────────────────────────────────────────────────────


def _days_stale(last_updated: str | None,
                reference: datetime.date | None = None) -> int | None:
    """Return days between ``last_updated`` and the reference date
    (defaults to today)."""
    if not last_updated:
        return None
    try:
        date = datetime.datetime.strptime(
            last_updated, _DATE_FMT
        ).date()
    except ValueError:
        return None
    ref = reference or datetime.date.today()
    return (ref - date).days


def freshness_summary(
    reference: datetime.date | None = None,
) -> dict[str, int]:
    """Return summary counts: total, with_frontmatter, fresh
    (≤30 days), stale (>180 days), undated."""
    entries = collect_doc_manifest()
    out = {
        "total": len(entries),
        "with_frontmatter": 0,
        "with_last_updated": 0,
        "undated": 0,
        "fresh_under_30d": 0,
        "fresh_under_90d": 0,
        "stale_over_90d": 0,
        "stale_over_180d": 0,
    }
    for e in entries:
        if e.has_frontmatter:
            out["with_frontmatter"] += 1
        if e.last_updated:
            out["with_last_updated"] += 1
        else:
            out["undated"] += 1
            continue
        days = _days_stale(e.last_updated, reference)
        if days is None:
            continue
        if days <= 30:
            out["fresh_under_30d"] += 1
        if days <= 90:
            out["fresh_under_90d"] += 1
        if days > 90:
            out["stale_over_90d"] += 1
        if days > 180:
            out["stale_over_180d"] += 1
    return out


def stale_normative_docs(
    threshold_days: int = 180,
    reference: datetime.date | None = None,
) -> tuple[DocEntry, ...]:
    """Return Normative-status docs whose last_updated is older than
    ``threshold_days``.  Catches the TSOL-style 3-week-stale failure
    mode that the audit was designed for."""
    out: list[DocEntry] = []
    for e in collect_doc_manifest():
        if e.status != "Normative":
            continue
        days = _days_stale(e.last_updated, reference)
        if days is not None and days > threshold_days:
            out.append(e)
    return tuple(out)


def undated_docs() -> tuple[DocEntry, ...]:
    """Docs that have no parseable last_updated date.  These are
    invisible to the freshness audit — fix by adding either YAML
    frontmatter with ``last_updated: YYYY-MM-DD`` or a body-form
    ``Last updated: YYYY-MM-DD`` line."""
    return tuple(e for e in collect_doc_manifest() if not e.last_updated)


def docs_by_root(
    entries: tuple[DocEntry, ...] | None = None,
) -> dict[str, tuple[DocEntry, ...]]:
    """Bucket entries by which root (``docs/spec``, ``docs/guides``,
    ...) they live under."""
    entries = entries if entries is not None else collect_doc_manifest()
    out: dict[str, list[DocEntry]] = {root: [] for root in _DOC_ROOTS}
    for e in entries:
        for root in _DOC_ROOTS:
            if e.path.startswith(root + "/"):
                out[root].append(e)
                break
    return {k: tuple(v) for k, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────
# Dashboard render
# ─────────────────────────────────────────────────────────────────────────


def render_dashboard(
    reference: datetime.date | None = None,
) -> str:
    """Render the freshness dashboard as Markdown text."""
    entries = collect_doc_manifest()
    summary = freshness_summary(reference)
    stale_normative = stale_normative_docs(reference=reference)
    undated = undated_docs()
    ref_date = (reference or datetime.date.today()).strftime(_DATE_FMT)

    lines: list[str] = []
    lines.append("# Documentation Freshness Dashboard")
    lines.append("")
    lines.append(
        "Generated from `python/tessera/compiler/docs_manifest.py`.  "
        "Don't edit by hand — regenerate via "
        "`python -c \"from tessera.compiler.docs_manifest import "
        "render_dashboard; "
        "open('docs/audit/generated/docs_freshness.md', 'w')"
        ".write(render_dashboard())\"`.  "
        "Drift gated by `tests/unit/test_docs_freshness.py`."
    )
    lines.append("")
    lines.append(f"Reference date for staleness: **{ref_date}**.")
    lines.append("")

    # ── Headline summary ──
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **{summary['total']}** docs catalogued across the "
                 f"canonical doc tree.")
    lines.append(
        f"- **{summary['with_last_updated']}** carry a "
        f"`last_updated:` marker; "
        f"**{summary['undated']}** are undated (invisible to the "
        f"freshness audit until tagged)."
    )
    lines.append(
        f"- **{summary['fresh_under_30d']}** updated within the "
        f"last 30 days."
    )
    lines.append(
        f"- **{summary['stale_over_90d']}** older than 90 days; "
        f"**{summary['stale_over_180d']}** older than 180 days."
    )
    n_stale_norm = len(stale_normative)
    if n_stale_norm:
        lines.append(
            f"- ⚠️  **{n_stale_norm}** Normative docs are stale "
            f"(>180 days).  These are the highest-leverage refresh "
            f"targets — Normative docs are user-facing contracts."
        )
    lines.append("")

    # ── Stale Normative section ──
    if stale_normative:
        lines.append("## Stale Normative docs (>180 days)")
        lines.append("")
        lines.append("| Path | last_updated | days stale |")
        lines.append("|------|--------------|-----------:|")
        for e in stale_normative:
            d = _days_stale(e.last_updated, reference)
            lines.append(f"| `{e.path}` | {e.last_updated} | {d} |")
        lines.append("")

    # ── Undated docs section ──
    if undated:
        lines.append("## Undated docs (no parseable `last_updated`)")
        lines.append("")
        lines.append(
            "These docs need either YAML frontmatter "
            "(`last_updated: YYYY-MM-DD`) or a body-form "
            "`Last updated:` line to participate in the audit.  Until "
            "tagged, the freshness signal is unavailable."
        )
        lines.append("")
        for e in undated:
            lines.append(f"- `{e.path}`")
        lines.append("")

    # ── Per-root detail ──
    lines.append("## Per-root inventory")
    lines.append("")
    grouped = docs_by_root(entries)
    for root, root_entries in grouped.items():
        if not root_entries:
            continue
        lines.append(f"### `{root}/`")
        lines.append("")
        lines.append(
            "| Path | status | last_updated | days stale | "
            "frontmatter |"
        )
        lines.append("|------|--------|--------------|-----------:|--|")
        for e in root_entries:
            d = _days_stale(e.last_updated, reference)
            d_str = "-" if d is None else str(d)
            status = e.status or "-"
            last_updated = e.last_updated or "_undated_"
            fm = "✓" if e.has_frontmatter else "_body_"
            short_path = e.path[len(root) + 1:]
            lines.append(
                f"| `{short_path}` | {status} | {last_updated} | "
                f"{d_str} | {fm} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_dashboard(path: Path | None = None) -> Path:
    target = path or (
        _REPO_ROOT / "docs" / "audit" / "generated" / "docs_freshness.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_dashboard())
    return target


__all__ = [
    "DocEntry",
    "collect_doc_manifest",
    "freshness_summary",
    "stale_normative_docs",
    "undated_docs",
    "docs_by_root",
    "render_dashboard",
    "write_dashboard",
]
