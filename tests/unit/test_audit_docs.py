"""Audit-prose doc contract gate.

The generated dashboards under ``docs/audit/generated/`` are count/status
truth and own their own drift gate (Decision #26).  This test owns the
*authored* layer: the hand-written audit prose that acts as the
"what's done / what's open / where next" guide.

Every authored doc under ``docs/audit/`` — the ``MASTER_AUDIT`` root plus the
theme subtrees (``compiler``, ``backend`` and its ``apple``/``nvidia``/``rocm``
sub-audits, ``coverage``, ``domain``, ``roadmap``) — must carry frontmatter
declaring:

  * ``last_updated: YYYY-MM-DD``     — feeds the freshness dashboard so a stale
                                       guide becomes visible instead of silently
                                       drifting (the failure mode that produced
                                       the MASTER_AUDIT reconciliation banners).
  * ``audit_role: <role>``           — lifecycle taxonomy so organization is
                                       machine-enforced, not vibes.
  * ``plan_state: open|landing|closed`` — REQUIRED iff ``audit_role == plan``.

EXCLUDED (not authored prose):
  * ``docs/audit/generated/**``       — generated dashboards (ground truth).
  * ``**/archive/**``                 — provenance only, not the live surface.
  * root drift-gated dashboards: ``op_target_conformance.md``,
    ``standalone_primitive_coverage.md``, ``stub_surface.md``.

What this catches: a new audit doc dropped in without a date or role; a plan
that never declares its state; a plan marked ``closed`` but left loitering in
the live tree instead of archived; a stale point-in-time ``snapshot`` that
should have been archived.
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT = _REPO_ROOT / "docs" / "audit"

# Lifecycle taxonomy.
#   root      — the single MASTER_AUDIT entry point.
#   index     — navigation README for a theme folder.
#   theme     — a living theme audit (compiler/backend/coverage/domain/roadmap).
#   sub_audit — a per-target audit under backend/ (apple/nvidia/rocm).
#   plan      — a work plan; carries plan_state.
#   reference — supporting reference/survey material (not a status surface).
#   snapshot  — a dated point-in-time audit; archive once superseded.
_ROLES = frozenset(
    {"root", "index", "theme", "sub_audit", "plan", "reference", "snapshot"}
)
_PLAN_STATES = frozenset({"open", "landing", "closed"})

# Root-level docs under docs/audit/ that are generated / drift-gated
# dashboards, NOT authored prose.  Owned by their own generators + gates.
_GENERATED_ROOT_DOCS = frozenset(
    {
        "op_target_conformance.md",
        "standalone_primitive_coverage.md",
        "stub_surface.md",
    }
)

# Point-in-time snapshots intentionally retained in the live tree (e.g. their
# closeout findings are still actively guarded by tests).  Every entry needs a
# reason.  Any OTHER snapshot older than the grace window must be archived.
#
# Empty today: the 2026-06-10 compiler snapshots were archived to
# compiler/archive/ (their guards in test_audit_closeout_guards.py parse the
# Python source, not the prose, so archiving the docs changed nothing).
_RETAINED_SNAPSHOTS: frozenset[str] = frozenset()

_SNAPSHOT_GRACE_DAYS = 45

_FM_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DATE_FMT = "%Y-%m-%d"


def _is_authored(rel: str) -> bool:
    if rel.startswith("generated/"):
        return False
    if "/archive/" in rel or rel.startswith("archive/"):
        return False
    if rel in _GENERATED_ROOT_DOCS:
        return False
    return True


def _authored_docs() -> list[Path]:
    out = []
    for path in sorted(_AUDIT.rglob("*.md")):
        rel = path.relative_to(_AUDIT).as_posix()
        if _is_authored(rel):
            out.append(path)
    return out


def _frontmatter(path: Path) -> dict[str, str]:
    m = _FM_RE.match(path.read_text(encoding="utf-8"))
    if not m:
        return {}
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()
    return fields


def _rel(path: Path) -> str:
    return path.relative_to(_AUDIT).as_posix()


# Status-surface roles whose claims must be traceable back to the generated
# count/status dashboards (Decision #26: "link them, never copy numbers").
_STATUS_ROLES = frozenset({"root", "theme", "sub_audit"})

# Root-level dashboards that are also count authority (not under generated/).
_ROOT_DASHBOARDS = frozenset(
    {"op_target_conformance.md", "standalone_primitive_coverage.md"}
)

_GENERATED_REF_RE = re.compile(r"generated/([A-Za-z0-9_]+\.md)")
_ROOT_DASH_REF_RE = re.compile(
    r"(op_target_conformance\.md|standalone_primitive_coverage\.md)"
)


def _generated_dashboard_names() -> set[str]:
    return {p.name for p in (_AUDIT / "generated").glob("*.md")}


def _dashboard_refs(path: Path) -> tuple[set[str], set[str]]:
    """Return (generated/* basenames, root-dashboard basenames) referenced
    anywhere in the doc (inline links or `authority:` frontmatter)."""
    text = path.read_text(encoding="utf-8")
    gen = set(_GENERATED_REF_RE.findall(text))
    root = set(_ROOT_DASH_REF_RE.findall(text))
    return gen, root


# ── discovery sanity ───────────────────────────────────────────────────────


def test_authored_audit_docs_discovered() -> None:
    """Guard the discovery itself: if this drops to ~zero the gate has
    been silently disarmed (e.g., the tree moved)."""
    docs = _authored_docs()
    assert len(docs) >= 18, (
        f"Expected the audit prose tree to hold >=18 authored docs, found "
        f"{len(docs)}.  Did the audit tree move or did discovery break?"
    )


# ── frontmatter contract ───────────────────────────────────────────────────


def test_every_authored_doc_has_last_updated_and_role() -> None:
    offenders: list[str] = []
    for path in _authored_docs():
        fm = _frontmatter(path)
        rel = _rel(path)
        lu = fm.get("last_updated")
        role = fm.get("audit_role")
        if not lu:
            offenders.append(f"{rel}: missing `last_updated`")
            continue
        try:
            _dt.datetime.strptime(lu, _DATE_FMT)
        except ValueError:
            offenders.append(f"{rel}: unparseable last_updated={lu!r}")
        if not role:
            offenders.append(f"{rel}: missing `audit_role`")
        elif role not in _ROLES:
            offenders.append(
                f"{rel}: audit_role={role!r} not in {sorted(_ROLES)}"
            )
    assert not offenders, (
        "Audit docs violating the frontmatter contract "
        "(see tests/unit/test_audit_docs.py):\n  " + "\n  ".join(offenders)
    )


def test_plan_docs_declare_a_valid_plan_state() -> None:
    offenders: list[str] = []
    for path in _authored_docs():
        fm = _frontmatter(path)
        rel = _rel(path)
        role = fm.get("audit_role")
        state = fm.get("plan_state")
        if role == "plan":
            if state not in _PLAN_STATES:
                offenders.append(
                    f"{rel}: plan must declare plan_state in "
                    f"{sorted(_PLAN_STATES)}, got {state!r}"
                )
        elif state is not None:
            offenders.append(
                f"{rel}: plan_state={state!r} is only meaningful for "
                f"audit_role: plan (this doc is {role!r})"
            )
    assert not offenders, "\n  ".join(["plan_state violations:"] + offenders)


# ── lifecycle enforcement ──────────────────────────────────────────────────


def test_closed_plans_are_archived_not_loitering() -> None:
    """A plan whose state is ``closed`` is finished work — it belongs in
    a theme-local ``archive/``, not in the live status surface."""
    offenders = [
        _rel(p)
        for p in _authored_docs()
        if _frontmatter(p).get("audit_role") == "plan"
        and _frontmatter(p).get("plan_state") == "closed"
    ]
    assert not offenders, (
        "Closed plans still in the live audit tree — move them to the theme "
        "archive/ and summarize in the theme audit (Decision #26):\n  "
        + "\n  ".join(offenders)
    )


def test_retained_snapshots_still_exist() -> None:
    """Keep the retain-list honest: an entry that's been archived/removed
    should be dropped from _RETAINED_SNAPSHOTS."""
    stale = sorted(
        rel
        for rel in _RETAINED_SNAPSHOTS
        if not (_AUDIT / rel).is_file()
    )
    assert not stale, (
        "Retained-snapshot allow-list names files that no longer exist — "
        f"remove them from _RETAINED_SNAPSHOTS: {stale}"
    )


def test_stale_snapshots_are_archived() -> None:
    """A dated snapshot older than the grace window must be archived unless
    explicitly retained (with a reason) in _RETAINED_SNAPSHOTS."""
    today = _dt.date.today()
    offenders: list[str] = []
    for path in _authored_docs():
        fm = _frontmatter(path)
        if fm.get("audit_role") != "snapshot":
            continue
        rel = _rel(path)
        if rel in _RETAINED_SNAPSHOTS:
            continue
        lu = fm.get("last_updated")
        if not lu:
            continue  # covered by the frontmatter-contract test
        age = (today - _dt.datetime.strptime(lu, _DATE_FMT).date()).days
        if age > _SNAPSHOT_GRACE_DAYS:
            offenders.append(f"{rel} ({age}d old)")
    assert not offenders, (
        f"Snapshots older than {_SNAPSHOT_GRACE_DAYS}d should be archived "
        "(or added to _RETAINED_SNAPSHOTS with a reason):\n  "
        + "\n  ".join(offenders)
    )


# ── claim-anchoring (Decision #26: never copy counts — link the truth) ──────


def test_status_docs_anchor_to_generated_truth() -> None:
    """Every root/theme/sub_audit doc must reference at least one generated
    dashboard, so its status claims are traceable to count authority instead
    of free-floating prose that silently drifts."""
    offenders: list[str] = []
    for path in _authored_docs():
        role = _frontmatter(path).get("audit_role")
        if role not in _STATUS_ROLES:
            continue
        gen, root = _dashboard_refs(path)
        if not gen and not root:
            offenders.append(_rel(path))
    assert not offenders, (
        "Status-surface audit docs with no link to any generated dashboard "
        "(Decision #26 — claims must trace to count authority, not restate "
        "it):\n  " + "\n  ".join(offenders)
    )


def test_referenced_generated_dashboards_exist() -> None:
    """A `generated/<x>.md` reference in any audit doc must resolve to a real
    dashboard — catches a claim anchor left dangling after a dashboard is
    renamed or removed."""
    real = _generated_dashboard_names()
    offenders: list[str] = []
    for path in _authored_docs():
        gen, _ = _dashboard_refs(path)
        for name in sorted(gen - real):
            offenders.append(f"{_rel(path)} -> generated/{name}")
    assert not offenders, (
        "Audit docs referencing generated dashboards that don't exist "
        "(dangling claim anchors):\n  " + "\n  ".join(offenders)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
