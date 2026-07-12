"""S-series primitive-contracts status dashboard (Slice 2, 2026-05-22).

Generates ``docs/audit/generated/s_series_status.md`` — a per-category ×
per-axis breakdown of the remaining ``partial`` and ``planned`` contract
entries across the S-series primitive surface.

Why a separate dashboard from the main support table?

The main support table is **op-centric** — one row per primitive, axes
across columns.  Useful for "does this op have a VJP?" lookups.

This dashboard is **category-centric** — one row per category, axis
totals across columns.  Useful for "where should the next batching-rule
promotion sprint focus?" prioritisation, which is the question the
S-series hardening ask is asking.

The dashboard is generated, not hand-written.  Drift gate:
``python -m tessera.compiler.s_series_status --check``.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from .primitive_coverage import (
    PrimitiveCoverage,
    all_primitive_coverages,
    is_contract_closed,
)
from . import backend_manifest as _backend_manifest


# Per the user direction (2026-05-22), prioritise the high-use S2/S5/
# S7/S10/S11 surface — reductions, scans/control flow, layers,
# optimizers, losses, and memory primitives.  This list anchors the
# dashboard's "priority" column so consumers can sort by user-visible
# impact rather than alphabetical accident.
S_SERIES_PRIORITY: dict[str, tuple[str, int]] = {
    # category               → (S-sprint, priority — lower is more urgent)
    "tensor_algebra":          ("S2", 10),
    "reduction":               ("S2", 11),
    "scalar_math":             ("S2", 12),
    "comparison":              ("S2", 13),
    "numeric_helper":          ("S2", 14),
    "stability":               ("S2", 15),

    "control_flow":            ("S5", 20),
    "autodiff_transform":      ("S5", 21),

    "layer":                   ("S7", 30),
    "attention":               ("S7", 31),
    "position_encoding":       ("S7", 32),
    "normalization":           ("S7", 33),
    "memory":                  ("S7", 34),
    "geometric_algebra":       ("S7", 35),

    "optimizer":               ("S10", 40),
    "schedule":                ("S10", 41),
    "grad_transform":          ("S10", 42),

    "loss":                    ("S11", 50),
    "rl_loss":                 ("S11", 51),

    "ebm":                     ("M6", 60),
    "visual_complex":          ("M7", 61),

    "data":                    ("S15", 70),
    "tokenizer":               ("S15", 71),
}

# The five contract axes we audit in this dashboard.  Order matters —
# left-to-right reflects "most likely to be closable without hardware"
# (batching) → "Phase G/H universal gate" (backend_kernel).
DASHBOARD_AXES: tuple[str, ...] = (
    "batching_rule",
    "transpose_rule",
    "sharding_rule",
    "lowering_rule",
    "backend_kernel",
)

# Statuses considered "open work" — partial OR planned.  ``complete``
# and ``not_applicable`` are closed; ``unknown`` ⇒ entry missing the
# axis, which we surface as a separate column for honesty.
OPEN_STATUSES: frozenset[str] = frozenset({"partial", "planned"})

# Per-target native proof statuses.  These mean "this architecture has a real
# backend path", not "every declared backend for the primitive is complete".
NATIVE_BACKEND_STATUSES: frozenset[str] = frozenset({
    "hardware_verified",
    "compiled",
    "fused",
    "packaged",
})

OPEN_BACKEND_STATUSES: frozenset[str] = frozenset({
    "artifact_only",
    "compileable",
    "planned",
})


# ─────────────────────────────────────────────────────────────────────────────
# Tally
# ─────────────────────────────────────────────────────────────────────────────


def tally_by_category(
    cov: dict[str, PrimitiveCoverage] | None = None,
) -> list[dict[str, object]]:
    """Walk the coverage registry and return one row per category.

    Each row is a dict with keys:
      category       : str
      sprint         : str — S2/S5/S7/S10/S11/M6/M7/S15 or "other"
      priority       : int — lower = more urgent (for sorting)
      total          : int — primitives in this category
      <axis>_open    : int — count of {partial, planned} on this axis
      <axis>_complete: int — count of complete plus explicit by-design terminals
    Sort order: priority ascending, then category alphabetically.
    """
    if cov is None:
        cov = all_primitive_coverages()

    per_cat_total: dict[str, int] = defaultdict(int)
    per_cat_axis_open: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    per_cat_axis_complete: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for entry in cov.values():
        cat = entry.category
        per_cat_total[cat] += 1
        for axis in DASHBOARD_AXES:
            status = entry.contract_status.get(axis, "unknown")
            if status in OPEN_STATUSES:
                per_cat_axis_open[cat][axis] += 1
            elif is_contract_closed(status):
                per_cat_axis_complete[cat][axis] += 1
            # ``unknown`` is silently dropped — it surfaces in the
            # support_table audit, not here.

    rows: list[dict[str, object]] = []
    for cat in sorted(per_cat_total):
        sprint, prio = S_SERIES_PRIORITY.get(cat, ("other", 999))
        row: dict[str, object] = {
            "category": cat,
            "sprint": sprint,
            "priority": prio,
            "total": per_cat_total[cat],
        }
        for axis in DASHBOARD_AXES:
            row[f"{axis}_open"] = per_cat_axis_open[cat].get(axis, 0)
            row[f"{axis}_complete"] = per_cat_axis_complete[cat].get(axis, 0)
        rows.append(row)

    rows.sort(key=lambda r: (r["priority"], r["category"]))
    return rows


def tally_backend_by_target(
    cov: dict[str, PrimitiveCoverage] | None = None,
) -> list[dict[str, object]]:
    """Return one row per backend target from ``BackendKernelEntry`` rows.

    This is the architecture-specific view the primitive-level
    ``backend_kernel`` axis intentionally cannot express.  The registry axis is
    still useful as a conservative compatibility flag; this target tally is the
    status users should read when asking "is S-series done on ROCm/x86/etc.?"
    """
    if cov is None:
        cov = all_primitive_coverages()

    targets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total = len(cov)
    for op_name in sorted(cov):
        seen: set[str] = set()
        for entry in _backend_manifest.manifest_for(op_name):
            target = entry.target
            seen.add(target)
            row = targets[target]
            row["declared"] += 1
            if entry.status in NATIVE_BACKEND_STATUSES:
                row["native_proven"] += 1
            elif entry.status == "reference":
                row["reference"] += 1
            elif entry.status in OPEN_BACKEND_STATUSES:
                row["open"] += 1
            else:
                row["other"] += 1
        for target in seen:
            targets[target]["_seen"] += 1

    rows: list[dict[str, object]] = []
    for target, counts in targets.items():
        declared = counts.get("declared", 0)
        rows.append({
            "target": target,
            "declared": declared,
            "native_proven": counts.get("native_proven", 0),
            "reference": counts.get("reference", 0),
            "open": counts.get("open", 0),
            "other": counts.get("other", 0),
            "missing": total - declared,
        })
    rows.sort(key=lambda r: _target_sort_key(str(r["target"])))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────


_AXIS_HEADERS: dict[str, str] = {
    "batching_rule":   "batching",
    "transpose_rule":  "transpose",
    "sharding_rule":   "sharding",
    "lowering_rule":   "lowering",
    "backend_kernel":  "backend",
}


def render_markdown(
    rows: list[dict[str, object]] | None = None,
) -> str:
    """Render the dashboard as Markdown.  Deterministic ⇒ drift-
    gateable."""
    if rows is None:
        rows = tally_by_category()

    # Section 1: header / context.
    lines: list[str] = []
    lines.append("# S-series primitive-contracts status (generated)")
    lines.append("")
    lines.append("> **Generated by `python -m tessera.compiler.s_series_status`.**")
    lines.append("> Drift gate: `python -m tessera.compiler.s_series_status --check`.")
    lines.append(">")
    lines.append("> One row per primitive category.  Counts are the number of")
    lines.append("> entries in that category whose contract axis is in `{partial,")
    lines.append("> planned}` (open) vs `complete`/explicit by-design terminals. Sprint")
    lines.append("> labels (S2/S5/S7/S10/S11/M6/M7/S15) anchor each category to")
    lines.append("> the milestone that owns it; priority sorts user-visible")
    lines.append("> impact (smaller = more urgent).")
    lines.append("")

    # Section 2: aggregate banner.
    aggregate = _aggregate(rows)
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Axis | Open (partial+planned) | Complete |")
    lines.append("|---|---:|---:|")
    for axis in DASHBOARD_AXES:
        lines.append(
            f"| `{axis}` | {aggregate['open'][axis]} | "
            f"{aggregate['complete'][axis]} |"
        )
    lines.append("")

    # Section 2b: backend proof by architecture.  This is the honest "S-series
    # done per target" view; the raw backend_kernel axis above remains a
    # conservative registry compatibility flag.
    target_rows = tally_backend_by_target()
    lines.append("## Backend Proof By Target")
    lines.append("")
    lines.append(
        "The registry-level `backend_kernel` axis is deliberately conservative "
        "and should not be read as an all-up veto.  Per-architecture completion "
        "comes from `BackendKernelEntry` rows: `hardware_verified`, `compiled`, "
        "`fused`, and `packaged` count as native proof for that target; "
        "`reference` is correct execution without a native kernel; "
        "`artifact_only` / `compileable` / `planned` remain open for that target."
    )
    lines.append("")
    lines.append("| Target | Declared | Native proven | Reference | Open artifact/planned | Missing target row |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in target_rows:
        lines.append(
            f"| `{row['target']}` | {row['declared']} | "
            f"{row['native_proven']} | {row['reference']} | "
            f"{row['open']} | {row['missing']} |"
        )
    lines.append("")

    # Section 3: per-category table sorted by priority.
    lines.append("## Per-category breakdown")
    lines.append("")
    headers = (
        ["sprint", "category", "total"]
        + [_AXIS_HEADERS[a] for a in DASHBOARD_AXES]
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append(
        "|" + "|".join(["---"] * 2 + ["---:"] * (1 + len(DASHBOARD_AXES))) + "|"
    )
    for row in rows:
        cells = [
            str(row["sprint"]),
            f"`{row['category']}`",
            str(row["total"]),
        ]
        for axis in DASHBOARD_AXES:
            open_n = row[f"{axis}_open"]
            cells.append(str(open_n) if open_n else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Section 4: per-sprint roll-up so consumers can pivot by sprint.
    by_sprint = _by_sprint(rows)
    lines.append("## Roll-up by sprint")
    lines.append("")
    lines.append(
        "| Sprint | Primitives | Open batching | Open transpose | "
        "Open sharding | Open backend |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for sprint in sorted(by_sprint, key=lambda s: _sprint_sort_key(s)):
        d = by_sprint[sprint]
        lines.append(
            f"| {sprint} | {d['total']} | "
            f"{d['batching_rule_open']} | {d['transpose_rule_open']} | "
            f"{d['sharding_rule_open']} | {d['backend_kernel_open']} |"
        )
    lines.append("")

    # Section 5: closure trajectory note.
    lines.append("## Closure trajectory")
    lines.append("")
    lines.append("* `lowering_rule` is closed project-wide today (0 open across all categories) — the multi-axis category-based hardening pass from Sprint A1+ landed this.")
    lines.append("* `backend_kernel` is a conservative registry-level compatibility axis, not the architecture completion signal.  Read **Backend Proof By Target** for ROCm/x86/Apple/NVIDIA status.")
    lines.append("* `batching_rule` / `transpose_rule` / `sharding_rule` are the closable axes today.  A category-by-category promotion sprint should focus on the rows above with `priority ≤ 50`.")
    lines.append("")
    return "\n".join(lines)


def _as_int(value: object) -> int:
    """Narrow a row cell (typed ``object``) back to int.  Rows are
    built with mixed-value cells; the count cells are always ints by
    construction in ``tally_by_category``."""
    return int(value)  # type: ignore[call-overload]


def _aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    open_totals: dict[str, int] = defaultdict(int)
    complete_totals: dict[str, int] = defaultdict(int)
    for row in rows:
        for axis in DASHBOARD_AXES:
            open_totals[axis] += _as_int(row[f"{axis}_open"])
            complete_totals[axis] += _as_int(row[f"{axis}_complete"])
    return {"open": dict(open_totals), "complete": dict(complete_totals)}


def _by_sprint(rows: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in rows:
        s = str(row["sprint"])
        out[s]["total"] += _as_int(row["total"])
        for axis in DASHBOARD_AXES:
            out[s][f"{axis}_open"] += _as_int(row[f"{axis}_open"])
    return {k: dict(v) for k, v in out.items()}


def _sprint_sort_key(s: str) -> tuple[int, str]:
    # Sort S2 < S5 < S7 < S10 < S11 < S15 < M6 < M7 < other.
    if s.startswith("S") and s[1:].isdigit():
        return (0, f"S{int(s[1:]):02d}")
    if s.startswith("M") and s[1:].isdigit():
        return (1, s)
    return (2, s)


def _target_sort_key(target: str) -> tuple[int, str]:
    order = {
        "cpu": 0,
        "x86": 1,
        "apple_cpu": 2,
        "apple_gpu": 3,
        "rocm": 4,
        "nvidia_sm80": 5,
        "nvidia_sm90": 6,
        "nvidia_sm100": 7,
        "nvidia_sm120": 8,
        "metalium": 9,
    }
    return (order.get(target, 100), target)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


_DEFAULT_OUT = Path("docs/audit/generated/s_series_status.md")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="S-series primitive-contracts status dashboard."
    )
    ap.add_argument(
        "--out", default=str(_DEFAULT_OUT),
        help="output path for the generated markdown",
    )
    ap.add_argument(
        "--check", action="store_true",
        help="exit non-zero if the file on disk doesn't match the "
             "generated output (drift gate)",
    )
    ap.add_argument(
        "--render", action="store_true",
        help="print the rendered markdown to stdout instead of writing",
    )
    args = ap.parse_args(argv)

    text = render_markdown()

    if args.render:
        print(text)
        return 0

    out_path = Path(args.out)

    if args.check:
        if not out_path.exists():
            print(
                f"s_series_status: {out_path} does not exist.\n"
                f"       run `python -m tessera.compiler.s_series_status` "
                "to generate it.",
                file=sys.stderr,
            )
            return 1
        existing = out_path.read_text()
        if existing != text:
            print(
                f"s_series_status: drift detected in {out_path}\n"
                "       regenerate with: "
                f"python -m tessera.compiler.s_series_status --out {out_path}",
                file=sys.stderr,
            )
            return 1
        print(f"s_series_status: {out_path} matches generated output")
        return 0

    _write(out_path, text)
    print(f"s_series_status: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
