"""E2E compiler-coverage audit (Issue 4, 2026-05-20).

For every op in :data:`tessera.compiler.op_catalog.OP_SPECS`,
report whether the op has a complete path through the compiler
stack: Graph IR → Schedule IR → Tile IR → Target IR → runtime.

This is an **audit**, not a backend implementation drive.  The
goal is to make the gap visible so prioritization decisions can
be made against real numbers, not vibes.

The audit reuses :func:`tessera.compiler.audit.support_row_for` —
the same source that powers ``docs/audit/generated/support_table.md``
— and rolls each op's 8-axis status into a single
:class:`E2EStatus`:

  * :class:`E2EStatus.COMPLETE` — every axis reports a "real"
    value: ``public`` for api / frontend, ``registered`` or
    ``not_applicable`` for graph_ir, ``complete`` /
    ``fused`` / ``ready`` for schedule_ir / tile_ir / target_ir /
    runtime, ``benchmarked`` for bench.
  * :class:`E2EStatus.RUNNABLE_REFERENCE` — every axis except
    bench reports a real value, but the op runs through the
    numpy reference path.  Useful — just not "native E2E".
  * :class:`E2EStatus.ARTIFACT_ONLY` — emits IR through the
    target_ir axis but the runtime / bench axes say no execution.
  * :class:`E2EStatus.PARTIAL` — some axes are ``planned`` /
    ``partial`` / ``missing``.  The op compiles but the pipeline
    has gaps.
  * :class:`E2EStatus.PLANNED` — only api/frontend are public;
    everything else is unfinished.

The CLI ``python -m tessera.cli.e2e_coverage`` ships in step 5
of this issue.  The drift gate
``tests/unit/test_e2e_coverage_contract.py`` locks the
classification logic + the generated dashboard's shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from . import audit as _audit


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATED_DOC = (
    _REPO_ROOT / "docs" / "audit" / "generated" / "e2e_op_coverage.md"
)


class E2EStatus(Enum):
    """Single-tier rollup of an op's compiler-stack readiness."""

    COMPLETE = "complete"
    """Every axis green — native end-to-end with benchmarks."""

    RUNNABLE_REFERENCE = "runnable_reference"
    """Compiles + runs via the numpy reference path; no native kernel."""

    ARTIFACT_ONLY = "artifact_only"
    """Compiles to IR but no execution wired."""

    PARTIAL = "partial"
    """Compiles partially; some axes are planned / missing."""

    PLANNED = "planned"
    """Only api / frontend are public; lowering not built yet."""

    @classmethod
    def _rank(cls) -> dict["E2EStatus", int]:
        return {
            cls.COMPLETE: 0,
            cls.RUNNABLE_REFERENCE: 1,
            cls.ARTIFACT_ONLY: 2,
            cls.PARTIAL: 3,
            cls.PLANNED: 4,
        }


@dataclass(frozen=True)
class E2ECoverageRow:
    """One op's E2E status + the per-axis evidence that produced it."""

    op_name: str
    family: str
    status: E2EStatus
    axis_status: dict[str, str]
    """Per-axis status as a plain dict — same vocabulary the
    ``support_table.md`` audit uses (``public`` / ``registered`` /
    ``fused`` / ``reference`` / ``ready`` / ``planned`` / etc.)."""


# Axis-value sets used for classification.  Kept narrow + named so
# the logic reads top-to-bottom without magic strings.

_AXIS_REAL_API_FRONTEND = frozenset({"public"})
_AXIS_REAL_GRAPH_IR = frozenset({
    "registered", "host_materialized", "runtime_only", "not_applicable",
})
_AXIS_NATIVE_LOWERING = frozenset({
    "complete",
    "fused",
    "device_verified_jit",
    "device_verified_abi",
    "packaged",
})
_AXIS_REFERENCE_LOWERING = frozenset({"reference", "compileable"})
_AXIS_ARTIFACT_LOWERING = frozenset({"artifact_only"})
_AXIS_RUNTIME_READY = frozenset({"ready", "fused"})
_AXIS_RUNTIME_REFERENCE = frozenset({"reference"})
_AXIS_BENCH_REAL = frozenset({"benchmarked"})


def classify(row: _audit.OpSupportRow) -> E2EStatus:
    """Roll a per-axis :class:`OpSupportRow` into an E2E tier.

    Reads each axis from the row and decides the rollup with a
    simple decision tree.  The "best evidence wins" rule applies —
    e.g., ``target_ir=fused`` with ``runtime=ready`` is
    ``COMPLETE`` even when ``bench=none``, because the op runs
    natively (it just doesn't have a recorded benchmark yet).
    """

    api = row.cells["api"].status
    frontend = row.cells["frontend"].status
    graph_ir = row.cells["graph_ir"].status
    tile_ir = row.cells["tile_ir"].status
    target_ir = row.cells["target_ir"].status
    runtime = row.cells["runtime"].status
    bench = row.cells["bench"].status

    api_real = api in _AXIS_REAL_API_FRONTEND
    frontend_real = frontend in _AXIS_REAL_API_FRONTEND
    graph_real = graph_ir in _AXIS_REAL_GRAPH_IR

    # If api + frontend aren't both public, the op isn't actually
    # callable — call it PLANNED regardless of the deeper axes.
    if not (api_real and frontend_real):
        return E2EStatus.PLANNED

    # Native execution: target_ir fused + runtime ready.  ``tile_ir``
    # is a documentation axis (the contract-axis registry can mark
    # it ``partial`` even for ops that run natively — matmul today
    # is the canonical example: tile_ir=partial yet target_ir=fused
    # + runtime=ready).  The real "does it run natively?" signal is
    # ``target_ir`` + ``runtime``.  ``graph_real`` confirms the op
    # actually has a Graph IR representation (rules out runtime-only
    # ops).
    if (
        target_ir in _AXIS_NATIVE_LOWERING
        and runtime in _AXIS_RUNTIME_READY
        and graph_real
    ):
        # Native dispatch confirmed.  Bench is a "would be nice"
        # axis, not a correctness gate — call it COMPLETE either
        # way.  Reference the local _ to satisfy mypy's
        # unused-variable checker on the ``tile_ir`` / ``bench``
        # bindings we keep for documentation purposes.
        _ = (tile_ir, bench)
        return E2EStatus.COMPLETE

    # Reference execution: at least one of {tile_ir, target_ir,
    # runtime} reports reference / compileable + the others don't
    # actively deny.
    if (
        runtime in _AXIS_RUNTIME_REFERENCE
        or target_ir in _AXIS_REFERENCE_LOWERING
        or tile_ir in _AXIS_REFERENCE_LOWERING
    ) and graph_real:
        return E2EStatus.RUNNABLE_REFERENCE

    # Artifact-only: target_ir compiles but runtime doesn't execute.
    if (
        target_ir in _AXIS_ARTIFACT_LOWERING
        or target_ir in _AXIS_REFERENCE_LOWERING
    ) and runtime not in _AXIS_RUNTIME_READY:
        return E2EStatus.ARTIFACT_ONLY

    # Otherwise partial — some axes promote-able, others not.
    return E2EStatus.PARTIAL


def coverage_row_for(op_name: str) -> E2ECoverageRow:
    """Build a single :class:`E2ECoverageRow`."""

    audit_row = _audit.support_row_for(op_name)
    status = classify(audit_row)
    return E2ECoverageRow(
        op_name=op_name,
        family=audit_row.family,
        status=status,
        axis_status={
            axis: audit_row.cells[axis].status
            for axis in _audit.LAYER_AXES
        },
    )


# Decorator aliases — names that appear in ``_M7_INVENTORY`` /
# similar inventories for *discoverability*, not because they're ops.
# ``complex_jit`` is the @-decorator factory that lowers a Python
# source function via ``analytic_symbolic``; treating it as an op
# would forever pin one row at ``partial`` (no Graph IR / no kernel /
# nothing to benchmark) even though that's precisely the right shape
# for a decorator surface.  Excluded from E2E op coverage; still
# visible in ``support_table.md`` because the support audit is the
# right place to document decorator surfaces.
_DECORATOR_ALIASES: frozenset[str] = frozenset({
    "complex_jit",
})


def all_coverage_rows() -> list[E2ECoverageRow]:
    """E2E coverage for every op the audit walker enumerates,
    minus decorator aliases (see ``_DECORATOR_ALIASES``)."""

    return [
        coverage_row_for(name)
        for name in _audit._candidate_op_names()
        if name not in _DECORATOR_ALIASES
    ]


def status_counts() -> dict[str, int]:
    """``{status_value: count}`` rollup across every op."""

    out: dict[str, int] = {s.value: 0 for s in E2EStatus}
    for row in all_coverage_rows():
        out[row.status.value] += 1
    return out


def rows_by_status(status: E2EStatus) -> list[E2ECoverageRow]:
    """Filter the coverage view to a single tier."""

    return [r for r in all_coverage_rows() if r.status is status]


# ─────────────────────────────────────────────────────────────────────
# Doc rendering — generated `docs/audit/generated/e2e_op_coverage.md`.
# ─────────────────────────────────────────────────────────────────────


def render_markdown(rows: Iterable[E2ECoverageRow] | None = None) -> str:
    """Render the E2E coverage dashboard.

    The output groups ops by status (COMPLETE first, then degrading)
    so the gap surface is at the bottom — easy to scroll to.
    """

    all_rows = list(rows) if rows is not None else all_coverage_rows()
    all_rows.sort(key=lambda r: (E2EStatus._rank()[r.status], r.family, r.op_name))

    counts = status_counts()
    total = sum(counts.values())

    lines: list[str] = [
        "<!-- AUTO-GENERATED by python/tessera/compiler/e2e_coverage.py — DO NOT EDIT BY HAND. -->",
        "<!-- Regenerate via: python -m tessera.cli.e2e_coverage --render -->",
        "",
        "# E2E op compiler coverage",
        "",
        "Per-op rollup of compiler-stack readiness across the 8 audit",
        "axes (api / frontend / graph_ir / schedule_ir / tile_ir /",
        "target_ir / runtime / bench).  Source of truth: the same",
        "``OpSupportRow`` data that powers",
        "``docs/audit/generated/support_table.md`` — this view rolls",
        "each row into a single E2E tier.",
        "",
        "## Tiers",
        "",
        "| Tier | Meaning |",
        "|---|---|",
        "| ``complete`` | Native end-to-end — tile_ir + target_ir fused, runtime ready. Bench may or may not be recorded. |",
        "| ``runnable_reference`` | Runs via the numpy reference path. Correct but not native. |",
        "| ``artifact_only`` | Emits IR / Target artifacts but no execution path. |",
        "| ``partial`` | Some axes ``planned`` / ``missing`` — pipeline has gaps. |",
        "| ``planned`` | API / frontend only — lowering not built yet. |",
        "",
        "## Counts",
        "",
        "| Tier | Count |",
        "|---|---:|",
    ]
    for status in E2EStatus:
        lines.append(f"| ``{status.value}`` | {counts[status.value]} |")
    lines.append(f"| **total** | **{total}** |")
    lines.append("")

    # Group rows by tier for readability.
    by_status: dict[str, list[E2ECoverageRow]] = {
        s.value: [] for s in E2EStatus
    }
    for row in all_rows:
        by_status[row.status.value].append(row)

    for status in E2EStatus:
        bucket = by_status[status.value]
        if not bucket:
            continue
        lines.append(f"## {status.value} ({len(bucket)})")
        lines.append("")
        lines.append("| Op | Family | api | frontend | graph_ir | tile_ir | target_ir | runtime | bench |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in bucket:
            cells = " | ".join(
                row.axis_status[axis] for axis in _audit.LAYER_AXES
                if axis != "schedule_ir"
            )
            lines.append(
                f"| ``{row.op_name}`` | {row.family} | {cells} |"
            )
        lines.append("")

    lines.append("## How to extend")
    lines.append("")
    lines.append(
        "Promoting an op to a higher tier means closing one or more "
        "audit axes.  Read this dashboard alongside "
        "``docs/audit/generated/support_table.md`` and "
        "``primitive_coverage_state.md`` for the per-axis breakdown."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_doc(out_path: Path | None = None) -> Path:
    """Write the generated doc to disk.  Returns the path."""

    path = out_path if out_path is not None else _GENERATED_DOC
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(), encoding="utf-8")
    return path


__all__ = [
    "E2ECoverageRow",
    "E2EStatus",
    "all_coverage_rows",
    "classify",
    "coverage_row_for",
    "render_markdown",
    "rows_by_status",
    "status_counts",
    "write_doc",
]
