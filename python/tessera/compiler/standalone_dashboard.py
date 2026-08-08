"""Generated sections for the standalone primitive coverage dashboard.

The authored document explains scope and policy.  Counts and compiler-status
claims come from the live registries and are replaced between explicit marker
pairs, so a new op, backend proof, or benchmark cannot leave stale prose behind.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Mapping

from . import audit
from . import backend_manifest as bm
from . import primitive_coverage as pc


REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_PATH = REPO_ROOT / "docs" / "audit" / "standalone_primitive_coverage.md"

REGISTRY_START = "<!-- BEGIN GENERATED LIVE REGISTRY SUMMARY -->"
REGISTRY_END = "<!-- END GENERATED LIVE REGISTRY SUMMARY -->"
COMPILER_START = "<!-- BEGIN GENERATED COMPILER FOUNDATION SUMMARY -->"
COMPILER_END = "<!-- END GENERATED COMPILER FOUNDATION SUMMARY -->"

_MANIFEST_STATUS_ORDER = (
    "device_verified_abi",
    "device_verified_jit",
    "packaged",
    "fused",
    "compileable",
    "artifact_only",
    "reference",
    "planned",
)


def _status_counts(values: Mapping[str, int], order: tuple[str, ...]) -> str:
    cells = [f"`{status}` {values[status]}" for status in order if values.get(status)]
    extras = sorted(set(values) - set(order))
    cells.extend(f"`{status}` {values[status]}" for status in extras)
    return ", ".join(cells) or "none"


def render_registry_summary() -> str:
    entries = pc.all_primitive_coverages()
    row_status = Counter(entry.status for entry in entries.values())
    graph_status = Counter(
        str(entry.metadata.get("graph_ir_lowering", "missing"))
        for entry in entries.values()
    )
    lines = [
        REGISTRY_START,
        "| Measure | Live value |",
        "|---|---:|",
        f"| Tracked primitives | {len(entries)} |",
        f"| Existing public/runtime primitives | {sum(e.existing_op for e in entries.values())} |",
        f"| Partial registry rows | {row_status.get('partial', 0)} |",
        f"| Planned registry rows | {row_status.get('planned', 0)} |",
        f"| Rows with at least one open contract axis | {sum(bool(e.missing_contracts()) for e in entries.values())} |",
        f"| Graph IR registered | {graph_status.get('registered', 0)} |",
        f"| Host-materialized | {graph_status.get('host_materialized', 0)} |",
        f"| Runtime-only | {graph_status.get('runtime_only', 0)} |",
        f"| Missing/stub Graph disposition | {graph_status.get('missing', 0) + graph_status.get('stub_required', 0)} |",
        "",
        "Contract axes distinguish implemented rules from explicit by-design closures; "
        "only `partial` and `planned` are open:",
        "",
        "| Contract axis | Complete | Closed by design | Partial | Planned |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis in pc.CONTRACT_FIELDS:
        counts = Counter(entry.contract_status[axis] for entry in entries.values())
        closed_by_design = sum(
            count
            for status, count in counts.items()
            if status != "complete" and pc.is_contract_closed(status)
        )
        lines.append(
            f"| `{axis}` | {counts.get('complete', 0)} | {closed_by_design} | "
            f"{counts.get('partial', 0)} | {counts.get('planned', 0)} |"
        )
    lines.append(REGISTRY_END)
    return "\n".join(lines)


def render_compiler_summary() -> str:
    rows = audit.all_support_rows()
    layer_counts = {
        axis: Counter(row.cells[axis].status for row in rows)
        for axis in audit.LAYER_AXES
    }
    tile_partial = sorted(
        row.op_name for row in rows if row.cells["tile_ir"].status == "partial"
    )
    native_without_bench = sorted(
        row.op_name
        for row in rows
        if row.cells["tile_ir"].status == "fused"
        and row.cells["bench"].status == "none"
    )
    entries = pc.all_primitive_coverages()
    planned_backend = sorted(
        entry.name
        for entry in entries.values()
        if entry.contract_status["backend_kernel"] == "planned"
    )
    planned_vjp = sum(
        entry.contract_status["vjp"] == "planned" for entry in entries.values()
    )
    planned_jvp = sum(
        entry.contract_status["jvp"] == "planned" for entry in entries.values()
    )
    partial_sharding = sum(
        entry.contract_status["sharding_rule"] == "partial"
        for entry in entries.values()
    )

    lines = [
        COMPILER_START,
        f"The compiler-facing table contains **{len(rows)}** callable/IR rows; "
        f"the remaining **{len(entries) - len(rows)}** registry entries are host APIs, "
        "runtime-only surfaces, or planned domain primitives. The aggregate table "
        "chooses the strongest evidence across targets and must not be read as support "
        "on every architecture.",
        "",
        "| Compiler layer | Live status counts |",
        "|---|---|",
    ]
    for axis in audit.LAYER_AXES:
        lines.append(
            f"| `{axis}` | {_status_counts(layer_counts[axis], tuple(audit.AXIS_VALUE_GLYPHS))} |"
        )

    lines.extend(
        [
            "",
            "Exact-target backend-manifest rows:",
            "",
            "| Target | Total | Status counts |",
            "|---|---:|---|",
        ]
    )
    for target, statuses in sorted(bm.manifest_summary().items()):
        lines.append(
            f"| `{target}` | {sum(statuses.values())} | "
            f"{_status_counts(statuses, _MANIFEST_STATUS_ORDER)} |"
        )

    lines.extend(
        [
            "",
            "Current open queues derived from the same sources:",
            "",
            "| Queue | Count | Scope |",
            "|---|---:|---|",
            f"| Tile IR partial | {len(tile_partial)} | "
            f"{', '.join(f'`{name}`' for name in tile_partial) or 'none'} |",
            f"| Backend contract planned | {len(planned_backend)} | "
            f"{', '.join(f'`{name}`' for name in planned_backend) or 'none'} |",
            f"| Native/fused aggregate rows without benchmark inventory | "
            f"{len(native_without_bench)} | See generated support table; this is an "
            "evidence queue, not proof of a missing kernel. |",
            f"| VJP / JVP planned | {planned_vjp} / {planned_jvp} | "
            "Autodiff rules, not forward compiler execution. |",
            f"| Sharding partial | {partial_sharding} | Multi-device/domain proof. |",
            "",
            "`device_verified_abi` and `device_verified_jit` are exact-target execution "
            "proofs. `fused` identifies an owned optimized implementation but still needs "
            "its target-specific execute/compare fixture before it is a complete conformance "
            "claim. `artifact_only` is compile evidence only. Benchmark presence does not "
            "imply selector promotion or timing eligibility.",
            COMPILER_END,
        ]
    )
    return "\n".join(lines)


def _replace_block(text: str, start: str, end: str, replacement: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"dashboard must contain exactly one {start!r}/{end!r} block")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return before + replacement + after


def render_document(text: str | None = None) -> str:
    source = DASHBOARD_PATH.read_text() if text is None else text
    source = _replace_block(
        source, REGISTRY_START, REGISTRY_END, render_registry_summary()
    )
    return _replace_block(
        source, COMPILER_START, COMPILER_END, render_compiler_summary()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    current = DASHBOARD_PATH.read_text()
    rendered = render_document(current)
    if args.check:
        if rendered != current:
            print(
                "standalone dashboard drifted; regenerate with "
                "python -m tessera.compiler.standalone_dashboard --write",
                file=sys.stderr,
            )
            return 1
        print(f"standalone dashboard: {DASHBOARD_PATH} matches live registries")
        return 0
    DASHBOARD_PATH.write_text(rendered)
    print(f"standalone dashboard: wrote {DASHBOARD_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "COMPILER_END",
    "COMPILER_START",
    "DASHBOARD_PATH",
    "REGISTRY_END",
    "REGISTRY_START",
    "render_compiler_summary",
    "render_document",
    "render_registry_summary",
]
