"""Generated compiler-progress rollup.

This dashboard answers the all-up question that the narrower audit tables do
not: how close is Tessera to an end-to-end optimizing compiler, and what is
actually open next?

It intentionally does not require every primitive to be native on every backend
before the compiler gets progress credit. Backend coverage is tracked as a
separate promotion axis so compiler/artifact progress, verifier/test evidence,
runtime integration, and hardware execution remain visible independently.
"""
from __future__ import annotations

import csv as _csv
import io as _io
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ProgressRow:
    scope: str
    item: str
    status: str
    ready: int
    total: int
    open: int
    detail: str
    source: str


def _read_csv(text: str) -> list[dict[str, str]]:
    return list(_csv.DictReader(_io.StringIO(text)))


def _breakdown(rows: Iterable[dict[str, str]], field: str) -> str:
    counts = Counter((r.get(field) or "unknown") for r in rows)
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _status(ready: int, total: int) -> str:
    if total == 0:
        return "missing"
    if ready == total:
        return "closed"
    if ready:
        return "mixed"
    return "open"


def _count_ready(
    rows: Iterable[dict[str, str]], field: str, ready_values: set[str]
) -> int:
    return sum(1 for r in rows if (r.get(field) or "") in ready_values)


def _phase_rows(support_rows: list[dict[str, str]]) -> list[ProgressRow]:
    phases = (
        ("Public Python API", "api", {"public"}),
        ("Frontend capture", "frontend", {"public"}),
        ("Graph IR registration", "graph_ir", {"registered", "not_applicable"}),
        ("Schedule IR", "schedule_ir", {"complete", "not_applicable"}),
        ("Tile IR", "tile_ir", {"complete", "fused", "not_applicable"}),
        ("Target IR native/fused codegen", "target_ir", {"fused"}),
        ("Runtime dispatch readiness", "runtime", {"ready", "fused"}),
        ("Benchmark evidence", "bench", {"benchmarked"}),
    )
    total = len(support_rows)
    out: list[ProgressRow] = []
    for item, field, ready_values in phases:
        ready = _count_ready(support_rows, field, ready_values)
        out.append(
            ProgressRow(
                scope="phase",
                item=item,
                status=_status(ready, total),
                ready=ready,
                total=total,
                open=total - ready,
                detail=_breakdown(support_rows, field),
                source="docs/audit/generated/support_table.csv",
            )
        )
    return out


def _primitive_axis_rows() -> list[ProgressRow]:
    from . import s_series_status

    rows = s_series_status.tally_by_category()
    out: list[ProgressRow] = []
    for axis in s_series_status.DASHBOARD_AXES:
        open_n = sum(int(r[f"{axis}_open"]) for r in rows)
        complete_n = sum(int(r[f"{axis}_complete"]) for r in rows)
        total = open_n + complete_n
        out.append(
            ProgressRow(
                scope="primitive_axis",
                item=axis,
                status=_status(complete_n, total),
                ready=complete_n,
                total=total,
                open=open_n,
                detail=(
                    "primitive contract axis; open means partial or planned, "
                    "not necessarily missing API support"
                ),
                source="docs/audit/generated/s_series_status.md",
            )
        )
    return out


def _verifier_row(verifier_rows: list[dict[str, str]]) -> ProgressRow:
    total = len(verifier_rows)
    ready = sum(1 for r in verifier_rows if r.get("impl_status") == "real")
    return ProgressRow(
        scope="integration",
        item="Verifier coverage",
        status=_status(ready, total),
        ready=ready,
        total=total,
        open=total - ready,
        detail=_breakdown(verifier_rows, "impl_status"),
        source="docs/audit/generated/verifier_coverage.csv",
    )


def _test_row(test_rows: list[dict[str, str]]) -> ProgressRow:
    total = len(test_rows)
    ready = sum(1 for r in test_rows if r.get("is_thinly_tested") == "0")
    return ProgressRow(
        scope="integration",
        item="Direct test evidence",
        status=_status(ready, total),
        ready=ready,
        total=total,
        open=total - ready,
        detail=_breakdown(test_rows, "bucket"),
        source="docs/audit/generated/test_coverage.csv",
    )


def _runtime_row(runtime_rows: list[dict[str, str]]) -> ProgressRow:
    total = len(runtime_rows)
    ready = sum(1 for r in runtime_rows if r.get("executable") == "1")
    return ProgressRow(
        scope="integration",
        item="Runtime execution matrix",
        status=_status(ready, total),
        ready=ready,
        total=total,
        open=total - ready,
        detail=_breakdown(runtime_rows, "target"),
        source="docs/audit/generated/runtime_execution_matrix.csv",
    )


def _surface_row(surface_rows: list[dict[str, str]]) -> ProgressRow:
    total = len(surface_rows)
    ready = sum(1 for r in surface_rows if r.get("status") == "runnable")
    return ProgressRow(
        scope="integration",
        item="Audited repo surfaces",
        status=_status(ready, total),
        ready=ready,
        total=total,
        open=total - ready,
        detail=_breakdown(surface_rows, "status"),
        source="docs/audit/generated/surface_status.csv",
    )


def _abi_row(runtime_abi_rows: list[dict[str, str]]) -> ProgressRow:
    total = len(runtime_abi_rows)
    ready = sum(
        1
        for r in runtime_abi_rows
        if r.get("path") and "stub" not in (r.get("path") or "")
    )
    return ProgressRow(
        scope="integration",
        item="Runtime ABI symbols",
        status=_status(ready, total),
        ready=ready,
        total=total,
        open=total - ready,
        detail=_breakdown(runtime_abi_rows, "backend"),
        source="docs/audit/generated/runtime_abi.csv",
    )


def _runtime_target_detail(runtime_rows: list[dict[str, str]], prefix: str) -> tuple[int, int, str]:
    selected = [r for r in runtime_rows if (r.get("target") or "").startswith(prefix)]
    ready = sum(1 for r in selected if r.get("executable") == "1")
    targets = _breakdown(selected, "target") if selected else "no runtime rows"
    return ready, len(selected), targets


def _target_status_detail(rows: list[dict[str, str]], field: str) -> str:
    if not rows:
        return "no target-map rows"
    return _breakdown(rows, field)


def _target_map_open_count(rows: list[dict[str, str]], field: str) -> int:
    return sum(1 for r in rows if r.get(field) in {"artifact_only", "absent"})


def _pathway_rows(
    runtime_rows: list[dict[str, str]],
    apple_rows: list[dict[str, str]],
    rocm_rows: list[dict[str, str]],
    nvidia_rows: list[dict[str, str]],
) -> list[ProgressRow]:
    specs = (
        (
            "Apple CPU",
            "apple_cpu",
            "cpu_status",
            apple_rows,
            "docs/audit/generated/apple_target_map.csv",
        ),
        (
            "Apple GPU",
            "apple_gpu",
            "gpu_status",
            apple_rows,
            "docs/audit/generated/apple_target_map.csv",
        ),
        (
            "x86 / CPU",
            "cpu",
            "",
            [],
            "docs/audit/generated/runtime_execution_matrix.csv",
        ),
        (
            "ROCm / HIP",
            "rocm",
            "status",
            rocm_rows,
            "docs/audit/generated/rocm_target_map.csv",
        ),
        (
            "CUDA / NVIDIA",
            "nvidia",
            "status",
            nvidia_rows,
            "docs/audit/generated/nvidia_sm90_target_map.csv",
        ),
    )
    out: list[ProgressRow] = []
    for item, target_prefix, target_field, target_rows, source in specs:
        runtime_ready, runtime_total, runtime_detail = _runtime_target_detail(
            runtime_rows, target_prefix
        )
        target_open = _target_map_open_count(target_rows, target_field) if target_field else 0
        target_total = len(target_rows) if target_field else 0
        ready = runtime_ready + max(target_total - target_open, 0)
        total = runtime_total + target_total
        open_n = (runtime_total - runtime_ready) + target_open
        detail = f"runtime: {runtime_detail}"
        if target_field:
            detail += f"; target_map: {_target_status_detail(target_rows, target_field)}"
        out.append(
            ProgressRow(
                scope="codegen_pathway",
                item=item,
                status=_status(ready, total),
                ready=ready,
                total=total,
                open=open_n,
                detail=detail,
                source=source,
            )
        )
    return out


def _target_map_open_rows(
    rocm_rows: list[dict[str, str]], nvidia_rows: list[dict[str, str]]
) -> list[ProgressRow]:
    out: list[ProgressRow] = []
    for item, rows, source in (
        ("ROCm target-map native promotion", rocm_rows, "docs/audit/generated/rocm_target_map.csv"),
        (
            "CUDA target-map native promotion",
            nvidia_rows,
            "docs/audit/generated/nvidia_sm90_target_map.csv",
        ),
    ):
        total = len(rows)
        ready = sum(1 for r in rows if r.get("status") in {"compiled", "hardware_verified"})
        out.append(
            ProgressRow(
                scope="open_work",
                item=item,
                status=_status(ready, total),
                ready=ready,
                total=total,
                open=total - ready,
                detail=_breakdown(rows, "status"),
                source=source,
            )
        )
    return out


def collect_rows() -> list[ProgressRow]:
    from . import audit
    from . import generated_docs as gd
    from . import gpu_target_map
    from . import runtime_abi_audit
    from . import verifier_coverage

    support_rows = _read_csv(audit.render_csv())
    runtime_rows = _read_csv(gd.get("runtime_execution_matrix").render_csv())
    verifier_rows = _read_csv(verifier_coverage.render_csv())
    test_rows = _read_csv(gd.get("test_coverage").render_csv())
    surface_rows = _read_csv(gd.get("surface_status").render_csv())
    runtime_abi_rows = _read_csv(runtime_abi_audit.render_csv())
    apple_rows = _read_csv(gd.get("apple_target_map").render_csv())
    rocm_rows = _read_csv(gpu_target_map.render_csv("rocm"))
    nvidia_rows = _read_csv(gpu_target_map.render_csv("nvidia_sm90"))

    phase = _phase_rows(support_rows)
    primitive = _primitive_axis_rows()
    integration = [
        _verifier_row(verifier_rows),
        _test_row(test_rows),
        _runtime_row(runtime_rows),
        _abi_row(runtime_abi_rows),
        _surface_row(surface_rows),
    ]
    pathways = _pathway_rows(runtime_rows, apple_rows, rocm_rows, nvidia_rows)

    def as_open_work(row: ProgressRow) -> ProgressRow:
        return ProgressRow(
            scope="open_work",
            item=row.item,
            status=row.status,
            ready=row.ready,
            total=row.total,
            open=row.open,
            detail=row.detail,
            source=row.source,
        )

    open_work = [
        as_open_work(next(r for r in phase if r.item == "Target IR native/fused codegen")),
        as_open_work(max(primitive, key=lambda r: r.open)),
        as_open_work(_verifier_row(verifier_rows)),
        as_open_work(_test_row(test_rows)),
        as_open_work(_surface_row(surface_rows)),
        *_target_map_open_rows(rocm_rows, nvidia_rows),
    ]

    return [
        ProgressRow(
            scope="overall",
            item="End-to-end optimizing compiler",
            status="mixed",
            ready=sum(r.ready for r in phase),
            total=sum(r.total for r in phase),
            open=sum(r.open for r in phase),
            detail=(
                "compiler pipeline is live, but native codegen, verifier proof, "
                "and per-backend promotion remain independent axes"
            ),
            source="docs/audit/generated/support_table.csv",
        ),
        *phase,
        *primitive,
        *integration,
        *pathways,
        *open_work,
    ]


def render_csv(rows: list[ProgressRow] | None = None) -> str:
    if rows is None:
        rows = collect_rows()
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(("scope", "item", "status", "ready", "total", "open", "detail", "source"))
    for r in rows:
        writer.writerow((r.scope, r.item, r.status, r.ready, r.total, r.open, r.detail, r.source))
    return buf.getvalue()


def _md_table(rows: list[ProgressRow], columns: tuple[str, ...]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    align = "|" + "|".join("---:" if c in {"Ready", "Total", "Open"} else "---" for c in columns) + "|"
    lines = [header, align]
    for r in rows:
        values: dict[str, str] = {
            "Scope": r.scope,
            "Item": f"`{r.item}`",
            "Status": r.status,
            "Ready": str(r.ready),
            "Total": str(r.total),
            "Open": str(r.open),
            "Detail": r.detail.replace("|", "\\|"),
            "Source": f"`{r.source}`",
        }
        lines.append("| " + " | ".join(values[c] for c in columns) + " |")
    return lines


def render_markdown(rows: list[ProgressRow] | None = None) -> str:
    if rows is None:
        rows = collect_rows()
    by_scope: dict[str, list[ProgressRow]] = {}
    for row in rows:
        by_scope.setdefault(row.scope, []).append(row)

    lines: list[str] = [
        "# Compiler Progress Dashboard (generated)",
        "",
        "> Generated by `python -m tessera.compiler.generated_docs --write compiler_progress`.",
        "> The canonical machine-readable artifact is `compiler_progress.csv`.",
        "",
        "This rollup is oriented around the goal of creating an end-to-end optimizing compiler.",
        "It deliberately separates compiler/artifact support from native execution on every backend.",
        "A row is not marked incomplete merely because Apple, x86, ROCm, and CUDA are not all green.",
        "",
        "## Overall",
        "",
    ]
    lines.extend(_md_table(by_scope.get("overall", []), ("Item", "Status", "Ready", "Total", "Open", "Detail")))
    lines.extend(
        [
            "",
            "## Compiler Phase And IR State",
            "",
        ]
    )
    lines.extend(_md_table(by_scope.get("phase", []), ("Item", "Status", "Ready", "Total", "Open", "Detail")))
    lines.extend(
        [
            "",
            "## Primitive Contract State",
            "",
        ]
    )
    lines.extend(_md_table(by_scope.get("primitive_axis", []), ("Item", "Status", "Ready", "Total", "Open", "Detail")))
    lines.extend(
        [
            "",
            "## Compiler Integration Evidence",
            "",
        ]
    )
    lines.extend(_md_table(by_scope.get("integration", []), ("Item", "Status", "Ready", "Total", "Open", "Detail")))
    lines.extend(
        [
            "",
            "## Code Generation Pathways",
            "",
        ]
    )
    lines.extend(_md_table(by_scope.get("codegen_pathway", []), ("Item", "Status", "Ready", "Total", "Open", "Detail")))
    lines.extend(
        [
            "",
            "## Open Work Summary",
            "",
        ]
    )
    open_rows = sorted(by_scope.get("open_work", []), key=lambda r: (-r.open, r.item))
    lines.extend(_md_table(open_rows, ("Item", "Status", "Open", "Detail", "Source")))
    lines.extend(
        [
            "",
            "## Reading Rules",
            "",
            "- Phase readiness comes from `support_table.csv`; it answers whether the compiler pipeline has that layer for each op.",
            "- Primitive contract readiness comes from `s_series_status.md`; `backend_kernel` is a promotion axis, not an all-up compiler veto.",
            "- Backend/pathway readiness comes from `runtime_execution_matrix.csv` and target maps; executable rows and artifact-only rows are both useful, but they mean different things.",
            "- Integration evidence comes from verifier, test, ABI, and surface dashboards; these are the places to look for the next real blockers.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["ProgressRow", "collect_rows", "render_csv", "render_markdown"]
