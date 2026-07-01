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
    next_action: str = ""


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
        ("Target IR native/fused codegen", "target_ir", {"fused", "compiled", "hardware_verified", "packaged", "not_applicable"}),
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
                next_action=_phase_next_action(item),
            )
        )
    return out


def _phase_next_action(item: str) -> str:
    return {
        "Tile IR": "Close partial Tile IR rows or explicitly classify them as fused/not-applicable.",
        "Target IR native/fused codegen": "Promote high-use reference rows into native/fused Target IR or mark intentional reference-only lanes.",
        "Benchmark evidence": "Attach benchmarks to native/hardware-promoted rows first.",
    }.get(item, "Keep this layer drift-gated through support_table.csv.")


def _primitive_axis_rows() -> list[ProgressRow]:
    from . import s_series_status

    rows = s_series_status.tally_by_category()
    out: list[ProgressRow] = []
    for axis in s_series_status.DASHBOARD_AXES:
        open_n = sum(int(str(r[f"{axis}_open"])) for r in rows)
        complete_n = sum(int(str(r[f"{axis}_complete"])) for r in rows)
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
                next_action=_primitive_axis_next_action(axis),
            )
        )
    return out


def _primitive_axis_next_action(axis: str) -> str:
    return {
        "backend_kernel": "Promote by backend/pathway; do not treat every target as an all-up compiler veto.",
        "sharding_rule": "Prioritize model-facing collectives, layout, memory, and optimizer rows.",
    }.get(axis, "No action unless this row reopens.")


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
        next_action="Add real verifier implementations for no_verifier ops, prioritizing native codegen lanes.",
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
        next_action="Convert structural_only and needs_direct_test rows into direct compare fixtures; keep hardware_gated tied to backend proof.",
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
        next_action="Add rows only when a launch path actually executes.",
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
        next_action="Graduate compile_only/scaffold entries that exercise compiler pathways; archive dead surfaces.",
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
        next_action="Reduce stub-only ABI rows where a backend claims native execution.",
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
    specs: tuple[tuple[str, str, str, list[dict[str, str]], str], ...] = (
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
                next_action=_pathway_next_action(item),
            )
        )
    return out


def _pathway_next_action(item: str) -> str:
    return {
        "Apple CPU": "Keep as regression baseline for CPU value-call/runtime ABI.",
        "Apple GPU": "Close the remaining absent target-map lane or document why it is host-only.",
        "x86 / CPU": "Keep native CPU and numpy reference lanes separate in runtime proofs.",
        "ROCm / HIP": "Close the artifact-only target-map tail and preserve CDNA as hardware-gated.",
        "CUDA / NVIDIA": "Promote artifact-only rows with execute-and-compare, starting from sm_120 matmul adjacency and attention.",
    }[item]


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
                next_action=(
                    "Promote artifact_only rows with hardware execute-and-compare "
                    "or move them to an explicit hardware-gated bucket."
                ),
            )
        )
    return out


def _dashboard_map_rows() -> list[ProgressRow]:
    entries = (
        (
            "compiler_progress",
            "primary",
            "Reader-facing all-up rollup: phase/IR state, primitive axes, integration evidence, codegen pathways, and open work.",
            "docs/audit/generated/compiler_progress.md",
            "Start here for compiler-progress status.",
        ),
        (
            "support_table",
            "drilldown",
            "Per-op phase support across API/frontend/Graph/Schedule/Tile/Target/runtime/bench.",
            "docs/audit/generated/support_table.csv",
            "Use when an open phase row needs the actual op list.",
        ),
        (
            "s_series_status",
            "drilldown",
            "Primitive contract axes by category; owns batching/transpose/sharding/lowering/backend-kernel status.",
            "docs/audit/generated/s_series_status.md",
            "Use for primitive-contract promotion planning.",
        ),
        (
            "standalone_primitive_coverage",
            "companion",
            "Historical primitive registry snapshot and S-series grouping; not the all-up completion gate.",
            "docs/audit/standalone_primitive_coverage.md",
            "Keep for primitive vocabulary/history; avoid using it as the primary progress summary.",
        ),
        (
            "op_target_conformance",
            "drilldown",
            "Op-by-target conformance cells; useful for target-specific holes, not overall compiler health.",
            "docs/audit/op_target_conformance.csv",
            "Use after a backend/pathway row points at a target-specific gap.",
        ),
        (
            "runtime_execution_matrix",
            "primary_evidence",
            "Executable compiler paths and launch outcomes by target.",
            "docs/audit/generated/runtime_execution_matrix.csv",
            "Use for native execution claims.",
        ),
        (
            "target_maps",
            "drilldown",
            "Apple/ROCm/CUDA native/artifact status per backend op family.",
            "docs/audit/generated/*_target_map.csv",
            "Use for backend promotion queues.",
        ),
        (
            "verifier_coverage",
            "integration",
            "ODS/C++ verifier implementation coverage.",
            "docs/audit/generated/verifier_coverage.csv",
            "Use for IR legality hardening work.",
        ),
        (
            "test_coverage",
            "integration",
            "Direct, structural, family, and hardware-gated test evidence by op.",
            "docs/audit/generated/test_coverage.csv",
            "Use for proof-quality triage.",
        ),
        (
            "runtime_abi",
            "integration",
            "C ABI symbols and implementation/stub split.",
            "docs/audit/generated/runtime_abi.csv",
            "Use when runtime/backend claims need symbol-level evidence.",
        ),
        (
            "surface_status",
            "integration",
            "Examples, benchmarks, research, tools, and tests surface status.",
            "docs/audit/generated/surface_status.csv",
            "Use to find runnable proof surfaces and stale scaffolds.",
        ),
        (
            "contract_consumers / effect_lattice / tsol",
            "specialized",
            "Focused contract-pass, effect-system, and TSOL views.",
            "docs/audit/generated/",
            "Use only when the specialized subsystem is the question.",
        ),
    )
    return [
        ProgressRow(
            scope="dashboard_map",
            item=item,
            status=status,
            ready=0,
            total=0,
            open=0,
            detail=detail,
            source=source,
            next_action=next_action,
        )
        for item, status, detail, source, next_action in entries
    ]


def _overall_row(rows: list[ProgressRow], open_work: list[ProgressRow]) -> ProgressRow:
    counts = Counter(r.status for r in rows)
    closed = counts.get("closed", 0)
    total = len(rows)
    largest = ", ".join(r.item for r in sorted(open_work, key=lambda r: -r.open)[:3])
    return ProgressRow(
        scope="overall",
        item="End-to-end optimizing compiler",
        status=_status(closed, total),
        ready=closed,
        total=total,
        open=total - closed,
        detail=(
            f"closed={closed}, mixed={counts.get('mixed', 0)}, "
            f"open={counts.get('open', 0)}, primary_open={largest}"
        ),
        source="docs/audit/generated/compiler_progress.csv",
        next_action="Drive the largest open-work rows without collapsing backend promotion into all-up compiler status.",
    )


def collect_rows() -> list[ProgressRow]:
    from . import audit
    from . import generated_docs as gd
    from . import gpu_target_map
    from . import runtime_abi_audit
    from . import verifier_coverage

    def _csv_of(name: str) -> list[dict[str, str]]:
        # The CSV-gated docs accessed here always carry a render_csv; assert it
        # so the (Optional) callable is non-None for the type checker.
        doc = gd.get(name)
        assert doc.render_csv is not None, f"{name} has no render_csv"
        return _read_csv(doc.render_csv())

    support_rows = _read_csv(audit.render_csv())
    runtime_rows = _csv_of("runtime_execution_matrix")
    verifier_rows = _read_csv(verifier_coverage.render_csv())
    test_rows = _csv_of("test_coverage")
    surface_rows = _csv_of("surface_status")
    runtime_abi_rows = _read_csv(runtime_abi_audit.render_csv())
    apple_rows = _csv_of("apple_target_map")
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
            next_action=row.next_action,
        )

    open_work = [
        as_open_work(next(r for r in phase if r.item == "Target IR native/fused codegen")),
        as_open_work(max(primitive, key=lambda r: r.open)),
        as_open_work(_verifier_row(verifier_rows)),
        as_open_work(_test_row(test_rows)),
        as_open_work(_surface_row(surface_rows)),
        *_target_map_open_rows(rocm_rows, nvidia_rows),
    ]

    dashboard_map = _dashboard_map_rows()
    summary_rows = [*phase, *primitive, *integration, *pathways]

    return [
        _overall_row(summary_rows, open_work),
        *phase,
        *primitive,
        *integration,
        *pathways,
        *open_work,
        *dashboard_map,
    ]


def render_csv(rows: list[ProgressRow] | None = None) -> str:
    if rows is None:
        rows = collect_rows()
    buf = _io.StringIO()
    writer = _csv.writer(buf, lineterminator="\n")
    writer.writerow(
        (
            "scope",
            "item",
            "status",
            "ready",
            "total",
            "open",
            "detail",
            "source",
            "next_action",
        )
    )
    for r in rows:
        writer.writerow(
            (
                r.scope,
                r.item,
                r.status,
                r.ready,
                r.total,
                r.open,
                r.detail,
                r.source,
                r.next_action,
            )
        )
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
            "Next": r.next_action.replace("|", "\\|"),
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
    lines.extend(
        _md_table(
            by_scope.get("overall", []),
            ("Item", "Status", "Ready", "Total", "Open", "Detail", "Next"),
        )
    )
    lines.extend(
        [
            "",
            "## Compiler Phase And IR State",
            "",
        ]
    )
    lines.extend(
        _md_table(
            by_scope.get("phase", []),
            ("Item", "Status", "Ready", "Total", "Open", "Detail", "Next"),
        )
    )
    lines.extend(
        [
            "",
            "## Primitive Contract State",
            "",
        ]
    )
    lines.extend(
        _md_table(
            by_scope.get("primitive_axis", []),
            ("Item", "Status", "Ready", "Total", "Open", "Detail", "Next"),
        )
    )
    lines.extend(
        [
            "",
            "## Compiler Integration Evidence",
            "",
        ]
    )
    lines.extend(
        _md_table(
            by_scope.get("integration", []),
            ("Item", "Status", "Ready", "Total", "Open", "Detail", "Next"),
        )
    )
    lines.extend(
        [
            "",
            "## Code Generation Pathways",
            "",
        ]
    )
    lines.extend(
        _md_table(
            by_scope.get("codegen_pathway", []),
            ("Item", "Status", "Ready", "Total", "Open", "Detail", "Next"),
        )
    )
    lines.extend(
        [
            "",
            "## Open Work Summary",
            "",
        ]
    )
    open_rows = sorted(by_scope.get("open_work", []), key=lambda r: (-r.open, r.item))
    lines.extend(_md_table(open_rows, ("Item", "Status", "Open", "Detail", "Next", "Source")))
    lines.extend(
        [
            "",
            "## Dashboard Map",
            "",
        ]
    )
    lines.extend(
        _md_table(
            by_scope.get("dashboard_map", []),
            ("Item", "Status", "Detail", "Next", "Source"),
        )
    )
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
