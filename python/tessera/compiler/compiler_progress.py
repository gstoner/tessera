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


def _manifest_int(row: dict[str, object], key: str) -> int:
    """Read a generated manifest count without accepting arbitrary objects."""
    value = row.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"manifest field {key!r} must be an integer, got {value!r}")


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
        ("Graph IR registration", "graph_ir",
         {"registered", "host_materialized", "runtime_only", "not_applicable"}),
        ("Schedule IR", "schedule_ir", {"complete", "not_applicable"}),
        ("Tile IR", "tile_ir", {"complete", "fused", "not_applicable", "no_kernel_required"}),
        ("Target IR native/fused codegen", "target_ir", {"fused", "device_verified_jit", "device_verified_abi", "packaged", "not_applicable", "no_kernel_required"}),
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
    open_n = total - ready
    return ProgressRow(
        scope="integration",
        item="Verifier coverage",
        status=_status(ready, total),
        ready=ready,
        total=total,
        open=open_n,
        detail=_breakdown(verifier_rows, "impl_status"),
        source="docs/audit/generated/verifier_coverage.csv",
        next_action=(
            "No action unless this row reopens."
            if open_n == 0
            else "Add real verifier implementations for no_verifier ops, prioritizing native codegen lanes."
        ),
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


def _runtime_target_detail(runtime_rows: list[dict[str, str]], target: str) -> tuple[int, int, str]:
    selected = [r for r in runtime_rows if (r.get("target") or "") == target]
    ready = sum(1 for r in selected if r.get("executable") == "1")
    targets = _breakdown(selected, "target") if selected else "no runtime rows"
    return ready, len(selected), targets


def _pathway_rows(
    runtime_rows: list[dict[str, str]],
    backend_rows: list[dict[str, object]] | None = None,
) -> list[ProgressRow]:
    from . import s_series_status

    if backend_rows is None:
        backend_rows = s_series_status.tally_backend_by_target()
    by_target = {str(row["target"]): row for row in backend_rows}
    # Each row below has one denominator: declared BackendKernelEntry
    # op×target grains. Runtime paths are reported as independent evidence in
    # detail and are never added to that denominator.
    specs: tuple[tuple[str, str, bool], ...] = (
        ("Portable CPU reference", "cpu", True),
        ("x86 / AVX-512", "x86", False),
        ("Apple CPU", "apple_cpu", True),
        ("Apple GPU", "apple_gpu", False),
        ("ROCm / gfx1151", "rocm", False),
        ("NVIDIA SM80", "nvidia_sm80", False),
        ("NVIDIA SM90", "nvidia_sm90", False),
        ("NVIDIA SM100", "nvidia_sm100", False),
        ("NVIDIA SM120", "nvidia_sm120", False),
    )
    out: list[ProgressRow] = []
    for item, target, reference_is_ready in specs:
        manifest = by_target.get(target, {})
        exact = _manifest_int(manifest, "exact_verified")
        implementation = _manifest_int(manifest, "implementation_present")
        reference = _manifest_int(manifest, "reference")
        declared = _manifest_int(manifest, "declared")
        open_artifact = _manifest_int(manifest, "open")
        other = _manifest_int(manifest, "other")
        missing = _manifest_int(manifest, "missing")
        ready = exact + (reference if reference_is_ready else 0)
        open_n = declared - ready
        runtime_ready, runtime_total, runtime_detail = _runtime_target_detail(
            runtime_rows, target
        )
        detail = (
            f"manifest: exact_verified={exact}, implementation_present={implementation}, "
            f"reference={reference}, artifact_or_planned={open_artifact}, other={other}, "
            f"missing_target_row={missing}; runtime_paths: executable={runtime_ready}/"
            f"{runtime_total} ({runtime_detail})"
        )
        out.append(
            ProgressRow(
                scope="codegen_pathway",
                item=item,
                status=_status(ready, declared),
                ready=ready,
                total=declared,
                open=open_n,
                detail=detail,
                source="docs/audit/generated/s_series_status.md",
                next_action=_pathway_next_action(item),
            )
        )
    return out


def _pathway_next_action(item: str) -> str:
    return {
        "Portable CPU reference": "Keep portable reference execution distinct from native x86 proof.",
        "Apple CPU": "Exact-device verify implementation-only rows or retain them explicitly as reference execution.",
        "Apple GPU": "Promote implementation/artifact rows only with exact-device execute-and-compare.",
        "x86 / AVX-512": "Promote implementation-only rows with exact Zen 5 execute-and-compare; keep AMX separately gated.",
        "ROCm / gfx1151": "Promote remaining reference rows only with exact gfx1151 execute-and-compare.",
        "NVIDIA SM80": "Retain as declared/open until architecture-owned execution evidence exists.",
        "NVIDIA SM90": "Keep compile/artifact evidence separate from SM120 exact-device execution.",
        "NVIDIA SM100": "Retain as declared/open until architecture-owned execution evidence exists.",
        "NVIDIA SM120": "Promote artifact rows with SM120 execute-and-compare evidence.",
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
        ready = sum(1 for r in rows if r.get("status") in {"device_verified_jit", "device_verified_abi"})
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
        item="End-to-end optimizing compiler dashboard checks",
        status=_status(closed, total),
        ready=closed,
        total=total,
        open=total - closed,
        detail=(
            f"dashboard_checks_closed={closed}, mixed={counts.get('mixed', 0)}, "
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
    pathways = _pathway_rows(runtime_rows)

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

    open_work_candidates = [
        as_open_work(next(r for r in phase if r.item == "Target IR native/fused codegen")),
        as_open_work(max(primitive, key=lambda r: r.open)),
        as_open_work(_verifier_row(verifier_rows)),
        as_open_work(_test_row(test_rows)),
        as_open_work(_surface_row(surface_rows)),
        *_target_map_open_rows(rocm_rows, nvidia_rows),
    ]
    open_work = [row for row in open_work_candidates if row.open > 0]

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
        "The overall ready/total value counts dashboard checks, not operations or target capability.",
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
            "- Backend/pathway denominators are declared `BackendKernelEntry` op×target grains from `s_series_status.md`; runtime-path counts are independent supporting detail and are never added to those totals.",
            "- Only `device_verified_abi` and `device_verified_jit` close a hardware pathway grain. `fused`/`packaged` prove implementation presence, while reference and artifact/planned rows remain visible without being promoted.",
            "- Integration evidence comes from verifier, test, ABI, and surface dashboards; these are the places to look for the next real blockers.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["ProgressRow", "collect_rows", "render_csv", "render_markdown"]
