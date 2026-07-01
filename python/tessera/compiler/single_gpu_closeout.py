"""Single-GPU closeout triage dashboard.

This module turns the live support/primitive dashboards into an operating
queue for ``docs/audit/roadmap/SINGLE_GPU_CLOSEOUT_PLAN.md``.  It does not
change status truth; it classifies open rows into the terminal buckets needed
before the compiler can honestly drive Tile IR / Target IR / verification /
runtime closeout on one GPU.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import audit
from . import backend_manifest as bm
from . import primitive_coverage as pc

_REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = _REPO_ROOT / "docs" / "audit" / "generated" / "single_gpu_closeout.csv"
MD_PATH = _REPO_ROOT / "docs" / "audit" / "generated" / "single_gpu_closeout.md"


CSV_COLUMNS: tuple[str, ...] = (
    "area",
    "op",
    "family",
    "current_status",
    "bucket",
    "owner",
    "next_action",
    "source",
)

_MULTI_GPU_FAMILIES = frozenset({
    "collective",
    "moe_transport",
    "sharding",
})

_MULTI_GPU_OP_PREFIXES = (
    "all_gather",
    "all_reduce",
    "all_to_all",
    "axis_",
    "mesh_",
    "pmap",
    "reduce_scatter",
)

_LOCAL_LAYOUT_FAMILIES = frozenset({
    "indexing",
    "layout_transform",
    "loop_nest",
    "stencil",
})

_DOMAIN_SHARDING_FAMILIES = frozenset({
    "attention",
    "ebm",
    "linalg_decomposition",
    "linalg_solver",
    "moe",
    "sparse",
    "spectral",
    "state_space",
    "state_update",
})


@dataclass(frozen=True)
class CloseoutRow:
    area: str
    op: str
    family: str
    current_status: str
    bucket: str
    owner: str
    next_action: str
    source: str

    def as_csv_row(self) -> list[str]:
        return [
            self.area,
            self.op,
            self.family,
            self.current_status,
            self.bucket,
            self.owner,
            self.next_action,
            self.source,
        ]


def _has_backend_manifest(op_name: str) -> bool:
    return bool(bm.manifest_for(op_name))


def _is_multi_gpu(family: str, op_name: str) -> bool:
    return family in _MULTI_GPU_FAMILIES or op_name.startswith(_MULTI_GPU_OP_PREFIXES)


def _tile_bucket(row: audit.OpSupportRow) -> tuple[str, str, str]:
    target = row.cells["target_ir"].status
    if _is_multi_gpu(row.family, row.op_name):
        return (
            "multi_gpu_deferred",
            "distributed_validation",
            "Move out of the single-GPU denominator; prove with distributed launch or mock-mesh oracle.",
        )
    if target == "fused":
        return (
            "fused_reclassify",
            "compiler_audit",
            "Reclassify Tile IR from partial to fused/not-applicable because Target IR already proves a fused lane.",
        )
    return (
        "single_gpu_closeable",
        "compiler_middle_end",
        "Add Tile IR lowering or explicitly mark fused/not-applicable with a generator-backed rationale.",
    )


def _target_bucket(row: audit.OpSupportRow) -> tuple[str, str, str]:
    if _is_multi_gpu(row.family, row.op_name):
        return (
            "multi_gpu_deferred",
            "distributed_validation",
            "Keep reference lane until a real collective/distributed proof exists.",
        )
    if row.family == "acceptance_verification":
        return (
            "intentional_reference_review",
            "compiler_audit",
            "Confirm this is a verifier/reference-only lane or attach a native target owner.",
        )
    return (
        "single_gpu_promote",
        "backend_codegen",
        "Promote to native/fused Target IR for the selected one-GPU backend or mark intentional reference-only.",
    )


def _benchmark_bucket(row: audit.OpSupportRow) -> tuple[str, str, str]:
    if _is_multi_gpu(row.family, row.op_name):
        return (
            "multi_gpu_deferred",
            "distributed_validation",
            "Benchmark after distributed execution proof; keep out of single-GPU closeout.",
        )
    return (
        "benchmark_required",
        "benchmarks",
        "Add smoke benchmark evidence for the fused/native single-GPU lane.",
    )


def _sharding_bucket(entry: pc.PrimitiveCoverage) -> tuple[str, str, str]:
    category = entry.category or "uncategorized"
    if _is_multi_gpu(category, entry.name):
        return (
            "multi_gpu_deferred",
            "distributed_validation",
            "Requires collective or multi-rank semantics; defer to distributed validation.",
        )
    if category in _LOCAL_LAYOUT_FAMILIES:
        return (
            "local_layout_transform",
            "compiler_middle_end",
            "Prove local layout/shard metadata preservation on one device.",
        )
    if category in _DOMAIN_SHARDING_FAMILIES:
        return (
            "needs_mesh_or_domain_proof",
            "primitive_registry",
            "Keep partial until the domain-specific mock-mesh or one-device shard proof lands.",
        )
    return (
        "single_device_identity",
        "primitive_registry",
        "Add direct one-device identity metadata/value test, then promote sharding_rule.",
    )


def _backend_bucket(entry: pc.PrimitiveCoverage) -> tuple[str, str, str]:
    category = entry.category or "uncategorized"
    if _is_multi_gpu(category, entry.name):
        return (
            "multi_gpu_deferred",
            "distributed_validation",
            "Backend kernel proof needs distributed or collective execution ownership.",
        )
    if entry.metadata.get("backend_kernel_manifest") or _has_backend_manifest(entry.name):
        return (
            "backend_pathway_owned",
            "backend_codegen",
            "Promote by backend/pathway evidence; keep registry axis conservative until target proof is complete.",
        )
    return (
        "backend_pathway_unowned",
        "backend_codegen",
        "Add BackendKernelEntry ownership or classify as intentional reference-only/not-applicable.",
    )


def collect_closeout_rows() -> tuple[CloseoutRow, ...]:
    rows: list[CloseoutRow] = []

    for row in audit.all_support_rows():
        tile_status = row.cells["tile_ir"].status
        if tile_status == "partial":
            bucket, owner, action = _tile_bucket(row)
            rows.append(CloseoutRow(
                "tile_ir",
                row.op_name,
                row.family,
                tile_status,
                bucket,
                owner,
                action,
                "support_table.csv",
            ))

        target_status = row.cells["target_ir"].status
        if target_status == "reference":
            bucket, owner, action = _target_bucket(row)
            rows.append(CloseoutRow(
                "target_ir",
                row.op_name,
                row.family,
                target_status,
                bucket,
                owner,
                action,
                "support_table.csv",
            ))

        bench_status = row.cells["bench"].status
        if row.cells["target_ir"].status == "fused" and bench_status == "none":
            bucket, owner, action = _benchmark_bucket(row)
            rows.append(CloseoutRow(
                "benchmark_evidence",
                row.op_name,
                row.family,
                bench_status,
                bucket,
                owner,
                action,
                "support_table.csv",
            ))

    for entry in pc.all_primitive_coverages().values():
        sharding_status = entry.contract_status.get("sharding_rule", "planned")
        if sharding_status in {"partial", "planned"}:
            bucket, owner, action = _sharding_bucket(entry)
            rows.append(CloseoutRow(
                "sharding_rule",
                entry.name,
                entry.category or "uncategorized",
                sharding_status,
                bucket,
                owner,
                action,
                "primitive_coverage.contract_status.sharding_rule",
            ))

        backend_status = entry.contract_status.get("backend_kernel", "planned")
        if backend_status in {"partial", "planned"}:
            bucket, owner, action = _backend_bucket(entry)
            rows.append(CloseoutRow(
                "backend_kernel",
                entry.name,
                entry.category or "uncategorized",
                backend_status,
                bucket,
                owner,
                action,
                "primitive_coverage.contract_status.backend_kernel",
            ))

    return tuple(sorted(rows, key=lambda r: (r.area, r.family, r.op, r.bucket)))


def render_csv(rows: tuple[CloseoutRow, ...] | None = None) -> str:
    rows = rows if rows is not None else collect_closeout_rows()
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in rows:
        writer.writerow(row.as_csv_row())
    return buf.getvalue()


def _breakdown(rows: tuple[CloseoutRow, ...], field: str) -> str:
    counts = Counter(getattr(row, field) for row in rows)
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-"


def render_markdown(rows: tuple[CloseoutRow, ...] | None = None) -> str:
    rows = rows if rows is not None else collect_closeout_rows()
    by_area: dict[str, list[CloseoutRow]] = {}
    for row in rows:
        by_area.setdefault(row.area, []).append(row)

    lines: list[str] = [
        "# Single-GPU Closeout Triage (generated)",
        "",
        "> **Generated by `python -m tessera.compiler.generated_docs --write single_gpu_closeout`.**",
        "> Do not edit by hand. The canonical machine-readable artifact is",
        "> `single_gpu_closeout.csv`.",
        "",
        "This dashboard classifies the open rows targeted by",
        "`docs/audit/roadmap/SINGLE_GPU_CLOSEOUT_PLAN.md`. It is a triage",
        "view only; status truth remains in the support, primitive, verifier,",
        "test, ABI, and surface dashboards.",
        "",
        "## Aggregate",
        "",
        "| Area | Rows | Buckets | Owners |",
        "|---|---:|---|---|",
    ]
    for area in sorted(by_area):
        area_rows = tuple(by_area[area])
        lines.append(
            f"| `{area}` | {len(area_rows)} | {_breakdown(area_rows, 'bucket')} | "
            f"{_breakdown(area_rows, 'owner')} |"
        )
    lines.extend([
        "",
        "## Rows",
        "",
        "| Area | Op | Family | Status | Bucket | Owner | Next action |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in rows:
        action = row.next_action.replace("|", "\\|")
        lines.append(
            f"| `{row.area}` | `{row.op}` | {row.family} | {row.current_status} | "
            f"`{row.bucket}` | {row.owner} | {action} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_dashboard(csv_path: Path = CSV_PATH) -> tuple[Path, Path]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = csv_path.with_suffix(".md")
    rows = collect_closeout_rows()
    csv_path.write_text(render_csv(rows))
    md_path.write_text(render_markdown(rows))
    return csv_path, md_path


__all__ = [
    "CSV_COLUMNS",
    "CSV_PATH",
    "MD_PATH",
    "CloseoutRow",
    "collect_closeout_rows",
    "render_csv",
    "render_markdown",
    "write_dashboard",
]
