"""Contract tests for the single-GPU closeout triage dashboard."""

from __future__ import annotations

import csv
import io
from collections import Counter

from tessera.compiler import backend_manifest as bm
from tessera.compiler import single_gpu_closeout as sgc


def _csv_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(sgc.render_csv())))


def test_single_gpu_closeout_csv_schema_is_stable() -> None:
    rows = _csv_rows()
    assert rows
    assert tuple(rows[0].keys()) == sgc.CSV_COLUMNS


def test_tile_ir_partials_are_all_classified() -> None:
    rows = [r for r in _csv_rows() if r["area"] == "tile_ir"]
    assert rows
    assert {r["current_status"] for r in rows} == {"partial"}
    assert {r["bucket"] for r in rows} <= {
        "fused_reclassify",
        "multi_gpu_deferred",
        "single_gpu_closeable",
    }
    assert all(r["owner"] and r["next_action"] for r in rows)


def test_target_ir_references_are_all_classified() -> None:
    rows = [r for r in _csv_rows() if r["area"] == "target_ir"]
    assert rows
    assert {r["current_status"] for r in rows} == {"reference"}
    assert {r["bucket"] for r in rows} <= {
        "intentional_reference_review",
        "multi_gpu_deferred",
        "single_gpu_promote",
    }
    assert any(r["bucket"] == "single_gpu_promote" for r in rows)


def test_sharding_rules_split_identity_layout_and_distributed() -> None:
    rows = [r for r in _csv_rows() if r["area"] == "sharding_rule"]
    buckets = Counter(r["bucket"] for r in rows)
    assert rows
    assert buckets["needs_mesh_or_domain_proof"] > 0
    assert buckets["local_layout_transform"] > 0
    assert buckets["multi_gpu_deferred"] > 0


def test_backend_kernel_rows_have_pathway_ownership() -> None:
    rows = [r for r in _csv_rows() if r["area"] == "backend_kernel"]
    assert rows
    assert {r["bucket"] for r in rows} <= {
        "backend_pathway_owned",
        "backend_pathway_unowned",
        "multi_gpu_deferred",
    }
    assert any(r["bucket"] == "backend_pathway_owned" for r in rows)


def test_x86_and_rocm_native_backend_rows_are_pathway_owned() -> None:
    native_statuses = {"fused", "compiled", "hardware_verified", "packaged"}
    native_x86_rocm = {
        op_name
        for op_name, entries in bm.all_manifests().items()
        if any(
            entry.target in {"x86", "rocm"}
            and entry.status in native_statuses
            for entry in entries
        )
    }
    assert native_x86_rocm

    offenders = [
        r
        for r in _csv_rows()
        if r["area"] == "backend_kernel"
        and r["op"] in native_x86_rocm
        and r["bucket"] == "backend_pathway_unowned"
    ]
    assert not offenders


def test_host_only_categories_are_not_backend_kernel_unowned() -> None:
    host_only_categories = {
        "aot", "conformance", "control_flow", "data", "diffusion_schedule",
        "extension", "schedule", "serialization", "state_tree", "tokenizer",
        "transform",
    }
    offenders = [
        r
        for r in _csv_rows()
        if r["area"] == "backend_kernel"
        and r["family"] in host_only_categories
        and r["bucket"] == "backend_pathway_unowned"
    ]
    assert not offenders


def test_markdown_points_back_to_plan_and_csv() -> None:
    md = sgc.render_markdown()
    assert "SINGLE_GPU_CLOSEOUT_PLAN.md" in md
    assert "single_gpu_closeout.csv" in md
    assert "## Aggregate" in md
