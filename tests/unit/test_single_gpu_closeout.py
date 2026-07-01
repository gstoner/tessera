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
    if not rows:
        return
    assert {r["current_status"] for r in rows} == {"partial"}
    assert {r["bucket"] for r in rows} <= {
        "fused_reclassify",
        "multi_gpu_deferred",
        "single_gpu_closeable",
    }
    assert all(r["owner"] and r["next_action"] for r in rows)


def test_target_ir_references_are_all_classified() -> None:
    rows = [r for r in _csv_rows() if r["area"] == "target_ir"]
    if not rows:
        return
    assert {r["current_status"] for r in rows} == {"reference"}
    assert {r["bucket"] for r in rows} <= {
        "intentional_reference_review",
        "multi_gpu_deferred",
        "single_gpu_promote",
    }


def test_single_gpu_tile_and_target_ir_closeout_is_empty() -> None:
    rows = [
        r for r in _csv_rows()
        if r["area"] in {"tile_ir", "target_ir"}
        and r["bucket"] in {"single_gpu_closeable", "single_gpu_promote"}
    ]
    assert rows == []


def test_sharding_rules_split_identity_layout_and_distributed() -> None:
    rows = [r for r in _csv_rows() if r["area"] == "sharding_rule"]
    buckets = Counter(r["bucket"] for r in rows)
    assert rows
    assert {r["current_status"] for r in rows} <= {"partial", "planned"}
    assert {r["bucket"] for r in rows} <= {
        "local_layout_transform",
        "multi_gpu_deferred",
        "needs_mesh_or_domain_proof",
        "single_device_identity",
    }
    assert buckets["needs_mesh_or_domain_proof"] > 0
    assert buckets["local_layout_transform"] > 0
    assert buckets["multi_gpu_deferred"] > 0
    assert all(r["owner"] and r["next_action"] for r in rows)


def test_batching_rule_closeout_is_empty() -> None:
    rows = [r for r in _csv_rows() if r["area"] == "batching_rule"]
    assert rows == []


def test_backend_kernel_rows_have_pathway_ownership() -> None:
    rows = [r for r in _csv_rows() if r["area"] == "backend_kernel"]
    assert rows
    assert {r["bucket"] for r in rows} <= {
        "backend_pathway_owned",
        "backend_pathway_unowned",
        "multi_gpu_deferred",
    }
    assert any(r["bucket"] == "backend_pathway_owned" for r in rows)
    assert all(r["owner"] and r["next_action"] for r in rows)


def test_backend_kernel_rows_are_not_unowned() -> None:
    offenders = [
        r for r in _csv_rows()
        if r["area"] == "backend_kernel"
        and r["bucket"] == "backend_pathway_unowned"
    ]
    assert offenders == []


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


def test_compute_tail_backend_rows_are_pathway_owned() -> None:
    compute_tail_families = {
        "attention",
        "diffusion",
        "indexing",
        "layout_transform",
        "loss",
        "memory",
        "model_layer",
        "normalization",
        "pooling",
        "position_encoding",
        "quantization",
        "recurrent",
        "rng",
        "rotary_embedding",
        "state_update",
        "stencil",
        "vision",
    }
    offenders = [
        r for r in _csv_rows()
        if r["area"] == "backend_kernel"
        and r["family"] in compute_tail_families
        and r["bucket"] == "backend_pathway_unowned"
    ]
    assert not offenders


def test_compute_tail_reference_manifests_cover_representative_ops() -> None:
    for op in (
        "rng_beta",
        "contrastive_loss",
        "quantize_int4",
        "image_resize",
        "adaptive_pool",
        "gru_cell",
        "conv1d",
        "patchify",
        "edm_precondition",
        "cross_attention",
        "depthwise_conv1d",
        "memory_read",
    ):
        entries = {entry.target: entry for entry in bm.manifest_for(op)}
        assert entries["cpu"].status == "reference"
        assert entries["x86"].status == "reference"
        assert entries["apple_cpu"].status == "reference"
        assert entries["rocm"].status == "planned"
        assert "planned_kernel" in entries["rocm"].feature_flags
        for target, arch in (
            ("nvidia_sm80", "sm_80"),
            ("nvidia_sm90", "sm_90a"),
            ("nvidia_sm100", "sm_100a"),
            ("nvidia_sm120", "sm_120a"),
        ):
            assert entries[target].status == "planned"
            assert entries[target].cuda_arch_min == arch
            assert entries[target].nvcc_version_min == "13.3"
            assert "planned_kernel" in entries[target].feature_flags


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
