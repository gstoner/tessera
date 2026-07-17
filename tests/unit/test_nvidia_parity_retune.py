from __future__ import annotations

import json
from pathlib import Path

from tessera.compiler.emit.nvidia_cuda import (
    _synthesize_flash_fwd_multiwarp_cuda,
)


ROOT = Path(__file__).resolve().parents[2]


def test_multiwarp_forward_attention_is_cuda_owned_online_softmax():
    for warps in (4, 8):
        source = _synthesize_flash_fwd_multiwarp_cuda(warps)
        assert f"#define WARPS {warps}" in source
        assert "threadIdx.x>>5" in source
        assert "__shfl_down_sync" in source
        assert "float mi=-INFINITY,li=0.f" in source
        assert "softcap*tanhf" in source


def test_committed_attention_forward_matrix_retains_dual_timing_and_resources():
    path = ROOT / "benchmarks/baselines/nvidia_sm120_attention_forward_schedules.json"
    data = json.loads(path.read_text())
    assert data["schema"] == "tessera.nvidia.attention-forward-schedules.v1"
    assert len(data["rows"]) == 8
    assert {row["candidate"] for row in data["rows"]} == {
        "warp_per_query_w4", "warp_per_query_w8"}
    assert {row["case"] for row in data["rows"]} == {
        "mha_512", "causal_ragged_1009", "gqa_window_ragged",
        "mqa_bias_softcap"}
    for row in data["rows"]:
        assert len(row["runs"]) == 2
        assert max(run["max_abs_error"] for run in row["runs"]) < 1e-5
        assert all(run["device_event_ms"] > 0 and run["end_to_end_ms"] > 0
                   for run in row["runs"])
        assert row["resource"]["spill_evidence_complete"]
        assert not row["resource"]["spills_detected"]
        assert row["resource"]["resource_fingerprint"].startswith("sha256:")
        assert row["stable"] == (row["device_stable"] and
                                  row["end_to_end_stable"])


def test_committed_legacy_retune_retains_launch_collapse_and_oracles():
    path = ROOT / "benchmarks/baselines/nvidia_sm120_legacy_retune.json"
    data = json.loads(path.read_text())
    assert data["schema"] == "tessera.nvidia.legacy-retune.v1"
    rows = {(row["case"], row["candidate"]): row for row in data["rows"]}
    assert rows[("grouped_gemm", "single_grouped_launch")]["launches_per_call"] == 1
    assert rows[("grouped_gemm", "legacy_per_expert")]["launches_per_call"] == 4
    assert rows[("grouped_swiglu", "collapsed_grouped")]["launches_per_call"] == 4
    assert rows[("grouped_swiglu", "legacy_per_expert")]["launches_per_call"] == 32
    for row in rows.values():
        assert len(row["runs"]) == 2
        assert all(run["device_event_ms"] > 0 and run["end_to_end_ms"] > 0
                   for run in row["runs"])
        assert max(run["max_abs_error"] for run in row["runs"]) < 5e-3
        assert row["resource_fingerprints"]
        assert row["stable"] == (row["device_stable"] and
                                  row["end_to_end_stable"])
    for candidate in ("collapsed_grouped", "legacy_per_expert"):
        resource_kinds = {resource.get("row_kind") for resource in
                          rows[("grouped_swiglu", candidate)]["resources"]}
        assert "generated_epilogue" in resource_kinds
    assert data["transport_dependency"] == "NVIDIA-PARITY-TRANSPORT"
    assert data["retained_transport_rows"] and data["retained_kv_rows"]
