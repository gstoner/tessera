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
        assert row["stable"]
        assert row["sampling"]["run_cohorts"] == (
            "disjoint_interleaved_samples")


def test_committed_legacy_retune_retains_launch_collapse_and_oracles():
    path = ROOT / "benchmarks/baselines/nvidia_sm120_legacy_retune.json"
    data = json.loads(path.read_text())
    assert data["schema"] == "tessera.nvidia.legacy-retune.v1"
    rows = {(row["case"], row["candidate"]): row for row in data["rows"]}
    assert rows[("grouped_gemm", "single_grouped_launch")]["launches_per_call"] == 1
    assert rows[("grouped_gemm", "legacy_per_expert")]["launches_per_call"] == 4
    assert rows[("grouped_swiglu", "collapsed_grouped")]["launches_per_call"] == 4
    assert rows[("grouped_swiglu", "legacy_per_expert")]["launches_per_call"] == 32
    assert rows[("f32_square", "shipped_tf32")]["shape"] == "512x512x512"
    assert rows[("grouped_gemm", "single_grouped_launch")]["shape"] == (
        "1024x384x256x5")
    for row in rows.values():
        assert len(row["runs"]) == 2
        assert all(run["device_event_ms"] > 0 and run["end_to_end_ms"] > 0
                   for run in row["runs"])
        assert max(run["max_abs_error"] for run in row["runs"]) < 5e-3
        assert row["resource_fingerprints"]
        assert row["stable"] == (row["device_stable"] and
                                  row["end_to_end_stable"])
        assert row["stable"]
        assert row["sampling"]["run_cohorts"] == (
            "disjoint_interleaved_samples")
        assert all(len(run["device_batch_medians_ms"]) == 10
                   for run in row["runs"])
    for candidate in ("collapsed_grouped", "legacy_per_expert"):
        resource_kinds = {resource.get("row_kind") for resource in
                          rows[("grouped_swiglu", candidate)]["resources"]}
        assert "generated_epilogue" in resource_kinds
    assert data["transport_dependency"] == "NVIDIA-PARITY-TRANSPORT"
    assert data["retained_transport_rows"] and data["retained_kv_rows"]


def test_committed_retune_stability_re_derives_from_its_own_runs():
    """`noise_policy` had a producer and no consumer.

    The ratchet asserted `noise_policy == 0.03` -- that the constant had not
    changed -- and never compared it to a measurement. So a regressed recording
    whose two runs disagree by 40% keeps `stable: true` and passes. Every
    stability and consensus flag is now recomputed from `runs`, the only field
    in the artifact that is measurement rather than conclusion.
    """
    import collections

    from tests._support.nvidia import (
        retune_stability_violations,
        retune_winner_consensus_violations,
    )

    root = Path(__file__).parents[2]
    payload = json.loads((root / "benchmarks/baselines"
                          / "nvidia_sm120_legacy_retune.json").read_text())
    noise = payload["noise_policy"]
    assert isinstance(noise, float) and 0.0 < noise < 1.0

    by_case = collections.defaultdict(list)
    for index, row in enumerate(payload["rows"]):
        assert retune_stability_violations(row, noise) == [], (
            f"row[{index}] ({row['case']}/{row['candidate']}) contradicts its "
            f"own runs")
        by_case[row["case"]].append(row)
    for case, rows in by_case.items():
        assert retune_winner_consensus_violations(rows) == [], case
    assert len(payload["rows"]) == 8 and len(by_case) == 4


def test_the_retune_re_derivation_catches_a_forged_row():
    """A checker that only ever passes proves nothing about the artifact."""
    import copy

    from tests._support.nvidia import (
        retune_stability_violations,
        retune_winner_consensus_violations,
    )

    root = Path(__file__).parents[2]
    payload = json.loads((root / "benchmarks/baselines"
                          / "nvidia_sm120_legacy_retune.json").read_text())
    noise = payload["noise_policy"]
    row = payload["rows"][0]
    assert retune_stability_violations(row, noise) == []

    # Runs pushed far apart while the flag still claims stability.
    drifted = copy.deepcopy(row)
    drifted["runs"][1]["device_event_ms"] = (
        drifted["runs"][0]["device_event_ms"] * 1.4)
    assert "device_stable" in retune_stability_violations(drifted, noise)

    # The composite flag disagreeing with its own components.
    inconsistent = copy.deepcopy(row)
    inconsistent["stable"] = not (inconsistent["device_stable"]
                                  and inconsistent["end_to_end_stable"])
    assert "stable" in retune_stability_violations(inconsistent, noise)

    # Runs removed entirely -- unverifiable is not the same as fine.
    stripped = copy.deepcopy(row)
    stripped["runs"] = []
    assert retune_stability_violations(stripped, noise) == ["missing_paired_runs"]

    # A case where the loser claims the winner consensus.
    case = [copy.deepcopy(r) for r in payload["rows"]
            if r["case"] == row["case"]]
    if len(case) > 1:
        for entry in case:
            entry["device_winner_consensus"] = True
        assert any("device_winner_consensus" in v
                   for v in retune_winner_consensus_violations(case))
