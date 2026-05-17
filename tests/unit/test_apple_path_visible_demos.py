"""CI guards for the visible GA + EBM Apple-path demos."""

from __future__ import annotations

from examples.conformance.apple_path_ga_ebm_demos import (
    ebt_tiny_inner_loop_demo,
    rotation_invariant_point_cloud_demo,
    run_all_demos,
)


def test_ga_point_cloud_demo_is_rotation_invariant_and_manifested() -> None:
    result = rotation_invariant_point_cloud_demo()
    assert result["name"] == "ga_rotation_invariant_point_cloud"
    assert result["max_abs_drift"] < 1e-9
    assert result["apple_gpu_manifest"]["clifford_inner"]["status"] == "fused"
    assert result["apple_gpu_manifest"]["clifford_rotor_sandwich"]["status"] == "fused"


def test_ebt_tiny_demo_inner_loop_improves_and_self_verifies() -> None:
    result = ebt_tiny_inner_loop_demo()
    assert result["name"] == "ebt_tiny_inner_loop_refinement"
    assert result["final_mean_energy"] < result["zero_shot_mean_energy"]
    assert result["chosen_mean_energy"] <= result["final_mean_energy"]
    assert result["improvement_ratio"] < 0.25
    assert len(result["self_verify_indices"]) == result["batch"]


def test_visible_demos_run_together() -> None:
    result = run_all_demos()
    assert result["ok"] is True
    assert "ga" in result
    assert "ebm" in result
