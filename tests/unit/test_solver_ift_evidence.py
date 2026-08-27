from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.compiler.implicit_solver import build_solver_ift_contract


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "target,filename",
    [
        ("x86", "x86_zen5_solver_ift_evidence.json"),
        ("rocm_gfx1151", "rocm_gfx1151_solver_ift_evidence.json"),
        ("nvidia_sm120", "nvidia_sm120_solver_ift_evidence.json"),
    ],
)
def test_solver_ift_packet_has_current_lineage_and_complete_backward(target: str, filename: str) -> None:
    packet = json.loads((ROOT / "benchmarks" / "baselines" / filename).read_text())
    assert packet["schema"] == "tessera.solver_ift.evidence.v1"
    assert packet["artifact_hash"] == build_solver_ift_contract(target=target, shape=packet["shape"])["artifact_hash"]
    assert packet["timing"]["complete_backward"] is True
    assert len(packet["timing"]["samples_ns"]) >= 20
    assert packet["retained_residual_bytes"] == 4 * 3 * 257
    assert packet["numerical"]["passed"] is True
    assert max(packet["numerical"]["max_abs_error_by_phase"].values()) <= 1e-6
    if target != "x86":
        assert packet["promotion"]["performance_eligible"] is False


def test_nvidia_general_solver_packet_binds_all_cuda_children() -> None:
    packet = json.loads(
        (ROOT / "benchmarks" / "baselines" /
         "nvidia_sm120_general_solver_evidence.json").read_text()
    )
    assert packet["schema"] == "tessera.general_solver.evidence.v1"
    assert packet["target"] == "nvidia_sm120"
    assert packet["architecture"] == "sm120"
    assert packet["matrix_free"] is True
    assert packet["true_residual_check"] is True
    assert set(packet["admitted_child_families"]) == {
        "binary_math", "unary_math", "reduction", "comparison", "where",
        "matmul_ieee",
    }
    assert "missing_matmul_math_mode" in packet["fail_closed_policies"]
    assert set(packet["child_digests"]) == {
        "residual", "solution_jvp", "solution_vjp",
        "parameter_jvp", "parameter_vjp",
    }
    assert all(len(value) == 64 for value in packet["child_digests"].values())
    assert len(packet["timing"]["samples_ns"]) >= 20
    assert packet["timing"]["complete_backward"] is True
    assert packet["numerical"]["passed"] is True
    assert max(packet["numerical"]["max_abs_error_by_phase"].values()) <= 1e-6
    assert packet["promotion"]["performance_eligible"] is False


def test_nvidia_krylov_packet_proves_device_resident_cg_state() -> None:
    packet = json.loads(
        (ROOT / "benchmarks" / "baselines" /
         "nvidia_sm120_solver_krylov_evidence.json").read_text()
    )
    assert packet["schema"] == "tessera.nvidia.krylov.evidence.v1"
    assert packet["target"] == "nvidia_sm120"
    assert packet["algorithm"] == "cg"
    assert packet["operator"] == "positive_diagonal_spd_v1"
    assert packet["state_residency"] == "single_launch_device_resident"
    assert set(packet["device_state"]) == {
        "solution", "residual", "direction", "matvec", "dot_reductions",
        "convergence",
    }
    assert packet["storage"] == "f32" and packet["accumulation"] == "f32"
    assert len(packet["timing"]["samples_ns"]) >= 20
    assert packet["timing"]["complete_solve"] is True
    assert packet["numerical"]["passed"] is True
    assert packet["numerical"]["max_abs_solution_error"] <= 2e-6
    assert packet["numerical"]["max_abs_equation_error"] <= 2e-6
    assert packet["promotion"]["performance_eligible"] is False


def test_nvidia_solver_child_packet_covers_all_typed_families() -> None:
    packet = json.loads(
        (ROOT / "benchmarks" / "baselines" /
         "nvidia_sm120_solver_children_evidence.json").read_text()
    )
    assert packet["schema"] == "tessera.nvidia.solver_children.evidence.v1"
    assert packet["target"] == "nvidia_sm120"
    assert set(packet["families"]) == {
        "unary", "reduction", "comparison", "where", "matmul_ieee",
        "matmul_native_lowp",
    }
    assert set(packet["storage"]) == {"f32", "f16", "bf16"}
    assert packet["accumulation"] == "f32"
    assert packet["matmul_math_mode"] == "ieee"
    assert set(packet["max_abs_error_by_storage_and_case"]) == {
        "f32", "f16", "bf16",
    }
    assert packet["passed"] is True
    assert packet["native_lowp_matmul"]["physical_route"] == "mma.sync"
    assert packet["native_lowp_matmul"]["missing_storage_policy"] == "fail_closed"
    assert set(packet["native_lowp_matmul"]["max_abs_error"]) == {"f16", "bf16"}
    assert packet["promotion"]["performance_eligible"] is False


def test_nvidia_dense_krylov_packet_proves_resident_arnoldi_and_multi_cta() -> None:
    packet = json.loads(
        (ROOT / "benchmarks" / "baselines" /
         "nvidia_sm120_solver_dense_krylov_evidence.json").read_text()
    )
    assert packet["schema"] == "tessera.nvidia.dense_krylov.evidence.v1"
    assert packet["target"] == "nvidia_sm120"
    assert packet["operator"] == "arbitrary_dense_row_major_v1"
    assert packet["orthogonalization"] == "twice_modified_gram_schmidt"
    assert packet["true_residual_required"] is True
    assert packet["passed"] is True
    assert {(case["algorithm"], case["storage"]) for case in packet["cases"]} == {
        ("cg", "f32"), ("gmres", "f32"),
        ("gmres", "f16"), ("gmres", "bf16"),
    }
    assert all(case["state_residency"] == "single_cooperative_launch_device_resident"
               for case in packet["cases"])
    assert all(case["reduction"] == "deterministic_multi_cta_two_level"
               for case in packet["cases"])
    assert all(case["reduction_ctas"] >= 2 for case in packet["cases"])
    assert all(case["numerical"]["passed"] for case in packet["cases"])
    assert packet["promotion"]["performance_eligible"] is True


def test_nvidia_dense_krylov_performance_packet_ratchets_scaling_matrix() -> None:
    packet = json.loads(
        (ROOT / "benchmarks" / "baselines" /
         "nvidia_sm120_solver_krylov_performance.json").read_text()
    )
    assert packet["schema"] == "tessera.benchmark.ratchet.v1"
    assert packet["device"] == "nvidia:sm_120"
    rows = packet["rows"]
    assert len(rows) == 12
    assert {row["op"] for row in rows} == {"dense_cg", "dense_gmres"}
    assert {row["shape"] for row in rows} == {
        "513x513", "1025x1025", "2049x2049",
    }
    assert {row["timing_domain"] for row in rows} == {
        "end_to_end", "device_event",
    }
    for algorithm in ("dense_cg", "dense_gmres"):
        device_rows = sorted(
            (row for row in rows if row["op"] == algorithm and
             row["timing_domain"] == "device_event"),
            key=lambda row: int(row["shape"].split("x")[0]),
        )
        assert [row["reduction_ctas"] for row in device_rows] == [3, 5, 9]
        assert all(row["max_latency_ms"] > row["median_ms"] > 0 for row in device_rows)
