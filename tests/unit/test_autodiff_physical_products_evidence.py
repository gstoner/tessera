from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "filename,target,architecture",
    [
        ("x86_zen5_autodiff_physical_products_evidence.json", "x86", "zen5_avx512"),
        ("rocm_gfx1151_autodiff_physical_products_evidence.json", "rocm_gfx1151", "gfx1151"),
    ],
)
def test_autodiff_physical_product_packet_is_complete_and_honest(
    filename: str, target: str, architecture: str
) -> None:
    packet = json.loads((ROOT / "benchmarks" / "baselines" / filename).read_text())
    assert packet["schema"] == "tessera.autodiff_physical_products.evidence.v1"
    assert packet["target"] == target
    assert packet["architecture"] == architecture
    assert packet["device"]
    assert packet["tessera_opt_sha256"] and len(packet["tessera_opt_sha256"]) == 64
    assert packet["host_clean"] is False
    assert packet["wsl"] is True
    assert packet["promotion"]["correctness_eligible"] is True
    assert packet["promotion"]["performance_eligible"] is False
    rows = {row["name"]: row for row in packet["rows"]}
    assert set(rows) == {
        "generated_nonlinear_residual_vjp",
        "generated_reduction_residual_vjp",
        "generated_matmul_residual_vjp",
        "generated_dynamic_mixed_residual_vjp",
        "generated_control_if_residual_vjp",
        "generated_control_while_residual_vjp",
        "istft_spectrum_window_jvp",
    }
    for row in rows.values():
        assert len(row["artifact_digest"]) == 64
        assert row["numerical"]["passed"] is True
        assert row["numerical"]["max_abs_error"] <= 5.0e-4
        timing = row["timing"]
        assert timing["source"] == "synchronized_host_wall"
        assert timing["device_clock"]["valid"] is False
        assert timing["cold_sample_ns"] > 0
        assert len(timing["samples_ns"]) >= 20
        assert timing["minimum_ns"] <= timing["median_ns"]
        assert row["resources"]

    solver = rows["generated_nonlinear_residual_vjp"]["resources"]
    assert solver["physical_children"] == 5
    assert solver["children_generated_from_graph"] is True
    assert solver["residual_expression"] == "x*x+sin(x)-theta"
    assert solver["matrix_materialized"] is False
    assert solver["last_solve"]["converged"] is True
    for name in (
        "generated_reduction_residual_vjp",
        "generated_matmul_residual_vjp",
        "generated_dynamic_mixed_residual_vjp",
        "generated_control_if_residual_vjp",
        "generated_control_while_residual_vjp",
    ):
        resources = rows[name]["resources"]
        assert resources["children_generated_from_graph"] is True
        assert resources["last_solve"]["converged"] is True
    for name in (
        "generated_control_if_residual_vjp",
        "generated_control_while_residual_vjp",
    ):
        replay = rows[name]["resources"]["predicate_replay"]
        assert replay["policy"] == "pure_recompute"
        assert replay["derivative_boundary"] == "reject_equal"
        assert replay["selection_count"] > 0
    assert rows["istft_spectrum_window_jvp"]["resources"][
        "window_energy_derivative"
    ] is True
