"""The MXFP4 graph selector stays closed on host-only wins."""
from __future__ import annotations

from tessera.compiler.rocm_mxfp4_graph_selection import assess_graph_pipeline_admission


def _row(shape: tuple[int, int, int], *, graph_us: float) -> dict[str, object]:
    rounds = [
        {
            "variant": variant,
            "sampled_bf16_exact": True,
            "direct_triple": {"kernel_event_us_median": 100.0},
            "graph_triple": {"kernel_event_us_median": graph_us},
        }
        for variant in ("block", "wave", "wave", "block")
    ]
    return {
        "shape": list(shape),
        "rounds": rounds,
        "summary": {
            "block": {
                "producer_stage": {"kernel_event_us_median": 20.0},
                "graph_triple": {"kernel_event_us_median": graph_us},
            },
            "wave": {
                "producer_stage": {"kernel_event_us_median": 19.8},
                "graph_triple": {"kernel_event_us_median": graph_us},
            },
        },
    }


def test_admission_refuses_wide_shape_regression_and_unproved_frontend() -> None:
    packet = {
        "host": "tajasarus",
        "device_name": "AMD Radeon RX 9070 XT",
        "target": "rocm_gfx1201",
        "tests": {"counts": {"tests": 18, "failures": 0, "errors": 0, "skipped": 0}},
        "benchmarks": [
            _row((256, 5120, 8704), graph_us=99.0),
            _row((1024, 17408, 5120), graph_us=107.0),
        ],
        "model_frontend_route": False,
        "matched_radiance_parity": False,
    }
    receipt = assess_graph_pipeline_admission(packet)
    assert receipt["automatic_selection"] is False
    assert receipt["admission_ready"] is False
    assert receipt["selected_producer_variant"] == "block"
    assert receipt["refusals"] == [
        "graph_device_time_regression",
        "matched_radiance_parity_missing",
        "model_frontend_route_missing",
    ]

    packet["benchmarks"][1] = _row((1024, 17408, 5120), graph_us=99.0)
    packet["model_frontend_route"] = True
    packet["matched_radiance_parity"] = True
    receipt = assess_graph_pipeline_admission(packet)
    assert receipt["admission_ready"] is True
    assert receipt["automatic_selection"] is False  # assessment never registers


def test_admission_refuses_missing_exact_device_or_shape_proof() -> None:
    receipt = assess_graph_pipeline_admission({})
    assert receipt["automatic_selection"] is False
    assert "exact_device_proof_missing" in receipt["refusals"]
    assert "prefill_shape_coverage_incomplete" in receipt["refusals"]
