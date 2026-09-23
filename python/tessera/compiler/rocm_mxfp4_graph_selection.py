"""Fail-closed assessment of gfx1201 MXFP4 graph production admission.

This evaluates exact-device evidence; it does not register a public selector.
Host enqueue wins never override a device-time or model-integration refusal.
"""
from __future__ import annotations

from typing import Any, Mapping


_SHAPES = {(256, 5120, 8704), (1024, 17408, 5120)}


def assess_graph_pipeline_admission(packet: Mapping[str, Any]) -> dict[str, object]:
    """Return explicit producer and route decisions from a proof packet."""
    reasons: list[str] = []
    if (packet.get("host") != "tajasarus"
            or packet.get("device_name") != "AMD Radeon RX 9070 XT"
            or packet.get("target") != "rocm_gfx1201"):
        reasons.append("exact_device_proof_missing")
    tests = packet.get("tests", {}).get("counts", {})
    if tests != {"tests": 18, "failures": 0, "errors": 0, "skipped": 0}:
        reasons.append("numerical_or_lifetime_proof_incomplete")
    rows = packet.get("benchmarks", [])
    if {tuple(row.get("shape", ())) for row in rows} != _SHAPES:
        reasons.append("prefill_shape_coverage_incomplete")
    wave_wins = True
    for row in rows:
        rounds = row.get("rounds", [])
        if len(rounds) != 4 or [item.get("variant") for item in rounds] != [
            "block", "wave", "wave", "block",
        ] or not all(item.get("sampled_bf16_exact") for item in rounds):
            reasons.append("matched_numerical_timing_missing")
            wave_wins = False
            continue
        for item in rounds:
            direct = item["direct_triple"]["kernel_event_us_median"]
            graph = item["graph_triple"]["kernel_event_us_median"]
            if direct <= 0 or graph <= 0 or graph > direct * 1.02:
                if "graph_device_time_regression" not in reasons:
                    reasons.append("graph_device_time_regression")
        summary = row["summary"]
        block = summary["block"]
        wave = summary["wave"]
        wave_wins &= (
            wave["producer_stage"]["kernel_event_us_median"]
            <= block["producer_stage"]["kernel_event_us_median"] * 0.95
            and wave["graph_triple"]["kernel_event_us_median"]
            <= block["graph_triple"]["kernel_event_us_median"] * 0.98
        )
    if not packet.get("model_frontend_route", False):
        reasons.append("model_frontend_route_missing")
    if not packet.get("matched_radiance_parity", False):
        reasons.append("matched_radiance_parity_missing")
    return {
        "selected_producer_variant": "wave" if wave_wins and rows else "block",
        "automatic_selection": False,
        "admission_ready": not reasons,
        "refusals": sorted(set(reasons)),
        "device_time_margin": 1.02,
        "wave_producer_margin": 0.95,
        "wave_graph_margin": 0.98,
    }


__all__ = ["assess_graph_pipeline_admission"]
