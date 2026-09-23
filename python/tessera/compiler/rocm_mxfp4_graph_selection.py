"""Fail-closed assessment of gfx1201 MXFP4 graph production admission.

This evaluates exact-device evidence; it does not register a public selector.
Host enqueue wins never override a device-time or model-integration refusal.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


_SHAPES = {(256, 5120, 8704), (1024, 17408, 5120)}


def _median(value: object) -> float | None:
    if not isinstance(value, Mapping):
        return None
    result = value.get("kernel_event_us_median")
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        return None
    result = float(result)
    return result if math.isfinite(result) and result > 0 else None


def _timing(record: object, name: str) -> float | None:
    if not isinstance(record, Mapping):
        return None
    return _median(record.get(name))


def assess_graph_pipeline_admission(packet: Mapping[str, Any]) -> dict[str, object]:
    """Return explicit producer and route decisions from a proof packet."""
    reasons: list[str] = []
    if (packet.get("host") != "tajasarus"
            or packet.get("device_name") != "AMD Radeon RX 9070 XT"
            or packet.get("target") != "rocm_gfx1201"):
        reasons.append("exact_device_proof_missing")
    tests_record = packet.get("tests")
    tests = tests_record.get("counts") if isinstance(tests_record, Mapping) else None
    if tests != {"tests": 18, "failures": 0, "errors": 0, "skipped": 0}:
        reasons.append("numerical_or_lifetime_proof_incomplete")
    raw_rows = packet.get("benchmarks")
    rows = raw_rows if isinstance(raw_rows, list) else []
    shapes = set()
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("shape"), (list, tuple)):
            shape = row["shape"]
            if len(shape) == 3 and all(type(value) is int for value in shape):
                shapes.add(tuple(shape))
    if shapes != _SHAPES or len(rows) != len(_SHAPES):
        reasons.append("prefill_shape_coverage_incomplete")
    wave_wins = True
    for row in rows:
        if not isinstance(row, Mapping):
            reasons.append("malformed_benchmark_timing")
            wave_wins = False
            continue
        rounds = row.get("rounds", [])
        if not isinstance(rounds, list) or len(rounds) != 4 or not all(
            isinstance(item, Mapping) for item in rounds
        ) or [item.get("variant") for item in rounds] != [
            "block", "wave", "wave", "block",
        ] or not all(item.get("sampled_bf16_exact") for item in rounds):
            reasons.append("matched_numerical_timing_missing")
            wave_wins = False
            continue
        for item in rounds:
            direct = _timing(item, "direct_triple")
            graph = _timing(item, "graph_triple")
            if direct is None or graph is None:
                reasons.append("malformed_benchmark_timing")
                wave_wins = False
            elif graph > direct * 1.02:
                if "graph_device_time_regression" not in reasons:
                    reasons.append("graph_device_time_regression")
        summary = row.get("summary")
        block = summary.get("block") if isinstance(summary, Mapping) else None
        wave = summary.get("wave") if isinstance(summary, Mapping) else None
        block_stage = _timing(block, "producer_stage")
        block_graph = _timing(block, "graph_triple")
        wave_stage = _timing(wave, "producer_stage")
        wave_graph = _timing(wave, "graph_triple")
        if any(value is None for value in (block_stage, block_graph, wave_stage, wave_graph)):
            reasons.append("malformed_benchmark_timing")
            wave_wins = False
        else:
            assert block_stage is not None and block_graph is not None
            assert wave_stage is not None and wave_graph is not None
            wave_wins &= wave_stage <= block_stage * 0.95 and wave_graph <= block_graph * 0.98
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
