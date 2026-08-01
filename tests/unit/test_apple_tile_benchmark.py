"""Portable contract checks for the APPLE-TILE-2 strict-v2 producer."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmarks" / "apple_gpu" / "benchmark_tile_simdgroup.py"


def _module():
    spec = importlib.util.spec_from_file_location("apple_tile_benchmark", BENCHMARK)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tile_producer_requires_three_interleaved_trials_and_current_schema():
    benchmark = _module()
    assert benchmark.ROUTE_REPORT_SCHEMA_VERSION == 1
    assert benchmark._shape("32x16x32") == (32, 16, 32)

    calls: list[str] = []

    def route(name: str):
        def call():
            calls.append(name)
            return None, True, {
                "device_time_ns": 10,
                "counter_sampling_supported": False,
                "counter_timestamp_delta": None,
                "resources": {"api": name},
            }
        return call

    mps, tile = benchmark._paired(route("mps"), route("tile"), reps=2, trials=3)
    assert mps[1] and tile[1]
    assert len(mps[2]["paired_trial_end_to_end_medians_ns"]) == 3
    assert len(tile[2]["paired_trial_device_medians_ns"]) == 3
    # One warm-up per route, then alternating trial order.
    assert calls[:2] == ["mps", "tile"]
    assert calls[2:6] == ["mps", "mps", "tile", "tile"]
    assert calls[6:10] == ["tile", "tile", "mps", "mps"]


def test_committed_tile_strict_ledger_admits_only_measured_mps_retentions():
    from tessera.compiler.apple_route_selector import AppleRouteContext, load_strict_route_ledger

    ledger = ROOT / "benchmarks" / "baselines" / "apple7_tile_strict_v2_route_ledger.json"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    measured = datetime.fromisoformat(payload["measured_at"].replace("Z", "+00:00"))
    admitted = load_strict_route_ledger(
        ledger,
        context=AppleRouteContext(**payload["context"]),
        now=measured + timedelta(days=1),
    )
    assert admitted.rejected == ()
    assert len(admitted.routes) == 16
    assert set(admitted.routes.values()) == {"mps"}
