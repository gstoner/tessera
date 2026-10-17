"""Host-free: synthetic-clock checks of sampler order and excluded oracle time; no device."""
from __future__ import annotations

import math

import pytest

from benchmarks.rocm import benchmark_gfx1201_fp8_blockscale as benchmark


@pytest.mark.parametrize("size", [2, 4])
def test_completed_host_sampler_counterbalances_and_excludes_oracle(monkeypatch, size):
    clock = [0]
    calls = []
    arms = {}

    def validate():
        clock[0] += 50_000_000

    for index in range(size):
        name = str(index)

        def launch(index=index, name=name):
            calls.append(name)
            clock[0] += (index + 1) * 1_000_000

        arms[name] = (launch, validate)
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: clock[0])
    result = benchmark.paired_completed_host_windows(
        arms, windows=2 * size, min_window_ms=20)
    for first, second in zip(result["window_orders"][::2], result["window_orders"][1::2]):
        assert second == list(reversed(first))
    for index in range(size):
        measured = result["arms"][str(index)]
        assert measured["median_ms_per_call"] == index + 1
        assert measured["admissible"]
    repeats = result["calls_per_window"]
    expected = [name for order in result["window_orders"] for name in order for _ in range(repeats)]
    assert calls[-len(expected):] == expected


@pytest.mark.parametrize("floor", [9.0, math.nan, math.inf])
def test_completed_host_sampler_rejects_invalid_window_before_calls(floor):
    def forbidden():
        pytest.fail("invalid timing arguments reached a launch")

    with pytest.raises(ValueError):
        benchmark.paired_completed_host_windows(
            {"a": (forbidden, forbidden)}, windows=2, min_window_ms=floor)
