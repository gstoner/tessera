"""Timing domain budgets and invalid arguments remain testable without CUDA."""
import sys
import pytest
from benchmarks.nvidia import record_lse_checkpoint as recorder

@pytest.mark.parametrize("cap,pilot,target,expected", [
    (100, 0.01, 20.0, 100),
    (100, 1.0, 20.0, 20),
    (100, 800.0, 20.0, 1),
    (100, 3.0, 20.0, 7),
    (100, 0.0, 20.0, 100),
])
def test_window_count_bounds_short_and_long_kernels(cap, pilot, target, expected):
    assert recorder._window_repetitions(cap, pilot, target) == expected

@pytest.mark.parametrize("pilot,target", [
    (-1.0, 20.0), (float("nan"), 20.0), (float("inf"), 20.0),
    (1.0, 0.0), (1.0, -1.0), (1.0, float("inf")),
])
def test_invalid_window_measurement_is_rejected(pilot, target):
    with pytest.raises(ValueError):
        recorder._window_repetitions(100, pilot, target)

@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_adaptive_argument_precedes_hardware_probe(value, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["recorder", "--adaptive-window-ms", value])
    monkeypatch.setattr(recorder, "nvidia_cuda_host_ready",
                        lambda: pytest.fail("probed hardware for invalid argument"))
    with pytest.raises(ValueError, match="adaptive window"):
        recorder.main()
