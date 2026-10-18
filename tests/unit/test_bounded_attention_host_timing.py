"""Host-free: synthetic clocks prove sampler boundaries and actual arm order."""
from types import SimpleNamespace

import pytest

from benchmarks.nvidia.benchmark_typed_attention_jvp import _paired_completed_windows


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_paired_windows_reverse_actual_calls_and_exclude_validation():
    clock = Clock()
    calls = []
    class Owner:
        @property
        def last_device_ms(self):
            clock.value += .05
            return (.3, .4)
    def arm(name):
        def invoke():
            calls.append(name)
            clock.value += .002
            return name
        return invoke, Owner()
    def validate(result):
        assert result in {"bounded", "static"}
        clock.value += .1
    result = _paired_completed_windows(
        {"bounded": arm("bounded"), "static": arm("static")}, validate,
        windows=4, min_window_ms=10, clock=clock)
    expected = [["bounded", "static"], ["static", "bounded"]]*2
    assert [x["order"] for x in result["windows"]] == expected
    repetitions = result["windows"][0]["arms"]["bounded"]["repetitions"]
    actual = [name for order in expected for name in order for _ in range(repetitions+1)]
    assert calls[-len(actual):] == actual
    for pair in result["windows"]:
        assert pair["arms"]["bounded"]["repetitions"] == pair["arms"]["static"]["repetitions"]
        for row in pair["arms"].values():
            assert row["completed_call_sum_ms"] >= 10
            assert row["completed_call_samples_ms"] == pytest.approx([2]*repetitions)
            assert row["native_event_samples_ms"] == [[.3, .4]]*repetitions


@pytest.mark.parametrize("windows,floor", [
    (3, 20), (2, 20), (True, 20), (4, float("nan")), (4, float("inf")),
    (4, 0), (4, 9.9), (4, True), (4, "20"),
])
def test_invalid_sampler_options_reject_before_work(windows, floor):
    def forbidden():
        pytest.fail("invalid arguments reached a timed arm")
    with pytest.raises(ValueError):
        _paired_completed_windows(
            {"bounded": (forbidden, None), "static": (forbidden, None)},
            forbidden, windows=windows, min_window_ms=floor)


def test_missing_native_event_receipt_refuses_timing():
    clock = Clock()
    def launch():
        clock.value += .002
        return "result"
    with pytest.raises(RuntimeError, match="native forward/JVP"):
        _paired_completed_windows(
            {"bounded": (launch, SimpleNamespace(last_device_ms=None)),
             "static": (launch, SimpleNamespace(last_device_ms=(1, 2)))},
            lambda result: None, windows=4, min_window_ms=10, clock=clock)
