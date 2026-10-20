"""Hardware-independent admission and measurement-limit checks for binary packets."""
import pytest
from benchmarks.nvidia.record_native_tensor_binary import options, event_windows


@pytest.mark.parametrize("arguments", [
    ["--rounds", "0"], ["--rounds", "3"], ["--rounds", "1002"],
    ["--calls", "0"], ["--calls", "10001"],
    ["--shape", "0", "1", "1"], ["--shape", "65536", "65536", "1"],
    ["--storage", "f16", "f16"], ["--case", "lhs", "lhs"],
])
def test_binary_recorder_validates_arguments_before_hardware(arguments, monkeypatch):
    from benchmarks.nvidia import record_native_tensor_binary as recorder
    monkeypatch.setattr(recorder, "identity", lambda: pytest.fail("invalid CLI probed hardware"))
    with pytest.raises(ValueError):
        recorder.main(["--output", "unused.json", *arguments])


def test_binary_recorder_preserves_unadmitted_regions_and_capture_limit():
    class Owner:
        def __init__(self):
            self.calls = []
        def profile(self, repeats, *, captured):
            assert repeats <= (4096 if captured else 10000)
            self.calls.append((repeats, captured))
            return {"stage_ms": [.001, .02], "program_ms": .03,
                    "measurement": "cuda_graph_events" if captured else "cuda_events"}
    owner = Owner()
    windows = event_windows(owner, 2)
    assert [row["captured"] for row in windows] == [False, True, True, False]
    assert all(row["repeats"] == 4096 and not row["duration_admitted"]
               for row in windows if row["captured"])
    assert all(row["duration_admitted"] for row in windows if not row["captured"])


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_binary_recorder_refuses_invalid_event_evidence(value):
    class Owner:
        def profile(self, repeats, *, captured):
            return {"stage_ms": [value], "program_ms": 1.}
    with pytest.raises(RuntimeError, match="finite and positive"):
        event_windows(Owner(), 2)


def test_binary_recorder_default_shape_and_order():
    args = options(["--output", "unused.json"])
    assert args.shape == [(256, 4096, 256)] and args.rounds == 6
    assert args.storage == ["f16", "bf16"] and args.case == ["lhs", "rhs", "both"]
