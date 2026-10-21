"""Invalid bounded measurements are rejected before querying hardware."""
import pytest

from benchmarks.nvidia import record_native_tensor_bounded_binary as recorder


def test_bounded_recorder_defaults_cover_shrink_and_grow():
    args = recorder.options(["--output", "unused.json"])
    assert args.rows == [17, 1, 256, 65, 5, 200]
    assert args.k == 4096 and args.n == 256


@pytest.mark.parametrize("arguments", [
    ["--k", "0"], ["--k", str(2**24)], ["--k", str(2**24 + 1)],
    ["--bound", "0"], ["--bound", str(16 * 65535 + 1)],
    ["--n", "0"], ["--n", str(2**31)],
    ["--rows", "0"], ["--rows", "257"], ["--rows", "17", "17"],
    ["--rounds", "1"], ["--rounds", "3"], ["--calls", "0"],
    ["--storage", "f16", "f16"],
])
def test_invalid_bounded_packet_does_not_probe_hardware(arguments, monkeypatch):
    def forbidden():
        pytest.fail("invalid arguments reached device identity")
    monkeypatch.setattr(recorder, "identity", forbidden)
    with pytest.raises(ValueError):
        recorder.main(["--output", "unused.json", *arguments])


def test_small_capacity_defaults_are_valid_and_unique():
    args = recorder.options(["--output", "unused.json", "--bound", "1"])
    assert args.rows == [1]
