"""Keep duration units and CLI errors independent of a live ROCm device."""
import pytest

from benchmarks.rocm import record_paired_exact_mxfp4 as recorder


@pytest.mark.parametrize("arguments", [
    ["--shapes", "256,31,64"], ["--shapes", "256,32"],
    ["--shapes", "bad"], ["--windows", "1"],
    ["--min-window-ms", "9"], ["--min-window-ms", "nan"],
])
def test_invalid_arguments_precede_hardware_probe(monkeypatch, tmp_path, arguments):
    def forbidden():
        raise AssertionError("invalid arguments reached the hardware gate")
    monkeypatch.setattr(recorder.runtime, "_rocm_live_arch", forbidden)
    monkeypatch.setattr("sys.argv", ["record_paired_exact_mxfp4", "--compiler", __file__,
        "--reference", __file__, "--output", str(tmp_path / "out.json"), *arguments])
    with pytest.raises(SystemExit) as error:
        recorder.main()
    assert error.value.code == 2


def test_native_average_is_converted_to_total_window_once():
    class Prepared:
        def invoke(self, *, repeats, timed):
            assert repeats == 40 and timed
            return 7, .25  # Runtime returns milliseconds PER program invocation.
    engine = recorder.Engine(Prepared())
    sample = recorder.ProgramWindows().window(engine, 40, bracketed=True)
    assert engine.generation == 7
    assert sample["device_window_ms"] == 10.
    assert sample["native_event_ms_per_invocation"] == .25
