"""Malformed recorder input cannot probe GPUs or overwrite measured evidence."""
import pytest
from benchmarks.nvidia import record_native_independent_shapes as recorder


@pytest.mark.parametrize("arguments", [
    ["--rounds", "1"], ["--rounds", "3"], ["--calls", "0"],
    ["--bounds", "0", "47", "64"], ["--bounds", "65", "47", "2147483648"],
    ["--shapes", "17x11"], ["--shapes", "bad"], ["--shapes", "66x11x23"],
    ["--shapes", "17x48x23"], ["--shapes", "17x11x65"], ["--shapes", "0x1x1"],
    ["--shapes", "17x11x23", "17x11x23"],
    ["--storage", "f16", "f16"], ["--case", "norm_lhs", "norm_lhs"],
])
def test_invalid_independent_shape_arguments_never_probe_hardware(arguments, monkeypatch, tmp_path):
    def forbidden():
        pytest.fail("invalid recorder input reached GPU/toolchain identity")
    monkeypatch.setattr(recorder, "identity", forbidden)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        recorder.main(["--output", str(output), *arguments])
    assert not output.exists()


def test_recording_requires_fresh_evidence_path(monkeypatch, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserved")
    monkeypatch.setattr(recorder, "identity", lambda: pytest.fail("existing evidence reached hardware identity"))
    with pytest.raises(FileExistsError):
        recorder.main(["--output", str(output)])
    assert output.read_text() == "preserved"
