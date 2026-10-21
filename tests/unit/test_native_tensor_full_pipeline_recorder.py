"""Malformed benchmark arguments must fail before probing an owning GPU."""
import pytest
from benchmarks.nvidia import record_native_tensor_full_pipeline as recorder


@pytest.mark.parametrize("arguments", [
    ["--rounds", "1"], ["--rounds", "3"], ["--rounds", "1002"],
    ["--calls", "0"], ["--calls", "10001"],
    ["--storage", "f16", "f16"], ["--case", "add", "add"],
])
def test_invalid_full_pipeline_recording_never_probes_hardware(arguments, monkeypatch, tmp_path):
    def forbidden():
        pytest.fail("invalid arguments reached GPU/toolchain identity")
    monkeypatch.setattr(recorder, "identity", forbidden)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        recorder.main(["--output", str(output), *arguments])
    assert not output.exists()


@pytest.mark.parametrize("arguments", [
    ["--bound", "16"], ["--rows", "0"], ["--rows", "66"],
    ["--rows", "1", "1"], ["--rounds", "3"], ["--calls", "0"],
    ["--storage", "bf16", "bf16"], ["--case", "chain", "chain"],
])
def test_invalid_bounded_dag_recording_never_probes_hardware(arguments, monkeypatch, tmp_path):
    from benchmarks.nvidia import record_native_tensor_bounded_dag as bounded

    def forbidden():
        pytest.fail("invalid bounded recorder arguments reached hardware identity")
    monkeypatch.setattr(bounded, "identity", forbidden)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        bounded.main(["--output", str(output), *arguments])
    assert not output.exists()


@pytest.mark.parametrize("arguments", [
    ["--bound", "16"], ["--rows", "0"], ["--rows", "66"],
    ["--rows", "1", "1"], ["--rounds", "3"], ["--calls", "0"],
    ["--storage", "f32", "f32"], ["--case", "fork_join", "fork_join"],
])
def test_invalid_semantic_dag_recording_never_probes_hardware(arguments, monkeypatch, tmp_path):
    from benchmarks.nvidia import record_native_semantic_dag as semantic

    def forbidden():
        pytest.fail("invalid semantic recorder arguments reached hardware identity")
    monkeypatch.setattr(semantic, "identity", forbidden)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError):
        semantic.main(["--output", str(output), *arguments])
    assert not output.exists()
