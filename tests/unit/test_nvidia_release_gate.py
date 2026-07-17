"""NVIDIA-TEST-7 local-proof gate contract."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts/run_nvidia_release_gate.sh"


def test_nvidia_release_gate_is_a_local_exact_device_proof():
    text = GATE.read_text(encoding="utf-8")
    assert "local exact-device proof gate" in text
    assert "NVIDIA exact-device proof bundle" in text
    assert "not GitHub-hosted or self-hosted-runner" in text
    assert "install_test_deps.sh --venv" in text


def test_nvidia_release_gate_separates_and_retains_all_proof_layers():
    text = GATE.read_text(encoding="utf-8")
    assert "/usr/local/cuda/bin" in text
    assert "/usr/local/cuda-*/bin" in text
    assert "/usr/lib/llvm-23" in text
    assert 'CMAKE_CUDA_COMPILER="$CUDA_BIN/nvcc"' in text
    assert "LOCAL_VENV" in text
    for report in (
        "machine-identity.txt",
        "cpu.xml",
        "compiler-artifact.xml",
        "device-correctness-1.xml",
        "device-correctness-2.xml",
        "performance.xml",
    ):
        assert report in text
    assert "hardware_nvidia and not performance" in text
    assert "hardware_nvidia and performance" in text
    assert "-n 0" in text
    assert "check-tessera-nvidia" in text
    assert "llvm-lit" in text


def test_nvidia_release_gate_supports_independent_workflow_layers() -> None:
    text = GATE.read_text(encoding="utf-8")
    assert "--layer cpu|compiler|device|performance|all" in text
    assert 'LAYER=all' in text
    assert 'layer_enabled performance' in text
    assert 'baseline-sha256.txt' in text
    assert "flock -n 9" in text
    assert "status=success" in text
    assert "status=failed" in text
    assert "-q -n 0 --durations=0" in text
