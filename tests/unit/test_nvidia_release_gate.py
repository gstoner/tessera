"""NVIDIA-TEST-7 release-gate ownership contract."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/nvidia-release-gate.yml"
GATE = ROOT / "scripts/run_nvidia_release_gate.sh"


def test_nvidia_release_workflow_serializes_the_sm120_box_and_retains_evidence():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "nvidia-sm120-release-gate" in text
    assert "cancel-in-progress: false" in text
    assert "runs-on: [self-hosted, linux, nvidia-rtx5070ti-sm120]" in text
    assert "scripts/run_nvidia_release_gate.sh" in text
    assert "actions/upload-artifact@v4" in text
    assert "retention-days: 30" in text


def test_nvidia_release_gate_separates_and_retains_all_proof_layers():
    text = GATE.read_text(encoding="utf-8")
    assert "/usr/local/cuda/bin" in text
    assert "/usr/local/cuda-*/bin" in text
    assert "/usr/lib/llvm-22/bin/lit" in text
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
