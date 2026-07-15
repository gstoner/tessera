"""Static contract for exact-device GitHub Actions runner routing."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER_SETUP = ROOT / "scripts/configure_backend_runner.sh"
RUNNER_DOC = ROOT / "docs/operations/self_hosted_backend_runners.md"
NVIDIA_WORKFLOW = ROOT / ".github/workflows/nvidia-release-gate.yml"


def test_backend_runner_profiles_have_distinct_exact_hardware_labels():
    text = RUNNER_SETUP.read_text(encoding="utf-8")
    labels = (
        "nvidia-rtx5070ti-sm120",
        "rocm-strix-halo-gfx1151",
        "apple-m1max-apple7",
    )
    assert all(label in text for label in labels)
    assert "registration-token" in text
    assert "--replace" in text
    assert "GITHUB_RUNNER_REGISTRATION_TOKEN" in text


def test_nvidia_workflow_targets_its_exact_physical_runner_and_docs_match():
    label = "nvidia-rtx5070ti-sm120"
    assert label in NVIDIA_WORKFLOW.read_text(encoding="utf-8")
    docs = RUNNER_DOC.read_text(encoding="utf-8")
    assert all(label in docs for label in (
        label,
        "rocm-strix-halo-gfx1151",
        "apple-m1max-apple7",
    ))
