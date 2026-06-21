from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_profiler_release_gates_document_native_availability_rules() -> None:
    doc = (ROOT / "docs/guides/Tessera_Profiler_Release_Gates.md").read_text()

    for phrase in (
        "Native profiler availability is never inferred from mock data",
        "TPROF_WITH_METAL=ON",
        "TPROF_WITH_ROCPROFILER=ON",
        "TPROF_WITH_CUPTI=ON",
        "Compiled adapter shells report `compiled_shell`",
        "fresh-process, out-of-sandbox `tprof-apple-metal-smoke`",
        "provider availability snapshot artifact",
        "profiler-native-proofs.yml",
        "tprof_rocm_native_smoke.py",
        "tprof_nvidia_cupti_smoke.py",
        "Mock, file, replay, and compile-only fixtures can demonstrate schema compatibility",
    ):
        assert phrase in doc


def test_profiler_native_proof_workflow_is_optional_and_uploads_snapshots() -> None:
    workflow = (ROOT / ".github/workflows/profiler-native-proofs.yml").read_text()

    for phrase in (
        "workflow_dispatch",
        "profiler-native-proof",
        "tprof_apple_metal_smoke.py",
        "tprof_rocm_native_smoke.py",
        "tprof_nvidia_cupti_smoke.py",
        "profiler-provider-status-apple",
        "profiler-provider-status-rocm",
        "profiler-provider-status-nvidia",
        "--allow-unavailable",
    ):
        assert phrase in workflow
