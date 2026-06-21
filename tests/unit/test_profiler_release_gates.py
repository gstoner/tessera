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
        "Mock, file, replay, and compile-only fixtures can demonstrate schema compatibility",
    ):
        assert phrase in doc
