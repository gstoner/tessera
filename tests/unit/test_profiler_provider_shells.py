from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_provider_shell_api_lists_native_and_heavy_lanes() -> None:
    header = (ROOT / "tools/profiler/include/tprof/provider_shells.h").read_text()
    source = (ROOT / "tools/profiler/src/runtime/provider_shells.cpp").read_text()
    cmake = (ROOT / "tools/profiler/CMakeLists.txt").read_text()

    for name in (
        "NVIDIA_SYSTEM_CONTEXT",
        "ROCM_SYSTEM_CONTEXT",
        "APPLE_SYSTEM_CONTEXT",
        "NVIDIA_CUPTI",
        "ROCM_ROCPROFILER",
        "APPLE_METAL_COUNTERS",
    ):
        assert name in header

    assert "bool runtime_api" in header
    assert "bool device_activity" in header
    assert "bool counters" in header
    assert "bool system_context" in header
    assert "bool command_correlation" in header
    assert "bool api_tracing" in header
    assert "bool activity_records" in header
    assert "bool counter_collection" in header
    assert "bool thread_trace" in header
    assert "bool external_correlation" in header
    assert "native_system_context_init" in header
    assert "heavy_provider_init" in header

    for provider in (
        "nvidia-system-context",
        "rocm-system-context",
        "apple-silicon-system-context",
        "cupti-activity-callbacks",
        "rocprofiler-sdk-dispatch-counters",
        "metal-command-buffer-counters",
    ):
        assert provider in source

    assert "src/runtime/provider_shells.cpp" in cmake
    assert "TPROF_WITH_ROCPROFILER" in cmake
    assert "find_package(ROCprofilerSDK QUIET)" in cmake
    assert "ROCprofiler-SDK requested but not found" in cmake
    assert (ROOT / "tools/profiler/cmake/FindROCprofilerSDK.cmake").is_file()
    assert "TPROF_WITH_APPLE_SYSTEM_CONTEXT" in cmake
    assert "TPROF_WITH_METAL" in cmake
    assert "ROCprofiler-SDK shell for HIP/HSA tracing" in source
    assert "thread-trace correlation" in source
    assert "CUPTI shell for runtime callbacks" in source
    assert "Metal shell for command-buffer timestamps" in source
