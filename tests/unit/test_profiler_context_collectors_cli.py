from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tessera.compiler.accelerator_profiler_context import AcceleratorProfilerContext
from tessera.compiler.profiler_collectors import (
    collect_profiler_context,
    sample_nvidia_nvml_context,
    sample_rocm_amdsmi_context,
)


ROOT = Path(__file__).resolve().parents[2]


def test_tprof_context_cli_mock_emits_context_artifact(tmp_path: Path) -> None:
    out = tmp_path / "context.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_context.py"),
            "--provider",
            "mock",
            "--target",
            "apple_gpu",
            "--out",
            str(out),
        ],
        check=True,
    )

    payload = json.loads(out.read_text())
    assert payload["schema"] == "tessera.profiler_context.v1"
    assert payload["provider"] == "apple-silicon-system-context"
    assert payload["source_status"] == "mock"
    assert payload["bottleneck_summary"]["dominant_bottleneck"] == "bandwidth_bound"


def test_tprof_context_cli_file_mode_normalizes_raw_sample(tmp_path: Path) -> None:
    raw = AcceleratorProfilerContext(
        vendor="rocm",
        gpu_utilization=0.95,
        memory_utilization=0.20,
        memory_used_fraction=0.10,
    ).to_dict()
    raw_path = tmp_path / "sample.json"
    out = tmp_path / "context.json"
    raw_path.write_text(json.dumps(raw))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_context.py"),
            "--provider",
            "mock",
            "--target",
            "rocm",
            "--input",
            str(raw_path),
            "--out",
            str(out),
        ],
        check=True,
    )

    payload = json.loads(out.read_text())
    assert payload["provider"] == "rocm-system-context"
    assert payload["source_status"] == "file"
    assert payload["bottleneck_summary"]["dominant_bottleneck"] == "compute_bound"


def test_tprof_context_cli_reports_malformed_input_cleanly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not-json")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/profiler/scripts/tprof_context.py"),
            "--provider",
            "mock",
            "--input",
            str(bad),
        ],
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "error:" in proc.stderr


def test_apple_collector_shell_is_host_safe() -> None:
    payload = collect_profiler_context("apple")

    assert payload["schema"] == "tessera.profiler_context.v1"
    assert payload["provider"] == "apple-silicon-system-context"
    assert payload["source_status"] in {"host_metadata_only", "unavailable"}
    assert payload["samples"][0]["raw"]["metadata"]["collector"] == "apple-host-metadata"


def test_rocm_amdsmi_collector_normalizes_fake_module() -> None:
    class FakeAmdSmi:
        def amdsmi_init(self):
            return None

        def amdsmi_shut_down(self):
            return None

        def amdsmi_get_processor_handles(self):
            return ["gpu0"]

        def amdsmi_get_gpu_activity(self, handle):
            assert handle == "gpu0"
            return {"gfx_activity": 95, "umc_activity": 20}

        def amdsmi_get_gpu_vram_usage(self, handle):
            return {"vram_used": 4, "vram_total": 8}

        def amdsmi_get_power_info(self, handle):
            return {"socket_power": 250, "power_limit": 400}

        def amdsmi_get_temp_metric(self, handle):
            return {"temperature": 64}

        def amdsmi_get_gpu_total_ecc_count(self, handle):
            return {"correctable_count": 2, "uncorrectable_count": 0}

    payload = sample_rocm_amdsmi_context(module=FakeAmdSmi())

    assert payload["source_status"] == "measured"
    raw = payload["samples"][0]["raw"]
    assert raw["vendor"] == "rocm"
    assert raw["gpu_utilization"] == 0.95
    assert raw["memory_used_fraction"] == 0.5
    assert raw["correctable_ecc_errors"] == 2
    assert raw["bottleneck"] == "compute_bound"
    assert raw["metadata"]["diagnostics"] == []


def test_rocm_amdsmi_collector_records_signature_mismatch_diagnostics() -> None:
    class FakeAmdSmi:
        def amdsmi_init(self):
            return None

        def amdsmi_shutdown(self):
            return None

        def amdsmi_get_processor_handles(self):
            return ["gpu0"]

        def amdsmi_get_gpu_activity(self, handle, extra):
            return {"gfx_activity": 100, "umc_activity": 0}

    payload = sample_rocm_amdsmi_context(module=FakeAmdSmi())

    assert payload["source_status"] == "measured"
    diagnostics = payload["samples"][0]["raw"]["metadata"]["diagnostics"]
    assert diagnostics
    assert diagnostics[0]["method"] == "amdsmi_get_gpu_activity"
    assert diagnostics[0]["error_type"] == "TypeError"


def test_nvidia_nvml_collector_normalizes_fake_dynamic_library() -> None:
    class FakeNvml:
        def nvmlInit_v2(self):
            return 0

        def nvmlShutdown(self):
            return 0

        def nvmlDeviceGetHandleByIndex_v2(self, index, handle_ref):
            assert index.value == 0
            handle_ref._obj.value = 7
            return 0

        def nvmlDeviceGetUtilizationRates(self, handle, util_ref):
            assert handle.value == 7
            util_ref._obj.gpu = 72
            util_ref._obj.memory = 88
            return 0

        def nvmlDeviceGetMemoryInfo(self, handle, memory_ref):
            memory_ref._obj.total = 16
            memory_ref._obj.used = 8
            memory_ref._obj.free = 8
            return 0

        def nvmlDeviceGetPowerUsage(self, handle, value_ref):
            value_ref._obj.value = 300000
            return 0

        def nvmlDeviceGetEnforcedPowerLimit(self, handle, value_ref):
            value_ref._obj.value = 400000
            return 0

        def nvmlDeviceGetTemperature(self, handle, sensor, value_ref):
            assert sensor.value == 0
            value_ref._obj.value = 65
            return 0

        def nvmlDeviceGetCurrentClocksThrottleReasons(self, handle, value_ref):
            value_ref._obj.value = 0
            return 0

        def nvmlDeviceGetTotalEccErrors(self, handle, error_type, counter_type, value_ref):
            value_ref._obj.value = 1 if error_type.value == 0 else 0
            return 0

    payload = sample_nvidia_nvml_context(lib=FakeNvml())

    assert payload["source_status"] == "measured"
    raw = payload["samples"][0]["raw"]
    assert raw["vendor"] == "nvidia"
    assert raw["gpu_utilization"] == 0.72
    assert raw["memory_utilization"] == 0.88
    assert raw["memory_used_fraction"] == 0.5
    assert raw["power_watts"] == 300
    assert raw["power_limit_watts"] == 400
    assert raw["correctable_ecc_errors"] == 1


def test_nvidia_unavailable_context_records_error_type() -> None:
    class BrokenNvml:
        def nvmlInit_v2(self):
            raise RuntimeError("boom")

    payload = sample_nvidia_nvml_context(lib=BrokenNvml())

    assert payload["source_status"] == "unavailable"
    metadata = payload["samples"][0]["raw"]["metadata"]
    assert metadata["collector"] == "nvml"
    assert metadata["error_type"] == "RuntimeError"
    assert "boom" in metadata["error"]


def test_rocm_unavailable_context_records_error_type() -> None:
    class BrokenAmdSmi:
        def amdsmi_init(self):
            raise RuntimeError("boom")

    payload = sample_rocm_amdsmi_context(module=BrokenAmdSmi())

    assert payload["source_status"] == "unavailable"
    metadata = payload["samples"][0]["raw"]["metadata"]
    assert metadata["collector"] == "amdsmi"
    assert metadata["error_type"] == "RuntimeError"
    assert "boom" in metadata["error"]
