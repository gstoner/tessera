from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_provider_adapter_headers_and_sources_are_wired() -> None:
    cmake = (ROOT / "tools/profiler/CMakeLists.txt").read_text()
    headers = {
        "rocprofiler": ROOT / "tools/profiler/include/tprof/rocprofiler_adapter.h",
        "metal": ROOT / "tools/profiler/include/tprof/metal_adapter.h",
        "cupti": ROOT / "tools/profiler/include/tprof/cupti_adapter.h",
    }
    sources = {
        "rocprofiler": ROOT / "tools/profiler/src/runtime/rocprofiler_adapter.cpp",
        "metal": ROOT / "tools/profiler/src/runtime/metal_adapter.cpp",
        "cupti": ROOT / "tools/profiler/src/runtime/cupti_adapter.cpp",
    }

    for provider, path in headers.items():
        text = path.read_text()
        assert f"{provider}_adapter_init" in text
        assert f"{provider}_adapter_shutdown" in text
        assert f"{provider}_record_" in text

    roc = sources["rocprofiler"].read_text()
    assert "hip_hsa_api_tracing" in roc
    assert "dispatch_activity_records" in roc
    assert "counter_collection" in roc
    assert "thread_trace" in roc
    assert "start_paused" in headers["rocprofiler"].read_text()
    assert "include_api" in headers["rocprofiler"].read_text()
    assert "exclude_api" in headers["rocprofiler"].read_text()
    assert "include_kernel" in headers["rocprofiler"].read_text()
    assert "exclude_kernel" in headers["rocprofiler"].read_text()
    assert "rocprofiler_adapter_pause" in headers["rocprofiler"].read_text()
    assert "passes_filters" in roc
    assert "runtime_api(name ? name : \"rocprofiler.api\"" in roc
    assert "device_activity(name ? name : \"rocprofiler.dispatch\"" in roc
    assert "intra_kernel_sample" in roc

    metal = sources["metal"].read_text()
    assert "command_buffer_spans" in metal
    assert "counter_sample_buffers" in metal
    assert "metal_adapter_pause" in headers["metal"].read_text()
    assert "include_label" in headers["metal"].read_text()
    assert "exclude_counter" in headers["metal"].read_text()
    assert "tprof_passes_filters" in metal
    assert "device_activity(label ? label : \"metal.command_buffer\"" in metal

    cupti = sources["cupti"].read_text()
    assert "runtime_driver_callbacks" in cupti
    assert "activity_records" in cupti
    assert "cupti_adapter_pause" in headers["cupti"].read_text()
    assert "activity_buffer_bytes" in headers["cupti"].read_text()
    assert "include_api" in headers["cupti"].read_text()
    assert "exclude_activity" in headers["cupti"].read_text()
    assert "tprof_passes_filters" in cupti
    assert "runtime_api(name ? name : \"cupti.callback\"" in cupti
    assert "device_activity(name ? name : \"cupti.activity\"" in cupti

    for source in (
        "src/runtime/rocprofiler_adapter.cpp",
        "src/runtime/metal_adapter.cpp",
        "src/runtime/cupti_adapter.cpp",
    ):
        assert source in cmake


def test_provider_adapter_harness_exports_tprof_categories(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("c++ compiler is not available")
    harness = tmp_path / "provider_adapter_harness.cpp"
    trace = tmp_path / "provider_adapter.trace.json"
    exe = tmp_path / "provider_adapter_harness"
    harness.write_text(
        r'''
#include "tprof/cupti_adapter.h"
#include "tprof/metal_adapter.h"
#include "tprof/rocprofiler_adapter.h"
#include "tprof/tprof_runtime.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  tprof::enable(tprof::config_t{});
  tprof::rocprofiler_adapter_init(tprof::rocprofiler_adapter_config_t{
      true, true, true, true});
  tprof::rocprofiler_record_api("hipLaunchKernel", "HIP_API", 7, 4.0,
                                "{\"phase\":\"complete\"}");
  tprof::rocprofiler_record_activity("matmul", "dispatch", 7, 25.0,
                                     "{\"dispatch_id\":11}");
  tprof::rocprofiler_record_counter("SQ_WAVES", 64.0, 11,
                                    "{\"unit\":\"waves\"}");
  tprof::rocprofiler_record_thread_trace("matmul", 11, 25.0,
                                         "{\"target_cu\":0}");
  tprof::metal_adapter_init(tprof::metal_adapter_config_t{true, true});
  tprof::metal_record_command_buffer("metal.matmul", 21, 30.0,
                                     "{\"probe\":\"mainloop\"}");
  tprof::metal_record_counter_sample("gpu_cycles", 4096.0, 21, "mainloop",
                                     "{\"sample_index\":0}");
  tprof::cupti_adapter_init(tprof::cupti_adapter_config_t{true, true});
  tprof::cupti_record_callback("cudaLaunchKernel", "runtime", 99, 3.0,
                               "{\"cbid\":123}");
  tprof::cupti_record_activity("cuda.matmul", "kernel", 99, 22.0,
                               "{\"stream_id\":2}");
  return tprof::export_chrome(argv[1]) ? 0 : 1;
}
'''
    )
    sources = [
        harness,
        ROOT / "tools/profiler/src/runtime/tprof_runtime.cpp",
        ROOT / "tools/profiler/src/runtime/rocprofiler_adapter.cpp",
        ROOT / "tools/profiler/src/runtime/metal_adapter.cpp",
        ROOT / "tools/profiler/src/runtime/cupti_adapter.cpp",
        ROOT / "tools/profiler/src/runtime/nvtx_shim.cpp",
        ROOT / "tools/profiler/src/exporters/chrome_trace_exporter.cpp",
        ROOT / "tools/profiler/src/exporters/perfetto_exporter.cpp",
    ]
    compile_proc = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "tools/profiler/include"),
            "-I",
            str(ROOT / "tools/profiler/src/runtime"),
            *[str(source) for source in sources],
            "-o",
            str(exe),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr

    run_proc = subprocess.run([str(exe), str(trace)], capture_output=True, text=True, timeout=10)
    assert run_proc.returncode == 0, run_proc.stderr
    payload = json.loads(trace.read_text())
    categories = {event.get("cat") for event in payload["traceEvents"] if event.get("cat")}
    assert {"runtime_api", "device_activity", "intra_kernel"} <= categories
    names = {event.get("name") for event in payload["traceEvents"]}
    assert {"hipLaunchKernel", "matmul", "metal.matmul", "cudaLaunchKernel", "cuda.matmul"} <= names
    payloads = [
        event.get("args", {}).get("payload", "")
        for event in payload["traceEvents"]
    ]
    assert any('"correlation_id":7' in value for value in payloads)
    assert any('"correlation_id":99' in value for value in payloads)


def test_rocprofiler_adapter_harness_honors_pause_and_filters(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("c++ compiler is not available")
    harness = tmp_path / "rocprofiler_filter_harness.cpp"
    trace = tmp_path / "rocprofiler_filter.trace.json"
    exe = tmp_path / "rocprofiler_filter_harness"
    harness.write_text(
        r'''
#include "tprof/rocprofiler_adapter.h"
#include "tprof/tprof_runtime.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  tprof::enable(tprof::config_t{});
  tprof::rocprofiler_adapter_config_t cfg{};
  cfg.start_paused = true;
  cfg.include_api = "hip";
  cfg.exclude_api = "Ignore";
  cfg.include_kernel = "matmul";
  cfg.exclude_kernel = "skip";
  tprof::rocprofiler_adapter_init(cfg);
  tprof::rocprofiler_record_api("hipLaunchKernel", "HIP_API", 1, 1.0, nullptr);
  tprof::rocprofiler_adapter_resume();
  tprof::rocprofiler_record_api("hipLaunchKernel", "HIP_API", 2, 1.0, nullptr);
  tprof::rocprofiler_record_api("hipIgnoreThis", "HIP_API", 3, 1.0, nullptr);
  tprof::rocprofiler_record_activity("matmul", "dispatch", 4, 5.0, nullptr);
  tprof::rocprofiler_record_activity("skip_matmul", "dispatch", 5, 5.0, nullptr);
  tprof::rocprofiler_adapter_pause();
  if (!tprof::rocprofiler_adapter_is_paused()) return 3;
  tprof::rocprofiler_record_activity("matmul", "dispatch", 6, 5.0, nullptr);
  return tprof::export_chrome(argv[1]) ? 0 : 1;
}
'''
    )
    sources = [
        harness,
        ROOT / "tools/profiler/src/runtime/tprof_runtime.cpp",
        ROOT / "tools/profiler/src/runtime/rocprofiler_adapter.cpp",
        ROOT / "tools/profiler/src/runtime/nvtx_shim.cpp",
        ROOT / "tools/profiler/src/exporters/chrome_trace_exporter.cpp",
        ROOT / "tools/profiler/src/exporters/perfetto_exporter.cpp",
    ]
    compile_proc = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "tools/profiler/include"),
            "-I",
            str(ROOT / "tools/profiler/src/runtime"),
            *[str(source) for source in sources],
            "-o",
            str(exe),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr
    run_proc = subprocess.run([str(exe), str(trace)], capture_output=True, text=True, timeout=10)
    assert run_proc.returncode == 0, run_proc.stderr

    payload = json.loads(trace.read_text())
    names = [event.get("name") for event in payload["traceEvents"]]
    assert names.count("hipLaunchKernel") == 1
    assert names.count("matmul") == 1
    assert "hipIgnoreThis" not in names
    assert "skip_matmul" not in names


def test_cupti_and_metal_adapters_honor_pause_and_filters(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("c++ compiler is not available")
    harness = tmp_path / "cupti_metal_filter_harness.cpp"
    trace = tmp_path / "cupti_metal_filter.trace.json"
    exe = tmp_path / "cupti_metal_filter_harness"
    harness.write_text(
        r'''
#include "tprof/cupti_adapter.h"
#include "tprof/metal_adapter.h"
#include "tprof/tprof_runtime.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  tprof::enable(tprof::config_t{});
  tprof::cupti_adapter_config_t cupti{};
  cupti.start_paused = true;
  cupti.include_api = "cuda";
  cupti.exclude_api = "Ignore";
  cupti.include_activity = "matmul";
  cupti.exclude_activity = "skip";
  tprof::cupti_adapter_init(cupti);
  tprof::cupti_record_callback("cudaLaunchKernel", "runtime", 1, 1.0, nullptr);
  tprof::cupti_adapter_resume();
  tprof::cupti_record_callback("cudaLaunchKernel", "runtime", 2, 1.0, nullptr);
  tprof::cupti_record_callback("cudaIgnore", "runtime", 3, 1.0, nullptr);
  tprof::cupti_record_activity("matmul", "kernel", 4, 5.0, nullptr);
  tprof::cupti_record_activity("skip_matmul", "kernel", 5, 5.0, nullptr);

  tprof::metal_adapter_config_t metal{};
  metal.command_buffer_spans = true;
  metal.counter_sample_buffers = true;
  metal.include_label = "matmul";
  metal.exclude_label = "skip";
  metal.include_counter = "gpu";
  metal.exclude_counter = "bad";
  tprof::metal_adapter_init(metal);
  tprof::metal_record_command_buffer("metal.matmul", 7, 6.0, nullptr);
  tprof::metal_record_command_buffer("metal.skip", 8, 6.0, nullptr);
  tprof::metal_record_counter_sample("gpu_cycles", 9.0, 7, "mainloop", nullptr);
  tprof::metal_record_counter_sample("bad_gpu_cycles", 9.0, 7, "mainloop", nullptr);
  tprof::metal_adapter_pause();
  if (!tprof::metal_adapter_is_paused()) return 3;
  tprof::metal_record_command_buffer("metal.matmul", 9, 6.0, nullptr);
  return tprof::export_chrome(argv[1]) ? 0 : 1;
}
'''
    )
    sources = [
        harness,
        ROOT / "tools/profiler/src/runtime/tprof_runtime.cpp",
        ROOT / "tools/profiler/src/runtime/cupti_adapter.cpp",
        ROOT / "tools/profiler/src/runtime/metal_adapter.cpp",
        ROOT / "tools/profiler/src/runtime/nvtx_shim.cpp",
        ROOT / "tools/profiler/src/exporters/chrome_trace_exporter.cpp",
        ROOT / "tools/profiler/src/exporters/perfetto_exporter.cpp",
    ]
    compile_proc = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-I",
            str(ROOT / "tools/profiler/include"),
            "-I",
            str(ROOT / "tools/profiler/src/runtime"),
            *[str(source) for source in sources],
            "-o",
            str(exe),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert compile_proc.returncode == 0, compile_proc.stderr
    run_proc = subprocess.run([str(exe), str(trace)], capture_output=True, text=True, timeout=10)
    assert run_proc.returncode == 0, run_proc.stderr

    payload = json.loads(trace.read_text())
    names = [event.get("name") for event in payload["traceEvents"]]
    assert names.count("cudaLaunchKernel") == 1
    assert names.count("matmul") == 1
    assert names.count("metal.matmul") == 1
    assert names.count("gpu_cycles") == 1
    assert "cudaIgnore" not in names
    assert "skip_matmul" not in names
    assert "metal.skip" not in names
    assert "bad_gpu_cycles" not in names
