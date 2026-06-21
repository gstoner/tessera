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
        assert f"{provider}_adapter_status" in text
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
    assert "include_counter" in headers["rocprofiler"].read_text()
    assert "thread_trace_max_bytes" in headers["rocprofiler"].read_text()
    assert "rocprofiler_api_record_t" in headers["rocprofiler"].read_text()
    assert "rocprofiler_replay_api_record" in headers["rocprofiler"].read_text()
    assert "rocprofiler_adapter_pause" in headers["rocprofiler"].read_text()
    assert "passes_filters" in roc
    assert "rocprofiler_adapter_status_t" in roc
    assert "thread trace record exceeded configured byte limit" in roc
    assert "rocprofiler_replay_thread_trace_record" in roc
    assert "runtime_api(name ? name : \"rocprofiler.api\"" in roc
    assert "device_activity(name ? name : \"rocprofiler.dispatch\"" in roc
    assert "intra_kernel_sample" in roc
    assert "context_created" in headers["rocprofiler"].read_text()
    assert "tool_registered" in headers["rocprofiler"].read_text()
    assert "hip_callbacks_configured" in headers["rocprofiler"].read_text()
    assert "hsa_callbacks_configured" in headers["rocprofiler"].read_text()
    assert "rocprofiler_adapter_start_collection" in headers["rocprofiler"].read_text()
    assert "buffered_activity_service_configured" in headers["rocprofiler"].read_text()
    assert "counter_request_validated" in headers["rocprofiler"].read_text()
    assert "unsupported_counter_requested" in headers["rocprofiler"].read_text()
    assert "dropped_records" in headers["rocprofiler"].read_text()
    assert "hardware proof required for native_available" in roc

    metal = sources["metal"].read_text()
    assert "command_buffer_spans" in metal
    assert "counter_sample_buffers" in metal
    assert "metal_adapter_pause" in headers["metal"].read_text()
    assert "include_label" in headers["metal"].read_text()
    assert "exclude_counter" in headers["metal"].read_text()
    assert "metal_command_buffer_record_t" in headers["metal"].read_text()
    assert "metal_replay_command_buffer_record" in headers["metal"].read_text()
    assert "tprof_passes_filters" in metal
    assert "metal_adapter_status_t" in metal
    assert "device_activity(label ? label : \"metal.command_buffer\"" in metal
    assert "metal_capture_command_buffer_timestamp" in headers["metal"].read_text()
    assert "metal_record_native_command_buffer" in headers["metal"].read_text()
    assert "metal_discover_counter_sets" in headers["metal"].read_text()
    assert "tprof_metal_capture_command_buffer_timestamp" in metal
    assert "tprof_metal_discover_counter_sets" in metal
    metal_objc = (ROOT / "tools/profiler/src/runtime/metal_command_buffer_probe.mm").read_text()
    assert "MTLCreateSystemDefaultDevice" in metal_objc
    assert "GPUStartTime" in metal_objc
    assert "GPUEndTime" in metal_objc
    assert "blitCommandEncoder" in metal_objc
    assert "MTLCounterSet" in metal_objc
    assert "counterSets" in metal_objc

    cupti = sources["cupti"].read_text()
    assert "runtime_driver_callbacks" in cupti
    assert "activity_records" in cupti
    assert "cupti_adapter_pause" in headers["cupti"].read_text()
    assert "activity_buffer_bytes" in headers["cupti"].read_text()
    assert "include_api" in headers["cupti"].read_text()
    assert "exclude_activity" in headers["cupti"].read_text()
    assert "cupti_callback_record_t" in headers["cupti"].read_text()
    assert "cupti_replay_callback_record" in headers["cupti"].read_text()
    assert "subscriber_created" in headers["cupti"].read_text()
    assert "activity_buffer_requested" in headers["cupti"].read_text()
    assert "activity_buffer_completed" in headers["cupti"].read_text()
    assert "metric_request_validated" in headers["cupti"].read_text()
    assert "unsupported_metric_requested" in headers["cupti"].read_text()
    assert "cupti_adapter_request_activity_buffer" in headers["cupti"].read_text()
    assert "tprof_passes_filters" in cupti
    assert "cupti_adapter_status_t" in cupti
    assert "runtime_api(name ? name : \"cupti.callback\"" in cupti
    assert "device_activity(name ? name : \"cupti.activity\"" in cupti
    assert "hardware proof required for native_available" in cupti

    for source in (
        "src/runtime/rocprofiler_adapter.cpp",
        "src/runtime/metal_adapter.cpp",
        "src/runtime/cupti_adapter.cpp",
    ):
        assert source in cmake
    assert "metal_command_buffer_probe.mm" in cmake
    assert "enable_language(OBJCXX)" in cmake


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


def test_rocprofiler_adapter_status_replay_and_thread_trace_guard(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("c++ compiler is not available")
    harness = tmp_path / "rocprofiler_replay_harness.cpp"
    trace = tmp_path / "rocprofiler_replay.trace.json"
    exe = tmp_path / "rocprofiler_replay_harness"
    harness.write_text(
        r'''
#include "tprof/rocprofiler_adapter.h"
#include "tprof/tprof_runtime.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  tprof::enable(tprof::config_t{});
  tprof::rocprofiler_adapter_config_t cfg{};
  cfg.counter_collection = true;
  cfg.thread_trace = true;
  cfg.counter_discovery = true;
  cfg.include_counter = "SQ";
  cfg.exclude_counter = "BAD";
  cfg.thread_trace_max_bytes = 64;
  cfg.requested_counters = "SQ_WAVES";
  (void)tprof::rocprofiler_adapter_init(cfg);
  auto status = tprof::rocprofiler_adapter_status();
  if (status.paused) return 3;
  if (!status.counter_collection || !status.thread_trace) return 4;
  if (status.context_created || status.tool_registered) return 8;
  if (status.hip_callbacks_configured || status.hsa_callbacks_configured) return 9;
  if (status.buffered_activity_service_configured) return 12;
  if (status.counter_discovery_configured) return 13;
  if (!status.counter_request_validated || status.unsupported_counter_requested) return 14;
  if (tprof::rocprofiler_adapter_start_collection()) return 10;
  if (tprof::rocprofiler_adapter_collection_started()) return 11;
  if (status.buffer_bytes == 0 || status.thread_trace_max_bytes != 64) return 5;
  if (status.source_status == nullptr || status.last_error == nullptr) return 6;
  if (status.lifecycle_stage == nullptr) return 15;
  if (tprof::rocprofiler_adapter_validate_counter_request("NOT_A_REAL_COUNTER")) return 16;
  status = tprof::rocprofiler_adapter_status();
  if (!status.unsupported_counter_requested || status.unsupported_counter == nullptr) return 17;
  tprof::rocprofiler_adapter_report_dropped_records(2);

  tprof::rocprofiler_replay_api_record(
      tprof::rocprofiler_api_record_t{"hipMemcpy", "HIP_API", 10, 1.0, 3.5,
                                      "{\"bytes\":128}"});
  tprof::rocprofiler_replay_activity_record(
      tprof::rocprofiler_activity_record_t{"matmul", "dispatch", 10, 4.0, 14.0, 77,
                                           "{\"queue_id\":2}"});
  tprof::rocprofiler_replay_counter_record(
      tprof::rocprofiler_counter_record_t{"SQ_WAVES", 8.0, 77, "{\"unit\":\"waves\"}"});
  tprof::rocprofiler_replay_counter_record(
      tprof::rocprofiler_counter_record_t{"BAD_SQ_WAVES", 8.0, 77, nullptr});
  tprof::rocprofiler_replay_thread_trace_record(
      tprof::rocprofiler_thread_trace_record_t{"matmul", 77, 4.0, 14.0, 32,
                                               "{\"target_cu\":0}"});
  tprof::rocprofiler_replay_thread_trace_record(
      tprof::rocprofiler_thread_trace_record_t{"huge", 78, 4.0, 14.0, 128,
                                               "{\"target_cu\":1}"});
  status = tprof::rocprofiler_adapter_status();
  if (!status.thread_trace_volume_limited) return 7;
  if (status.dropped_records != 3 || tprof::rocprofiler_adapter_dropped_records() != 3) return 18;
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
    assert "hipMemcpy" in names
    assert "matmul" in names
    assert "SQ_WAVES" in names
    assert "BAD_SQ_WAVES" not in names
    assert "huge" not in names


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
  cupti.requested_metrics = "sm__cycles_active";
  tprof::cupti_adapter_init(cupti);
  auto cupti_status = tprof::cupti_adapter_status();
  if (!cupti_status.paused || !cupti_status.runtime_driver_callbacks) return 4;
  if (cupti_status.subscriber_created) return 6;
  if (cupti_status.activity_buffer_service_configured) return 7;
  if (!cupti_status.metric_request_validated || cupti_status.unsupported_metric_requested) return 8;
  if (tprof::cupti_adapter_start_collection()) return 9;
  if (tprof::cupti_adapter_collection_started()) return 10;
  if (tprof::cupti_adapter_request_activity_buffer(4096)) return 11;
  tprof::cupti_adapter_complete_activity_buffer(128, 3);
  if (tprof::cupti_adapter_validate_metric_request("not_a_real_metric")) return 12;
  cupti_status = tprof::cupti_adapter_status();
  if (!cupti_status.activity_buffer_completed || cupti_status.dropped_records != 3) return 13;
  if (!cupti_status.unsupported_metric_requested || cupti_status.unsupported_metric == nullptr) return 14;
  tprof::cupti_record_callback("cudaLaunchKernel", "runtime", 1, 1.0, nullptr);
  tprof::cupti_adapter_resume();
  tprof::cupti_replay_callback_record(
      tprof::cupti_callback_record_t{"cudaLaunchKernel", "runtime", 2, 1.0, 2.0,
                                     nullptr});
  tprof::cupti_record_callback("cudaIgnore", "runtime", 3, 1.0, nullptr);
  tprof::cupti_replay_activity_record(
      tprof::cupti_activity_record_t{"matmul", "kernel", 4, 4.0, 9.0, nullptr});
  tprof::cupti_record_activity("skip_matmul", "kernel", 5, 5.0, nullptr);

  tprof::metal_adapter_config_t metal{};
  metal.command_buffer_spans = true;
  metal.counter_sample_buffers = true;
  metal.include_label = "matmul";
  metal.exclude_label = "skip";
  metal.include_counter = "gpu";
  metal.exclude_counter = "bad";
  tprof::metal_adapter_init(metal);
  auto metal_status = tprof::metal_adapter_status();
  if (!metal_status.command_buffer_spans || !metal_status.counter_sample_buffers) return 5;
  tprof::metal_replay_command_buffer_record(
      tprof::metal_command_buffer_record_t{"metal.matmul", 7, 10.0, 16.0, nullptr});
  tprof::metal_record_command_buffer("metal.skip", 8, 6.0, nullptr);
  tprof::metal_replay_counter_sample_record(
      tprof::metal_counter_sample_record_t{"gpu_cycles", 9.0, 7, "mainloop", nullptr});
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
