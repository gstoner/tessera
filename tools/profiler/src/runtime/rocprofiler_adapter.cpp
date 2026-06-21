#include "tprof/rocprofiler_adapter.h"

#include "provider_adapter_utils.h"
#include "tprof/tprof_runtime.h"

#include <atomic>
#include <string>

#ifdef TPROF_WITH_ROCPROFILER
#include <rocprofiler-sdk/rocprofiler.h>
#endif

namespace tprof {
namespace {
rocprofiler_adapter_config_t g_rocprofiler_cfg{};
std::atomic<bool> g_rocprofiler_paused{false};
std::atomic<bool> g_rocprofiler_initialized{false};
std::atomic<bool> g_rocprofiler_thread_trace_volume_limited{false};
std::atomic<bool> g_rocprofiler_context_created{false};
std::atomic<bool> g_rocprofiler_tool_registered{false};
std::atomic<bool> g_rocprofiler_hip_callbacks_configured{false};
std::atomic<bool> g_rocprofiler_hsa_callbacks_configured{false};
std::atomic<bool> g_rocprofiler_collection_started{false};
const char* g_rocprofiler_last_error = "not initialized";

double duration_from_us(double start_us, double end_us) {
  return end_us > start_us ? end_us - start_us : 0.0;
}
} // namespace

bool rocprofiler_adapter_init(const rocprofiler_adapter_config_t& cfg) {
  g_rocprofiler_cfg = cfg;
  g_rocprofiler_paused.store(cfg.start_paused);
  g_rocprofiler_thread_trace_volume_limited.store(false);
#ifdef TPROF_WITH_ROCPROFILER
  g_rocprofiler_initialized.store(true);
  g_rocprofiler_context_created.store(true);
  g_rocprofiler_tool_registered.store(true);
  g_rocprofiler_hip_callbacks_configured.store(cfg.hip_hsa_api_tracing);
  g_rocprofiler_hsa_callbacks_configured.store(cfg.hip_hsa_api_tracing);
  g_rocprofiler_collection_started.store(!cfg.start_paused);
  g_rocprofiler_last_error = "compiled ROCprofiler-SDK lifecycle shell; native HIP/HSA callback registration pending";
  return true;
#else
  g_rocprofiler_initialized.store(false);
  g_rocprofiler_context_created.store(false);
  g_rocprofiler_tool_registered.store(false);
  g_rocprofiler_hip_callbacks_configured.store(false);
  g_rocprofiler_hsa_callbacks_configured.store(false);
  g_rocprofiler_collection_started.store(false);
  g_rocprofiler_last_error = "ROCprofiler-SDK was not found at build time";
  return false;
#endif
}

void rocprofiler_adapter_shutdown() {
  g_rocprofiler_initialized.store(false);
  g_rocprofiler_context_created.store(false);
  g_rocprofiler_tool_registered.store(false);
  g_rocprofiler_hip_callbacks_configured.store(false);
  g_rocprofiler_hsa_callbacks_configured.store(false);
  g_rocprofiler_collection_started.store(false);
  g_rocprofiler_last_error = "shutdown";
}

void rocprofiler_adapter_pause() {
  g_rocprofiler_paused.store(true);
}

void rocprofiler_adapter_resume() {
  g_rocprofiler_paused.store(false);
}

bool rocprofiler_adapter_is_paused() {
  return g_rocprofiler_paused.load();
}

bool rocprofiler_adapter_start_collection() {
#ifdef TPROF_WITH_ROCPROFILER
  if (!g_rocprofiler_initialized.load()) {
    g_rocprofiler_last_error = "ROCprofiler adapter is not initialized";
    return false;
  }
  g_rocprofiler_collection_started.store(true);
  g_rocprofiler_paused.store(false);
  return true;
#else
  g_rocprofiler_collection_started.store(false);
  g_rocprofiler_last_error = "ROCprofiler-SDK was not found at build time";
  return false;
#endif
}

void rocprofiler_adapter_stop_collection() {
  g_rocprofiler_collection_started.store(false);
}

bool rocprofiler_adapter_collection_started() {
  return g_rocprofiler_collection_started.load();
}

rocprofiler_adapter_status_t rocprofiler_adapter_status() {
  const bool sdk_compiled =
#ifdef TPROF_WITH_ROCPROFILER
      true;
#else
      false;
#endif
  return rocprofiler_adapter_status_t{
      sdk_compiled,
      g_rocprofiler_initialized.load(),
      g_rocprofiler_paused.load(),
      g_rocprofiler_context_created.load(),
      g_rocprofiler_tool_registered.load(),
      g_rocprofiler_hip_callbacks_configured.load(),
      g_rocprofiler_hsa_callbacks_configured.load(),
      g_rocprofiler_collection_started.load(),
      g_rocprofiler_cfg.hip_hsa_api_tracing,
      g_rocprofiler_cfg.dispatch_activity_records,
      g_rocprofiler_cfg.counter_collection,
      g_rocprofiler_cfg.thread_trace,
      g_rocprofiler_thread_trace_volume_limited.load(),
      g_rocprofiler_cfg.buffer_bytes,
      g_rocprofiler_cfg.thread_trace_max_bytes,
      sdk_compiled ? "compiled_shell" : "planned",
      g_rocprofiler_last_error,
  };
}

void rocprofiler_record_api(const char* name,
                            const char* domain,
                            uint64_t correlation_id,
                            double duration_us,
                            const char* args_json) {
  if (!g_rocprofiler_cfg.hip_hsa_api_tracing) return;
  if (g_rocprofiler_paused.load()) return;
  if (!tprof_passes_filters(name, g_rocprofiler_cfg.include_api, g_rocprofiler_cfg.exclude_api)) return;
  const auto payload = provider_payload("rocprofiler", "runtime_api", name,
                                        correlation_id, "domain", domain, args_json);
  runtime_api(name ? name : "rocprofiler.api", payload.c_str());
  if (duration_us > 0.0) {
    device_activity("rocprofiler.api.duration", duration_us, payload.c_str());
  }
}

void rocprofiler_record_activity(const char* name,
                                 const char* activity,
                                 uint64_t correlation_id,
                                 double duration_us,
                                 const char* args_json) {
  if (!g_rocprofiler_cfg.dispatch_activity_records) return;
  if (g_rocprofiler_paused.load()) return;
  if (!tprof_passes_filters(name, g_rocprofiler_cfg.include_kernel, g_rocprofiler_cfg.exclude_kernel)) return;
  const auto payload = provider_payload("rocprofiler", "device_activity", name,
                                        correlation_id, "activity", activity, args_json);
  device_activity(name ? name : "rocprofiler.dispatch", duration_us, payload.c_str());
}

void rocprofiler_record_counter(const char* metric,
                                double value,
                                uint64_t correlation_id,
                                const char* args_json) {
  if (!g_rocprofiler_cfg.counter_collection) return;
  if (g_rocprofiler_paused.load()) return;
  if (!tprof_passes_filters(metric, g_rocprofiler_cfg.include_counter, g_rocprofiler_cfg.exclude_counter)) return;
  const auto payload = provider_payload("rocprofiler", "counter", metric,
                                        correlation_id, "metric", metric, args_json);
  (void)payload;
  counter_add(metric ? metric : "rocprofiler.counter", value);
}

void rocprofiler_record_thread_trace(const char* kernel,
                                     uint64_t dispatch_id,
                                     double duration_us,
                                     const char* args_json) {
  if (!g_rocprofiler_cfg.thread_trace) return;
  if (g_rocprofiler_paused.load()) return;
  if (!tprof_passes_filters(kernel, g_rocprofiler_cfg.include_kernel, g_rocprofiler_cfg.exclude_kernel)) return;
  const auto payload = provider_payload("rocprofiler", "thread_trace", kernel,
                                        dispatch_id, "kernel", kernel, args_json);
  intra_kernel_sample(kernel ? kernel : "rocprofiler.thread_trace", duration_us, payload.c_str());
}

void rocprofiler_replay_api_record(const rocprofiler_api_record_t& record) {
  rocprofiler_record_api(record.name,
                         record.domain,
                         record.correlation_id,
                         duration_from_us(record.start_us, record.end_us),
                         record.args_json);
}

void rocprofiler_replay_activity_record(const rocprofiler_activity_record_t& record) {
  const uint64_t correlation_id =
      record.correlation_id != 0 ? record.correlation_id : record.dispatch_id;
  rocprofiler_record_activity(record.name,
                              record.activity,
                              correlation_id,
                              duration_from_us(record.start_us, record.end_us),
                              record.args_json);
}

void rocprofiler_replay_counter_record(const rocprofiler_counter_record_t& record) {
  rocprofiler_record_counter(record.metric, record.value, record.correlation_id, record.args_json);
}

void rocprofiler_replay_thread_trace_record(const rocprofiler_thread_trace_record_t& record) {
  if (record.trace_bytes > 0 &&
      g_rocprofiler_cfg.thread_trace_max_bytes > 0 &&
      record.trace_bytes > g_rocprofiler_cfg.thread_trace_max_bytes) {
    g_rocprofiler_thread_trace_volume_limited.store(true);
    g_rocprofiler_last_error = "thread trace record exceeded configured byte limit";
    return;
  }
  rocprofiler_record_thread_trace(record.kernel,
                                  record.dispatch_id,
                                  duration_from_us(record.start_us, record.end_us),
                                  record.args_json);
}

} // namespace tprof
