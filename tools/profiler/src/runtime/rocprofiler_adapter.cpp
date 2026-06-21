#include "tprof/rocprofiler_adapter.h"

#include "provider_adapter_utils.h"
#include "tprof/tprof_runtime.h"

#include <atomic>
#include <string>

namespace tprof {
namespace {
rocprofiler_adapter_config_t g_rocprofiler_cfg{};
std::atomic<bool> g_rocprofiler_paused{false};

bool contains_token(const char* value, const char* token) {
  if (!token || token[0] == '\0') return false;
  if (!value) return false;
  return std::string(value).find(token) != std::string::npos;
}

bool passes_filters(const char* value, const char* include_token, const char* exclude_token) {
  if (include_token && include_token[0] != '\0' && !contains_token(value, include_token)) {
    return false;
  }
  if (exclude_token && exclude_token[0] != '\0' && contains_token(value, exclude_token)) {
    return false;
  }
  return true;
}
} // namespace

bool rocprofiler_adapter_init(const rocprofiler_adapter_config_t& cfg) {
  g_rocprofiler_cfg = cfg;
  g_rocprofiler_paused.store(cfg.start_paused);
#ifdef TPROF_WITH_ROCPROFILER
  return true;
#else
  return false;
#endif
}

void rocprofiler_adapter_shutdown() {}

void rocprofiler_adapter_pause() {
  g_rocprofiler_paused.store(true);
}

void rocprofiler_adapter_resume() {
  g_rocprofiler_paused.store(false);
}

bool rocprofiler_adapter_is_paused() {
  return g_rocprofiler_paused.load();
}

void rocprofiler_record_api(const char* name,
                            const char* domain,
                            uint64_t correlation_id,
                            double duration_us,
                            const char* args_json) {
  if (!g_rocprofiler_cfg.hip_hsa_api_tracing) return;
  if (g_rocprofiler_paused.load()) return;
  if (!passes_filters(name, g_rocprofiler_cfg.include_api, g_rocprofiler_cfg.exclude_api)) return;
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
  if (!passes_filters(name, g_rocprofiler_cfg.include_kernel, g_rocprofiler_cfg.exclude_kernel)) return;
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
  if (!passes_filters(kernel, g_rocprofiler_cfg.include_kernel, g_rocprofiler_cfg.exclude_kernel)) return;
  const auto payload = provider_payload("rocprofiler", "thread_trace", kernel,
                                        dispatch_id, "kernel", kernel, args_json);
  intra_kernel_sample(kernel ? kernel : "rocprofiler.thread_trace", duration_us, payload.c_str());
}

} // namespace tprof
