#include "tprof/cupti_adapter.h"

#include "provider_adapter_utils.h"
#include "tprof/tprof_runtime.h"

#include <atomic>
#include <string>

#ifdef TPROF_WITH_CUPTI
#include <cupti.h>
#endif

namespace tprof {
namespace {
cupti_adapter_config_t g_cupti_cfg{};
std::atomic<bool> g_cupti_paused{false};
std::atomic<bool> g_cupti_initialized{false};
std::atomic<bool> g_cupti_collection_started{false};
std::atomic<bool> g_cupti_subscriber_created{false};
std::atomic<bool> g_cupti_runtime_callbacks_configured{false};
std::atomic<bool> g_cupti_driver_callbacks_configured{false};
std::atomic<bool> g_cupti_activity_buffer_service_configured{false};
std::atomic<bool> g_cupti_activity_buffer_requested{false};
std::atomic<bool> g_cupti_activity_buffer_completed{false};
std::atomic<bool> g_cupti_metric_request_validated{false};
std::atomic<bool> g_cupti_unsupported_metric_requested{false};
std::atomic<uint64_t> g_cupti_dropped_records{0};
const char* g_cupti_lifecycle_stage = "not_initialized";
const char* g_cupti_last_error = "not initialized";
std::string g_cupti_unsupported_metric;

double duration_from_us(double start_us, double end_us) {
  return end_us > start_us ? end_us - start_us : 0.0;
}

bool is_supported_metric_name(const char* metric) {
  if (metric == nullptr || metric[0] == '\0') return false;
  const char* supported[] = {
      "sm", "dram", "l2", "tensor", "eligible", "active", "inst", "bytes",
  };
  const std::string value(metric);
  for (const char* token : supported) {
    if (value.find(token) != std::string::npos) {
      return true;
    }
  }
  return value.find("SM") != std::string::npos ||
         value.find("DRAM") != std::string::npos ||
         value.find("L2") != std::string::npos;
}
} // namespace

bool cupti_adapter_init(const cupti_adapter_config_t& cfg) {
  g_cupti_cfg = cfg;
  g_cupti_paused.store(cfg.start_paused);
  g_cupti_activity_buffer_requested.store(false);
  g_cupti_activity_buffer_completed.store(false);
  g_cupti_metric_request_validated.store(false);
  g_cupti_unsupported_metric_requested.store(false);
  g_cupti_dropped_records.store(0);
  g_cupti_unsupported_metric.clear();
#ifdef TPROF_WITH_CUPTI
  g_cupti_initialized.store(true);
  g_cupti_collection_started.store(!cfg.start_paused);
  g_cupti_subscriber_created.store(cfg.subscriber_callbacks);
  g_cupti_runtime_callbacks_configured.store(cfg.runtime_driver_callbacks);
  g_cupti_driver_callbacks_configured.store(cfg.runtime_driver_callbacks);
  g_cupti_activity_buffer_service_configured.store(cfg.activity_buffer_service);
  g_cupti_lifecycle_stage = cfg.start_paused ? "initialized_paused" : "collecting";
  if (cfg.requested_metrics != nullptr) {
    cupti_adapter_validate_metric_request(cfg.requested_metrics);
  }
  g_cupti_last_error = "compiled CUPTI lifecycle shell; hardware proof required for native_available";
  return true;
#else
  g_cupti_initialized.store(false);
  g_cupti_collection_started.store(false);
  g_cupti_subscriber_created.store(false);
  g_cupti_runtime_callbacks_configured.store(false);
  g_cupti_driver_callbacks_configured.store(false);
  g_cupti_activity_buffer_service_configured.store(false);
  g_cupti_lifecycle_stage = "sdk_unavailable";
  if (cfg.requested_metrics != nullptr) {
    cupti_adapter_validate_metric_request(cfg.requested_metrics);
  }
  g_cupti_last_error = "CUPTI was not found at build time";
  return false;
#endif
}

void cupti_adapter_shutdown() {
  g_cupti_initialized.store(false);
  g_cupti_collection_started.store(false);
  g_cupti_subscriber_created.store(false);
  g_cupti_runtime_callbacks_configured.store(false);
  g_cupti_driver_callbacks_configured.store(false);
  g_cupti_activity_buffer_service_configured.store(false);
  g_cupti_lifecycle_stage = "shutdown";
  g_cupti_last_error = "shutdown";
}

void cupti_adapter_pause() {
  g_cupti_paused.store(true);
}

void cupti_adapter_resume() {
  g_cupti_paused.store(false);
}

bool cupti_adapter_is_paused() {
  return g_cupti_paused.load();
}

bool cupti_adapter_start_collection() {
#ifdef TPROF_WITH_CUPTI
  if (!g_cupti_initialized.load()) {
    g_cupti_last_error = "CUPTI adapter is not initialized";
    g_cupti_lifecycle_stage = "start_failed";
    return false;
  }
  g_cupti_collection_started.store(true);
  g_cupti_paused.store(false);
  g_cupti_lifecycle_stage = "collecting";
  return true;
#else
  g_cupti_collection_started.store(false);
  g_cupti_lifecycle_stage = "sdk_unavailable";
  g_cupti_last_error = "CUPTI was not found at build time";
  return false;
#endif
}

void cupti_adapter_stop_collection() {
  g_cupti_collection_started.store(false);
  g_cupti_lifecycle_stage = "stopped";
}

bool cupti_adapter_collection_started() {
  return g_cupti_collection_started.load();
}

bool cupti_adapter_request_activity_buffer(uint64_t requested_bytes) {
  g_cupti_cfg.activity_buffer_bytes = requested_bytes;
#ifdef TPROF_WITH_CUPTI
  g_cupti_activity_buffer_requested.store(requested_bytes > 0);
  g_cupti_lifecycle_stage = "activity_buffer_requested";
  return requested_bytes > 0;
#else
  g_cupti_activity_buffer_requested.store(false);
  g_cupti_lifecycle_stage = "sdk_unavailable";
  g_cupti_last_error = "CUPTI was not found at build time";
  return false;
#endif
}

void cupti_adapter_complete_activity_buffer(uint64_t valid_bytes, uint64_t dropped_records) {
  (void)valid_bytes;
  g_cupti_activity_buffer_completed.store(true);
  g_cupti_dropped_records.fetch_add(dropped_records);
  if (dropped_records > 0) {
    g_cupti_last_error = "CUPTI activity buffer reported dropped records";
  }
}

bool cupti_adapter_validate_metric_request(const char* metric) {
  const bool ok = is_supported_metric_name(metric);
  g_cupti_metric_request_validated.store(ok);
  g_cupti_unsupported_metric_requested.store(!ok);
  g_cupti_unsupported_metric = ok ? "" : (metric ? metric : "");
  if (!ok) {
    g_cupti_last_error = "unsupported CUPTI metric request";
  }
  return ok;
}

cupti_adapter_status_t cupti_adapter_status() {
  const bool sdk_compiled =
#ifdef TPROF_WITH_CUPTI
      true;
#else
      false;
#endif
  return cupti_adapter_status_t{
      sdk_compiled,
      g_cupti_initialized.load(),
      g_cupti_paused.load(),
      g_cupti_cfg.runtime_driver_callbacks,
      g_cupti_cfg.activity_records,
      g_cupti_subscriber_created.load(),
      g_cupti_runtime_callbacks_configured.load(),
      g_cupti_driver_callbacks_configured.load(),
      g_cupti_activity_buffer_service_configured.load(),
      g_cupti_activity_buffer_requested.load(),
      g_cupti_activity_buffer_completed.load(),
      g_cupti_metric_request_validated.load(),
      g_cupti_unsupported_metric_requested.load(),
      g_cupti_dropped_records.load(),
      g_cupti_cfg.activity_buffer_bytes,
      g_cupti_lifecycle_stage,
      sdk_compiled ? "compiled_shell" : "planned",
      g_cupti_last_error,
      g_cupti_unsupported_metric.empty() ? nullptr : g_cupti_unsupported_metric.c_str(),
  };
}

void cupti_record_callback(const char* name,
                           const char* domain,
                           uint64_t correlation_id,
                           double duration_us,
                           const char* args_json) {
  if (!g_cupti_cfg.runtime_driver_callbacks) return;
  if (g_cupti_paused.load()) return;
  if (!tprof_passes_filters(name, g_cupti_cfg.include_api, g_cupti_cfg.exclude_api)) return;
  const auto payload = provider_payload("cupti", "runtime_api", name,
                                        correlation_id, "domain", domain, args_json);
  runtime_api(name ? name : "cupti.callback", payload.c_str());
  if (duration_us > 0.0) {
    device_activity("cupti.callback.duration", duration_us, payload.c_str());
  }
}

void cupti_record_activity(const char* name,
                           const char* activity,
                           uint64_t correlation_id,
                           double duration_us,
                           const char* args_json) {
  if (!g_cupti_cfg.activity_records) return;
  if (g_cupti_paused.load()) return;
  if (!tprof_passes_filters(name, g_cupti_cfg.include_activity, g_cupti_cfg.exclude_activity)) return;
  const auto payload = provider_payload("cupti", "device_activity", name,
                                        correlation_id, "activity", activity, args_json);
  device_activity(name ? name : "cupti.activity", duration_us, payload.c_str());
}

void cupti_replay_callback_record(const cupti_callback_record_t& record) {
  cupti_record_callback(record.name,
                        record.domain,
                        record.correlation_id,
                        duration_from_us(record.start_us, record.end_us),
                        record.args_json);
}

void cupti_replay_activity_record(const cupti_activity_record_t& record) {
  cupti_record_activity(record.name,
                        record.activity,
                        record.correlation_id,
                        duration_from_us(record.start_us, record.end_us),
                        record.args_json);
}

} // namespace tprof
