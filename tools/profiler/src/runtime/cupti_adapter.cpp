#include "tprof/cupti_adapter.h"

#include "provider_adapter_utils.h"
#include "tprof/tprof_runtime.h"

#include <atomic>

#ifdef TPROF_WITH_CUPTI
#include <cupti.h>
#endif

namespace tprof {
namespace {
cupti_adapter_config_t g_cupti_cfg{};
std::atomic<bool> g_cupti_paused{false};
std::atomic<bool> g_cupti_initialized{false};
const char* g_cupti_last_error = "not initialized";

double duration_from_us(double start_us, double end_us) {
  return end_us > start_us ? end_us - start_us : 0.0;
}
} // namespace

bool cupti_adapter_init(const cupti_adapter_config_t& cfg) {
  g_cupti_cfg = cfg;
  g_cupti_paused.store(cfg.start_paused);
#ifdef TPROF_WITH_CUPTI
  g_cupti_initialized.store(true);
  g_cupti_last_error = "compiled CUPTI shell; native subscriber registration pending";
  return true;
#else
  g_cupti_initialized.store(false);
  g_cupti_last_error = "CUPTI was not found at build time";
  return false;
#endif
}

void cupti_adapter_shutdown() {
  g_cupti_initialized.store(false);
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
      g_cupti_cfg.activity_buffer_bytes,
      sdk_compiled ? "compiled_shell" : "planned",
      g_cupti_last_error,
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
