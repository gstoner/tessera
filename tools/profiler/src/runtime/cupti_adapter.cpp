#include "tprof/cupti_adapter.h"

#include "provider_adapter_utils.h"
#include "tprof/tprof_runtime.h"

#include <atomic>

namespace tprof {
namespace {
cupti_adapter_config_t g_cupti_cfg{};
std::atomic<bool> g_cupti_paused{false};
} // namespace

bool cupti_adapter_init(const cupti_adapter_config_t& cfg) {
  g_cupti_cfg = cfg;
  g_cupti_paused.store(cfg.start_paused);
#ifdef TPROF_WITH_CUPTI
  return true;
#else
  return false;
#endif
}

void cupti_adapter_shutdown() {}

void cupti_adapter_pause() {
  g_cupti_paused.store(true);
}

void cupti_adapter_resume() {
  g_cupti_paused.store(false);
}

bool cupti_adapter_is_paused() {
  return g_cupti_paused.load();
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

} // namespace tprof
