#include "tprof/metal_adapter.h"

#include "provider_adapter_utils.h"
#include "tprof/tprof_runtime.h"

#include <atomic>

namespace tprof {
namespace {
metal_adapter_config_t g_metal_cfg{};
std::atomic<bool> g_metal_paused{false};
} // namespace

bool metal_adapter_init(const metal_adapter_config_t& cfg) {
  g_metal_cfg = cfg;
  g_metal_paused.store(cfg.start_paused);
#if defined(TPROF_WITH_METAL) && defined(__APPLE__)
  return true;
#else
  return false;
#endif
}

void metal_adapter_shutdown() {}

void metal_adapter_pause() {
  g_metal_paused.store(true);
}

void metal_adapter_resume() {
  g_metal_paused.store(false);
}

bool metal_adapter_is_paused() {
  return g_metal_paused.load();
}

void metal_record_command_buffer(const char* label,
                                 uint64_t command_buffer_id,
                                 double duration_us,
                                 const char* args_json) {
  if (!g_metal_cfg.command_buffer_spans) return;
  if (g_metal_paused.load()) return;
  if (!tprof_passes_filters(label, g_metal_cfg.include_label, g_metal_cfg.exclude_label)) return;
  const auto payload = provider_payload("metal", "command_buffer", label,
                                        command_buffer_id, "label", label, args_json);
  device_activity(label ? label : "metal.command_buffer", duration_us, payload.c_str());
}

void metal_record_counter_sample(const char* counter,
                                 double value,
                                 uint64_t command_buffer_id,
                                 const char* probe,
                                 const char* args_json) {
  if (!g_metal_cfg.counter_sample_buffers) return;
  if (g_metal_paused.load()) return;
  if (!tprof_passes_filters(counter, g_metal_cfg.include_counter, g_metal_cfg.exclude_counter)) return;
  const auto payload = provider_payload("metal", "counter", counter,
                                        command_buffer_id, "probe", probe, args_json);
  (void)payload;
  counter_add(counter ? counter : "metal.counter", value);
}

} // namespace tprof
