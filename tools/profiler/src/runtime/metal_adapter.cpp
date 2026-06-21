#include "tprof/metal_adapter.h"

#include "provider_adapter_utils.h"
#include "tprof/tprof_runtime.h"

#include <atomic>

namespace tprof {
namespace {
metal_adapter_config_t g_metal_cfg{};
std::atomic<bool> g_metal_paused{false};
std::atomic<bool> g_metal_initialized{false};
const char* g_metal_last_error = "not initialized";

double duration_from_us(double start_us, double end_us) {
  return end_us > start_us ? end_us - start_us : 0.0;
}
} // namespace

#if defined(TPROF_WITH_METAL) && defined(__APPLE__)
extern "C" bool tprof_metal_command_buffer_probe_compiled();
#endif

bool metal_adapter_init(const metal_adapter_config_t& cfg) {
  g_metal_cfg = cfg;
  g_metal_paused.store(cfg.start_paused);
#if defined(TPROF_WITH_METAL) && defined(__APPLE__)
  g_metal_initialized.store(true);
  g_metal_last_error = "compiled Metal shell; native command-buffer attachment pending";
  return true;
#else
  g_metal_initialized.store(false);
  g_metal_last_error = "Metal support was not enabled for this build";
  return false;
#endif
}

void metal_adapter_shutdown() {
  g_metal_initialized.store(false);
  g_metal_last_error = "shutdown";
}

void metal_adapter_pause() {
  g_metal_paused.store(true);
}

void metal_adapter_resume() {
  g_metal_paused.store(false);
}

bool metal_adapter_is_paused() {
  return g_metal_paused.load();
}

metal_adapter_status_t metal_adapter_status() {
  const bool compiled_for_apple =
#ifdef __APPLE__
      true;
#else
      false;
#endif
  const bool metal_framework_compiled =
#if defined(TPROF_WITH_METAL) && defined(__APPLE__)
      tprof_metal_command_buffer_probe_compiled();
#else
      false;
#endif
  return metal_adapter_status_t{
      compiled_for_apple,
      metal_framework_compiled,
      g_metal_initialized.load(),
      g_metal_paused.load(),
      g_metal_cfg.command_buffer_spans,
      g_metal_cfg.counter_sample_buffers,
      metal_framework_compiled ? "compiled_shell" : "planned",
      g_metal_last_error,
  };
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

void metal_replay_command_buffer_record(const metal_command_buffer_record_t& record) {
  metal_record_command_buffer(record.label,
                              record.command_buffer_id,
                              duration_from_us(record.start_us, record.end_us),
                              record.args_json);
}

void metal_replay_counter_sample_record(const metal_counter_sample_record_t& record) {
  metal_record_counter_sample(record.counter,
                              record.value,
                              record.command_buffer_id,
                              record.probe,
                              record.args_json);
}

} // namespace tprof
