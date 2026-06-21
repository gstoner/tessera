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
extern "C" bool tprof_metal_capture_command_buffer_timestamp(const char* label,
                                                             uint64_t command_buffer_id,
                                                             double* start_us,
                                                             double* end_us,
                                                             const char** error);
extern "C" bool tprof_metal_discover_counter_sets(uint64_t* counter_set_count,
                                                  const char** first_counter_set,
                                                  const char** error);
#endif

bool metal_adapter_init(const metal_adapter_config_t& cfg) {
  g_metal_cfg = cfg;
  g_metal_paused.store(cfg.start_paused);
#if defined(TPROF_WITH_METAL) && defined(__APPLE__)
  g_metal_initialized.store(true);
  g_metal_last_error = "compiled Metal timestamp/counter-discovery shell; hardware proof required for native_available";
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
  const bool native_timestamp_capture = metal_framework_compiled && g_metal_initialized.load();
  const bool counter_set_discovery = metal_framework_compiled && g_metal_initialized.load();
  return metal_adapter_status_t{
      compiled_for_apple,
      metal_framework_compiled,
      native_timestamp_capture,
      counter_set_discovery,
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

bool metal_capture_command_buffer_timestamp(const char* label,
                                            uint64_t command_buffer_id,
                                            metal_command_buffer_timestamp_t* out) {
  if (out == nullptr) return false;
  out->command_buffer_id = command_buffer_id;
  out->start_us = 0.0;
  out->end_us = 0.0;
  out->label = label;
  out->error = nullptr;
#if defined(TPROF_WITH_METAL) && defined(__APPLE__)
  const char* error = nullptr;
  const bool ok = tprof_metal_capture_command_buffer_timestamp(
      label, command_buffer_id, &out->start_us, &out->end_us, &error);
  out->error = error;
  if (!ok) {
    g_metal_last_error = error != nullptr ? error : "Metal command-buffer timestamp capture failed";
  }
  return ok;
#else
  out->error = "Metal support was not enabled for this build";
  g_metal_last_error = out->error;
  return false;
#endif
}

bool metal_record_native_command_buffer(const char* label,
                                        uint64_t command_buffer_id,
                                        const char* args_json) {
  metal_command_buffer_timestamp_t timestamp{};
  if (!metal_capture_command_buffer_timestamp(label, command_buffer_id, &timestamp)) {
    return false;
  }
  metal_record_command_buffer(label,
                              command_buffer_id,
                              duration_from_us(timestamp.start_us, timestamp.end_us),
                              args_json);
  return true;
}

bool metal_discover_counter_sets(metal_counter_set_discovery_t* out) {
  if (out == nullptr) return false;
  out->available = false;
  out->counter_set_count = 0;
  out->first_counter_set = nullptr;
  out->error = nullptr;
#if defined(TPROF_WITH_METAL) && defined(__APPLE__)
  const char* first = nullptr;
  const char* error = nullptr;
  const bool ok = tprof_metal_discover_counter_sets(&out->counter_set_count, &first, &error);
  out->available = ok;
  out->first_counter_set = first;
  out->error = error;
  if (!ok) {
    g_metal_last_error = error != nullptr ? error : "Metal counter-set discovery failed";
  }
  return ok;
#else
  out->error = "Metal support was not enabled for this build";
  g_metal_last_error = out->error;
  return false;
#endif
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
