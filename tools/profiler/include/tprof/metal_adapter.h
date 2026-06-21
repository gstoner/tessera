#pragma once

#include <cstdint>

namespace tprof {

struct metal_adapter_config_t {
  bool command_buffer_spans = true;
  bool counter_sample_buffers = false;
  bool start_paused = false;
  const char* include_label = nullptr;
  const char* exclude_label = nullptr;
  const char* include_counter = nullptr;
  const char* exclude_counter = nullptr;
};

struct metal_adapter_status_t {
  bool compiled_for_apple;
  bool metal_framework_compiled;
  bool initialized;
  bool paused;
  bool command_buffer_spans;
  bool counter_sample_buffers;
  const char* source_status;
  const char* last_error;
};

struct metal_command_buffer_record_t {
  const char* label;
  uint64_t command_buffer_id;
  double start_us;
  double end_us;
  const char* args_json;
};

struct metal_counter_sample_record_t {
  const char* counter;
  double value;
  uint64_t command_buffer_id;
  const char* probe;
  const char* args_json;
};

bool metal_adapter_init(const metal_adapter_config_t& cfg = {});
void metal_adapter_shutdown();
void metal_adapter_pause();
void metal_adapter_resume();
bool metal_adapter_is_paused();
metal_adapter_status_t metal_adapter_status();

void metal_record_command_buffer(const char* label,
                                 uint64_t command_buffer_id,
                                 double duration_us,
                                 const char* args_json = nullptr);
void metal_record_counter_sample(const char* counter,
                                 double value,
                                 uint64_t command_buffer_id,
                                 const char* probe,
                                 const char* args_json = nullptr);
void metal_replay_command_buffer_record(const metal_command_buffer_record_t& record);
void metal_replay_counter_sample_record(const metal_counter_sample_record_t& record);

} // namespace tprof
