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

bool metal_adapter_init(const metal_adapter_config_t& cfg = {});
void metal_adapter_shutdown();
void metal_adapter_pause();
void metal_adapter_resume();
bool metal_adapter_is_paused();

void metal_record_command_buffer(const char* label,
                                 uint64_t command_buffer_id,
                                 double duration_us,
                                 const char* args_json = nullptr);
void metal_record_counter_sample(const char* counter,
                                 double value,
                                 uint64_t command_buffer_id,
                                 const char* probe,
                                 const char* args_json = nullptr);

} // namespace tprof
