#pragma once

#include <cstdint>

namespace tprof {

struct cupti_adapter_config_t {
  bool runtime_driver_callbacks = true;
  bool activity_records = true;
  bool start_paused = false;
  uint64_t activity_buffer_bytes = 4 * 1024 * 1024;
  const char* include_api = nullptr;
  const char* exclude_api = nullptr;
  const char* include_activity = nullptr;
  const char* exclude_activity = nullptr;
};

struct cupti_adapter_status_t {
  bool sdk_compiled;
  bool initialized;
  bool paused;
  bool runtime_driver_callbacks;
  bool activity_records;
  uint64_t activity_buffer_bytes;
  const char* source_status;
  const char* last_error;
};

struct cupti_callback_record_t {
  const char* name;
  const char* domain;
  uint64_t correlation_id;
  double start_us;
  double end_us;
  const char* args_json;
};

struct cupti_activity_record_t {
  const char* name;
  const char* activity;
  uint64_t correlation_id;
  double start_us;
  double end_us;
  const char* args_json;
};

bool cupti_adapter_init(const cupti_adapter_config_t& cfg = {});
void cupti_adapter_shutdown();
void cupti_adapter_pause();
void cupti_adapter_resume();
bool cupti_adapter_is_paused();
cupti_adapter_status_t cupti_adapter_status();

void cupti_record_callback(const char* name,
                           const char* domain,
                           uint64_t correlation_id,
                           double duration_us,
                           const char* args_json = nullptr);
void cupti_record_activity(const char* name,
                           const char* activity,
                           uint64_t correlation_id,
                           double duration_us,
                           const char* args_json = nullptr);
void cupti_replay_callback_record(const cupti_callback_record_t& record);
void cupti_replay_activity_record(const cupti_activity_record_t& record);

} // namespace tprof
