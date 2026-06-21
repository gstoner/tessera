#pragma once

#include <cstdint>

namespace tprof {

struct cupti_adapter_config_t {
  bool runtime_driver_callbacks = true;
  bool activity_records = true;
  bool subscriber_callbacks = true;
  bool activity_buffer_service = true;
  bool start_paused = false;
  uint64_t activity_buffer_bytes = 4 * 1024 * 1024;
  const char* include_api = nullptr;
  const char* exclude_api = nullptr;
  const char* include_activity = nullptr;
  const char* exclude_activity = nullptr;
  const char* requested_metrics = nullptr;
};

struct cupti_adapter_status_t {
  bool sdk_compiled;
  bool initialized;
  bool paused;
  bool runtime_driver_callbacks;
  bool activity_records;
  bool subscriber_created;
  bool runtime_callbacks_configured;
  bool driver_callbacks_configured;
  bool activity_buffer_service_configured;
  bool activity_buffer_requested;
  bool activity_buffer_completed;
  bool metric_request_validated;
  bool unsupported_metric_requested;
  uint64_t dropped_records;
  uint64_t activity_buffer_bytes;
  const char* lifecycle_stage;
  const char* source_status;
  const char* last_error;
  const char* unsupported_metric;
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
bool cupti_adapter_start_collection();
void cupti_adapter_stop_collection();
bool cupti_adapter_collection_started();
bool cupti_adapter_request_activity_buffer(uint64_t requested_bytes);
void cupti_adapter_complete_activity_buffer(uint64_t valid_bytes, uint64_t dropped_records);
bool cupti_adapter_validate_metric_request(const char* metric);
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
