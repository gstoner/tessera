#pragma once

#include <cstdint>

namespace tprof {

struct rocprofiler_adapter_config_t {
  bool hip_hsa_api_tracing = true;
  bool dispatch_activity_records = true;
  bool counter_collection = false;
  bool thread_trace = false;
  bool start_paused = false;
  uint64_t buffer_bytes = 4 * 1024 * 1024;
  uint64_t thread_trace_max_bytes = 256 * 1024 * 1024;
  bool thread_trace_serialize_all = false;
  const char* include_api = nullptr;
  const char* exclude_api = nullptr;
  const char* include_kernel = nullptr;
  const char* exclude_kernel = nullptr;
  const char* include_counter = nullptr;
  const char* exclude_counter = nullptr;
};

struct rocprofiler_adapter_status_t {
  bool sdk_compiled;
  bool initialized;
  bool paused;
  bool hip_hsa_api_tracing;
  bool dispatch_activity_records;
  bool counter_collection;
  bool thread_trace;
  bool thread_trace_volume_limited;
  uint64_t buffer_bytes;
  uint64_t thread_trace_max_bytes;
  const char* source_status;
  const char* last_error;
};

struct rocprofiler_api_record_t {
  const char* name;
  const char* domain;
  uint64_t correlation_id;
  double start_us;
  double end_us;
  const char* args_json;
};

struct rocprofiler_activity_record_t {
  const char* name;
  const char* activity;
  uint64_t correlation_id;
  double start_us;
  double end_us;
  uint64_t dispatch_id;
  const char* args_json;
};

struct rocprofiler_counter_record_t {
  const char* metric;
  double value;
  uint64_t correlation_id;
  const char* args_json;
};

struct rocprofiler_thread_trace_record_t {
  const char* kernel;
  uint64_t dispatch_id;
  double start_us;
  double end_us;
  uint64_t trace_bytes;
  const char* args_json;
};

bool rocprofiler_adapter_init(const rocprofiler_adapter_config_t& cfg = {});
void rocprofiler_adapter_shutdown();
void rocprofiler_adapter_pause();
void rocprofiler_adapter_resume();
bool rocprofiler_adapter_is_paused();
rocprofiler_adapter_status_t rocprofiler_adapter_status();

void rocprofiler_record_api(const char* name,
                            const char* domain,
                            uint64_t correlation_id,
                            double duration_us,
                            const char* args_json = nullptr);
void rocprofiler_record_activity(const char* name,
                                 const char* activity,
                                 uint64_t correlation_id,
                                 double duration_us,
                                 const char* args_json = nullptr);
void rocprofiler_record_counter(const char* metric,
                                double value,
                                uint64_t correlation_id,
                                const char* args_json = nullptr);
void rocprofiler_record_thread_trace(const char* kernel,
                                     uint64_t dispatch_id,
                                     double duration_us,
                                     const char* args_json = nullptr);

void rocprofiler_replay_api_record(const rocprofiler_api_record_t& record);
void rocprofiler_replay_activity_record(const rocprofiler_activity_record_t& record);
void rocprofiler_replay_counter_record(const rocprofiler_counter_record_t& record);
void rocprofiler_replay_thread_trace_record(const rocprofiler_thread_trace_record_t& record);

} // namespace tprof
