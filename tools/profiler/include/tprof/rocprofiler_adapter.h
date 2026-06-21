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
  const char* include_api = nullptr;
  const char* exclude_api = nullptr;
  const char* include_kernel = nullptr;
  const char* exclude_kernel = nullptr;
};

bool rocprofiler_adapter_init(const rocprofiler_adapter_config_t& cfg = {});
void rocprofiler_adapter_shutdown();
void rocprofiler_adapter_pause();
void rocprofiler_adapter_resume();
bool rocprofiler_adapter_is_paused();

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

} // namespace tprof
