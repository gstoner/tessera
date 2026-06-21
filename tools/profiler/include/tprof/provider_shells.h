#pragma once

#include <vector>

namespace tprof {

enum class provider_kind_t {
  NVIDIA_SYSTEM_CONTEXT,
  ROCM_SYSTEM_CONTEXT,
  APPLE_SYSTEM_CONTEXT,
  NVIDIA_CUPTI,
  ROCM_ROCPROFILER,
  APPLE_METAL_COUNTERS,
};

struct provider_shell_t {
  provider_kind_t kind;
  const char* name;
  const char* status;
  const char* artifact;
  bool runtime_api;
  bool device_activity;
  bool counters;
  bool system_context;
  bool command_correlation;
  bool api_tracing;
  bool activity_records;
  bool counter_collection;
  bool thread_trace;
  bool external_correlation;
  const char* notes;
};

provider_shell_t provider_shell(provider_kind_t kind);
std::vector<provider_shell_t> provider_shells();

bool native_system_context_init(provider_kind_t kind);
void native_system_context_shutdown(provider_kind_t kind);

bool heavy_provider_init(provider_kind_t kind);
void heavy_provider_shutdown(provider_kind_t kind);

} // namespace tprof
