#include "tprof/tessera_runtime_adapter.h"

#include "tessera/tessera_runtime.h"
#include "tprof/tprof_runtime.h"

#include <atomic>

namespace {

std::atomic<bool> g_attached{false};

void tessera_profile_callback(TsrProfileEventKind kind,
                              const char* name,
                              const char* payload_json,
                              double value,
                              void*) {
  switch (kind) {
    case TSR_PROFILE_RUNTIME_API:
      tprof::runtime_api(name, payload_json);
      break;
    case TSR_PROFILE_DEVICE_ACTIVITY:
      tprof::device_activity(name, value, payload_json);
      break;
  }
}

} // namespace

namespace tprof {

bool attach_tessera_runtime_trace(bool enable_tessera_profiling) {
  TsrStatus st = tsrSetProfileEventCallback(tessera_profile_callback, nullptr);
  if (st != TSR_STATUS_SUCCESS) {
    return false;
  }
  g_attached.store(true, std::memory_order_release);
  if (enable_tessera_profiling) {
    tsrEnableProfiling(1);
  }
  return true;
}

void detach_tessera_runtime_trace() {
  if (!g_attached.exchange(false, std::memory_order_acq_rel)) {
    return;
  }
  tsrSetProfileEventCallback(nullptr, nullptr);
}

} // namespace tprof
