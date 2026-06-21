#include "tprof/provider_shells.h"

#include <iterator>

namespace tprof {
namespace {

constexpr provider_shell_t kShells[] = {
    {
        provider_kind_t::NVIDIA_SYSTEM_CONTEXT,
        "nvidia-system-context",
        "planned",
        "nvidia_context_json",
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        "Native shell for NVML/DCGM utilization, memory, power, thermal, "
        "PCIe/NVLink, and reliability samples.",
    },
    {
        provider_kind_t::ROCM_SYSTEM_CONTEXT,
        "rocm-system-context",
        "planned",
        "rocm_context_json",
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        "Native shell for AMD SMI/RDC utilization, memory, power, thermal, "
        "PCIe/XGMI, and RAS samples.",
    },
    {
        provider_kind_t::APPLE_SYSTEM_CONTEXT,
        "apple-silicon-system-context",
#if defined(TPROF_WITH_APPLE_SYSTEM_CONTEXT) && defined(__APPLE__)
        "compiled_shell",
#else
        "planned",
#endif
        "apple_context_json",
        false,
        false,
        false,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        "Native shell for IOReport/SMC/HID-style Apple system context; this "
        "is separate from Metal command-buffer proof.",
    },
    {
        provider_kind_t::NVIDIA_CUPTI,
        "cupti-activity-callbacks",
#ifdef TPROF_WITH_CUPTI
        "compiled_shell",
#else
        "planned",
#endif
        "chrome_trace/perfetto_json+metrics_json",
        true,
        true,
        true,
        false,
        true,
        true,
        true,
        true,
        false,
        true,
        "CUPTI shell for runtime callbacks, activity records, range metrics, "
        "and PC sampling correlation.",
    },
    {
        provider_kind_t::ROCM_ROCPROFILER,
        "rocprofiler-sdk-dispatch-counters",
#ifdef TPROF_WITH_ROCPROFILER
        "compiled_shell",
#else
        "planned",
#endif
        "chrome_trace/perfetto_json+metrics_json",
        true,
        true,
        true,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        "ROCprofiler-SDK shell for HIP/HSA tracing, dispatch records, counter "
        "collection, and thread-trace correlation.",
    },
    {
        provider_kind_t::APPLE_METAL_COUNTERS,
        "metal-command-buffer-counters",
#if defined(TPROF_WITH_METAL) && defined(__APPLE__)
        "compiled_shell",
#else
        "planned",
#endif
        "chrome_trace/perfetto_json+metal_counter_json",
        false,
        true,
        true,
        false,
        true,
        false,
        true,
        true,
        false,
        true,
        "Metal shell for command-buffer timestamps and counter sample buffer "
        "correlation.",
    },
};

} // namespace

provider_shell_t provider_shell(provider_kind_t kind) {
  for (const auto& shell : kShells) {
    if (shell.kind == kind) {
      return shell;
    }
  }
  return {
      kind,
      "unknown-provider",
      "unsupported",
      "none",
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
      "No provider shell exists for this kind.",
  };
}

std::vector<provider_shell_t> provider_shells() {
  return std::vector<provider_shell_t>(std::begin(kShells), std::end(kShells));
}

bool native_system_context_init(provider_kind_t kind) {
  const auto shell = provider_shell(kind);
  return shell.system_context && false;
}

void native_system_context_shutdown(provider_kind_t) {}

bool heavy_provider_init(provider_kind_t kind) {
  const auto shell = provider_shell(kind);
  return shell.command_correlation && false;
}

void heavy_provider_shutdown(provider_kind_t) {}

} // namespace tprof
