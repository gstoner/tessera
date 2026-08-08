#include "tprof/rocm_timing_provider.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

namespace tprof {
namespace {

__global__ void empty_probe_kernel() {}

__global__ void wall_clock_probe_kernel(unsigned long long* envelope,
                                        int spin_iterations) {
  const unsigned long long entry =
      static_cast<unsigned long long>(wall_clock64());
  atomicMin(envelope, entry);
  unsigned long long value =
      entry ^ (static_cast<unsigned long long>(blockIdx.x) << 32) ^ threadIdx.x;
  for (int iteration = 0; iteration < spin_iterations; ++iteration) {
    value = value * 6364136223846793005ull + 1442695040888963407ull;
  }
  if (value == 0) envelope[2] = value;
  const unsigned long long exit =
      static_cast<unsigned long long>(wall_clock64());
  atomicMax(envelope + 1, exit);
}

bool is_wsl_host() {
  std::ifstream input("/proc/version");
  std::ostringstream stream;
  stream << input.rdbuf();
  std::string text = stream.str();
  for (char& ch : text) ch = static_cast<char>(std::tolower(ch));
  return text.find("microsoft") != std::string::npos ||
         text.find("wsl") != std::string::npos;
}

std::string architecture_name(const hipDeviceProp_t& properties) {
  std::string name = properties.gcnArchName;
  const auto feature = name.find(':');
  return feature == std::string::npos ? name : name.substr(0, feature);
}

bool valid_event(float milliseconds) {
  return std::isfinite(milliseconds) && milliseconds > 0.0f;
}

uint64_t ticks_to_ns(uint64_t ticks, int rate_khz) {
  if (rate_khz <= 0) return 0;
  return static_cast<uint64_t>(
      static_cast<long double>(ticks) * 1000000.0L / rate_khz);
}

}  // namespace

rocm_timing_probe_result_t collect_rocm_timing_probe(
    const rocm_timing_probe_config_t& config) {
  rocm_timing_probe_result_t result;
  result.repetitions = config.repetitions;
  result.threads = 256;
  result.wsl = is_wsl_host();
  if (config.repetitions <= 0 || config.spin_iterations < 0) {
    result.error = "invalid probe configuration";
    return result;
  }
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess) {
    result.error = "hipGetDevice failed";
    return result;
  }
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, device) != hipSuccess) {
    result.error = "hipGetDeviceProperties failed";
    return result;
  }
  result.architecture = architecture_name(properties);
  result.device_name = properties.name;
  result.exact_gfx1151 = result.architecture == "gfx1151";
  result.blocks = properties.multiProcessorCount * 2;
  if (!result.exact_gfx1151) {
    result.error = "native timing probe requires exact gfx1151";
    return result;
  }
  if (hipDeviceGetAttribute(&result.wall_clock_rate_khz,
                            hipDeviceAttributeWallClockRate, device) !=
      hipSuccess) {
    result.error = "hipDeviceAttributeWallClockRate unavailable";
    return result;
  }

  unsigned long long* device_envelope = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (hipMalloc(&device_envelope, 3 * sizeof(unsigned long long)) != hipSuccess ||
      hipEventCreate(&start) != hipSuccess ||
      hipEventCreate(&stop) != hipSuccess) {
    result.error = "HIP timing resource creation failed";
    if (device_envelope) (void)hipFree(device_envelope);
    if (start) (void)hipEventDestroy(start);
    if (stop) (void)hipEventDestroy(stop);
    return result;
  }
  const auto cleanup = [&] {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    (void)hipFree(device_envelope);
  };

  const unsigned long long warmup_initial[3] = {
      std::numeric_limits<unsigned long long>::max(), 0, 0};
  if (hipMemcpy(device_envelope, warmup_initial, sizeof(warmup_initial),
                hipMemcpyHostToDevice) != hipSuccess) {
    result.error = "HIP warmup initialization failed";
    cleanup();
    return result;
  }
  wall_clock_probe_kernel<<<result.blocks, result.threads>>>(
      device_envelope, config.spin_iterations);
  if (hipDeviceSynchronize() != hipSuccess) {
    result.error = "HIP warmup failed";
  } else {
    const unsigned long long initial[3] = {
        std::numeric_limits<unsigned long long>::max(), 0, 0};
    if (hipMemcpy(device_envelope, initial, sizeof(initial),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipEventRecord(start) != hipSuccess) {
      result.error = "HIP measurement initialization failed";
      cleanup();
      return result;
    }
    const auto host_start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < config.repetitions; ++repetition) {
      wall_clock_probe_kernel<<<result.blocks, result.threads>>>(
          device_envelope, config.spin_iterations);
    }
    const hipError_t launch_status = hipGetLastError();
    const hipError_t stop_status = hipEventRecord(stop);
    if (launch_status != hipSuccess || stop_status != hipSuccess) {
      result.error = "HIP instrumented launch failed";
      cleanup();
      return result;
    }
    const hipError_t synchronization = hipEventSynchronize(stop);
    const auto host_stop = std::chrono::steady_clock::now();
    result.host_wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(host_stop - host_start)
            .count());
    result.host_wall_valid = synchronization == hipSuccess && result.host_wall_ns > 0;
    if (synchronization == hipSuccess)
      (void)hipEventElapsedTime(&result.hip_event_ms, start, stop);
    unsigned long long envelope[3] = {};
    if (synchronization == hipSuccess &&
        hipMemcpy(envelope, device_envelope, sizeof(envelope),
                  hipMemcpyDeviceToHost) == hipSuccess &&
        envelope[1] >= envelope[0]) {
      result.device_wall_clock_ticks = envelope[1] - envelope[0];
      result.device_wall_clock_ns = ticks_to_ns(
          result.device_wall_clock_ticks, result.wall_clock_rate_khz);
    }

    if (hipEventRecord(start) != hipSuccess) {
      result.error = "HIP empty-launch timing initialization failed";
      cleanup();
      return result;
    }
    const auto empty_start = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < config.repetitions; ++repetition)
      empty_probe_kernel<<<1, 1>>>();
    const hipError_t empty_launch_status = hipGetLastError();
    const hipError_t empty_stop_status = hipEventRecord(stop);
    if (empty_launch_status != hipSuccess || empty_stop_status != hipSuccess) {
      result.error = "HIP empty launch failed";
      cleanup();
      return result;
    }
    const hipError_t empty_sync = hipEventSynchronize(stop);
    const auto empty_stop = std::chrono::steady_clock::now();
    result.empty_host_wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(empty_stop - empty_start)
            .count());
    if (empty_sync == hipSuccess)
      (void)hipEventElapsedTime(&result.empty_hip_event_ms, start, stop);

    result.hip_event_valid = valid_event(result.hip_event_ms);
    result.device_wall_clock_valid = result.device_wall_clock_ns > 0;
    if (result.hip_event_valid && result.device_wall_clock_valid) {
      const long double event_ns = result.hip_event_ms * 1000000.0L;
      const long double difference = std::fabs(
          event_ns - static_cast<long double>(result.device_wall_clock_ns));
      result.clock_agreement_valid = difference / event_ns <= 0.05L;
    }
    result.eligible_for_regression =
        result.host_wall_valid && result.device_wall_clock_valid;
    result.eligible_for_promotion =
        result.eligible_for_regression && result.hip_event_valid &&
        result.clock_agreement_valid && !result.wsl;
  }
  cleanup();
  return result;
}

}  // namespace tprof
