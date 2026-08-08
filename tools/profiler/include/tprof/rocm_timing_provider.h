#pragma once

#include <cstdint>
#include <string>

namespace tprof {

struct rocm_timing_probe_config_t {
  int repetitions = 100;
  int spin_iterations = 4096;
};

struct rocm_timing_probe_result_t {
  std::string architecture;
  std::string device_name;
  std::string error;
  uint64_t host_wall_ns = 0;
  uint64_t empty_host_wall_ns = 0;
  uint64_t device_wall_clock_ticks = 0;
  uint64_t device_wall_clock_ns = 0;
  int wall_clock_rate_khz = 0;
  float hip_event_ms = 0.0f;
  float empty_hip_event_ms = 0.0f;
  int repetitions = 0;
  int blocks = 0;
  int threads = 0;
  bool exact_gfx1151 = false;
  bool wsl = false;
  bool host_wall_valid = false;
  bool hip_event_valid = false;
  bool device_wall_clock_valid = false;
  bool clock_agreement_valid = false;
  bool eligible_for_regression = false;
  bool eligible_for_promotion = false;
};

rocm_timing_probe_result_t collect_rocm_timing_probe(
    const rocm_timing_probe_config_t& config = {});

}  // namespace tprof
