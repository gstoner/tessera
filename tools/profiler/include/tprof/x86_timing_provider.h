#pragma once

#include <cstdint>

namespace tprof {

enum x86_perf_event_bit : uint32_t {
  X86_PERF_TASK_CLOCK = 1u << 0,
  X86_PERF_CPU_CYCLES = 1u << 1,
  X86_PERF_REF_CPU_CYCLES = 1u << 2,
  X86_PERF_INSTRUCTIONS = 1u << 3,
  X86_PERF_BRANCHES = 1u << 4,
  X86_PERF_BRANCH_MISSES = 1u << 5,
  X86_PERF_CACHE_REFERENCES = 1u << 6,
  X86_PERF_CACHE_MISSES = 1u << 7,
};

inline constexpr uint32_t X86_PERF_PORTABLE_EVENT_MASK = 0xffu;

struct x86_timing_capabilities_t {
  bool x86_64 = false;
  bool avx512f = false;
  bool monotonic_raw = false;
  bool rdtscp = false;
  bool invariant_tsc = false;
  bool perf_event_open = false;
  int perf_event_paranoid = -1;
  int perf_error = 0;
};

struct x86_clock_snapshot_t {
  uint64_t steady_ns = 0;
  uint64_t monotonic_raw_ns = 0;
  uint64_t tsc_cycles = 0;
  uint32_t logical_cpu = 0;
  bool monotonic_raw_valid = false;
  bool tsc_valid = false;
};

struct x86_perf_snapshot_t {
  uint64_t task_clock_ns = 0;
  uint64_t cpu_cycles = 0;
  uint64_t ref_cpu_cycles = 0;
  uint64_t instructions = 0;
  uint64_t branches = 0;
  uint64_t branch_misses = 0;
  uint64_t cache_references = 0;
  uint64_t cache_misses = 0;
  uint64_t time_enabled_ns = 0;
  uint64_t time_running_ns = 0;
  double multiplex_scale = 1.0;
  uint32_t available_events_mask = 0;
  uint32_t unavailable_events_mask = X86_PERF_PORTABLE_EVENT_MASK;
  bool valid = false;
  bool multiplexed = false;
  int error_code = 0;
};

// Capabilities are observed, not inferred from the target triple. In
// particular, perf_event_open remains false when the host policy denies it.
x86_timing_capabilities_t x86_timing_capabilities();
x86_clock_snapshot_t x86_clock_snapshot();

class x86_perf_group_t {
 public:
  x86_perf_group_t();
  ~x86_perf_group_t();
  x86_perf_group_t(const x86_perf_group_t&) = delete;
  x86_perf_group_t& operator=(const x86_perf_group_t&) = delete;

  bool open();
  bool start();
  x86_perf_snapshot_t stop();
  void close();
  int error_code() const { return error_code_; }

 private:
  int leader_fd_ = -1;
  int event_fds_[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  unsigned event_slots_[8] = {};
  unsigned event_count_ = 0;
  uint32_t available_events_mask_ = 0;
  int error_code_ = 0;
};

}  // namespace tprof
