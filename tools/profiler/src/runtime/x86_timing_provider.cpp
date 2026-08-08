#include "tprof/x86_timing_provider.h"

#include <cerrno>
#include <chrono>
#include <fstream>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) && defined(__GNUC__)
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace tprof {
namespace {

#if defined(__linux__)
int perf_event_open(struct perf_event_attr* attr, int group_fd) {
  return static_cast<int>(
      syscall(__NR_perf_event_open, attr, 0, -1, group_fd, 0));
}

perf_event_attr make_event(uint32_t type, uint64_t config, bool leader) {
  perf_event_attr attr{};
  attr.type = type;
  attr.size = sizeof(attr);
  attr.config = config;
  attr.disabled = leader ? 1 : 0;
  attr.inherit = 0;
  attr.exclude_kernel = 0;
  attr.exclude_hv = 0;
  attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED |
                     PERF_FORMAT_TOTAL_TIME_RUNNING;
  return attr;
}

int read_perf_paranoid() {
  std::ifstream input("/proc/sys/kernel/perf_event_paranoid");
  int value = -1;
  input >> value;
  return input ? value : -1;
}
#endif

#if defined(__x86_64__) && defined(__GNUC__)
bool has_extended_edx_bit(unsigned leaf, unsigned bit) {
  unsigned max_leaf = __get_cpuid_max(0x80000000, nullptr);
  if (max_leaf < leaf) return false;
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  __cpuid(leaf, eax, ebx, ecx, edx);
  (void)eax;
  (void)ebx;
  (void)ecx;
  return edx & (1u << bit);
}

bool has_rdtscp() { return has_extended_edx_bit(0x80000001, 27); }
bool has_invariant_tsc() { return has_extended_edx_bit(0x80000007, 8); }
bool has_avx512f() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f");
}
#endif

}  // namespace

x86_timing_capabilities_t x86_timing_capabilities() {
  x86_timing_capabilities_t result;
#if defined(__x86_64__) && defined(__GNUC__)
  result.x86_64 = true;
  result.avx512f = has_avx512f();
  result.rdtscp = has_rdtscp();
  result.invariant_tsc = has_invariant_tsc();
#endif
#if defined(CLOCK_MONOTONIC_RAW)
  timespec ts{};
  result.monotonic_raw = clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0;
#endif
#if defined(__linux__)
  result.perf_event_paranoid = read_perf_paranoid();
  perf_event_attr attr =
      make_event(PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, true);
  const int fd = perf_event_open(&attr, -1);
  if (fd >= 0) {
    result.perf_event_open = true;
    ::close(fd);
  } else {
    result.perf_error = errno;
  }
#else
  result.perf_error = ENOTSUP;
#endif
  return result;
}

x86_clock_snapshot_t x86_clock_snapshot() {
  x86_clock_snapshot_t result;
  result.steady_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
#if defined(CLOCK_MONOTONIC_RAW)
  timespec raw{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &raw) == 0) {
    result.monotonic_raw_ns = static_cast<uint64_t>(raw.tv_sec) * 1000000000ull +
                              static_cast<uint64_t>(raw.tv_nsec);
    result.monotonic_raw_valid = true;
  }
#endif
#if defined(__x86_64__) && defined(__GNUC__)
  if (has_rdtscp()) {
    unsigned aux = 0;
    _mm_lfence();
    result.tsc_cycles = __rdtscp(&aux);
    _mm_lfence();
    result.logical_cpu = aux;
    result.tsc_valid = true;
  }
#endif
  return result;
}

x86_perf_group_t::x86_perf_group_t() = default;
x86_perf_group_t::~x86_perf_group_t() { close(); }

bool x86_perf_group_t::open() {
  close();
  error_code_ = 0;
#if defined(__linux__)
  constexpr uint32_t types[] = {
      PERF_TYPE_SOFTWARE, PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE,
      PERF_TYPE_HARDWARE, PERF_TYPE_HARDWARE};
  constexpr uint64_t configs[] = {
      PERF_COUNT_SW_TASK_CLOCK, PERF_COUNT_HW_CPU_CYCLES,
      PERF_COUNT_HW_REF_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS,
      PERF_COUNT_HW_BRANCH_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_MISSES,
      PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES};
  for (unsigned i = 0; i < sizeof(configs) / sizeof(configs[0]); ++i) {
    perf_event_attr attr = make_event(types[i], configs[i], i == 0);
    const int fd = perf_event_open(&attr, i == 0 ? -1 : leader_fd_);
    if (fd < 0 && i == 0) {
      error_code_ = errno;
      close();
      return false;
    }
    if (fd < 0) {
      if (error_code_ == 0) error_code_ = errno;
      continue;
    }
    if (i == 0) leader_fd_ = fd;
    event_fds_[event_count_] = fd;
    event_slots_[event_count_] = i;
    ++event_count_;
    available_events_mask_ |= 1u << i;
  }
  return true;
#else
  error_code_ = ENOTSUP;
  return false;
#endif
}

bool x86_perf_group_t::start() {
#if defined(__linux__)
  if (leader_fd_ < 0 && !open()) return false;
  if (ioctl(leader_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0 ||
      ioctl(leader_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
    error_code_ = errno;
    return false;
  }
  return true;
#else
  error_code_ = ENOTSUP;
  return false;
#endif
}

x86_perf_snapshot_t x86_perf_group_t::stop() {
  x86_perf_snapshot_t result;
#if defined(__linux__)
  if (leader_fd_ < 0) {
    result.error_code = error_code_ ? error_code_ : EBADF;
    return result;
  }
  if (ioctl(leader_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
    result.error_code = errno;
    return result;
  }
  struct group_read_t {
    uint64_t nr;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t values[8];
  } readout{};
  const ssize_t bytes = ::read(leader_fd_, &readout, sizeof(readout));
  const auto expected_bytes = static_cast<ssize_t>(
      sizeof(uint64_t) * (3 + event_count_));
  if (bytes < expected_bytes || readout.nr != event_count_) {
    result.error_code = bytes < 0 ? errno : EIO;
    return result;
  }
  uint64_t* destinations[] = {
      &result.task_clock_ns, &result.cpu_cycles, &result.ref_cpu_cycles,
      &result.instructions, &result.branches, &result.branch_misses,
      &result.cache_references, &result.cache_misses};
  for (unsigned i = 0; i < event_count_; ++i) {
    *destinations[event_slots_[i]] = readout.values[i];
  }
  result.time_enabled_ns = readout.time_enabled;
  result.time_running_ns = readout.time_running;
  result.error_code = error_code_;
  result.available_events_mask = available_events_mask_;
  result.unavailable_events_mask =
      X86_PERF_PORTABLE_EVENT_MASK & ~available_events_mask_;
  result.valid = readout.time_running > 0;
  if (result.valid) {
    result.multiplex_scale = static_cast<double>(readout.time_enabled) /
                             static_cast<double>(readout.time_running);
    result.multiplexed = readout.time_running < readout.time_enabled;
  }
#else
  result.error_code = ENOTSUP;
#endif
  return result;
}

void x86_perf_group_t::close() {
#if defined(__linux__)
  for (unsigned i = 0; i < event_count_; ++i) {
    if (event_fds_[i] >= 0) ::close(event_fds_[i]);
    event_fds_[i] = -1;
    event_slots_[i] = 0;
  }
#endif
  leader_fd_ = -1;
  event_count_ = 0;
  available_events_mask_ = 0;
}

}  // namespace tprof
