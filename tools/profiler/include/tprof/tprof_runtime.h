#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tprof {

struct config_t {
  enum mode_t { FAST = 0, DEEP = 1 };
  mode_t mode = FAST;
  int sampling_us = 0; // stub
};

struct event_t {
  enum type_t { RANGE_B, RANGE_E, MARKER, COUNTER };
  type_t type;
  const char* name;
  uint64_t ts_ns;
  uint64_t tid;
  double value;
};

void enable(const config_t& cfg);
void disable();
void push(const char* name);
void pop();
void marker(const char* name);
void counter_add(const char* name, double v);

bool export_chrome(const std::string& path);
bool export_perfetto(const std::string& path); // minimal JSON compatible with Perfetto UI

struct range_t {
  explicit range_t(const char* name): name_(name) { push(name_); }
  ~range_t() { pop(); }
private:
  const char* name_;
};

void nvtx_push(const char* name);
void nvtx_pop();
bool cupti_init();
void cupti_shutdown();

uint64_t now_ns();
uint64_t this_thread_id();

} // namespace tprof
