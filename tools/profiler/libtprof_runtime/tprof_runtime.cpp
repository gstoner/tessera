#include "tprof/tprof_runtime.h"
#include <chrono>
#include <thread>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
static uint64_t get_tid() { return static_cast<uint64_t>(::GetCurrentThreadId()); }
#else
#include <unistd.h>
#include <sys/syscall.h>
static uint64_t get_tid() {
#ifdef SYS_gettid
  return static_cast<uint64_t>(::syscall(SYS_gettid));
#else
  return static_cast<uint64_t>(::getpid());
#endif
}
#endif

namespace tprof {

static std::atomic<bool> g_enabled{false};
static Config g_cfg{};

static std::mutex g_mu;
static std::vector<Event> g_events;

uint64_t now_ns() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

uint64_t this_thread_id() { return get_tid(); }

void enable(const Config& cfg) {
  g_cfg = cfg;
  g_enabled.store(true, std::memory_order_release);
}

void disable() {
  g_enabled.store(false, std::memory_order_release);
}

void push(const char* name) {
  if (!g_enabled.load(std::memory_order_acquire)) return;
  Event e{Event::RANGE_B, name, now_ns(), this_thread_id(), 0.0};
  std::lock_guard<std::mutex> lk(g_mu);
  g_events.emplace_back(e);
  nvtx_push(name);
}

void pop() {
  if (!g_enabled.load(std::memory_order_acquire)) return;
  Event e{Event::RANGE_E, "", now_ns(), this_thread_id(), 0.0};
  std::lock_guard<std::mutex> lk(g_mu);
  g_events.emplace_back(e);
  nvtx_pop();
}

void marker(const char* name) {
  if (!g_enabled.load(std::memory_order_acquire)) return;
  Event e{Event::MARKER, name, now_ns(), this_thread_id(), 0.0};
  std::lock_guard<std::mutex> lk(g_mu);
  g_events.emplace_back(e);
}

void counter_add(const char* name, double v) {
  if (!g_enabled.load(std::memory_order_acquire)) return;
  Event e{Event::COUNTER, name, now_ns(), this_thread_id(), v};
  std::lock_guard<std::mutex> lk(g_mu);
  g_events.emplace_back(e);
}

// Forward decl from exporter
bool chrome_export(const std::vector<Event>&, const std::string&);

bool export_chrome(const std::string& path) {
  std::lock_guard<std::mutex> lk(g_mu);
  return chrome_export(g_events, path);
}

} // namespace tprof
