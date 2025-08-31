#include <vector>
#include <memory>
#include <mutex>
#include <cstring>
#include <string>
#include <cassert>
#include <chrono>
#include "backend/base_backend.h"
#include "../include/tessera/tessera_runtime.h"
#include "../include/tessera/tsr_types.h"
#include "util/logging.h"

using namespace tsr;

struct tsrDevice_t { std::unique_ptr<Backend> be; };
struct tsrStream_t { tsr::Stream* impl; tsrDevice_t* dev; };
struct tsrEvent_t  { tsr::Event*  impl; tsrDevice_t* dev; };
struct tsrBuffer_t { tsr::Buffer* impl; tsrDevice_t* dev; };

static std::vector<tsrDevice_t*> g_devices;
static std::once_flag g_init_once;
static std::mutex g_mu;

static thread_local std::string g_last_error;
static bool g_profiling_enabled = true;

static void SetLastError(const char* msg) { g_last_error = msg ? msg : ""; }
static uint64_t NowNs() {
  using namespace std::chrono;
  static const auto t0 = steady_clock::now();
  return duration_cast<nanoseconds>(steady_clock::now() - t0).count();
}

extern "C" {

// ---- Version & profiling ----
void tsrGetVersion(int* major, int* minor, int* patch) {
  if (major) *major = TESSERA_VERSION_MAJOR;
  if (minor) *minor = TESSERA_VERSION_MINOR;
  if (patch) *patch = TESSERA_VERSION_PATCH;
}
void tsrEnableProfiling(int enable) { g_profiling_enabled = (enable != 0); }
uint64_t tsrTimestampNowNs(void) { return NowNs(); }

// ---- Status & error strings ----
const char* tsrStatusString(TsrStatus s) {
  switch (s) {
    case TSR_STATUS_SUCCESS: return "SUCCESS";
    case TSR_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case TSR_STATUS_NOT_FOUND: return "NOT_FOUND";
    case TSR_STATUS_ALREADY_EXISTS: return "ALREADY_EXISTS";
    case TSR_STATUS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    case TSR_STATUS_UNIMPLEMENTED: return "UNIMPLEMENTED";
    case TSR_STATUS_INTERNAL: return "INTERNAL";
    case TSR_STATUS_DEVICE_ERROR: return "DEVICE_ERROR";
    default: return "UNKNOWN";
  }
}
const char* tsrGetLastError(void) { return g_last_error.c_str(); }
void tsrClearLastError(void) { g_last_error.clear(); }

// ---- Init / Shutdown ----
TsrStatus tsrInit(void) {
  std::call_once(g_init_once, [](){
    std::lock_guard<std::mutex> lk(g_mu);
    // Always have a CPU backend
    auto* dev_cpu = new tsrDevice_t();
    dev_cpu->be = CreateCpuBackend();
    g_devices.push_back(dev_cpu);

    // Optional CUDA/HIP backends (stubs) if compiled in
    if (auto cuda = CreateCudaBackend()) {
      auto* d = new tsrDevice_t(); d->be = std::move(cuda); g_devices.push_back(d);
    }
    if (auto hip = CreateHipBackend()) {
      auto* d = new tsrDevice_t(); d->be = std::move(hip); g_devices.push_back(d);
    }
  });
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrShutdown(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  for (auto* d : g_devices) delete d;
  g_devices.clear();
  return TSR_STATUS_SUCCESS;
}

// ---- Devices ----
TsrStatus tsrGetDeviceCount(int* count) {
  if (!count) { SetLastError("count==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  *count = (int)g_devices.size();
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrGetDevice(int index, tsrDevice* out) {
  if (!out) { SetLastError("out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  if (index < 0 || index >= (int)g_devices.size()) {
    SetLastError("device index out of range"); return TSR_STATUS_NOT_FOUND;
  }
  *out = g_devices[index];
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrGetDeviceProps(tsrDevice dev, tsrDeviceProps* props) {
  if (!dev || !props) { SetLastError("dev/props==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  auto p = dev->be->props();
  props->kind = p.kind;
  std::snprintf(props->name, sizeof(props->name), "%s", p.name.c_str());
  props->logical_tile_threads_max = p.logical_tile_threads_max;
  props->concurrent_tiles_hint = p.concurrent_tiles_hint;
  return TSR_STATUS_SUCCESS;
}

// ---- Streams / Events ----
TsrStatus tsrCreateStream(tsrDevice dev, tsrStream* out) {
  if (!dev || !out) { SetLastError("dev/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  auto s = new tsrStream_t();
  s->impl = dev->be->createStream();
  s->dev = dev;
  *out = s;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrDestroyStream(tsrStream s) {
  if (!s) { SetLastError("s==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  s->dev->be->destroyStream(s->impl);
  delete s;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrStreamSynchronize(tsrStream s) {
  if (!s) { SetLastError("s==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  s->dev->be->streamSync(s->impl);
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrCreateEvent(tsrDevice dev, tsrEvent* out) {
  if (!dev || !out) { SetLastError("dev/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  auto e = new tsrEvent_t();
  e->impl = dev->be->createEvent();
  e->dev = dev;
  *out = e;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrRecordEvent(tsrEvent e, tsrStream s) {
  if (!e || !s) { SetLastError("e/s==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  e->dev->be->recordEvent(e->impl, s->impl);
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrWaitEvent(tsrEvent e, tsrStream s) {
  if (!e || !s) { SetLastError("e/s==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  e->dev->be->waitEvent(e->impl, s->impl);
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrEventSynchronize(tsrEvent e) {
  if (!e) { SetLastError("e==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  e->dev->be->eventSync(e->impl);
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrDestroyEvent(tsrEvent e) {
  if (!e) { SetLastError("e==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  e->dev->be->destroyEvent(e->impl);
  delete e;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrEventGetTimestamp(tsrEvent e, uint64_t* ns_out) {
  if (!e || !ns_out) { SetLastError("e/ns_out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  *ns_out = e->impl->timestamp_ns;
  return TSR_STATUS_SUCCESS;
}

// ---- Memory ----
TsrStatus tsrMalloc(tsrDevice dev, size_t bytes, tsrBuffer* out) {
  if (!dev || !out) { SetLastError("dev/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  auto impl = dev->be->malloc(bytes);
  if (!impl) { SetLastError("allocation failed"); return TSR_STATUS_OUT_OF_MEMORY; }
  auto b = new tsrBuffer_t();
  b->impl = impl;
  b->dev = dev;
  *out = b;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrFree(tsrBuffer b) {
  if (!b) { SetLastError("b==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  b->dev->be->free(b->impl);
  delete b;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrMemset(tsrBuffer b, int value, size_t bytes) {
  if (!b) { SetLastError("b==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  b->dev->be->memset(b->impl, value, bytes);
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrMemcpy(tsrBuffer dst, const tsrBuffer src, size_t bytes, TsrMemcpyKind kind) {
  if (!dst || !src) { SetLastError("dst/src==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  dst->dev->be->memcpy(dst->impl, src->impl, bytes, kind);
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrMap(tsrBuffer b, void** host_ptr, size_t* bytes) {
  if (!b || !host_ptr) { SetLastError("b/host_ptr==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  *host_ptr = b->dev->be->map(b->impl);
  if (bytes) *bytes = b->impl->bytes;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrUnmap(tsrBuffer b) {
  if (!b) { SetLastError("b==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  b->dev->be->unmap(b->impl);
  return TSR_STATUS_SUCCESS;
}

// ---- Shape helpers ----
TsrStatus tsrValidateLaunch(const tsrDeviceProps* props, const tsrLaunchParams* p) {
  if (!props || !p) { SetLastError("props/p==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  if (p->tile.x == 0 || p->tile.y == 0 || p->tile.z == 0) {
    SetLastError("tile dims must be non-zero"); return TSR_STATUS_INVALID_ARGUMENT;
  }
  uint64_t threads = (uint64_t)p->tile.x * p->tile.y * p->tile.z;
  if (threads > props->logical_tile_threads_max) {
    SetLastError("tile threads exceed device limit"); return TSR_STATUS_INVALID_ARGUMENT;
  }
  if (p->grid.x == 0 || p->grid.y == 0 || p->grid.z == 0) {
    SetLastError("grid dims must be non-zero"); return TSR_STATUS_INVALID_ARGUMENT;
  }
  return TSR_STATUS_SUCCESS;
}

void tsrSuggestTile(const tsrDeviceProps* props, uint32_t logical_threads, tsrDim3* out_tile) {
  if (!out_tile) return;
  uint32_t maxT = props ? props->logical_tile_threads_max : 1024;
  uint32_t t = logical_threads > maxT ? maxT : logical_threads;
  out_tile->x = t; out_tile->y = 1; out_tile->z = 1;
}

// ---- Host kernel launch ----
TsrStatus tsrLaunchHostTileKernel(tsrStream s,
                                  const tsrLaunchParams* params,
                                  tsrHostKernelFn kernel,
                                  void* user_payload) {
  if (!s || !params || !kernel) { SetLastError("s/params/kernel==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  s->dev->be->launchHostKernel(s->impl, params, kernel, user_payload);
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrLaunchHostTileKernelSync(tsrDevice dev,
                                      const tsrLaunchParams* params,
                                      tsrHostKernelFn kernel,
                                      void* user_payload) {
  if (!dev) { SetLastError("dev==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  tsrStream s;
  TsrStatus st = tsrCreateStream(dev, &s);
  if (st != TSR_STATUS_SUCCESS) return st;
  st = tsrLaunchHostTileKernel(s, params, kernel, user_payload);
  if (st != TSR_STATUS_SUCCESS) { tsrDestroyStream(s); return st; }
  tsrStreamSynchronize(s);
  tsrDestroyStream(s);
  return TSR_STATUS_SUCCESS;
}

} // extern "C"
