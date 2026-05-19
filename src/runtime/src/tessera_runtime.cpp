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
struct tsrArtifact_t { std::string payload; };
struct tsrKernel_t { std::string name; tsrArtifact_t* artifact; };

static std::vector<tsrDevice_t*> g_devices;
// Explicit init flag (protected by `g_mu`).  We deliberately do NOT use
// `std::once_flag` here: once a `std::once_flag` has fired, it never
// re-arms, which would mean a later `tsrInit()` after `tsrShutdown()`
// reported success but couldn't repopulate `g_devices`.  Long-running
// processes (notebooks, embedded runtimes, reload tests) need clean
// re-initialization after explicit shutdown, so we track the state
// explicitly under the mutex.
static bool g_initialized = false;
static std::mutex g_mu;

// Live-handle accounting (protected by `g_mu`).  Streams, events, and
// buffers all carry a raw `tsrDevice_t*` back-pointer.  If
// `tsrShutdown()` deletes the underlying devices while handles are
// still alive, any subsequent `tsrFree` / `tsrDestroyStream` / ...
// dereferences a freed pointer.  We count outstanding handles so
// `tsrShutdown` can refuse with a clear diagnostic instead of
// silently corrupting memory.
static uint64_t g_live_streams = 0;
static uint64_t g_live_events = 0;
static uint64_t g_live_buffers = 0;

static uint64_t _liveHandleCount() {
  // Caller must hold ``g_mu``.
  return g_live_streams + g_live_events + g_live_buffers;
}

static thread_local std::string g_last_error;
static bool g_profiling_enabled = true;

static void SetLastError(const char* msg) { g_last_error = msg ? msg : ""; }

// Consult the backend's per-thread last-error slot after a
// ``void``-returning backend call.  If the backend reports a
// device error, surface it via the C ABI's ``tsrGetLastError()``
// channel and return the matching ``TsrStatus``.  This is how
// CUDA/HIP errors that previously got swallowed by the C ABI now
// propagate (P1 #3 from the 2026-05-19 static audit).
static TsrStatus _PropagateBackendError(tsrDevice_t* dev) {
  if (!dev) return TSR_STATUS_SUCCESS;
  std::string msg;
  TsrStatus st = dev->be->consumeLastError(&msg);
  if (st != TSR_STATUS_SUCCESS) {
    SetLastError(msg.c_str());
  }
  return st;
}
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
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_initialized) {
    // Idempotent: second `tsrInit()` without an intervening shutdown is
    // a benign success.
    return TSR_STATUS_SUCCESS;
  }

  // Always have a CPU backend.
  auto* dev_cpu = new tsrDevice_t();
  dev_cpu->be = CreateCpuBackend();
  g_devices.push_back(dev_cpu);

  // Optional CUDA/HIP backends (stubs) if compiled in.
  if (auto cuda = CreateCudaBackend()) {
    auto* d = new tsrDevice_t(); d->be = std::move(cuda); g_devices.push_back(d);
  }
  if (auto hip = CreateHipBackend()) {
    auto* d = new tsrDevice_t(); d->be = std::move(hip); g_devices.push_back(d);
  }

  g_initialized = true;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrShutdown(void) {
  std::lock_guard<std::mutex> lk(g_mu);
  // Refuse to tear down devices while streams / events / buffers are
  // still live — those handles carry raw ``tsrDevice_t*`` back-pointers
  // and would dereference freed memory on a subsequent ``tsrFree`` /
  // ``tsrDestroyStream`` / ``tsrDestroyEvent`` call.  This is the
  // notebook-style use-after-free P1 the audit flagged; we surface it
  // here as ``INVALID_ARGUMENT`` with a precise diagnostic instead of
  // letting it become memory corruption.
  if (uint64_t live = _liveHandleCount()) {
    static thread_local std::string buf;
    buf = "tsrShutdown: refusing to destroy devices with live handles "
          "(streams=" + std::to_string(g_live_streams) +
          ", events="  + std::to_string(g_live_events)  +
          ", buffers=" + std::to_string(g_live_buffers) +
          ", total="   + std::to_string(live) +
          "); call tsrDestroyStream / tsrDestroyEvent / tsrFree for "
          "every outstanding handle first";
    SetLastError(buf.c_str());
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  for (auto* d : g_devices) delete d;
  g_devices.clear();
  // Flip the initialized flag so a later `tsrInit()` repopulates the
  // device list.  Without this, `std::call_once`'s "fire once forever"
  // semantics would leave the runtime in a non-functional state after
  // shutdown — a real glass jaw for notebooks / reload tests / embedded
  // runtimes.
  g_initialized = false;
  return TSR_STATUS_SUCCESS;
}

// Internal: returns whether the runtime currently has any devices.
// Exposed to tests via the C ABI under `tsrIsInitialized`.
TsrStatus tsrIsInitialized(int* out) {
  if (!out) { SetLastError("out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  std::lock_guard<std::mutex> lk(g_mu);
  *out = g_initialized ? 1 : 0;
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
  { std::lock_guard<std::mutex> lk(g_mu); ++g_live_streams; }
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrDestroyStream(tsrStream s) {
  if (!s) { SetLastError("s==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  s->dev->be->destroyStream(s->impl);
  delete s;
  { std::lock_guard<std::mutex> lk(g_mu);
    if (g_live_streams) --g_live_streams; }
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrStreamSynchronize(tsrStream s) {
  if (!s) { SetLastError("s==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  s->dev->be->streamSync(s->impl);
  return _PropagateBackendError(s->dev);
}

TsrStatus tsrCreateEvent(tsrDevice dev, tsrEvent* out) {
  if (!dev || !out) { SetLastError("dev/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  auto e = new tsrEvent_t();
  e->impl = dev->be->createEvent();
  e->dev = dev;
  *out = e;
  { std::lock_guard<std::mutex> lk(g_mu); ++g_live_events; }
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrRecordEvent(tsrEvent e, tsrStream s) {
  if (!e || !s) { SetLastError("e/s==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  e->dev->be->recordEvent(e->impl, s->impl);
  return _PropagateBackendError(e->dev);
}

TsrStatus tsrWaitEvent(tsrEvent e, tsrStream s) {
  if (!e || !s) { SetLastError("e/s==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  e->dev->be->waitEvent(e->impl, s->impl);
  return _PropagateBackendError(e->dev);
}

TsrStatus tsrEventSynchronize(tsrEvent e) {
  if (!e) { SetLastError("e==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  e->dev->be->eventSync(e->impl);
  return _PropagateBackendError(e->dev);
}

TsrStatus tsrDestroyEvent(tsrEvent e) {
  if (!e) { SetLastError("e==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  e->dev->be->destroyEvent(e->impl);
  delete e;
  { std::lock_guard<std::mutex> lk(g_mu);
    if (g_live_events) --g_live_events; }
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
  { std::lock_guard<std::mutex> lk(g_mu); ++g_live_buffers; }
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrFree(tsrBuffer b) {
  if (!b) { SetLastError("b==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  tsrDevice_t* dev = b->dev;
  dev->be->free(b->impl);
  delete b;
  { std::lock_guard<std::mutex> lk(g_mu);
    if (g_live_buffers) --g_live_buffers; }
  return _PropagateBackendError(dev);
}

TsrStatus tsrMemset(tsrBuffer b, int value, size_t bytes) {
  if (!b) { SetLastError("b==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  b->dev->be->memset(b->impl, value, bytes);
  return _PropagateBackendError(b->dev);
}

TsrStatus tsrMemcpy(tsrBuffer dst, const tsrBuffer src, size_t bytes, TsrMemcpyKind kind) {
  if (!dst || !src) { SetLastError("dst/src==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  dst->dev->be->memcpy(dst->impl, src->impl, bytes, kind);
  return _PropagateBackendError(dst->dev);
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

// ---- Generated artifact ABI skeleton ----
TsrStatus tsrCompileArtifact(const char* module_ir,
                             const tsrCompileOptions* options,
                             tsrArtifact* out) {
  (void)module_ir;
  (void)options;
  (void)out;
  SetLastError("generated artifact compilation is not wired to the runtime ABI yet");
  return TSR_STATUS_UNIMPLEMENTED;
}

TsrStatus tsrLoadArtifact(const void* bytes, size_t bytes_len, tsrArtifact* out) {
  (void)bytes;
  (void)bytes_len;
  (void)out;
  SetLastError("generated artifact loading is not wired to the runtime ABI yet");
  return TSR_STATUS_UNIMPLEMENTED;
}

TsrStatus tsrDestroyArtifact(tsrArtifact artifact) {
  delete artifact;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrGetKernel(tsrArtifact artifact, const char* name, tsrKernel* out) {
  (void)artifact;
  (void)name;
  (void)out;
  SetLastError("generated artifact kernel lookup is not wired to the runtime ABI yet");
  return TSR_STATUS_UNIMPLEMENTED;
}

TsrStatus tsrLaunchKernel(tsrStream s, tsrKernel kernel, void** args, size_t nargs) {
  (void)s;
  (void)kernel;
  (void)args;
  (void)nargs;
  SetLastError("generated artifact kernel launch is not wired to the runtime ABI yet");
  return TSR_STATUS_UNIMPLEMENTED;
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
  TsrStatus st = s->dev->be->launchHostKernel(s->impl, params, kernel, user_payload);
  if (st == TSR_STATUS_UNIMPLEMENTED) {
    SetLastError(
      "launchHostTileKernel: this backend cannot honor the host tile "
      "kernel ABI; route to the CPU device for host tile kernels");
  }
  return st;
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

TsrStatus tsrNativeGemmF32(tsrDevice dev,
                           const float* a,
                           const float* b,
                           float* c,
                           int32_t m,
                           int32_t n,
                           int32_t k) {
  if (!dev || !a || !b || !c) {
    SetLastError("dev/a/b/c==NULL");
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  if (m < 0 || n < 0 || k < 0) {
    SetLastError("gemm dimensions must be non-negative");
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  if (!dev->be->gemmF32(a, b, c, m, n, k)) {
    SetLastError("native f32 GEMM is not available on this backend");
    return TSR_STATUS_UNIMPLEMENTED;
  }
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrGetWorkerThreadCount(tsrDevice dev, uint32_t* out) {
  if (!dev || !out) {
    SetLastError("dev/out==NULL");
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  *out = dev->be->workerThreadCount();
  return TSR_STATUS_SUCCESS;
}

} // extern "C"
