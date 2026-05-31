#include <vector>
#include <memory>
#include <mutex>
#include <cstring>
#include <string>
#include <sstream>
#include <unordered_map>
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

// G5 — Artifact / Kernel lifecycle for the CPU backend.
//
// An artifact carries a name->host-fn map and a serializable payload. The
// payload is the canonical text form ("TSRART1\n<n>\n<name>\t<fn_ptr_hex>\n..."),
// so it round-trips through tsrCompileArtifact -> tsrLoadArtifact bit-exactly.
// Host functions are registered process-wide via tsrRegisterHostKernel(name, fn);
// tsrCompileArtifact takes the *names* to bundle (passed in `module_ir` as a
// comma-separated list of registered names — the natural minimal "compile" for
// pre-registered CPU host kernels). A full MLIR-to-host-fn JIT is a separate
// gap; this lane is honest about being the lifecycle layer over host kernels.
struct tsrArtifact_t {
  std::string payload;                                // canonical text form
  std::unordered_map<std::string, tsrHostKernelFn> kernels;   // name -> fn
};
struct tsrKernel_t { std::string name; tsrHostKernelFn fn; tsrArtifact_t* artifact; };

namespace {
// Process-wide registry of host kernels available for artifact bundling.
std::mutex g_host_kernel_mu;
std::unordered_map<std::string, tsrHostKernelFn> g_host_kernels;

constexpr const char *kArtifactMagic = "TSRART1";

std::string serializeArtifact(
    const std::unordered_map<std::string, tsrHostKernelFn>& kernels) {
  std::ostringstream out;
  out << kArtifactMagic << '\n' << kernels.size() << '\n';
  // Deterministic order: sort names. Critical for bit-exact round-trip.
  std::vector<std::string> names;
  names.reserve(kernels.size());
  for (auto& kv : kernels) names.push_back(kv.first);
  std::sort(names.begin(), names.end());
  for (auto& n : names) {
    out << n << '\t' << reinterpret_cast<std::uintptr_t>(kernels.at(n)) << '\n';
  }
  return out.str();
}

bool parseArtifact(const std::string& payload,
                   std::unordered_map<std::string, tsrHostKernelFn>& out) {
  std::istringstream in(payload);
  std::string magic; std::getline(in, magic);
  if (magic != kArtifactMagic) return false;
  std::size_t n = 0; { std::string nstr; std::getline(in, nstr);
    try { n = std::stoull(nstr); } catch (...) { return false; } }
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::string line; std::getline(in, line);
    auto tab = line.find('\t');
    if (tab == std::string::npos) return false;
    std::string name = line.substr(0, tab);
    std::uintptr_t addr = 0;
    try { addr = std::stoull(line.substr(tab + 1)); } catch (...) { return false; }
    out[std::move(name)] = reinterpret_cast<tsrHostKernelFn>(addr);
  }
  return true;
}
}  // namespace

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
//
// The switch below covers every value of ``TsrStatus``; we deliberately
// omit a ``default:`` label so a future enum addition triggers
// ``-Wswitch`` instead of being silently bucketed into "UNKNOWN".
// The trailing ``return`` after the switch handles the out-of-range
// case where a C ABI caller passed an integer cast from outside the
// enum (still a legal TsrStatus value at the language level since
// ``TsrStatus`` is an int-shaped enum at the ABI boundary).
const char* tsrStatusString(TsrStatus s) {
  switch (s) {
    case TSR_STATUS_SUCCESS:          return "SUCCESS";
    case TSR_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case TSR_STATUS_NOT_FOUND:        return "NOT_FOUND";
    case TSR_STATUS_ALREADY_EXISTS:   return "ALREADY_EXISTS";
    case TSR_STATUS_OUT_OF_MEMORY:    return "OUT_OF_MEMORY";
    case TSR_STATUS_UNIMPLEMENTED:    return "UNIMPLEMENTED";
    case TSR_STATUS_INTERNAL:         return "INTERNAL";
    case TSR_STATUS_DEVICE_ERROR:     return "DEVICE_ERROR";
  }
  return "UNKNOWN";
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
  // Allocate the backend stream FIRST and validate before we take any
  // ownership.  If the backend reports failure (returns nullptr, e.g.,
  // CUDA/HIP path on `cudaStreamCreate` failure), we must NOT:
  //   * write a wrapper into ``*out`` — caller would think they own
  //     a handle, but ``tsrDestroyStream`` would dereference a null
  //     ``impl`` and the matching backend ``destroyStream(nullptr)``
  //     is not contracted to tolerate that;
  //   * increment ``g_live_streams`` — a failed create that bumped
  //     the counter would trap ``tsrShutdown`` forever (the ratchet
  //     refuses while any handle counter is non-zero).
  // Propagate the backend error via ``consumeLastError`` so callers
  // see *why* it failed.
  tsr::Stream* impl = dev->be->createStream();
  if (impl == nullptr) {
    std::string msg;
    TsrStatus backend_st = dev->be->consumeLastError(&msg);
    if (backend_st != TSR_STATUS_SUCCESS && !msg.empty()) {
      SetLastError(msg.c_str());
      return backend_st;
    }
    SetLastError("backend->createStream returned NULL");
    return TSR_STATUS_INTERNAL;
  }
  auto s = new tsrStream_t();
  s->impl = impl;
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
  // See ``tsrCreateStream`` for the rationale — validate the backend
  // event before taking ownership, so a CUDA/HIP create failure
  // doesn't smuggle a NULL ``impl`` handle out to the caller or
  // increment the live-event counter (which would trap tsrShutdown).
  tsr::Event* impl = dev->be->createEvent();
  if (impl == nullptr) {
    std::string msg;
    TsrStatus backend_st = dev->be->consumeLastError(&msg);
    if (backend_st != TSR_STATUS_SUCCESS && !msg.empty()) {
      SetLastError(msg.c_str());
      return backend_st;
    }
    SetLastError("backend->createEvent returned NULL");
    return TSR_STATUS_INTERNAL;
  }
  auto e = new tsrEvent_t();
  e->impl = impl;
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
// G5 — Artifact lifecycle for the CPU host-kernel ABI.
//
// `module_ir` is interpreted as a comma-separated list of names previously
// registered via tsrRegisterHostKernel. The bundled artifact captures those
// (name -> fn) entries and serializes a canonical text payload for round-trip
// through tsrLoadArtifact. Non-CPU codegen JIT remains a separate gap (we
// honestly return UNIMPLEMENTED for any name that isn't in the host registry).
// `options` is reserved for future flags (e.g. dtype/target); for the CPU lane
// the registry plus the request names are sufficient.
TsrStatus tsrCompileArtifact(const char* module_ir,
                             const tsrCompileOptions* options,
                             tsrArtifact* out) {
  (void)options;
  if (!module_ir || !out) { SetLastError("module_ir/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  std::unique_ptr<tsrArtifact_t> a(new tsrArtifact_t);
  std::string spec(module_ir);
  // Split on commas; trim whitespace. Empty spec -> empty artifact (valid).
  std::lock_guard<std::mutex> lk(g_host_kernel_mu);
  std::size_t i = 0;
  while (i <= spec.size()) {
    std::size_t j = spec.find(',', i);
    if (j == std::string::npos) j = spec.size();
    // trim
    std::size_t a0 = i, b0 = j;
    while (a0 < b0 && std::isspace((unsigned char)spec[a0])) ++a0;
    while (b0 > a0 && std::isspace((unsigned char)spec[b0 - 1])) --b0;
    if (a0 < b0) {
      std::string name = spec.substr(a0, b0 - a0);
      auto it = g_host_kernels.find(name);
      if (it == g_host_kernels.end()) {
        std::string msg = "tsrCompileArtifact: kernel '" + name +
            "' is not registered (call tsrRegisterHostKernel first; non-CPU "
            "codegen JIT is a separate gap)";
        SetLastError(msg.c_str());
        return TSR_STATUS_UNIMPLEMENTED;
      }
      a->kernels[name] = it->second;
    }
    if (j == spec.size()) break;
    i = j + 1;
  }
  a->payload = serializeArtifact(a->kernels);
  *out = a.release();
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrLoadArtifact(const void* bytes, size_t bytes_len, tsrArtifact* out) {
  if (!bytes || !out) { SetLastError("bytes/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  std::unique_ptr<tsrArtifact_t> a(new tsrArtifact_t);
  a->payload.assign(static_cast<const char*>(bytes), bytes_len);
  if (!parseArtifact(a->payload, a->kernels)) {
    SetLastError("tsrLoadArtifact: payload is not a valid Tessera artifact "
                 "(expected TSRART1 magic + name/fn table)");
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  *out = a.release();
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrDestroyArtifact(tsrArtifact artifact) {
  delete artifact;
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrGetKernel(tsrArtifact artifact, const char* name, tsrKernel* out) {
  if (!artifact || !name || !out) { SetLastError("artifact/name/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  auto it = artifact->kernels.find(name);
  if (it == artifact->kernels.end() || !it->second) {
    std::string msg = "tsrGetKernel: artifact does not contain kernel '";
    msg += name; msg += "'";
    SetLastError(msg.c_str());
    return TSR_STATUS_NOT_FOUND;
  }
  std::unique_ptr<tsrKernel_t> k(new tsrKernel_t{name, it->second, artifact});
  *out = k.release();
  return TSR_STATUS_SUCCESS;
}

// G5 — Launch a kernel obtained via tsrGetKernel. The `args` convention for the
// CPU host-kernel ABI: args[0] = const tsrLaunchParams* (non-null);
// args[1] = void* user_payload (may be null). nargs must be >= 1.
// The kernel itself is a tsrHostKernelFn that the CPU backend's
// launchHostKernel already knows how to execute.
TsrStatus tsrLaunchKernel(tsrStream s, tsrKernel kernel, void** args, size_t nargs) {
  if (!s || !kernel || !kernel->fn) { SetLastError("s/kernel==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  if (!args || nargs < 1) {
    SetLastError("tsrLaunchKernel(host): args[0] must be a tsrLaunchParams* "
                 "(args[1] optional user payload)");
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  const tsrLaunchParams* params = static_cast<const tsrLaunchParams*>(args[0]);
  void* user_payload = nargs >= 2 ? args[1] : nullptr;
  return tsrLaunchHostTileKernel(s, params, kernel->fn, user_payload);
}

// G5 — Register a host kernel under a name so tsrCompileArtifact can bundle it.
// Idempotent: re-registering the same name with the same fn is a no-op; a
// conflicting re-registration returns INVALID_ARGUMENT.
TsrStatus tsrRegisterHostKernel(const char* name, tsrHostKernelFn fn) {
  if (!name || !fn) { SetLastError("name/fn==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  std::lock_guard<std::mutex> lk(g_host_kernel_mu);
  auto it = g_host_kernels.find(name);
  if (it != g_host_kernels.end() && it->second != fn) {
    std::string msg = "tsrRegisterHostKernel: '"; msg += name;
    msg += "' is already registered with a different function";
    SetLastError(msg.c_str());
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  g_host_kernels[name] = fn;
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
