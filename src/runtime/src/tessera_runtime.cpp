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

// G5/G6 — Artifact / Kernel lifecycle.
//
// G5 added CPU end-to-end (compile -> load -> getkernel -> launch via the host
// kernel ABI). G6 adds **target tagging**: an artifact carries (target,
// compiler_path, execution_kind) and a kernel handle is a tagged union over
// {CPU host fn, GPU artifact (no native ABI bridge yet)}. tsrLaunchKernel
// routes by kind and returns precise UNIMPLEMENTED for non-bridged backends —
// so a CPU passing test cannot mask a GPU gap. The payload format gained a v2:
//
//   v1 (legacy, CPU-only): "TSRART1\n<n>\n<name>\t<fn_ptr_hex>\n..."
//   v2 (G6, target-tagged): "TSRART2\ntarget\t<t>\ncompiler_path\t<c>\n"
//                           "execution_kind\t<e>\n<n>\n<name>\t<fn_ptr_hex>\n..."
//
// Both round-trip cleanly through tsrCompileArtifact -> tsrLoadArtifact; v1 is
// still accepted on load (treated as target=cpu) so older artifacts keep working.
struct tsrArtifact_t {
  std::string payload;                                  // canonical text form
  std::string target;          // e.g. "cpu" / "apple_gpu" — set by Compile
  std::string compiler_path;   // e.g. "native_cpu" / "apple_gpu_mps"
  std::string execution_kind;  // "native_cpu" / "native_gpu" / "artifact_only"
  std::unordered_map<std::string, tsrHostKernelFn> kernels;   // populated when target=cpu
};

// G6 — tagged-union kernel handle.
enum class tsrKernelKind {
  kHostCpu,         // CPU host kernel (G5 path) — fn is the callable.
  kGpuUnbridged,    // GPU artifact with no native ABI launch bridge yet.
};
struct tsrKernel_t {
  std::string name;
  tsrKernelKind kind;
  tsrHostKernelFn fn;             // valid iff kind == kHostCpu
  tsrArtifact_t* artifact;        // back-pointer (does not own)
};

namespace {
// Process-wide registry of host kernels available for artifact bundling.
std::mutex g_host_kernel_mu;
std::unordered_map<std::string, tsrHostKernelFn> g_host_kernels;

constexpr const char *kArtifactMagicV1 = "TSRART1";
constexpr const char *kArtifactMagicV2 = "TSRART2";

std::string serializeArtifact(
    const std::string& target, const std::string& compiler_path,
    const std::string& execution_kind,
    const std::unordered_map<std::string, tsrHostKernelFn>& kernels) {
  std::ostringstream out;
  out << kArtifactMagicV2 << '\n';
  out << "target\t" << target << '\n';
  out << "compiler_path\t" << compiler_path << '\n';
  out << "execution_kind\t" << execution_kind << '\n';
  out << kernels.size() << '\n';
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

// Parse v1 (CPU-only legacy) or v2 (target-tagged). On v1, target defaults to
// "cpu" / "native_cpu" / "native_cpu" so the artifact behaves as before.
bool parseArtifact(const std::string& payload, tsrArtifact_t& out) {
  std::istringstream in(payload);
  std::string magic; std::getline(in, magic);
  if (magic == kArtifactMagicV1) {
    out.target = "cpu"; out.compiler_path = "native_cpu"; out.execution_kind = "native_cpu";
  } else if (magic == kArtifactMagicV2) {
    // Read 3 key/tab/value header lines.
    auto read_kv = [&](const char *key, std::string &dst) -> bool {
      std::string line; if (!std::getline(in, line)) return false;
      auto tab = line.find('\t');
      if (tab == std::string::npos) return false;
      if (line.substr(0, tab) != key) return false;
      dst = line.substr(tab + 1); return true;
    };
    if (!read_kv("target", out.target)) return false;
    if (!read_kv("compiler_path", out.compiler_path)) return false;
    if (!read_kv("execution_kind", out.execution_kind)) return false;
  } else {
    return false;
  }
  std::size_t n = 0; { std::string nstr; std::getline(in, nstr);
    try { n = std::stoull(nstr); } catch (...) { return false; } }
  out.kernels.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::string line; std::getline(in, line);
    auto tab = line.find('\t');
    if (tab == std::string::npos) return false;
    std::string name = line.substr(0, tab);
    std::uintptr_t addr = 0;
    try { addr = std::stoull(line.substr(tab + 1)); } catch (...) { return false; }
    out.kernels[std::move(name)] = reinterpret_cast<tsrHostKernelFn>(addr);
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
// G5/G6 — Artifact lifecycle.
//
// CPU lane (G5): `module_ir` is a comma-separated list of names previously
// registered via tsrRegisterHostKernel; the artifact bundles their function
// pointers, and tsrLaunchKernel routes them through tsrLaunchHostTileKernel.
//
// GPU lane (G6): `options->target` selects a non-CPU target (e.g. "apple_gpu").
// The artifact is created with the right target tag but, because no native
// (C-ABI-level) GPU launch bridge exists yet, getkernel returns a `kGpuUnbridged`
// handle and tsrLaunchKernel returns TSR_STATUS_UNIMPLEMENTED with a precise
// reason. This makes the C ABI honest about non-CPU lanes — a CPU passing test
// cannot mask the GPU gap; the GPU artifact contract is testable end-to-end
// (compile -> load -> getkernel -> honest unimplemented launch).
TsrStatus tsrCompileArtifact(const char* module_ir,
                             const tsrCompileOptions* options,
                             tsrArtifact* out) {
  if (!module_ir || !out) { SetLastError("module_ir/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  std::unique_ptr<tsrArtifact_t> a(new tsrArtifact_t);
  // G6 — read the target tag from options. Default "cpu" preserves the G5 path.
  a->target = (options && options->target) ? std::string(options->target) : "cpu";
  if (a->target == "cpu") {
    a->compiler_path = "native_cpu";
    a->execution_kind = "native_cpu";
  } else {
    // Non-CPU: still build the artifact (it serializes/round-trips), but the
    // payload's "kernels" are *placeholders* (name -> null fn) recording WHICH
    // kernels the artifact would launch. The native GPU launch bridge is
    // separate work; until then tsrLaunchKernel reports UNIMPLEMENTED.
    a->compiler_path = a->target + "_artifact";
    a->execution_kind = "artifact_only";
  }
  std::string spec(module_ir);
  // Split on commas; trim whitespace. Empty spec -> empty artifact (valid).
  std::lock_guard<std::mutex> lk(g_host_kernel_mu);
  std::size_t i = 0;
  while (i <= spec.size()) {
    std::size_t j = spec.find(',', i);
    if (j == std::string::npos) j = spec.size();
    std::size_t a0 = i, b0 = j;
    while (a0 < b0 && std::isspace((unsigned char)spec[a0])) ++a0;
    while (b0 > a0 && std::isspace((unsigned char)spec[b0 - 1])) --b0;
    if (a0 < b0) {
      std::string name = spec.substr(a0, b0 - a0);
      if (a->target == "cpu") {
        auto it = g_host_kernels.find(name);
        if (it == g_host_kernels.end()) {
          std::string msg = "tsrCompileArtifact: kernel '" + name +
              "' is not registered (call tsrRegisterHostKernel first; non-CPU "
              "codegen JIT is a separate gap)";
          SetLastError(msg.c_str());
          return TSR_STATUS_UNIMPLEMENTED;
        }
        a->kernels[name] = it->second;
      } else {
        // GPU artifact: record the kernel name with a null fn-pointer
        // placeholder. getkernel still succeeds; launch returns UNIMPLEMENTED.
        a->kernels[name] = nullptr;
      }
    }
    if (j == spec.size()) break;
    i = j + 1;
  }
  a->payload = serializeArtifact(a->target, a->compiler_path,
                                 a->execution_kind, a->kernels);
  *out = a.release();
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrLoadArtifact(const void* bytes, size_t bytes_len, tsrArtifact* out) {
  if (!bytes || !out) { SetLastError("bytes/out==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  std::unique_ptr<tsrArtifact_t> a(new tsrArtifact_t);
  a->payload.assign(static_cast<const char*>(bytes), bytes_len);
  if (!parseArtifact(a->payload, *a)) {
    SetLastError("tsrLoadArtifact: payload is not a valid Tessera artifact "
                 "(expected TSRART1/TSRART2 magic + target + name/fn table)");
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
  if (it == artifact->kernels.end()) {
    std::string msg = "tsrGetKernel: artifact (target=" + artifact->target +
                      ") does not contain kernel '" + name + "'";
    SetLastError(msg.c_str());
    return TSR_STATUS_NOT_FOUND;
  }
  std::unique_ptr<tsrKernel_t> k(new tsrKernel_t);
  k->name = name;
  k->artifact = artifact;
  if (artifact->target == "cpu") {
    if (!it->second) {
      // CPU artifact with a null fn-pointer is corrupt (shouldn't happen via
      // the public API but defensive guard).
      SetLastError("tsrGetKernel: CPU artifact has a null kernel function");
      return TSR_STATUS_INVALID_ARGUMENT;
    }
    k->kind = tsrKernelKind::kHostCpu;
    k->fn = it->second;
  } else {
    k->kind = tsrKernelKind::kGpuUnbridged;
    k->fn = nullptr;
  }
  *out = k.release();
  return TSR_STATUS_SUCCESS;
}

TsrStatus tsrDestroyKernel(tsrKernel kernel) {
  delete kernel;
  return TSR_STATUS_SUCCESS;
}

// G6.2 — public read-only view of the canonical payload. Returns the same
// bytes tsrLoadArtifact accepts; the pointer is owned by the artifact.
TsrStatus tsrGetArtifactBytes(tsrArtifact artifact, const void** out_bytes,
                              size_t* out_len) {
  if (!artifact || !out_bytes || !out_len) {
    SetLastError("artifact/out_bytes/out_len==NULL");
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  *out_bytes = artifact->payload.data();
  *out_len = artifact->payload.size();
  return TSR_STATUS_SUCCESS;
}

// G6.2 — public read-only target tag.
TsrStatus tsrGetArtifactTarget(tsrArtifact artifact, const char** out) {
  if (!artifact || !out) {
    SetLastError("artifact/out==NULL");
    return TSR_STATUS_INVALID_ARGUMENT;
  }
  *out = artifact->target.c_str();
  return TSR_STATUS_SUCCESS;
}

// G5/G6 — Launch a kernel obtained via tsrGetKernel. Dispatch by kernel kind:
//   kHostCpu      : route through tsrLaunchHostTileKernel.
//                   args[0] = const tsrLaunchParams* (required);
//                   args[1] = void* user_payload (optional); nargs >= 1.
//   kGpuUnbridged : honest TSR_STATUS_UNIMPLEMENTED with a precise reason.
//                   The Python runtime still executes Apple GPU artifacts via
//                   its own dispatcher; until a native ABI-level launch bridge
//                   exists, the C ABI must not silently succeed here.
TsrStatus tsrLaunchKernel(tsrStream s, tsrKernel kernel, void** args, size_t nargs) {
  if (!s || !kernel) { SetLastError("s/kernel==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
  if (kernel->kind == tsrKernelKind::kGpuUnbridged) {
    std::string msg = "tsrLaunchKernel: no native C-ABI launch bridge for ";
    msg += "target='" + kernel->artifact->target +
           "' kernel='" + kernel->name + "'. The Python runtime executes this "
           "artifact via execution_matrix dispatch; the C ABI launch bridge is "
           "a separate gap.";
    SetLastError(msg.c_str());
    return TSR_STATUS_UNIMPLEMENTED;
  }
  if (!kernel->fn) { SetLastError("tsrLaunchKernel: kernel.fn==NULL"); return TSR_STATUS_INVALID_ARGUMENT; }
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
