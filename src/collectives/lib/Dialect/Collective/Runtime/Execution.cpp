#include "tessera/Dialect/Collective/Runtime/Execution.h"
#include <memory>
#include <mutex>

namespace tessera { namespace collective {

// Function-local storage with explicit synchronization.  The previous
// `static ExecRuntime*& _globalRt()` raw-pointer global was:
//   * unsynchronized — two threads hitting any `tessera_*` entry point
//     before initialization could each see ``nullptr`` and each
//     allocate a new ``ExecRuntime``, leaking one;
//   * never released — even orderly shutdown left the runtime in
//     place because nothing ever ``delete``s it.
//
// We now hold a ``unique_ptr`` under a mutex.  A `tessera_shutdown_runtime`
// teardown hook is provided for embedded/notebook loads/unloads.
static std::mutex& _rtMutex() {
  static std::mutex m;
  return m;
}
static std::unique_ptr<ExecRuntime>& _globalRt() {
  static std::unique_ptr<ExecRuntime> rt;
  return rt;
}

static ExecRuntime& _ensureRuntime(int tokens) {
  // Caller holds `_rtMutex()`.
  auto& slot = _globalRt();
  if (!slot) {
    slot = std::make_unique<ExecRuntime>(tokens, Policy::fromEnv(),
                                          /*pidBase*/1000);
  }
  return *slot;
}

extern "C" void tessera_qos_limit_set(int tokens) {
  std::lock_guard<std::mutex> lk(_rtMutex());
  ExecRuntime& rt = _ensureRuntime(tokens);
  rt.setMaxInflight(tokens);
}
extern "C" void tessera_qos_acquire() {
  std::lock_guard<std::mutex> lk(_rtMutex());
  (void)_ensureRuntime(/*tokens*/1);
  // acquire happens implicitly on submit in this model; keep for symmetry.
}
extern "C" void tessera_qos_release() {
  // release is handled in submit callback; keep for symmetry.
}
extern "C" void tessera_submit_chunk_async(const void* ptr, uint64_t bytes, int device, int stream) {
  std::lock_guard<std::mutex> lk(_rtMutex());
  ExecRuntime& rt = _ensureRuntime(/*tokens*/1);
  ChunkDesc d{ptr, bytes, device, stream, /*intraNode*/true};
  rt.submit(d);
}
extern "C" void tessera_trace_write(const char* path) {
  std::lock_guard<std::mutex> lk(_rtMutex());
  if (auto& rt = _globalRt()) {
    rt->trace().write(path);
  }
}

// Explicit teardown for embedded/notebook lifecycles.  Safe to call
// even if the runtime was never initialized (no-op).  After this
// returns, subsequent ``tessera_*`` calls will re-initialize the
// runtime on demand.
extern "C" void tessera_shutdown_runtime() {
  std::lock_guard<std::mutex> lk(_rtMutex());
  _globalRt().reset();
}

}} // ns
