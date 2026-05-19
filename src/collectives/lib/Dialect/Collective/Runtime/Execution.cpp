#include "tessera/Dialect/Collective/Runtime/Execution.h"
#include <memory>
#include <mutex>

namespace tessera { namespace collective {

// The collective runtime is a process-singleton owned by an internal
// ``shared_ptr``.  The mutex protects only the *slot* — i.e., the
// pointer write — and is **not held across** any call to
// ``ExecRuntime::submit()``.
//
// Safety preconditions (every component below is thread-safe today):
//
//   * ``TokenLimiter`` — own ``std::mutex`` + ``condition_variable``.
//   * ``PerfettoTraceWriter`` — own ``std::mutex``; ``write()`` takes
//     an event-vector snapshot under the lock before flushing.
//   * ``NCCLAdapter`` / ``RCCLAdapter`` slots — written only via
//     ``setNCCL`` / ``setRCCL`` at runtime construction time; the
//     read inside ``submit()`` is benign.
//
// Why dropping the global lock from ``submit()`` matters:
//
//   * **Re-entrancy.**  NCCL/RCCL completion callbacks may call
//     back into a ``tessera_*`` entry point on the same thread; if
//     we held the global mutex through ``submit()`` we'd
//     self-deadlock the moment any adapter ever did so.
//   * **Submit latency.**  Holding a process-global mutex through
//     every submit serializes every collective across every stream
//     and device.  Multi-stream callers couldn't overlap submits
//     with each other.
//   * **Shutdown safety.**  By holding ``shared_ptr<ExecRuntime>``
//     we keep an in-flight submitter's runtime alive even if
//     another thread calls ``tessera_shutdown_runtime()``
//     mid-submit — the last reference drops cleanly when the
//     in-flight call returns.
static std::mutex& _rtMutex() {
  static std::mutex m;
  return m;
}
static std::shared_ptr<ExecRuntime>& _rtSlot() {
  static std::shared_ptr<ExecRuntime> rt;
  return rt;
}

/// Ensure a runtime exists and return a strong reference.
/// Caller must hold ``_rtMutex()``.
static std::shared_ptr<ExecRuntime> _ensureRuntimeLocked(int tokens) {
  auto& slot = _rtSlot();
  if (!slot) {
    slot = std::make_shared<ExecRuntime>(tokens, Policy::fromEnv(),
                                          /*pidBase*/1000);
  }
  return slot;
}

/// Convenience: take the global mutex, ensure the runtime, return a
/// strong reference.  The mutex is released on return so the caller
/// can invoke ``submit()`` (or other long-running methods) without
/// serializing on the global lock.
static std::shared_ptr<ExecRuntime> _grabRuntime(int tokens) {
  std::lock_guard<std::mutex> lk(_rtMutex());
  return _ensureRuntimeLocked(tokens);
}

extern "C" void tessera_qos_limit_set(int tokens) {
  auto rt = _grabRuntime(tokens);
  // Mutex released — ``TokenLimiter::set`` takes its own internal lock.
  rt->setMaxInflight(tokens);
}
extern "C" void tessera_qos_acquire() {
  // Acquire happens implicitly on submit in this model; we just need
  // to make sure the runtime exists.  Don't hold the mutex past the
  // creation point.
  (void)_grabRuntime(/*tokens*/1);
}
extern "C" void tessera_qos_release() {
  // release is handled in submit callback; keep for symmetry.
}
extern "C" void tessera_submit_chunk_async(const void* ptr, uint64_t bytes, int device, int stream) {
  auto rt = _grabRuntime(/*tokens*/1);
  // Mutex released here.  The local ``shared_ptr`` keeps the
  // runtime alive even if another thread calls
  // ``tessera_shutdown_runtime()`` while we're inside ``submit()``.
  ChunkDesc d{ptr, bytes, device, stream, /*intraNode*/true};
  rt->submit(d);
}
extern "C" void tessera_trace_write(const char* path) {
  // Snapshot the slot under the lock; do the (potentially heavy)
  // ``trace().write(...)`` outside the critical section.
  // ``PerfettoTraceWriter::write`` internally snapshots its own
  // event vector under its own lock.
  std::shared_ptr<ExecRuntime> rt;
  {
    std::lock_guard<std::mutex> lk(_rtMutex());
    rt = _rtSlot();
  }
  if (rt) {
    rt->trace().write(path);
  }
}

// Explicit teardown for embedded/notebook lifecycles.  Safe to call
// even if the runtime was never initialized (no-op).  After this
// returns, subsequent ``tessera_*`` calls will re-initialize the
// runtime on demand.
//
// Lifetime contract: it is safe to call ``tessera_shutdown_runtime()``
// while another thread is inside ``tessera_submit_chunk_async()``.
// The in-flight submitter holds a ``shared_ptr`` strong reference, so
// the runtime is only destroyed when the last submit returns.  The
// caller is, however, responsible for not relying on any specific
// runtime *identity* after shutdown (a subsequent submit may
// re-initialize a fresh runtime with a different ``TokenLimiter``).
extern "C" void tessera_shutdown_runtime() {
  std::lock_guard<std::mutex> lk(_rtMutex());
  _rtSlot().reset();
}

}} // ns
