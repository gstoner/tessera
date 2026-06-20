#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include "../scheduler/tile_scheduler.h"
#include "../../include/tessera/tsr_types.h"
#include "../../include/tessera/tsr_status.h"
#include "../../include/tessera/tsr_kernel.h"

namespace tsr {

struct Buffer { void* ptr=nullptr; size_t bytes=0; };

struct Stream { explicit Stream(ThreadPool* p):pool(p){} ThreadPool* pool; };

struct Event {
  std::mutex mu;
  std::condition_variable cv;       // notified on recordEvent (CPU waitEvent)
  bool signaled=false;
  uint64_t timestamp_ns=0; // time of recordEvent (profiling)
};

struct DeviceProps {
  TsrDeviceKind kind;
  std::string name;
  uint32_t logical_tile_threads_max;
  uint32_t concurrent_tiles_hint;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual DeviceProps props() const = 0;

  virtual Buffer* malloc(size_t bytes) = 0;
  virtual void free(Buffer* b) = 0;
  virtual void memset(Buffer* b, int value, size_t bytes) = 0;
  virtual void memcpy(Buffer* dst, const Buffer* src, size_t bytes, TsrMemcpyKind kind) = 0;
  virtual void* map(Buffer* b) = 0;
  virtual void unmap(Buffer* b) = 0;

  virtual Stream* createStream() = 0;
  virtual void destroyStream(Stream* s) = 0;
  virtual void streamSync(Stream* s) = 0;

  virtual Event* createEvent() = 0;
  virtual void destroyEvent(Event* e) = 0;
  virtual void recordEvent(Event* e, Stream* s) = 0;
  virtual void waitEvent(Event* e, Stream* s) = 0;
  virtual void eventSync(Event* e) = 0;

  // Launch a host-portable tile kernel.  Returns TSR_STATUS_SUCCESS on
  // dispatch (the kernel may still execute asynchronously on the
  // backend's stream model), or a non-SUCCESS status when the backend
  // cannot honor the per-tile/per-thread contract (e.g., GPU backends
  // that have no equivalent host-side iteration semantics).  The CPU
  // backend implements the full nested grid×tile iteration; the
  // CUDA/HIP backends report TSR_STATUS_UNIMPLEMENTED so callers can
  // route to the CPU device explicitly.
  virtual TsrStatus launchHostKernel(Stream* s,
                                     const tsrLaunchParams* params,
                                     tsrHostKernelFn kernel,
                                     void* user_payload) = 0;

  virtual bool gemmF32(const float* a,
                       const float* b,
                       float* c,
                       int32_t m,
                       int32_t n,
                       int32_t k) {
    (void)a; (void)b; (void)c; (void)m; (void)n; (void)k;
    return false;
  }

  virtual uint32_t workerThreadCount() const { return 0; }

  // Consult-and-clear the backend's last device-level error.
  // Returns ``TSR_STATUS_SUCCESS`` when no error is pending; writes
  // the cleared error's human-readable message into ``*msg`` (when
  // non-null) for any other status.  Called by the C ABI after
  // ``void``-returning backend methods (``memcpy`` / ``memset`` /
  // ``free`` / ``streamSync`` / ``recordEvent`` / ``waitEvent`` /
  // ``eventSync``) so a failure inside the backend surfaces as a
  // real ``TsrStatus`` instead of being swallowed.  The default
  // implementation (CPU + reference backends) reports no error.
  virtual TsrStatus consumeLastError(std::string* /*msg*/) {
    return TSR_STATUS_SUCCESS;
  }
};

// Factories
std::unique_ptr<Backend> CreateCpuBackend();
std::unique_ptr<Backend> CreateCudaBackend(); // may return nullptr if not built
std::unique_ptr<Backend> CreateHipBackend();  // may return nullptr if not built

} // namespace tsr
