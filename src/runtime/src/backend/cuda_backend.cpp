// cuda_backend.cpp — Tessera CUDA backend (Phase 6)
//
// Compiled only when -DTESSERA_ENABLE_CUDA is set.
// Exposes CreateCudaBackend(); returns nullptr when not enabled.
//
// Each public method follows the pattern:
//   1. Call the CUDA runtime API.
//   2. Check the return code with TSR_CUDA_CHECK.
//   3. On failure, store the error in thread-local state; do not throw.
//
// Buffer layout:
//   Buffer::ptr  — device pointer (cudaMalloc result)
//   Buffer::bytes — allocated size
//
// Stream layout:
//   Stream::pool — unused for CUDA (nullptr); cudaStream_t is stored
//   alongside the base struct via CudaStream wrapper.

#include "base_backend.h"

#ifdef TESSERA_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <string>
#include <atomic>

namespace tsr {

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------

namespace {

thread_local cudaError_t g_last_cuda_err = cudaSuccess;

inline void set_cuda_error(cudaError_t e) noexcept { g_last_cuda_err = e; }

#define TSR_CUDA_CHECK(expr)                        \
  do {                                              \
    cudaError_t _e = (expr);                        \
    if (_e != cudaSuccess) {                        \
      set_cuda_error(_e);                           \
      return;                                       \
    }                                               \
  } while (0)

#define TSR_CUDA_CHECK_RET(expr, retval)            \
  do {                                              \
    cudaError_t _e = (expr);                        \
    if (_e != cudaSuccess) {                        \
      set_cuda_error(_e);                           \
      return (retval);                              \
    }                                               \
  } while (0)

} // anonymous namespace

// ---------------------------------------------------------------------------
// Extended buffer / stream / event wrappers
// ---------------------------------------------------------------------------

struct CudaBuffer : Buffer {
  // Buffer::ptr holds the device pointer; Buffer::bytes holds the size.
};

struct CudaStream : Stream {
  cudaStream_t cu_stream{nullptr};
  explicit CudaStream() : Stream(nullptr) {}
};

struct CudaEvent : Event {
  cudaEvent_t cu_event{nullptr};
};

// ---------------------------------------------------------------------------
// CudaBackend
// ---------------------------------------------------------------------------

class CudaBackend final : public Backend {
 public:
  // ------------------------------------------------------------------ props
  DeviceProps props() const override {
    cudaDeviceProp dp{};
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e == cudaSuccess) e = cudaGetDeviceProperties(&dp, dev);
    if (e != cudaSuccess) {
      // Don't return zero-initialized bogus props silently — surface the error.
      set_cuda_error(e);
      return DeviceProps{TSR_DEVICE_CUDA, "<unknown>", 0, 0};
    }
    return DeviceProps{
        TSR_DEVICE_CUDA,
        std::string(dp.name),
        static_cast<uint32_t>(dp.maxThreadsPerBlock),
        static_cast<uint32_t>(dp.multiProcessorCount),
    };
  }

  // ------------------------------------------------------------------ memory
  Buffer* malloc(size_t bytes) override {
    void* ptr = nullptr;
    cudaError_t e = cudaMalloc(&ptr, bytes);
    if (e != cudaSuccess) {
      set_cuda_error(e);
      return nullptr;
    }
    auto* buf = new CudaBuffer();
    buf->ptr = ptr;
    buf->bytes = bytes;
    return buf;
  }

  void free(Buffer* b) override {
    if (!b) return;
    // Always delete the host wrapper, even if cudaFree fails — returning early
    // (as TSR_CUDA_CHECK does) would leak the wrapper while the live-handle
    // ratchet still decrements.
    cudaError_t e = cudaFree(b->ptr);
    if (e != cudaSuccess) set_cuda_error(e);
    delete static_cast<CudaBuffer*>(b);
  }

  void memset(Buffer* b, int value, size_t bytes) override {
    if (!b || !b->ptr) return;
    size_t n = (bytes == 0 || bytes > b->bytes) ? b->bytes : bytes;
    TSR_CUDA_CHECK(cudaMemset(b->ptr, value, n));
  }

  void memcpy(Buffer* dst, const Buffer* src,
              size_t bytes, TsrMemcpyKind kind) override {
    if (!dst || !src) return;
    // Bound the copy against both buffer sizes (the CPU backend clamps; the
    // GPU path must not pass an oversized length straight to cudaMemcpy →
    // device buffer overflow).
    if (bytes > dst->bytes || bytes > src->bytes) {
      set_cuda_error(cudaErrorInvalidValue);
      return;
    }
    cudaMemcpyKind ck;
    switch (kind) {
      case TSR_MEMCPY_HOST_TO_DEVICE: ck = cudaMemcpyHostToDevice;   break;
      case TSR_MEMCPY_DEVICE_TO_HOST: ck = cudaMemcpyDeviceToHost;   break;
      case TSR_MEMCPY_DEVICE_TO_DEVICE: ck = cudaMemcpyDeviceToDevice; break;
      case TSR_MEMCPY_HOST_TO_HOST:   ck = cudaMemcpyHostToHost;     break;
      default:                        ck = cudaMemcpyDefault;         break;
    }
    TSR_CUDA_CHECK(cudaMemcpy(dst->ptr, src->ptr, bytes, ck));
  }

  void* map(Buffer* b) override {
    // Device buffers are not host-mappable via standard CUDA.
    // Callers that need host access should use pinned memory or cudaMemcpy.
    // Return the device pointer so the caller can at least inspect the address.
    return b ? b->ptr : nullptr;
  }

  void unmap(Buffer* /*b*/) override {
    // No-op: map() does not create a host mapping.
  }

  // ------------------------------------------------------------------ streams
  Stream* createStream() override {
    auto* s = new CudaStream();
    cudaError_t e = cudaStreamCreateWithFlags(&s->cu_stream,
                                              cudaStreamNonBlocking);
    if (e != cudaSuccess) {
      set_cuda_error(e);
      delete s;
      return nullptr;
    }
    return s;
  }

  void destroyStream(Stream* s) override {
    if (!s) return;
    auto* cs = static_cast<CudaStream*>(s);
    if (cs->cu_stream) {
      cudaError_t e = cudaStreamDestroy(cs->cu_stream);
      if (e != cudaSuccess) set_cuda_error(e);
      cs->cu_stream = nullptr;
    }
    delete cs;
  }

  void streamSync(Stream* s) override {
    if (!s) return;
    auto* cs = static_cast<CudaStream*>(s);
    TSR_CUDA_CHECK(cudaStreamSynchronize(cs->cu_stream));
  }

  // ------------------------------------------------------------------ events
  Event* createEvent() override {
    auto* ev = new CudaEvent();
    cudaError_t e = cudaEventCreateWithFlags(&ev->cu_event,
                                             cudaEventBlockingSync |
                                             cudaEventDisableTiming);
    if (e != cudaSuccess) {
      // Try again with timing enabled (needed for event_get_timestamp)
      e = cudaEventCreate(&ev->cu_event);
    }
    if (e != cudaSuccess) {
      set_cuda_error(e);
      delete ev;
      return nullptr;
    }
    return ev;
  }

  void destroyEvent(Event* e) override {
    if (!e) return;
    auto* ce = static_cast<CudaEvent*>(e);
    if (ce->cu_event) {
      cudaError_t err = cudaEventDestroy(ce->cu_event);
      if (err != cudaSuccess) set_cuda_error(err);
      ce->cu_event = nullptr;
    }
    delete ce;
  }

  void recordEvent(Event* e, Stream* s) override {
    if (!e || !s) return;
    auto* ce = static_cast<CudaEvent*>(e);
    auto* cs = static_cast<CudaStream*>(s);
    TSR_CUDA_CHECK(cudaEventRecord(ce->cu_event, cs->cu_stream));
    // Mirror into the base Event for host-side queries
    using namespace std::chrono;
    ce->timestamp_ns = static_cast<uint64_t>(
        duration_cast<nanoseconds>(
            system_clock::now().time_since_epoch()).count());
  }

  void waitEvent(Event* e, Stream* s) override {
    if (!e || !s) return;
    auto* ce = static_cast<CudaEvent*>(e);
    auto* cs = static_cast<CudaStream*>(s);
    TSR_CUDA_CHECK(cudaStreamWaitEvent(cs->cu_stream, ce->cu_event, 0));
  }

  void eventSync(Event* e) override {
    if (!e) return;
    auto* ce = static_cast<CudaEvent*>(e);
    TSR_CUDA_CHECK(cudaEventSynchronize(ce->cu_event));
    std::lock_guard<std::mutex> lk(ce->mu);
    ce->signaled = true;
  }

  // ------------------------------------------------------------------ kernel
  TsrStatus launchHostKernel(Stream* /*s*/,
                             const tsrLaunchParams* /*params*/,
                             tsrHostKernelFn /*kernel*/,
                             void* /*user_payload*/) override {
    // Host tile kernels follow the
    //   ``fn(tsrKernelCtx*, const tsrTileCoord*, const tsrThreadCoord*)``
    // contract from ``tsr_kernel.h``.  The CPU backend implements the
    // full nested grid×tile iteration to honor that ABI; the CUDA
    // stream model has no equivalent (a ``cudaLaunchHostFunc``
    // callback only receives a single ``void*`` payload, not the
    // per-thread coords).  Returning ``TSR_STATUS_UNIMPLEMENTED``
    // here lets callers route to the CPU device explicitly and
    // avoids the pre-2026-05-19 bug where the launch site silently
    // invoked ``fn(params, payload)`` with the wrong arg count.
    return TSR_STATUS_UNIMPLEMENTED;
  }

  TsrStatus consumeLastError(std::string* msg) override {
    cudaError_t e = g_last_cuda_err;
    if (e == cudaSuccess) return TSR_STATUS_SUCCESS;
    g_last_cuda_err = cudaSuccess;   // consume — caller now owns it
    if (msg) {
      *msg = "cuda: ";
      *msg += cudaGetErrorString(e);
    }
    return TSR_STATUS_DEVICE_ERROR;
  }
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<Backend> CreateCudaBackend() {
  // Verify at least one CUDA device is present before constructing.
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess || count == 0) return nullptr;
  return std::make_unique<CudaBackend>();
}

} // namespace tsr

#else  // TESSERA_ENABLE_CUDA not defined

namespace tsr {
std::unique_ptr<Backend> CreateCudaBackend() { return nullptr; }
} // namespace tsr

#endif // TESSERA_ENABLE_CUDA
