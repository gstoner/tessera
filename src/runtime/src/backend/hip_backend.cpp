// hip_backend.cpp — Tessera HIP/ROCm backend (Phase 6)
//
// Compiled only when -DTESSERA_ENABLE_HIP is set.
// Exposes CreateHipBackend(); returns nullptr when not enabled.
//
// The structure mirrors cuda_backend.cpp.  HIP provides a CUDA-compatible
// API so the mapping is almost 1-to-1:
//
//   cudaMalloc        → hipMalloc
//   cudaMemcpy        → hipMemcpy
//   cudaStream_t      → hipStream_t
//   cudaEvent_t       → hipEvent_t
//   cudaLaunchHostFunc → hipLaunchHostFunc  (HIP 5.6+)
//
// Buffer layout:
//   Buffer::ptr   — device pointer (hipMalloc result)
//   Buffer::bytes — allocated size

#include "base_backend.h"

#ifdef TESSERA_ENABLE_HIP

#include <hip/hip_runtime.h>
#include <cstring>
#include <chrono>
#include <string>
#include <atomic>

namespace tsr {

// ---------------------------------------------------------------------------
// Error handling helpers
// ---------------------------------------------------------------------------

namespace {

thread_local hipError_t g_last_hip_err = hipSuccess;

inline void set_hip_error(hipError_t e) noexcept { g_last_hip_err = e; }

#define TSR_HIP_CHECK(expr)                         \
  do {                                              \
    hipError_t _e = (expr);                         \
    if (_e != hipSuccess) {                         \
      set_hip_error(_e);                            \
      return;                                       \
    }                                               \
  } while (0)

#define TSR_HIP_CHECK_RET(expr, retval)             \
  do {                                              \
    hipError_t _e = (expr);                         \
    if (_e != hipSuccess) {                         \
      set_hip_error(_e);                            \
      return (retval);                              \
    }                                               \
  } while (0)

} // anonymous namespace

// ---------------------------------------------------------------------------
// Extended wrappers
// ---------------------------------------------------------------------------

struct HipBuffer : Buffer {};

struct HipStream : Stream {
  hipStream_t hip_stream{nullptr};
  HipStream() : Stream(nullptr) {}
};

struct HipEvent : Event {
  hipEvent_t hip_event{nullptr};
};

// ---------------------------------------------------------------------------
// HipBackend
// ---------------------------------------------------------------------------

class HipBackend final : public Backend {
 public:
  // ------------------------------------------------------------------ props
  DeviceProps props() const override {
    hipDeviceProp_t dp{};
    int dev = 0;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dp, dev);

    return DeviceProps{
        TSR_DEVICE_HIP,
        std::string(dp.name),
        static_cast<uint32_t>(dp.maxThreadsPerBlock),
        static_cast<uint32_t>(dp.multiProcessorCount),
    };
  }

  // ------------------------------------------------------------------ memory
  Buffer* malloc(size_t bytes) override {
    void* ptr = nullptr;
    hipError_t e = hipMalloc(&ptr, bytes);
    if (e != hipSuccess) {
      set_hip_error(e);
      return nullptr;
    }
    auto* buf = new HipBuffer();
    buf->ptr = ptr;
    buf->bytes = bytes;
    return buf;
  }

  void free(Buffer* b) override {
    if (!b) return;
    TSR_HIP_CHECK(hipFree(b->ptr));
    delete static_cast<HipBuffer*>(b);
  }

  void memset(Buffer* b, int value, size_t bytes) override {
    if (!b || !b->ptr) return;
    size_t n = (bytes == 0 || bytes > b->bytes) ? b->bytes : bytes;
    TSR_HIP_CHECK(hipMemset(b->ptr, value, n));
  }

  void memcpy(Buffer* dst, const Buffer* src,
              size_t bytes, TsrMemcpyKind kind) override {
    if (!dst || !src) return;
    hipMemcpyKind hk;
    switch (kind) {
      case TSR_MEMCPY_HOST_TO_DEVICE:   hk = hipMemcpyHostToDevice;   break;
      case TSR_MEMCPY_DEVICE_TO_HOST:   hk = hipMemcpyDeviceToHost;   break;
      case TSR_MEMCPY_DEVICE_TO_DEVICE: hk = hipMemcpyDeviceToDevice; break;
      case TSR_MEMCPY_HOST_TO_HOST:     hk = hipMemcpyHostToHost;     break;
      default:                          hk = hipMemcpyDefault;         break;
    }
    TSR_HIP_CHECK(hipMemcpy(dst->ptr, src->ptr, bytes, hk));
  }

  void* map(Buffer* b) override {
    // HIP device memory is not host-mappable via hipMalloc.
    // Return the device pointer so callers can inspect the address.
    return b ? b->ptr : nullptr;
  }

  void unmap(Buffer* /*b*/) override {}

  // ------------------------------------------------------------------ streams
  Stream* createStream() override {
    auto* s = new HipStream();
    hipError_t e = hipStreamCreateWithFlags(&s->hip_stream,
                                            hipStreamNonBlocking);
    if (e != hipSuccess) {
      set_hip_error(e);
      delete s;
      return nullptr;
    }
    return s;
  }

  void destroyStream(Stream* s) override {
    if (!s) return;
    auto* hs = static_cast<HipStream*>(s);
    if (hs->hip_stream) {
      hipStreamDestroy(hs->hip_stream);
      hs->hip_stream = nullptr;
    }
    delete hs;
  }

  void streamSync(Stream* s) override {
    if (!s) return;
    auto* hs = static_cast<HipStream*>(s);
    TSR_HIP_CHECK(hipStreamSynchronize(hs->hip_stream));
  }

  // ------------------------------------------------------------------ events
  Event* createEvent() override {
    auto* ev = new HipEvent();
    // hipEventBlockingSync ensures eventSync blocks the CPU thread.
    hipError_t e = hipEventCreateWithFlags(&ev->hip_event,
                                           hipEventBlockingSync);
    if (e != hipSuccess) {
      set_hip_error(e);
      delete ev;
      return nullptr;
    }
    return ev;
  }

  void destroyEvent(Event* e) override {
    if (!e) return;
    auto* he = static_cast<HipEvent*>(e);
    if (he->hip_event) {
      hipEventDestroy(he->hip_event);
      he->hip_event = nullptr;
    }
    delete he;
  }

  void recordEvent(Event* e, Stream* s) override {
    if (!e || !s) return;
    auto* he = static_cast<HipEvent*>(e);
    auto* hs = static_cast<HipStream*>(s);
    TSR_HIP_CHECK(hipEventRecord(he->hip_event, hs->hip_stream));
    using namespace std::chrono;
    he->timestamp_ns = static_cast<uint64_t>(
        duration_cast<nanoseconds>(
            system_clock::now().time_since_epoch()).count());
  }

  void waitEvent(Event* e, Stream* s) override {
    if (!e || !s) return;
    auto* he = static_cast<HipEvent*>(e);
    auto* hs = static_cast<HipStream*>(s);
    TSR_HIP_CHECK(hipStreamWaitEvent(hs->hip_stream, he->hip_event, 0));
  }

  void eventSync(Event* e) override {
    if (!e) return;
    auto* he = static_cast<HipEvent*>(e);
    TSR_HIP_CHECK(hipEventSynchronize(he->hip_event));
    std::lock_guard<std::mutex> lk(he->mu);
    he->signaled = true;
  }

  // ------------------------------------------------------------------ kernel
  void launchHostKernel(Stream* s,
                        const tsrLaunchParams* params,
                        tsrHostKernelFn kernel,
                        void* user_payload) override {
    if (!kernel) return;
    if (s) {
      auto* hs = static_cast<HipStream*>(s);
      struct Ctx { tsrHostKernelFn fn; const tsrLaunchParams* p; void* payload; };
      auto* ctx = new Ctx{kernel, params, user_payload};
      // hipLaunchHostFunc is available in ROCm 5.6+.
      hipError_t e = hipLaunchHostFunc(
          hs->hip_stream,
          [](void* arg) {
            auto* c = reinterpret_cast<Ctx*>(arg);
            c->fn(c->p, c->payload);
            delete c;
          },
          ctx);
      if (e == hipSuccess) return;
      delete ctx;
    }
    kernel(params, user_payload);
  }
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<Backend> CreateHipBackend() {
  int count = 0;
  hipError_t e = hipGetDeviceCount(&count);
  if (e != hipSuccess || count == 0) return nullptr;
  return std::make_unique<HipBackend>();
}

} // namespace tsr

#else  // TESSERA_ENABLE_HIP not defined

namespace tsr {
std::unique_ptr<Backend> CreateHipBackend() { return nullptr; }
} // namespace tsr

#endif // TESSERA_ENABLE_HIP
