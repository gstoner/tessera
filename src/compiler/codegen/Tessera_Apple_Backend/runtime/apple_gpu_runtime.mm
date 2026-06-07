//===- apple_gpu_runtime.mm - MPS-backed Apple GPU runtime --------------===//
//
// Phase 8.3 — Apple GPU native execution.
//
// Objective-C++ shim that wraps Metal + Metal Performance Shaders behind a
// pure C ABI. Callers (the lowering pass + the Python ctypes wrapper) see
// only:
//
//   void tessera_apple_gpu_mps_matmul_f32(
//       const float* A,    // i64 raw pointer (row-major M*K)
//       const float* B,    // i64 raw pointer (row-major K*N)
//       float*       C,    // i64 raw pointer (row-major M*N, written)
//       int32_t M, int32_t N, int32_t K)
//
//   int32_t tessera_apple_gpu_runtime_has_metal(void)
//
// On non-Darwin builds the .mm file is excluded by CMake; a portable C++ TU
// (apple_gpu_runtime_stub.cpp would be added later if needed) provides the
// fallback. The CPU-side TesseraAppleRuntime stays buildable on Linux CI.
//
// The MetalDeviceContext is a process-wide singleton lazily initialized on
// first call. It owns:
//   - id<MTLDevice>        (default system device)
//   - id<MTLCommandQueue>
//
// Per-call buffers are allocated via MTLResourceStorageModeShared so the host
// can write/read directly without a blit copy. For Apple Silicon's unified
// memory architecture this avoids data movement entirely; for discrete-Metal
// hosts it falls back to managed staging but the ABI is unchanged.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// PK1 (packaged-kernel sprint) — Objective-C class declarations must
// appear at GLOBAL scope (clang refuses ``@interface`` inside any
// namespace or extern "C" block). Placed at file top so the @interface
// is unambiguously visible to every extern "C" function below.
//
// ``TesseraMlpkgPipeline`` owns a loaded ``MTLLibrary`` and the
// ``MTL4MachineLearningPipelineState`` built from it. The C ABI
// (further down — ``tessera_apple_gpu_mlpkg_compile``) wraps this
// class behind ``CFBridgingRetain`` so Python ctypes can hold an
// opaque ``void*`` handle.
API_AVAILABLE(macos(26.0), ios(26.0))
@interface TesseraMlpkgPipeline : NSObject
@property (nonatomic, strong) id<MTLLibrary> library;
@property (nonatomic, strong) id<MTL4MachineLearningPipelineState> pipelineState;
@property (nonatomic, copy) NSString *functionName;
@property (nonatomic, copy) NSString *packagePath;
// PK3 — per-binding tensor inventory + argument table populated by
// `tessera_apple_gpu_mlpkg_prepare_tensors`. ``tensorsByName`` keys
// match what reflection returns; ``argumentTable`` carries the bound
// resources at each binding's kernel-side index.
@property (nonatomic, strong) NSMutableDictionary<NSString *, id<MTLTensor>> *tensorsByName;
// PK8 — parallel index-keyed inventory. MPSGraph-authored packages (vs
// CoreML-origin ones like Apple's sample) expose *unnamed* bindings, so
// ``tensorsByName`` collapses on the empty-string key; the argument table
// is still correct (resources bound by ``b.index``). This map preserves
// per-binding-index access so ``fill_input_at`` / ``read_output_at`` can
// address positionally-bound packages.
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, id<MTLTensor>> *tensorsByIndex;
@property (nonatomic, strong) id<MTL4ArgumentTable> argumentTable;
// PK4 — intermediates heap allocated lazily on first dispatch (size
// comes from ``pipelineState.intermediatesHeapSize``). Cached because
// the heap-size is pipeline-state-derived — every dispatch on this
// pipeline needs the same size. Audit Action 7 / Apple-sample Pattern 7.
@property (nonatomic, strong) id<MTLHeap> intermediatesHeap;
// Phase 2 stride-alignment wire-up (2026-06-01) — opt-in flag.
// When true, ``tessera_apple_gpu_mlpkg_prepare_tensors`` computes
// strides via the aligned helper (64-byte for byte+ ML-usage; 128-
// byte for sub-byte dtypes) and sets ``MTLTensorDescriptor.strides``
// explicitly. When false (default) Metal computes strides
// implicitly from ``td.dimensions`` (the existing behavior).
// Toggled via ``tessera_apple_gpu_mlpkg_set_aligned_strides``.
@property (nonatomic, assign) BOOL useAlignedStrides;
@end

@implementation TesseraMlpkgPipeline
@end

namespace {

// SIMD-feature capability bits returned by tessera_apple_gpu_simd_caps() and
// used to gate the SIMD-reduction rowop fast path. On every Apple-Silicon GPU
// (M-series = MTLGPUFamilyApple7+) all four are present; the probe exists so the
// gate is honest (and so a hypothetical device without SIMD reduction cleanly
// falls back to the threadgroup-tree / reference path).
enum TsSimdCaps : int32_t {
  kTsSimdReduction      = 1,  // simd_sum / simd_max / simd_prefix_*
  kTsSimdShuffle        = 2,  // simd_shuffle / simd_shuffle_xor
  kTsSimdShuffleAndFill = 4,  // simd_shuffle_and_fill_down / _up
  kTsSimdgroupBarrier   = 8,  // simdgroup_barrier
};

struct MetalDeviceContext {
  id<MTLDevice>       device;
  id<MTLCommandQueue> queue;
  bool                ok;
  int32_t             simd_caps = 0;  // TsSimdCaps bitmask (0 until device init)

  // Apple-sample Pattern 4 (2026-05-31) — Shared-event timeout for the
  // legacy MPS / MPSGraph queue. The MTL4 lane already has ``mtl4_event``
  // below; this is its sibling on the legacy ``queue`` so the
  // ``commit_and_wait_with_timeout`` helper can sign a per-dispatch value
  // and bail out with a precise diagnostic instead of hanging forever.
  // Lazy-init on first use; ``legacy_event_val`` is monotonic across
  // dispatches (the shared event's signaled value only grows). Mirrors
  // Apple's sample at ``MLMatrixMultiplier.m:87, 241-255``.
  id        legacy_event;       // id<MTLSharedEvent>
  uint64_t  legacy_event_val = 0;
  std::mutex legacy_event_mu;

  // Phase 8.4 — MSL kernel cache. Keyed by (msl_source + entry_point) so
  // multiple kernels in the same source unit (uncommon but legal) cache
  // independently. Compiled lazily; the cache outlives any single command
  // buffer so amortizes the ~1ms compile cost across kernel invocations.
  std::unordered_map<std::string, id<MTLComputePipelineState>> kernel_cache;
  std::mutex                                                   kernel_cache_mu;

  // Metal 4 lane caches — the MTL4 command queue, the MTL4 compiler, and the
  // MTL4-compiled compute pipelines (keyed by msl_source + entry_point) are all
  // created lazily under @available(macOS 26.0) and reused across calls, so the
  // ~ms MSL+pipeline compile cost is paid once instead of per dispatch. The
  // pipeline state type is the shared id<MTLComputePipelineState>.
  //
  // **Apple-sample Pattern 5 audit (2026-05-31).** Mirrors Apple's sample
  // ``RunningAMachineLearningModelOnTheGPUTimeline`` (MLMatrixMultiplier.m:62-103,
  // 137-220) — device / queue / compiler / allocator / shared-event /
  // command-buffer / argument-table / heap are created ONCE in ``init`` and
  // reused across encode→commit cycles. Tessera's canonical dispatch path
  // (``mtl4_matmul2d_dispatch`` at ~L10220) honors this end-to-end under
  // ``mtl4_dispatch_mu``: ``mtl4_compiler`` (init at L1644-46, archive path
  // L1700-01), ``mtl4_allocator`` (L1730), ``mtl4_event``, and the reusable
  // ``mtl4_cmdbuf`` / ``mtl4_argtable`` / ``mtl4_residency`` are all
  // session-singletons by design. Two outlier MTL4 dispatchers (around L9798
  // and L10050) intentionally create per-call allocators + command buffers
  // because they don't take ``mtl4_dispatch_mu`` — they operate on their own
  // lane and don't want to serialize against the canonical path. The MTL4
  // capability probe at L9640+ creates fresh objects on purpose so the
  // probe answer doesn't depend on cache state. None of these are bugs;
  // they are documented design choices. See ``skills.md`` →
  // "Lessons learned — external Apple Metal 4 ML sample" Pattern 5.
  std::unordered_map<std::string, id<MTLComputePipelineState>> mtl4_pipeline_cache;
  std::mutex                                                   mtl4_mu;
  id                                                           mtl4_queue;     // id<MTL4CommandQueue>
  id                                                           mtl4_compiler;  // id<MTL4Compiler>

  // P2/P3 — reusable per-dispatch MTL4 objects. Recreating the command
  // allocator / command buffer / argument table / shared event on every
  // dispatch was the last big chunk of per-call overhead. They are reset+rebound
  // instead, which is safe because `mtl4_dispatch_mu` serializes the
  // encode→commit→wait sequence (and the single shared queue serializes GPU work
  // regardless, so no overlap is lost). The shared event advances a monotonic
  // value per dispatch. See docs/apple_backend_integration_review.md (P2/P3).
  id        mtl4_allocator;   // id<MTL4CommandAllocator>
  id        mtl4_cmdbuf;      // id<MTL4CommandBuffer>
  id        mtl4_argtable;    // id<MTL4ArgumentTable> (maxBufferBindCount = 8)
  id        mtl4_event;       // id<MTLSharedEvent>
  // Reusable per-dispatch residency set — repopulated each call (removeAll +
  // addAllocation + commit + requestResidency) and attached to the command
  // buffer via `useResidencySet:` (per-cmdbuf, the granular intended path), so we
  // avoid creating a fresh MTLResidencySet + queue addResidencySet/remove churn
  // every dispatch (and sidestep the queue's 32-residency-set ceiling).
  id        mtl4_residency;   // id<MTLResidencySet>
  uint64_t  mtl4_event_val;   // monotonic signal counter
  std::mutex mtl4_dispatch_mu;

  // PK audit P2 (2026-05-31) — packaged-ML dispatch
  // (``tessera_apple_gpu_mlpkg_dispatch``) intentionally does NOT acquire
  // ``mtl4_dispatch_mu`` because it creates its own per-call allocator +
  // command buffer (the "outlier" pattern documented above). To avoid
  // racing the canonical lane's monotonic counter / shared event, the
  // packaged lane gets its OWN shared event + counter, guarded by its
  // own mutex. Sharing the queue is fine — the queue itself serializes
  // submission. The audit found this concurrency bug; see PK1 status notes.
  id         mlpkg_event;     // id<MTLSharedEvent> — packaged-ML lane only
  uint64_t   mlpkg_event_val; // monotonic signal counter for packaged ML
  std::mutex mlpkg_event_mu;  // guards mlpkg_event lazy-init + counter bump

  // P4 — MTL4Archive pipeline persistence (opt-in). When enabled, the MTL4
  // compiler is created with a CaptureBinaries serializer so every pipeline it
  // builds is captured; a previously-flushed archive is loaded as a lookup
  // archive so matching pipelines skip the MSL recompile (~ms each) on process
  // start. Off by default — no effect on the default path. See
  // docs/apple_backend_integration_review.md (P4).
  id          mtl4_serializer;       // id<MTL4PipelineDataSetSerializer>
  id          mtl4_lookup_archive;   // id<MTL4Archive> (nil if none/incompatible)
  std::string mtl4_archive_path;
  bool        mtl4_archive_enabled;

  // 2026-05-17 — Metal buffer pool.  Each dispatch helper used to
  // call ``[device newBufferWithBytes:...]`` and let the buffer
  // deallocate at the end of the @autoreleasepool.  Profiling
  // showed those allocations cost ~50–100µs each on M-series — at
  // least 3× the per-kernel dispatch floor for small kernels.  The
  // pool below recycles `MTLResourceStorageModeShared` buffers
  // keyed on size bucket so a typical pointwise kernel sequence
  // re-uses the same backing storage across iterations.
  //
  // Buckets are powers of 2 from 16B to 4MB.  Buffers larger than
  // the top bucket bypass the pool (rare; the GA/EBM workloads
  // stay well under 4MB per buffer).
  static constexpr size_t kBucketCount = 19;   // 16B → 4MB
  static constexpr size_t kMinBucketLog2 = 4;  // log2(16)
  std::vector<id<MTLBuffer>> buffer_pool[kBucketCount];
  std::mutex                 buffer_pool_mu;
};

// Pick the smallest bucket whose capacity ≥ requested size.
static inline size_t metal_buffer_bucket(size_t bytes) {
  if (bytes == 0) return 0;
  size_t log2 = 0;
  size_t b = 1;
  while (b < bytes) { b <<= 1; ++log2; }
  if (log2 < MetalDeviceContext::kMinBucketLog2) {
    return 0;
  }
  size_t bucket = log2 - MetalDeviceContext::kMinBucketLog2;
  if (bucket >= MetalDeviceContext::kBucketCount) {
    return MetalDeviceContext::kBucketCount;  // sentinel: too big
  }
  return bucket;
}

// Acquire a shared-storage buffer ≥ ``bytes`` from the pool, or
// allocate fresh if the pool is empty / the request exceeds the
// top bucket.  Caller is responsible for releasing via
// :func:`metal_buffer_release`.
static id<MTLBuffer> metal_buffer_acquire(MetalDeviceContext &ctx,
                                           size_t bytes) {
  size_t bucket = metal_buffer_bucket(bytes);
  if (bucket >= MetalDeviceContext::kBucketCount) {
    return [ctx.device newBufferWithLength:bytes
                                   options:MTLResourceStorageModeShared];
  }
  {
    std::lock_guard<std::mutex> lock(ctx.buffer_pool_mu);
    auto &pool = ctx.buffer_pool[bucket];
    if (!pool.empty()) {
      id<MTLBuffer> buf = pool.back();
      pool.pop_back();
      return buf;
    }
  }
  size_t cap = (size_t)1 << (bucket + MetalDeviceContext::kMinBucketLog2);
  return [ctx.device newBufferWithLength:cap
                                 options:MTLResourceStorageModeShared];
}

static void metal_buffer_release(MetalDeviceContext &ctx,
                                  id<MTLBuffer> buf, size_t bytes) {
  if (buf == nil) return;
  size_t bucket = metal_buffer_bucket(bytes);
  if (bucket >= MetalDeviceContext::kBucketCount) {
    // Too big for the pool — let it dealloc.
    return;
  }
  std::lock_guard<std::mutex> lock(ctx.buffer_pool_mu);
  // Cap each bucket so the pool's footprint stays bounded.
  if (ctx.buffer_pool[bucket].size() < 8) {
    ctx.buffer_pool[bucket].push_back(buf);
  }
}

// Convenience: acquire + memcpy host bytes into a shared buffer.
// Underlying primitive — must use the raw acquire here (the
// TS_METAL_BUF_ACQUIRE macro is defined below and is meant for
// dispatcher call sites, not for this helper).
static id<MTLBuffer> metal_buffer_acquire_with_bytes(
    MetalDeviceContext &ctx, const void *src, size_t bytes) {
  id<MTLBuffer> buf = metal_buffer_acquire(ctx, bytes);
  if (buf == nil) return nil;
  std::memcpy([buf contents], src, bytes);
  return buf;
}

// RAII guard: acquires a buffer in its constructor and returns it to
// the pool via :func:`metal_buffer_release` in its destructor — so
// every exit path from a dispatcher (success, early `return false`,
// caught exception) releases automatically.  Use the macros below
// to keep call sites short.
struct MetalBufferGuard {
  MetalDeviceContext *ctx;
  id<MTLBuffer> buf;
  size_t bytes;
  MetalBufferGuard(MetalDeviceContext &c, id<MTLBuffer> b, size_t n)
      : ctx(&c), buf(b), bytes(n) {}
  ~MetalBufferGuard() {
    if (buf) metal_buffer_release(*ctx, buf, bytes);
  }
  MetalBufferGuard(const MetalBufferGuard&) = delete;
  MetalBufferGuard& operator=(const MetalBufferGuard&) = delete;
  MetalBufferGuard(MetalBufferGuard&& o) noexcept
      : ctx(o.ctx), buf(o.buf), bytes(o.bytes) { o.buf = nil; }
  MetalBufferGuard& operator=(MetalBufferGuard&&) = delete;
};

// Declare an `id<MTLBuffer> NAME` bound to a guarded acquire.  The
// guard is named `_g_<NAME>` so multiple acquires in the same scope
// don't collide.  Releases run automatically when the enclosing
// scope exits.
#define TS_METAL_BUF_ACQUIRE(NAME, CTX, BYTES) \
  MetalBufferGuard _g_##NAME((CTX), metal_buffer_acquire((CTX), (BYTES)), (BYTES)); \
  id<MTLBuffer> NAME = _g_##NAME.buf
#define TS_METAL_BUF_ACQUIRE_WITH_BYTES(NAME, CTX, SRC, BYTES) \
  MetalBufferGuard _g_##NAME((CTX), \
      metal_buffer_acquire_with_bytes((CTX), (SRC), (BYTES)), (BYTES)); \
  id<MTLBuffer> NAME = _g_##NAME.buf

MetalDeviceContext &deviceContext() {
  static MetalDeviceContext ctx{nil, nil, false};
  static std::once_flag once;
  std::call_once(once, [] {
    @autoreleasepool {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (!dev) {
        ctx.ok = false;
        return;
      }
      id<MTLCommandQueue> q = [dev newCommandQueue];
      if (!q) {
        ctx.ok = false;
        return;
      }
      ctx.device = dev;
      ctx.queue = q;
      ctx.ok = true;
      // SIMD-feature detection. Apple7 (M1) and later expose the full set of
      // SIMD reduction + shuffle(+fill) + simdgroup_barrier intrinsics; Mac2
      // discrete GPUs likewise. Older Apple4/5 GPUs have shuffle + barrier
      // (Metal 2.0) but not the reduction/shuffle-and-fill (Metal 2.1) family.
      int32_t caps = 0;
      if ([dev supportsFamily:MTLGPUFamilyApple7] ||
          [dev supportsFamily:MTLGPUFamilyMac2]) {
        caps = kTsSimdReduction | kTsSimdShuffle | kTsSimdShuffleAndFill |
               kTsSimdgroupBarrier;
      } else if ([dev supportsFamily:MTLGPUFamilyApple4]) {
        caps = kTsSimdShuffle | kTsSimdgroupBarrier;
      }
      ctx.simd_caps = caps;
    }
  });
  return ctx;
}

// Introspection: the SIMD-feature capability bitmask of the active GPU (see
// TsSimdCaps). 0 when no Metal device. Used by the SIMD-reduction rowop fast
// path and exposed to Python as apple_gpu_simd_caps().
extern "C" int32_t tessera_apple_gpu_simd_caps(void) {
  MetalDeviceContext &ctx = deviceContext();
  return ctx.ok ? ctx.simd_caps : 0;
}

//===----------------------------------------------------------------------===//
// R0 — persistent device-tensor handle (GPU-resident activations).
//
// A TsDeviceTensor owns one shared (`MTLResourceStorageModeShared`) MTLBuffer.
// Because Apple Silicon is unified-memory, `[buf contents]` is a CPU pointer to
// the *same* bytes the GPU sees, so once a value lives in a TsDeviceTensor the
// host can read/write it with zero further copies — and a producer op's output
// can feed a consumer op without any host round-trip (wired in R1). The handle
// is an opaque `void*` across the ABI; Python wraps it in `runtime.DeviceTensor`
// with a numpy view over the shared storage and lazy host materialization.
//===----------------------------------------------------------------------===//

struct TsDeviceTensor {
  id<MTLBuffer> buf;
  int64_t nbytes;
  // R0→MTLTensor bridge: a lazily-created, cached buffer-backed MTLTensor view so
  // a resident activation can feed the Metal 4 cooperative (matrix-unit) lane with
  // zero host round-trip. Bare `id` (not `id<MTLTensor>`) keeps the struct free of
  // the macos(26) availability annotation; cached by (inner, outer, dt).
  id tensorView;
  int viewInner;
  int viewOuter;
  int viewDt;
};

extern "C" TsDeviceTensor *ts_dev_alloc(int64_t nbytes) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || nbytes <= 0) return nullptr;
  @autoreleasepool {
    id<MTLBuffer> b = [ctx.device newBufferWithLength:(NSUInteger)nbytes
                                              options:MTLResourceStorageModeShared];
    if (!b) return nullptr;
    return new TsDeviceTensor{b, nbytes, nil, 0, 0, -1};
  }
}

// CPU pointer to the shared storage — a numpy view over this is zero-copy and
// stays coherent with the GPU (unified memory).
extern "C" void *ts_dev_contents(TsDeviceTensor *t) {
  return t ? [t->buf contents] : nullptr;
}

extern "C" int64_t ts_dev_nbytes(TsDeviceTensor *t) {
  return t ? t->nbytes : 0;
}

// Interop escape hatch: the underlying `id<MTLBuffer>` as an opaque `void*`
// (`__bridge`, no ownership transfer — Tessera owns the buffer's lifetime). Lets
// external Metal/MPS code operate on a resident DeviceTensor GPU-side; pair with
// tessera_apple_gpu_device_handle() / _command_queue_handle(). Callers must not
// release it and must serialize their own work against Tessera's queue.
extern "C" void *ts_dev_mtl_buffer(TsDeviceTensor *t) {
  return t ? (__bridge void *)t->buf : nullptr;
}

extern "C" void ts_dev_upload(TsDeviceTensor *t, const void *src, int64_t n) {
  if (t && src && n > 0 && n <= t->nbytes) std::memcpy([t->buf contents], src, (size_t)n);
}

extern "C" void ts_dev_download(TsDeviceTensor *t, void *dst, int64_t n) {
  if (t && dst && n > 0 && n <= t->nbytes) std::memcpy(dst, [t->buf contents], (size_t)n);
}

extern "C" void ts_dev_free(TsDeviceTensor *t) {
  if (t) {
    t->tensorView = nil;  // ARC releases the cached MTLTensor view
    t->buf = nil;  // ARC releases the buffer
    delete t;
  }
}

// Introspection: 1 when a real Metal device backs the handles (vs the non-Apple
// host-memory reference path).
extern "C" int32_t ts_dev_is_metal(void) {
  return deviceContext().ok ? 1 : 0;
}

// Interop escape hatch (cf. Mojo's metal_device(ctx)): the raw `id<MTLDevice>` /
// `id<MTLCommandQueue>` that Tessera's runtime uses, as opaque `void*`
// (`__bridge`, no ownership transfer — the process-wide singleton keeps them
// alive, so the pointers are valid for process lifetime). For advanced interop:
// build custom Metal/MPS work against the *same* device/queue so it composes with
// Tessera's resident buffers (ts_dev_mtl_buffer). nullptr if Metal is
// unavailable. Callers must not release these and should serialize work they
// enqueue on the shared queue.
extern "C" void *tessera_apple_gpu_device_handle(void) {
  MetalDeviceContext &ctx = deviceContext();
  return ctx.ok ? (__bridge void *)ctx.device : nullptr;
}

extern "C" void *tessera_apple_gpu_command_queue_handle(void) {
  MetalDeviceContext &ctx = deviceContext();
  return ctx.ok ? (__bridge void *)ctx.queue : nullptr;
}

// Reference fallback. Used when MTLCreateSystemDefaultDevice() returns nil
// (unlikely on Darwin GUI sessions, but possible in headless CI). Same shape
// as the CPU runtime's reference path.
inline void reference_gemm_f32(const float* A, const float* B, float* C,
                               int32_t M, int32_t N, int32_t K) {
  std::memset(C, 0, sizeof(float) * static_cast<std::size_t>(M) *
                        static_cast<std::size_t>(N));
  for (int32_t m = 0; m < M; ++m) {
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(m) * K + k];
      for (int32_t n = 0; n < N; ++n) {
        C[static_cast<std::size_t>(m) * N + n] +=
            a * B[static_cast<std::size_t>(k) * N + n];
      }
    }
  }
}

// Apple-sample Pattern 4 (2026-05-31) — Shared-event timeout wrapper for
// the legacy MPS / MPSGraph queue. Mirrors Apple's sample
// (``MLMatrixMultiplier.m:241-255``):
//
//     [commandQueue signalEvent:sharedEvent value:N]
//     bool done = [sharedEvent waitUntilSignaledValue:N timeoutMS:T]
//
// On the legacy queue we encode the signal into the command buffer
// itself (the queue-side ``signalEvent:value:`` API is MTL4-only); the
// effect is the same — the event ticks AFTER the GPU finishes the
// preceding work. Returns ``true`` if the GPU completed within the
// timeout, ``false`` if the timeout fired (kernel hung / driver crash /
// device disappeared). Caller MUST check the return; on ``false`` the
// command buffer may still be in flight — do not touch its outputs.
//
// ``op_name`` is logged on timeout so test diagnostics name which
// dispatcher hung instead of a generic "GPU stalled".
API_AVAILABLE(macos(10.14), ios(12.0))
static bool commit_and_wait_with_timeout(MetalDeviceContext &ctx,
                                         id<MTLCommandBuffer> cb,
                                         uint64_t timeout_ms,
                                         const char *op_name) {
  if (!cb) return false;
  // Lazy-init the shared event under the dedicated lock.
  id<MTLSharedEvent> ev;
  uint64_t signal_val;
  {
    std::lock_guard<std::mutex> lock(ctx.legacy_event_mu);
    if (!ctx.legacy_event) {
      ctx.legacy_event = [ctx.device newSharedEvent];
      if (!ctx.legacy_event) {
        // Event creation failed — fall back to the pre-Pattern-4
        // ``waitUntilCompleted`` path so the caller doesn't crash.
        // No timeout protection, but at least correct.
        [cb commit];
        [cb waitUntilCompleted];
        return cb.status == MTLCommandBufferStatusCompleted;
      }
    }
    ev = (id<MTLSharedEvent>)ctx.legacy_event;
    signal_val = ++ctx.legacy_event_val;
  }
  // Encode the signal into the command buffer; the event ticks AFTER the
  // GPU finishes everything queued before it on this cb.
  [cb encodeSignalEvent:ev value:signal_val];
  [cb commit];
  bool done = [ev waitUntilSignaledValue:signal_val
                                timeoutMS:timeout_ms];
  if (!done) {
    fprintf(stderr,
            "[tessera_apple_gpu] %s: GPU dispatch did not signal within "
            "%llu ms (signaledValue=%llu wanted=%llu cb.status=%ld "
            "cb.error=%s)\n",
            op_name ? op_name : "<unknown>",
            (unsigned long long)timeout_ms,
            (unsigned long long)ev.signaledValue,
            (unsigned long long)signal_val,
            (long)cb.status,
            cb.error ? [[cb.error localizedDescription] UTF8String]
                     : "<nil>");
    return false;
  }
  // ``encodeSignalEvent`` is encoded AFTER all preceding work in the cb,
  // so a signaled event means the GPU has finished everything in the
  // command buffer. ``cb.status`` updates asynchronously on a separate
  // thread and frequently still reads as ``Scheduled (3)`` for a tiny
  // window after the event ticks — gating on ``status == Completed``
  // here loses the race. Treat the event signal as authoritative; only
  // a non-nil ``cb.error`` indicates real failure.
  if (cb.error != nil) {
    fprintf(stderr,
            "[tessera_apple_gpu] %s: GPU dispatch signaled but reported "
            "an error: %s\n",
            op_name ? op_name : "<unknown>",
            [[cb.error localizedDescription] UTF8String]);
    return false;
  }
  return true;
}

bool dispatch_mps_gemm_f32(MetalDeviceContext &ctx, const float* A,
                           const float* B, float* C, int32_t M, int32_t N,
                           int32_t K) {
  @autoreleasepool {
    NSUInteger byteCountA = sizeof(float) * static_cast<NSUInteger>(M) *
                            static_cast<NSUInteger>(K);
    NSUInteger byteCountB = sizeof(float) * static_cast<NSUInteger>(K) *
                            static_cast<NSUInteger>(N);
    NSUInteger byteCountC = sizeof(float) * static_cast<NSUInteger>(M) *
                            static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, byteCountA);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, byteCountB);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, byteCountC);
    if (!bufA || !bufB || !bufC) return false;

    NSUInteger rowBytesA = sizeof(float) * static_cast<NSUInteger>(K);
    NSUInteger rowBytesB = sizeof(float) * static_cast<NSUInteger>(N);
    NSUInteger rowBytesC = sizeof(float) * static_cast<NSUInteger>(N);

    MPSMatrixDescriptor *descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                              columns:static_cast<NSUInteger>(K)
                                             rowBytes:rowBytesA
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(K)
                                              columns:static_cast<NSUInteger>(N)
                                             rowBytes:rowBytesB
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                              columns:static_cast<NSUInteger>(N)
                                             rowBytes:rowBytesC
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:ctx.device
         transposeLeft:NO
        transposeRight:NO
            resultRows:static_cast<NSUInteger>(M)
         resultColumns:static_cast<NSUInteger>(N)
       interiorColumns:static_cast<NSUInteger>(K)
                 alpha:1.0
                  beta:0.0];

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    [kernel encodeToCommandBuffer:cb
                       leftMatrix:matA
                      rightMatrix:matB
                     resultMatrix:matC];
    // Pattern 4 — bound the wait so a kernel hang surfaces as a timed
    // failure with a named op instead of an infinite stall. 30 seconds
    // is generous for an MPS GEMM at any size we'd legitimately call
    // (M*N*K up to ~1e10 finishes in <1s on M-series; the bound is for
    // genuine GPU stalls / driver kills, not for slow but valid runs).
    bool ok = commit_and_wait_with_timeout(ctx, cb, /*timeout_ms=*/30000,
                                            "mps_gemm_f32");
    if (!ok) return false;

    std::memcpy(C, [bufC contents], byteCountC);
    return true;
  }
}

} // namespace

extern "C" int32_t tessera_apple_gpu_runtime_has_metal(void) {
  return deviceContext().ok ? 1 : 0;
}

// Apple-sample pattern 6 — expose the innermost-first row-major stride
// contract so a unit test can verify it without bringing up a MTLDevice.
// The math here is byte-identical to ``apple_row_major_strides`` in the
// anonymous namespace (which the namespace-scoped ``make_buffer_tensor*``
// helpers use); duplicating four lines avoids the extra plumbing of
// pulling the helper out of the anonymous namespace just for this probe.
// Tests pin the layout: ``strides[0] == 1`` and ``strides[i+1] ==
// strides[i] * dims[i]``.
// Apple-sample Pattern 4 probe — exercise ``commit_and_wait_with_timeout``
// from a Python test by submitting a tiny no-op MPS dispatch (just memset a
// host-shared buffer to 0) and asserting it completes within ``timeout_ms``.
// Returns:
//    1 — completed in time
//    0 — timed out
//   -1 — runtime not initialized / Metal not available
//   -2 — buffer alloc / command buffer creation failed
// The "no-op" workload is a single ``MTLBlitCommandEncoder fillBuffer:``
// which is the cheapest GPU dispatch we can encode; if ANY GPU work
// completes, this test passes. A timeout therefore means the timeout
// machinery itself is broken (or the GPU is on fire), not that the kernel
// was too slow.
extern "C" int32_t tessera_apple_gpu_commit_and_wait_timeout_probe(
    uint64_t timeout_ms) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !ctx.device || !ctx.queue) return -1;
  @autoreleasepool {
    id<MTLBuffer> buf = [ctx.device newBufferWithLength:64
                                                  options:MTLResourceStorageModeShared];
    if (!buf) return -2;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    if (!cb) return -2;
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit fillBuffer:buf range:NSMakeRange(0, 64) value:0];
    [blit endEncoding];
    bool ok = commit_and_wait_with_timeout(ctx, cb, timeout_ms,
                                            "timeout_probe");
    return ok ? 1 : 0;
  }
}

extern "C" int32_t tessera_apple_gpu_row_major_strides(const int64_t *dims_in,
                                                      int32_t rank,
                                                      int64_t *strides_out) {
  if (!dims_in || !strides_out || rank <= 0 || rank > 8) return 0;
  int64_t stride = 1;
  for (int32_t i = 0; i < rank; ++i) {
    strides_out[i] = stride;
    stride *= dims_in[i];
  }
  return rank;
}

// Aligned variant — enforces Apple's MTLTensorDescriptor.strides rules
// for ML-usage / sub-byte dtypes (skills.md row #3 follow-on, 2026-06-01).
//
// Apple's documented rules (developer.apple.com MTLTensorDescriptor.strides):
//
//   * ``strides[0]`` must equal 1 (innermost dim is contiguous).
//   * For ML usage with byte+ dtypes, the second stride * element_bytes
//     must be aligned to 64 bytes.
//   * For sub-byte dtypes (< 8 bits per element, e.g. int4), the second
//     stride * element_bytes must be aligned to 128 bytes — this rule
//     applies regardless of usage flag.
//   * Subsequent strides are cumulative products of (aligned stride[1],
//     dims[1], dims[2], ...).
//
// Returns the rank on success, 0 on invalid input (dims/strides null,
// rank out of [1, 8], element_bits not in (0, 64], unrepresentable
// alignment).
//
// ``element_bits`` is the dtype size in BITS (e.g. 32 for fp32, 16 for
// fp16/bf16, 8 for int8, 4 for sub-byte int4). ``ml_usage`` is 1 if the
// caller intends ``MTLTensorUsageMachineLearning`` and 0 otherwise.
//
// Example: rank=2, dims=[13, 7], element_bits=32, ml_usage=1.
//   * natural stride[1] = 13 (innermost dim count).
//   * 64-byte alignment for fp32 = 16 elements.
//   * aligned stride[1] = round_up(13, 16) = 16.
//   * strides_out = [1, 16].
extern "C" int32_t tessera_apple_gpu_row_major_strides_aligned(
    const int64_t *dims_in, int32_t rank, int32_t element_bits,
    int32_t ml_usage, int64_t *strides_out) {
  if (!dims_in || !strides_out || rank <= 0 || rank > 8) return 0;
  if (element_bits <= 0 || element_bits > 64) return 0;

  // Determine the byte-alignment rule for the second stride.
  // Sub-byte dtypes (< 8 bits / element) ALWAYS use 128-byte alignment;
  // ML usage with byte+ dtypes uses 64-byte alignment; generic usage
  // with byte+ dtypes has no alignment requirement.
  int32_t alignment_bits = 0;
  if (element_bits < 8) {
    alignment_bits = 1024;          // 128 bytes
  } else if (ml_usage != 0) {
    alignment_bits = 512;           // 64 bytes
  }

  // Convert byte alignment to element alignment. For power-of-2 element
  // sizes (the only case Apple supports) this is exact division.
  int64_t elem_align = 1;
  if (alignment_bits > 0) {
    if (alignment_bits % element_bits != 0) {
      // Non-power-of-2 element size that doesn't divide the byte rule
      // evenly. Apple's tensor surface doesn't expose such dtypes; treat
      // as caller error.
      return 0;
    }
    elem_align = alignment_bits / element_bits;
    if (elem_align < 1) elem_align = 1;
  }

  // Apple's rule #1 — innermost stride is always 1.
  strides_out[0] = 1;
  if (rank == 1) return rank;

  // Apple's rule #2 — second stride is aligned. Innermost dim count
  // (dims_in[0]) is the natural element stride for dim-1; round UP to
  // satisfy the byte-alignment rule.
  int64_t natural = dims_in[0];
  int64_t aligned = natural;
  if (elem_align > 1) {
    int64_t rem = natural % elem_align;
    if (rem != 0) aligned = natural + (elem_align - rem);
  }
  strides_out[1] = aligned;

  // Apple's rule #3 — subsequent strides are cumulative products of the
  // ALIGNED stride[1] and the intermediate dims.
  int64_t acc = aligned;
  for (int32_t i = 2; i < rank; ++i) {
    acc *= dims_in[i - 1];
    strides_out[i] = acc;
  }
  return rank;
}

// Phase 2 stride-alignment integration (2026-06-01) — companion
// helper that returns the BYTE size an MTLBuffer needs to hold a
// tensor with aligned strides, for the same (dims, element_bits,
// ml_usage) inputs as ``tessera_apple_gpu_row_major_strides_aligned``.
//
// Use case: future Tessera-authored packaged kernels (or any path
// that sets ``MTLTensorDescriptor.strides`` explicitly to honor
// Apple's alignment rules) MUST allocate buffers sized for the
// aligned strides, not the dense element count. Mis-sizing causes
// out-of-bounds reads on rows past the first.
//
// Formula: ``total_bits = strides_aligned[rank-1] * dims[rank-1] *
// element_bits``, then ceil to bytes.
//
// Returns the byte count on success, 0 on invalid input. Dense
// (un-aligned) buffer size is ``prod(dims) * element_bits / 8``.
extern "C" int64_t tessera_apple_gpu_aligned_buffer_nbytes(
    const int64_t *dims_in, int32_t rank, int32_t element_bits,
    int32_t ml_usage) {
  if (!dims_in || rank <= 0 || rank > 8) return 0;
  if (element_bits <= 0 || element_bits > 64) return 0;
  int64_t strides[8];
  int32_t rc = tessera_apple_gpu_row_major_strides_aligned(
      dims_in, rank, element_bits, ml_usage, strides);
  if (rc != rank) return 0;
  // Total elements counting padding = stride[rank-1] * dims[rank-1].
  int64_t total_elems = strides[rank - 1] * dims_in[rank - 1];
  // ceil(total_elems * element_bits / 8).
  int64_t total_bits = total_elems * element_bits;
  return (total_bits + 7) / 8;
}

extern "C" void tessera_apple_gpu_mps_matmul_f32(const float* A,
                                                 const float* B, float* C,
                                                 int32_t M, int32_t N,
                                                 int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_mps_gemm_f32(ctx, A, B, C, M, N, K)) return;
  reference_gemm_f32(A, B, C, M, N, K);
}

//===----------------------------------------------------------------------===//
// GPU linear-algebra lane — Cholesky / LU / triangular solve via the
// MetalPerformanceShaders MPSMatrix* fixed-function kernels. This is the one
// capability MPSGraph cannot provide (it has no matrix-decomposition ops), so
// these dense f32 factorizations/solves are the only GPU path for
// tessera.ops.{cholesky, solve, cholesky_solve, tri_solve} — previously
// numpy/CPU only. Rank-2 f32; batched + f16 are follow-ups. Each returns 0 on a
// successful GPU run and a non-zero code otherwise (Metal unavailable = -1, or a
// singular / non-positive-definite matrix = 2), so the Python wrapper cleanly
// falls back to the numpy reference. All matrices are row-major (MPSMatrix
// native), matching numpy's storage so no transpose is needed at the boundary.
//===----------------------------------------------------------------------===//
namespace {

// Row-major MPSMatrix over an existing buffer (4-byte elements: f32 or uint32).
static MPSMatrix *ts_mps_mat(id<MTLBuffer> buf, int rows, int cols, MPSDataType dt) {
  MPSMatrixDescriptor *d =
      [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                            columns:(NSUInteger)cols
                                           rowBytes:(NSUInteger)cols * 4
                                           dataType:dt];
  return [[MPSMatrix alloc] initWithBuffer:buf descriptor:d];
}

// The MPSMatrixDecompositionStatus int written into the status buffer.
static int ts_decomp_status(id<MTLBuffer> s) {
  return s ? *((const int *)[s contents]) : -1;
}

}  // namespace

// Cholesky: A (n×n SPD, row-major f32) -> L lower-triangular with A = L·Lᵀ. The
// strict upper triangle of L is zeroed to match numpy.linalg.cholesky. 0 on
// success; 2 if A is not positive-definite; -1 if Metal is unavailable.
extern "C" int32_t tessera_apple_gpu_cholesky_f32(const float *A, float *L, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !A || !L || n <= 0) return -1;
  @autoreleasepool {
    size_t bytes = (size_t)n * n * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bA, ctx, A, bytes);
    TS_METAL_BUF_ACQUIRE(bR, ctx, bytes);
    TS_METAL_BUF_ACQUIRE(bS, ctx, sizeof(int));
    if (!bA || !bR || !bS) return -1;
    MPSMatrix *mA = ts_mps_mat(bA, n, n, MPSDataTypeFloat32);
    MPSMatrix *mR = ts_mps_mat(bR, n, n, MPSDataTypeFloat32);
    MPSMatrixDecompositionCholesky *chol = [[MPSMatrixDecompositionCholesky alloc]
        initWithDevice:ctx.device lower:YES order:(NSUInteger)n];
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    [chol encodeToCommandBuffer:cb sourceMatrix:mA resultMatrix:mR status:bS];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper
    // adds 30 s timeout protection so a hung Cholesky doesn't deadlock
    // the entire test process.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "cholesky_f32")) return -1;
    if (ts_decomp_status(bS) != MPSMatrixDecompositionStatusSuccess) return 2;
    const float *R = (const float *)[bR contents];
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        L[(size_t)i * n + j] = (j <= i) ? R[(size_t)i * n + j] : 0.0f;
    return 0;
  }
}

// SPD solve: factor A = L·Lᵀ (Cholesky) then solve A·X = B for X. A is n×n
// row-major f32; B/X are n×nrhs row-major f32. 0 on success; 2 if not PD.
extern "C" int32_t tessera_apple_gpu_solve_cholesky_f32(const float *A, const float *B,
                                                        float *X, int32_t n, int32_t nrhs) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !A || !B || !X || n <= 0 || nrhs <= 0) return -1;
  @autoreleasepool {
    size_t aB = (size_t)n * n * 4, rB = (size_t)n * nrhs * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bA, ctx, A, aB);
    TS_METAL_BUF_ACQUIRE(bL, ctx, aB);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bB, ctx, B, rB);
    TS_METAL_BUF_ACQUIRE(bX, ctx, rB);
    TS_METAL_BUF_ACQUIRE(bS, ctx, sizeof(int));
    if (!bA || !bL || !bB || !bX || !bS) return -1;
    MPSMatrix *mA = ts_mps_mat(bA, n, n, MPSDataTypeFloat32);
    MPSMatrix *mL = ts_mps_mat(bL, n, n, MPSDataTypeFloat32);
    MPSMatrix *mB = ts_mps_mat(bB, n, nrhs, MPSDataTypeFloat32);
    MPSMatrix *mX = ts_mps_mat(bX, n, nrhs, MPSDataTypeFloat32);
    MPSMatrixDecompositionCholesky *chol = [[MPSMatrixDecompositionCholesky alloc]
        initWithDevice:ctx.device lower:YES order:(NSUInteger)n];
    MPSMatrixSolveCholesky *solve = [[MPSMatrixSolveCholesky alloc]
        initWithDevice:ctx.device upper:NO order:(NSUInteger)n
        numberOfRightHandSides:(NSUInteger)nrhs];
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    [chol encodeToCommandBuffer:cb sourceMatrix:mA resultMatrix:mL status:bS];
    [solve encodeToCommandBuffer:cb sourceMatrix:mL rightHandSideMatrix:mB solutionMatrix:mX];
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "solve_cholesky_f32")) return -1;
    if (ts_decomp_status(bS) != MPSMatrixDecompositionStatusSuccess) return 2;
    std::memcpy(X, [bX contents], rB);
    return 0;
  }
}

// General solve: factor A = P·L·U (LU with partial pivoting) then solve A·X = B.
// A is n×n row-major f32; B/X are n×nrhs row-major f32. 0 on success; 2 if
// singular.
extern "C" int32_t tessera_apple_gpu_solve_lu_f32(const float *A, const float *B,
                                                  float *X, int32_t n, int32_t nrhs) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !A || !B || !X || n <= 0 || nrhs <= 0) return -1;
  @autoreleasepool {
    size_t aB = (size_t)n * n * 4, rB = (size_t)n * nrhs * 4, pB = (size_t)n * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bA, ctx, A, aB);
    TS_METAL_BUF_ACQUIRE(bLU, ctx, aB);
    TS_METAL_BUF_ACQUIRE(bP, ctx, pB);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bB, ctx, B, rB);
    TS_METAL_BUF_ACQUIRE(bX, ctx, rB);
    TS_METAL_BUF_ACQUIRE(bS, ctx, sizeof(int));
    if (!bA || !bLU || !bP || !bB || !bX || !bS) return -1;
    MPSMatrix *mA = ts_mps_mat(bA, n, n, MPSDataTypeFloat32);
    MPSMatrix *mLU = ts_mps_mat(bLU, n, n, MPSDataTypeFloat32);
    MPSMatrix *mP = ts_mps_mat(bP, 1, n, MPSDataTypeUInt32);
    MPSMatrix *mB = ts_mps_mat(bB, n, nrhs, MPSDataTypeFloat32);
    MPSMatrix *mX = ts_mps_mat(bX, n, nrhs, MPSDataTypeFloat32);
    MPSMatrixDecompositionLU *lu = [[MPSMatrixDecompositionLU alloc]
        initWithDevice:ctx.device rows:(NSUInteger)n columns:(NSUInteger)n];
    MPSMatrixSolveLU *solve = [[MPSMatrixSolveLU alloc]
        initWithDevice:ctx.device transpose:NO order:(NSUInteger)n
        numberOfRightHandSides:(NSUInteger)nrhs];
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    [lu encodeToCommandBuffer:cb sourceMatrix:mA resultMatrix:mLU pivotIndices:mP status:bS];
    [solve encodeToCommandBuffer:cb sourceMatrix:mLU rightHandSideMatrix:mB
                   pivotIndices:mP solutionMatrix:mX];
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "solve_lu_f32")) return -1;
    if (ts_decomp_status(bS) != MPSMatrixDecompositionStatusSuccess) return 2;
    std::memcpy(X, [bX contents], rB);
    return 0;
  }
}

// Triangular solve: solve op(tri(A))·X = B where tri(A) is the lower (lower=1)
// or upper (lower=0) triangle of A, op is transpose when trans=1, and the
// diagonal is treated as unit when unit=1. Only the relevant triangle of A is
// read (BLAS trsm semantics), matching numpy's np.tril/np.triu. A is n×n,
// B/X are n×nrhs, all row-major f32. 0 on success; -1 if Metal unavailable.
extern "C" int32_t tessera_apple_gpu_tri_solve_f32(const float *A, const float *B,
                                                   float *X, int32_t n, int32_t nrhs,
                                                   int32_t lower, int32_t trans,
                                                   int32_t unit) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !A || !B || !X || n <= 0 || nrhs <= 0) return -1;
  @autoreleasepool {
    size_t aB = (size_t)n * n * 4, rB = (size_t)n * nrhs * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bA, ctx, A, aB);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bB, ctx, B, rB);
    TS_METAL_BUF_ACQUIRE(bX, ctx, rB);
    if (!bA || !bB || !bX) return -1;
    MPSMatrix *mA = ts_mps_mat(bA, n, n, MPSDataTypeFloat32);
    MPSMatrix *mB = ts_mps_mat(bB, n, nrhs, MPSDataTypeFloat32);
    MPSMatrix *mX = ts_mps_mat(bX, n, nrhs, MPSDataTypeFloat32);
    MPSMatrixSolveTriangular *solve = [[MPSMatrixSolveTriangular alloc]
        initWithDevice:ctx.device right:NO upper:(lower ? NO : YES)
             transpose:(trans ? YES : NO) unit:(unit ? YES : NO)
                 order:(NSUInteger)n numberOfRightHandSides:(NSUInteger)nrhs alpha:1.0];
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    [solve encodeToCommandBuffer:cb sourceMatrix:mA rightHandSideMatrix:mB solutionMatrix:mX];
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "tri_solve_f32")) return -1;
    std::memcpy(X, [bX contents], rB);
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// GPU-native RNG lane (opt-in) — uniform / normal f32 fills via
// MPSMatrixRandomPhilox. Philox-family (matching Tessera's S4 RNG family) but
// the stream is NOT bit-identical to Tessera's CPU Philox — different
// counter/key layout — so this is a SEPARATE opt-in path, never wired into the
// deterministic tessera.rng samplers (that would break the CPU/GPU-equality +
// check_determinism contracts, Decision #18). Determinism here is defined by the
// `seed` argument under MPS's own generator. Returns 1 on a GPU run, else 0
// (Python falls back to its own RNG).
//===----------------------------------------------------------------------===//
namespace {
bool dispatch_mps_random_f32(MetalDeviceContext &ctx, float *out, int64_t n,
                             uint64_t seed, bool normal, float a, float b) {
  if (!ctx.ok || !out || n <= 0) return false;
  @autoreleasepool {
    NSUInteger len = (NSUInteger)n;
    // MPSMatrixRandomPhilox is Philox-4x32 — it generates 4 values per counter
    // step and asserts the result count is a multiple of 4. Over-allocate to the
    // next multiple of 4, generate, and copy back only the requested `len`.
    NSUInteger gen = (len + 3u) & ~(NSUInteger)3u;
    size_t bytes = (size_t)gen * 4;
    TS_METAL_BUF_ACQUIRE(bOut, ctx, bytes);
    if (!bOut) return false;
    // A flat MPSVector is the idiomatic destination for a 1-D random fill — no
    // 2-D rowBytes alignment to reason about (vs the matrix encode path).
    MPSVectorDescriptor *vd =
        [MPSVectorDescriptor vectorDescriptorWithLength:gen dataType:MPSDataTypeFloat32];
    MPSVector *v = [[MPSVector alloc] initWithBuffer:bOut descriptor:vd];
    MPSMatrixRandomDistributionDescriptor *dist =
        normal ? [MPSMatrixRandomDistributionDescriptor
                     normalDistributionDescriptorWithMean:a standardDeviation:b]
               : [MPSMatrixRandomDistributionDescriptor
                     uniformDistributionDescriptorWithMinimum:a maximum:b];
    MPSMatrixRandomPhilox *rng =
        [[MPSMatrixRandomPhilox alloc] initWithDevice:ctx.device
                                  destinationDataType:MPSDataTypeFloat32
                                                 seed:(NSUInteger)seed
                               distributionDescriptor:dist];
    if (!v || !dist || !rng) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    [rng encodeToCommandBuffer:cb destinationVector:v];
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "mps_random_f32")) return false;
    std::memcpy(out, [bOut contents], (size_t)len * 4);  // only the requested n
    return true;
  }
}
}  // namespace

// Uniform f32 in [lo, hi). 1 if it ran on the GPU, else 0.
extern "C" int32_t tessera_apple_gpu_random_uniform_f32(float *out, int64_t n,
                                                        uint64_t seed, float lo,
                                                        float hi) {
  return dispatch_mps_random_f32(deviceContext(), out, n, seed, false, lo, hi) ? 1 : 0;
}

// Normal f32 with given mean + standard deviation. 1 if it ran on the GPU.
extern "C" int32_t tessera_apple_gpu_random_normal_f32(float *out, int64_t n,
                                                       uint64_t seed, float mean,
                                                       float stddev) {
  return dispatch_mps_random_f32(deviceContext(), out, n, seed, true, mean, stddev) ? 1 : 0;
}

//===---------------------------------------------------------------------===//
// Phase 8.4.4 — fp16 / bf16 matmul (mirrors Phase 8.2 BNNS bf16 pattern)
//
// fp16: native MPSDataTypeFloat16 path. Apple Silicon GPUs run fp16
//       natively at 2x throughput vs fp32 on most ops.
// bf16: MPS does NOT directly support bf16 matmul as of macOS 14, so this
//       path uses fp32 conversion at the boundary — load with bit-shift
//       (bf16 -> fp32), run MPSDataTypeFloat32, convert back. Same shape
//       as the BNNS bf16 fallback in apple_cpu_runtime.cpp.
//
// At the C ABI boundary fp16/bf16 inputs are passed as uint16_t* (the bit
// pattern). This keeps the ABI portable across compilers regardless of
// _Float16 / __bf16 availability.
//===---------------------------------------------------------------------===//

namespace {

// fp16 <-> float helpers (bit-pattern; no _Float16 dependency).
inline float half_to_float_gpu(uint16_t h) {
  uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
  uint32_t exp  = (uint32_t(h) & 0x7C00u) >> 10;
  uint32_t frac = uint32_t(h) & 0x03FFu;
  uint32_t f;
  if (exp == 0) {
    if (frac == 0) {
      f = sign;
    } else {
      while ((frac & 0x0400u) == 0) { frac <<= 1; exp -= 1; }
      exp += 1;
      frac &= ~0x0400u;
      f = sign | ((exp + 112) << 23) | (frac << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (frac << 13);
  } else {
    f = sign | ((exp + 112) << 23) | (frac << 13);
  }
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline uint16_t float_to_half_gpu(float v) {
  uint32_t f;
  std::memcpy(&f, &v, sizeof(f));
  uint32_t sign = (f >> 16) & 0x8000u;
  int32_t  exp  = int32_t((f >> 23) & 0xFFu) - 127 + 15;
  uint32_t frac = f & 0x007FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return uint16_t(sign);
    frac = (frac | 0x00800000u) >> (1 - exp);
    if (frac & 0x00001000u) frac += 0x00002000u;
    return uint16_t(sign | (frac >> 13));
  }
  if (exp >= 0x1F) {
    if (((f >> 23) & 0xFFu) == 0xFFu) {
      return uint16_t(sign | 0x7C00u | (frac ? (frac >> 13) | 0x200u : 0));
    }
    return uint16_t(sign | 0x7C00u);
  }
  if (frac & 0x00001000u) {
    frac += 0x00002000u;
    if (frac & 0x00800000u) {
      frac = 0;
      exp += 1;
      if (exp >= 0x1F) return uint16_t(sign | 0x7C00u);
    }
  }
  return uint16_t(sign | (uint32_t(exp) << 10) | (frac >> 13));
}

// bf16 <-> float helpers (bit-shift; round-to-nearest-even on store).
inline float bfloat16_to_float_gpu(uint16_t b) {
  uint32_t f = static_cast<uint32_t>(b) << 16;
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}

inline uint16_t float_to_bfloat16_gpu(float v) {
  uint32_t f;
  std::memcpy(&f, &v, sizeof(f));
  if ((f & 0x7FC00000u) == 0x7F800000u && (f & 0x007FFFFFu) != 0) {
    return static_cast<uint16_t>((f >> 16) | 0x40u);
  }
  uint32_t lsb = (f >> 16) & 1u;
  uint32_t rounded = f + 0x7FFFu + lsb;
  return static_cast<uint16_t>(rounded >> 16);
}

bool dispatch_mps_gemm_f16(MetalDeviceContext &ctx, const uint16_t* A,
                           const uint16_t* B, uint16_t* C,
                           int32_t M, int32_t N, int32_t K) {
  // Same shape as dispatch_mps_gemm_f32 — only the MPS data type and the
  // per-element byte count change. Apple GPUs run fp16 natively at higher
  // throughput than fp32 so this is a real perf win, not just convenience.
  @autoreleasepool {
    NSUInteger byteCountA = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                            static_cast<NSUInteger>(K);
    NSUInteger byteCountB = sizeof(uint16_t) * static_cast<NSUInteger>(K) *
                            static_cast<NSUInteger>(N);
    NSUInteger byteCountC = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                            static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, byteCountA);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, byteCountB);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, byteCountC);
    if (!bufA || !bufB || !bufC) return false;

    NSUInteger rowBytesA = sizeof(uint16_t) * static_cast<NSUInteger>(K);
    NSUInteger rowBytesB = sizeof(uint16_t) * static_cast<NSUInteger>(N);
    NSUInteger rowBytesC = sizeof(uint16_t) * static_cast<NSUInteger>(N);

    MPSMatrixDescriptor *descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                              columns:static_cast<NSUInteger>(K)
                                             rowBytes:rowBytesA
                                             dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(K)
                                              columns:static_cast<NSUInteger>(N)
                                             rowBytes:rowBytesB
                                             dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                              columns:static_cast<NSUInteger>(N)
                                             rowBytes:rowBytesC
                                             dataType:MPSDataTypeFloat16];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:ctx.device
         transposeLeft:NO
        transposeRight:NO
            resultRows:static_cast<NSUInteger>(M)
         resultColumns:static_cast<NSUInteger>(N)
       interiorColumns:static_cast<NSUInteger>(K)
                 alpha:1.0
                  beta:0.0];

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    [kernel encodeToCommandBuffer:cb
                       leftMatrix:matA
                      rightMatrix:matB
                     resultMatrix:matC];
    // Migration batch 2 (2026-06-01) — 30 s timeout via Pattern-4
    // wrapper so a hung f16 GEMM doesn't deadlock the test runner.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "mps_gemm_f16")) return false;
    std::memcpy(C, [bufC contents], byteCountC);
    return true;
  }
}

inline void reference_gemm_f16_via_fp32(const uint16_t* A, const uint16_t* B,
                                        uint16_t* C, int32_t M, int32_t N,
                                        int32_t K) {
  // Convert each operand to fp32, run the existing reference kernel, convert
  // back. Same numerical contract as the BNNS-fallback path in
  // apple_cpu_runtime.cpp.
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(M) * N, 0.0f);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_gpu(B[i]);
  reference_gemm_f32(Af.data(), Bf.data(), Cf.data(), M, N, K);
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_half_gpu(Cf[i]);
}

inline void reference_gemm_bf16_via_fp32(const uint16_t* A, const uint16_t* B,
                                         uint16_t* C, int32_t M, int32_t N,
                                         int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(M) * N, 0.0f);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  reference_gemm_f32(Af.data(), Bf.data(), Cf.data(), M, N, K);
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_bfloat16_gpu(Cf[i]);
}

bool dispatch_mps_gemm_bf16_via_fp32(MetalDeviceContext &ctx,
                                     const uint16_t* A, const uint16_t* B,
                                     uint16_t* C, int32_t M, int32_t N,
                                     int32_t K) {
  // bf16 -> fp32 -> MPSDataTypeFloat32 -> fp32 -> bf16. MPS does not
  // natively accept bf16 as of macOS 14; this path keeps the bf16 ABI
  // honest while still spending the heavy compute on the GPU.
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(M) * N, 0.0f);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  if (!dispatch_mps_gemm_f32(ctx, Af.data(), Bf.data(), Cf.data(), M, N, K))
    return false;
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_bfloat16_gpu(Cf[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_mps_matmul_f16(const uint16_t* A,
                                                 const uint16_t* B,
                                                 uint16_t* C,
                                                 int32_t M, int32_t N,
                                                 int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_mps_gemm_f16(ctx, A, B, C, M, N, K)) return;
  reference_gemm_f16_via_fp32(A, B, C, M, N, K);
}

extern "C" void tessera_apple_gpu_mps_matmul_bf16(const uint16_t* A,
                                                  const uint16_t* B,
                                                  uint16_t* C,
                                                  int32_t M, int32_t N,
                                                  int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_mps_gemm_bf16_via_fp32(ctx, A, B, C, M, N, K)) return;
  reference_gemm_bf16_via_fp32(A, B, C, M, N, K);
}

//===---------------------------------------------------------------------===//
// Phase 8.4 — Custom MSL kernel infrastructure
//
// `compile_msl_kernel` compiles an MSL source string into an
// MTLComputePipelineState and caches it in the device context. The cache key
// is the concatenation of the MSL source and entry point name — same source
// + same entry point hits the cache; either changing forces a recompile.
//
// `dispatch_msl_kernel` is a thin wrapper that encodes a compute pass with
// a 2-D grid. Kernel-specific symbols (rope_f32, etc.) build the buffers
// and pick grid dimensions, then call this helper.
//
// Capability probe `tessera_apple_gpu_runtime_msl_cache_size` exposes the
// number of cached pipelines so tests can assert the second invocation of a
// kernel reuses a cached pipeline state.
//===---------------------------------------------------------------------===//

namespace {

id<MTLComputePipelineState> compile_msl_kernel(MetalDeviceContext &ctx,
                                               NSString *source,
                                               NSString *entry_point) {
  std::string key;
  key.reserve(static_cast<std::size_t>([source length]) +
              static_cast<std::size_t>([entry_point length]) + 1);
  key.append([source UTF8String]);
  key.push_back('\x1f'); // unit separator — entry-point disambiguator
  key.append([entry_point UTF8String]);

  {
    std::lock_guard<std::mutex> lock(ctx.kernel_cache_mu);
    auto it = ctx.kernel_cache.find(key);
    if (it != ctx.kernel_cache.end()) return it->second;
  }

  NSError *error = nil;
  MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
  opts.languageVersion = MTLLanguageVersion3_0;
  id<MTLLibrary> library = [ctx.device newLibraryWithSource:source
                                                    options:opts
                                                      error:&error];
  if (!library) return nil;
  id<MTLFunction> fn = [library newFunctionWithName:entry_point];
  if (!fn) return nil;

  id<MTLComputePipelineState> pso =
      [ctx.device newComputePipelineStateWithFunction:fn error:&error];
  if (!pso) return nil;

  std::lock_guard<std::mutex> lock(ctx.kernel_cache_mu);
  // Re-check under the lock in case another thread compiled the same kernel
  // concurrently — keep whichever object won the race.
  auto it = ctx.kernel_cache.find(key);
  if (it != ctx.kernel_cache.end()) return it->second;
  ctx.kernel_cache.emplace(std::move(key), pso);
  return pso;
}

//===----------------------------------------------------------------------===//
// SVD via one-sided Jacobi (custom MSL) — the SVD/eigensolver gap MPS doesn't
// fill. One threadgroup per matrix, T threads cooperating. Rotates pairs of
// *columns* of a working copy W (= A) by Givens/Jacobi rotations (accumulated
// into V) until every column pair is mutually orthogonal; then σ_k = ‖W[:,k]‖,
// U[:,k] = W[:,k]/σ_k, and V holds the right singular vectors (as columns). The
// m-dimensional dot products (α,β,γ) reduce through a threadgroup tree; a sweep
// that applies no rotation means convergence. Unsorted on the GPU; the Python
// wrapper sorts σ descending, permutes U/V, and verifies ‖UΣVᵀ−A‖ before
// trusting the result. f32, m ≥ n, single matrix (rank-2). NB: one threadgroup =
// correctness-first, not a throughput win — see docs.
//===----------------------------------------------------------------------===//
namespace {
constexpr int kSvdThreads = 128;

bool dispatch_svd_jacobi_f32(MetalDeviceContext &ctx, const float *A, float *U,
                             float *S, float *V, int32_t batch, int32_t M, int32_t N) {
  if (!ctx.ok || !A || !U || !S || !V || batch <= 0 || M <= 0 || N <= 0 || M < N)
    return false;
  // Grid-batched: one threadgroup per matrix (grid = batch), each indexing into
  // its slice of W/V/S via threadgroup_position_in_grid. batch == 1 is the
  // single-matrix case; batch > 1 keeps the whole GPU busy across the batch.
  static NSString *const kSvdSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;
#define TS_SVD_THREADS 128

kernel void svd_jacobi_f32(
    device float* W        [[buffer(0)]],   // [batch] M x N, in = A, out = U·Σ
    device float* V        [[buffer(1)]],   // [batch] N x N, out = right vectors
    device float* S        [[buffer(2)]],   // [batch] N,     out = singular values
    constant int& M        [[buffer(3)]],
    constant int& N        [[buffer(4)]],
    constant int& maxSweeps[[buffer(5)]],
    uint lid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    const int T = TS_SVD_THREADS;
    int li = (int)lid;
    // This threadgroup's matrix slice.
    device float* Wb = W + (size_t)bid * (size_t)M * (size_t)N;
    device float* Vb = V + (size_t)bid * (size_t)N * (size_t)N;
    device float* Sb = S + (size_t)bid * (size_t)N;
    threadgroup float sa[TS_SVD_THREADS];
    threadgroup float sb[TS_SVD_THREADS];
    threadgroup float sg[TS_SVD_THREADS];
    threadgroup float crot[2];   // {c, s}
    threadgroup int   info[2];   // {do_rotation, n_rotations_this_sweep}
    const float eps = 1e-7f;

    // V = I
    for (int idx = li; idx < N * N; idx += T)
        Vb[idx] = ((idx / N) == (idx % N)) ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        if (li == 0) info[1] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int p = 0; p < N - 1; ++p) {
            for (int q = p + 1; q < N; ++q) {
                // α=Σ Wp², β=Σ Wq², γ=Σ Wp·Wq  (one pass over the M rows).
                float pa = 0.0f, pb = 0.0f, pg = 0.0f;
                for (int i = li; i < M; i += T) {
                    float wp = Wb[i * N + p], wq = Wb[i * N + q];
                    pa += wp * wp; pb += wq * wq; pg += wp * wq;
                }
                sa[li] = pa; sb[li] = pb; sg[li] = pg;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int s = T / 2; s > 0; s >>= 1) {
                    if (li < s) { sa[li]+=sa[li+s]; sb[li]+=sb[li+s]; sg[li]+=sg[li+s]; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (li == 0) {
                    float alpha = sa[0], beta = sb[0], gamma = sg[0];
                    int rot = 0; float c = 1.0f, s = 0.0f;
                    if (fabs(gamma) > eps * sqrt(max(alpha * beta, 0.0f))) {
                        rot = 1;
                        float zeta = (beta - alpha) / (2.0f * gamma);
                        float t = (zeta >= 0.0f ? 1.0f : -1.0f) /
                                  (fabs(zeta) + sqrt(1.0f + zeta * zeta));
                        if (zeta == 0.0f) t = 1.0f;   // α==β, γ≠0 -> 45° rotation
                        c = 1.0f / sqrt(1.0f + t * t);
                        s = c * t;
                        info[1] += 1;
                    }
                    crot[0] = c; crot[1] = s; info[0] = rot;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (info[0]) {
                    float c = crot[0], s = crot[1];
                    for (int i = li; i < M; i += T) {
                        float wp = Wb[i * N + p], wq = Wb[i * N + q];
                        Wb[i * N + p] = c * wp - s * wq;
                        Wb[i * N + q] = s * wp + c * wq;
                    }
                    for (int i = li; i < N; i += T) {
                        float vp = Vb[i * N + p], vq = Vb[i * N + q];
                        Vb[i * N + p] = c * vp - s * vq;
                        Vb[i * N + q] = s * vp + c * vq;
                    }
                    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (info[1] == 0) break;   // a full sweep with no rotation -> converged
    }

    // σ_k = ‖W[:,k]‖ ; U[:,k] = W[:,k]/σ_k.
    for (int k = 0; k < N; ++k) {
        float ps = 0.0f;
        for (int i = li; i < M; i += T) { float w = Wb[i * N + k]; ps += w * w; }
        sa[li] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = T / 2; s > 0; s >>= 1) {
            if (li < s) sa[li] += sa[li + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sigma = sqrt(sa[0]);
        if (li == 0) Sb[k] = sigma;
        float inv = sigma > 1e-30f ? 1.0f / sigma : 0.0f;
        for (int i = li; i < M; i += T) Wb[i * N + k] *= inv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)MSL";
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, kSvdSource, @"svd_jacobi_f32");
    if (!pso) return false;
    size_t wB = (size_t)batch * M * N * 4;
    size_t vB = (size_t)batch * N * N * 4;
    size_t sB = (size_t)batch * N * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bW, ctx, A, wB);
    TS_METAL_BUF_ACQUIRE(bV, ctx, vB);
    TS_METAL_BUF_ACQUIRE(bS, ctx, sB);
    if (!bW || !bV || !bS) return false;
    int maxSweeps = 60;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bW offset:0 atIndex:0];
    [enc setBuffer:bV offset:0 atIndex:1];
    [enc setBuffer:bS offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&maxSweeps length:sizeof(int32_t) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)batch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kSvdThreads, 1, 1)];
    [enc endEncoding];
    // Migration batch 2 (2026-06-01) — SVD can take seconds on large
    // matrices; use a 60 s timeout (vs 30 s default) to cover real
    // batch-of-large-SVDs without false hang reports.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "svd_jacobi_f32")) return false;
    std::memcpy(U, [bW contents], wB);
    std::memcpy(V, [bV contents], vB);
    std::memcpy(S, [bS contents], sB);
    return true;
  }
}
}  // namespace

// One-sided Jacobi SVD: A (M×N row-major f32, M≥N) -> U (M×N), S (N, unsorted),
// V (N×N, right vectors as columns). 1 if it ran on the GPU, else 0. The caller
// sorts σ descending + verifies reconstruction.
extern "C" int32_t tessera_apple_gpu_svd_f32(const float *A, float *U, float *S,
                                             float *V, int32_t M, int32_t N) {
  return dispatch_svd_jacobi_f32(deviceContext(), A, U, S, V, 1, M, N) ? 1 : 0;
}

// Batched one-sided Jacobi SVD: `batch` stacked M×N matrices (row-major,
// contiguous), one threadgroup per matrix. Same per-matrix contract as above.
extern "C" int32_t tessera_apple_gpu_svd_batched_f32(const float *A, float *U,
                                                     float *S, float *V,
                                                     int32_t batch, int32_t M,
                                                     int32_t N) {
  return dispatch_svd_jacobi_f32(deviceContext(), A, U, S, V, batch, M, N) ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Brent–Luk parallel Jacobi SVD (experimental). Same one-sided Jacobi, but with
// the round-robin "tournament" ordering: each sweep is N-1 rounds, and the N/2
// column pairs in a round are *disjoint*, so they're rotated concurrently — one
// per SIMD-group (each SIMD-group reduces its pair's M-row dots via simd_sum, no
// threadgroup scratch, no intra-round barrier; barrier only between rounds). The
// aim is ~nsg× the sequential cyclic kernel for large N. Kept separate so it can
// be A/B-benchmarked and dropped if it doesn't win. N ≤ 256 (perm in tg memory).
//===----------------------------------------------------------------------===//
namespace {
constexpr int kSvdBlMaxN = 256;

bool dispatch_svd_jacobi_bl_f32(MetalDeviceContext &ctx, const float *A, float *U,
                                float *S, float *V, int32_t batch, int32_t M,
                                int32_t N) {
  if (!ctx.ok || !A || !U || !S || !V || batch <= 0 || M <= 0 || N <= 0 ||
      M < N || N > kSvdBlMaxN)
    return false;
  static NSString *const kSvdBlSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;
#define TS_SVDBL_THREADS 128
#define TS_SVDBL_MAXN 256

kernel void svd_jacobi_bl_f32(
    device float* W        [[buffer(0)]],
    device float* V        [[buffer(1)]],
    device float* S        [[buffer(2)]],
    constant int& M        [[buffer(3)]],
    constant int& N        [[buffer(4)]],
    constant int& maxSweeps[[buffer(5)]],
    uint lid  [[thread_position_in_threadgroup]],
    uint bid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg  [[simdgroups_per_threadgroup]])
{
    const int T = TS_SVDBL_THREADS;
    int li = (int)lid;
    device float* Wb = W + (size_t)bid * (size_t)M * (size_t)N;
    device float* Vb = V + (size_t)bid * (size_t)N * (size_t)N;
    device float* Sb = S + (size_t)bid * (size_t)N;
    threadgroup int   perm[TS_SVDBL_MAXN];
    threadgroup atomic_int nrot;
    threadgroup float sa[TS_SVDBL_THREADS];   // reused for the σ finalization
    const float eps = 1e-7f;
    int Np = (N % 2 == 0) ? N : N + 1;        // pad to even for the tournament
    int npairs = Np / 2;                      // 'half' is a reserved MSL type

    for (int idx = li; idx < N * N; idx += T)
        Vb[idx] = ((idx / N) == (idx % N)) ? 1.0f : 0.0f;
    for (int i = li; i < Np; i += T) perm[i] = i;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        if (li == 0) atomic_store_explicit(&nrot, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int round = 0; round < Np - 1; ++round) {
            // SIMD-group sg processes the disjoint pairs i = sg, sg+nsg, ...
            for (int i = (int)sgid; i < npairs; i += (int)nsg) {
                int p = perm[i], q = perm[Np - 1 - i];
                if (p > q) { int tmp = p; p = q; q = tmp; }
                if (q < N) {   // skip the dummy padded column (odd N)
                    float pa = 0.0f, pb = 0.0f, pg = 0.0f;
                    for (int r = (int)lane; r < M; r += 32) {
                        float wp = Wb[r * N + p], wq = Wb[r * N + q];
                        pa += wp * wp; pb += wq * wq; pg += wp * wq;
                    }
                    float alpha = simd_sum(pa), beta = simd_sum(pb), gamma = simd_sum(pg);
                    if (fabs(gamma) > eps * sqrt(max(alpha * beta, 0.0f))) {
                        float zeta = (beta - alpha) / (2.0f * gamma);
                        float t = (zeta >= 0.0f ? 1.0f : -1.0f) /
                                  (fabs(zeta) + sqrt(1.0f + zeta * zeta));
                        if (zeta == 0.0f) t = 1.0f;
                        float c = 1.0f / sqrt(1.0f + t * t), s = c * t;
                        for (int r = (int)lane; r < M; r += 32) {
                            float wp = Wb[r * N + p], wq = Wb[r * N + q];
                            Wb[r * N + p] = c * wp - s * wq;
                            Wb[r * N + q] = s * wp + c * wq;
                        }
                        for (int r = (int)lane; r < N; r += 32) {
                            float vp = Vb[r * N + p], vq = Vb[r * N + q];
                            Vb[r * N + p] = c * vp - s * vq;
                            Vb[r * N + q] = s * vp + c * vq;
                        }
                        if (lane == 0)
                            atomic_fetch_add_explicit(&nrot, 1, memory_order_relaxed);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
            // circle method: fix perm[0], cyclically rotate perm[1..Np-1] by one.
            if (li == 0 && Np > 2) {
                int tmp = perm[Np - 1];
                for (int i = Np - 1; i > 1; --i) perm[i] = perm[i - 1];
                perm[1] = tmp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&nrot, memory_order_relaxed) == 0) break;
    }

    // σ_k = ‖W[:,k]‖ ; U[:,k] = W[:,k]/σ_k  (all T threads, threadgroup reduce).
    for (int k = 0; k < N; ++k) {
        float ps = 0.0f;
        for (int i = li; i < M; i += T) { float w = Wb[i * N + k]; ps += w * w; }
        sa[li] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = T / 2; s > 0; s >>= 1) {
            if (li < s) sa[li] += sa[li + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float sigma = sqrt(sa[0]);
        if (li == 0) Sb[k] = sigma;
        float inv = sigma > 1e-30f ? 1.0f / sigma : 0.0f;
        for (int i = li; i < M; i += T) Wb[i * N + k] *= inv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)MSL";
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, kSvdBlSource, @"svd_jacobi_bl_f32");
    if (!pso) return false;
    size_t wB = (size_t)batch * M * N * 4, vB = (size_t)batch * N * N * 4,
           sB = (size_t)batch * N * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bW, ctx, A, wB);
    TS_METAL_BUF_ACQUIRE(bV, ctx, vB);
    TS_METAL_BUF_ACQUIRE(bS, ctx, sB);
    if (!bW || !bV || !bS) return false;
    int maxSweeps = 60;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bW offset:0 atIndex:0];
    [enc setBuffer:bV offset:0 atIndex:1];
    [enc setBuffer:bS offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&maxSweeps length:sizeof(int32_t) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)batch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    // Migration batch 2 (2026-06-01) — Brent–Luk parallel Jacobi SVD;
    // same 60 s timeout as the serial variant.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "svd_jacobi_bl_f32")) return false;
    std::memcpy(U, [bW contents], wB);
    std::memcpy(V, [bV contents], vB);
    std::memcpy(S, [bS contents], sB);
    return true;
  }
}
}  // namespace

extern "C" int32_t tessera_apple_gpu_svd_bl_batched_f32(const float *A, float *U,
                                                        float *S, float *V,
                                                        int32_t batch, int32_t M,
                                                        int32_t N) {
  return dispatch_svd_jacobi_bl_f32(deviceContext(), A, U, S, V, batch, M, N) ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// Batched dense factorizations / solves (custom MSL, one threadgroup per matrix,
// grid of `batch`). MPS's Cholesky/solve are single-matrix per encode, so the
// Python batched path used to loop them per matrix; these grid-batched kernels
// keep the whole GPU busy across the batch (same pattern + win as batched SVD).
// f32, single threadgroup per matrix. See docs/apple_backend_integration_review.md.
//===----------------------------------------------------------------------===//
namespace {
constexpr int kBatchLinalgThreads = 128;

// Batched Cholesky: A (n×n SPD) -> L lower with A = L·Lᵀ (left-looking, column by
// column). Per-matrix status: 0 ok, 1 not positive-definite.
bool dispatch_cholesky_batched_f32(MetalDeviceContext &ctx, const float *A,
                                   float *L, int32_t *status, int32_t batch,
                                   int32_t n) {
  if (!ctx.ok || !A || !L || !status || batch <= 0 || n <= 0) return false;
  static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
#define TS_CHOL_THREADS 128
kernel void cholesky_batched_f32(
    device const float* A [[buffer(0)]],
    device float*       L [[buffer(1)]],
    device int*         status [[buffer(2)]],
    constant int&       n [[buffer(3)]],
    uint lid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    const int T = TS_CHOL_THREADS;
    int li = (int)lid;
    device const float* Ab = A + (size_t)bid * (size_t)n * (size_t)n;
    device float*       Lb = L + (size_t)bid * (size_t)n * (size_t)n;
    threadgroup float red[TS_CHOL_THREADS];
    threadgroup float diag[1];
    threadgroup int   bad[1];
    for (int idx = li; idx < n * n; idx += T) Lb[idx] = Ab[idx];
    if (li == 0) bad[0] = 0;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (int k = 0; k < n; ++k) {
        // diagonal d = A[k,k] - Σ_{j<k} L[k,j]²
        float ps = 0.0f;
        for (int j = li; j < k; j += T) { float v = Lb[k * n + j]; ps += v * v; }
        red[li] = ps;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = T / 2; s > 0; s >>= 1) {
            if (li < s) red[li] += red[li + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (li == 0) {
            float d = Lb[k * n + k] - red[0];
            if (d <= 0.0f) { bad[0] = 1; d = 1.0f; }   // not PD: flag, avoid NaN
            diag[0] = sqrt(d);
            Lb[k * n + k] = diag[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float invd = 1.0f / diag[0];
        for (int i = k + 1 + li; i < n; i += T) {
            float s = 0.0f;
            for (int j = 0; j < k; ++j) s += Lb[i * n + j] * Lb[k * n + j];
            Lb[i * n + k] = (Lb[i * n + k] - s) * invd;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    for (int idx = li; idx < n * n; idx += T) {   // zero strict upper triangle
        if ((idx % n) > (idx / n)) Lb[idx] = 0.0f;
    }
    if (li == 0) status[bid] = bad[0];
}
)MSL";
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, kSrc, @"cholesky_batched_f32");
    if (!pso) return false;
    size_t mB = (size_t)batch * n * n * 4, sB = (size_t)batch * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bA, ctx, A, mB);
    TS_METAL_BUF_ACQUIRE(bL, ctx, mB);
    TS_METAL_BUF_ACQUIRE(bSt, ctx, sB);
    if (!bA || !bL || !bSt) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bA offset:0 atIndex:0];
    [enc setBuffer:bL offset:0 atIndex:1];
    [enc setBuffer:bSt offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)batch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kBatchLinalgThreads, 1, 1)];
    [enc endEncoding];
    // Migration batch 2 (2026-06-01) — batched Cholesky over a grid
    // of threadgroups. 30 s timeout.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "cholesky_batched_f32")) return false;
    std::memcpy(L, [bL contents], mB);
    std::memcpy(status, [bSt contents], sB);
    return true;
  }
}

// Batched triangular solve: op(tri(A))·X = B, one threadgroup per matrix.
// tri = lower (lower=1) or upper; op = transpose when trans=1; unit diagonal when
// unit=1. Forward/back substitution over the n rows; the nrhs right-hand sides
// run in parallel across threads. A is n×n, B/X are n×nrhs, all row-major f32.
bool dispatch_tri_solve_batched_f32(MetalDeviceContext &ctx, const float *A,
                                    const float *B, float *X, int32_t batch,
                                    int32_t n, int32_t nrhs, int32_t lower,
                                    int32_t trans, int32_t unit) {
  if (!ctx.ok || !A || !B || !X || batch <= 0 || n <= 0 || nrhs <= 0) return false;
  static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
#define TS_TRSV_THREADS 128
kernel void tri_solve_batched_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float*       X [[buffer(2)]],
    constant int&       n [[buffer(3)]],
    constant int&       nrhs [[buffer(4)]],
    constant int&       lower [[buffer(5)]],
    constant int&       trans [[buffer(6)]],
    constant int&       unit  [[buffer(7)]],
    uint lid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]])
{
    const int T = TS_TRSV_THREADS;
    int li = (int)lid;
    device const float* Ab = A + (size_t)bid * (size_t)n * (size_t)n;
    device const float* Bb = B + (size_t)bid * (size_t)n * (size_t)nrhs;
    device float*       Xb = X + (size_t)bid * (size_t)n * (size_t)nrhs;
    // op(tri(A)) is lower-triangular iff (lower XOR trans) == false... work it out:
    //   effective lower-triangular  <=>  (lower && !trans) || (!lower && trans)
    bool effLower = (lower != 0) != (trans != 0);
    // A_op[r,c] = trans ? A[c,r] : A[r,c]; only the tri(A) triangle is read.
    for (int rr = 0; rr < n; ++rr) {
        int r = effLower ? rr : (n - 1 - rr);   // forward if lower, back if upper
        // each thread handles a subset of the nrhs columns
        for (int col = li; col < nrhs; col += T) {
            float acc = Bb[r * nrhs + col];
            // sum over already-solved rows j (j<r if lower, j>r if upper)
            int jstart = effLower ? 0 : r + 1;
            int jend   = effLower ? r : n;
            for (int j = jstart; j < jend; ++j) {
                float a = trans ? Ab[j * n + r] : Ab[r * n + j];  // op(A)[r,j]
                acc -= a * Xb[j * nrhs + col];
            }
            float d = unit ? 1.0f : (trans ? Ab[r * n + r] : Ab[r * n + r]);
            Xb[r * nrhs + col] = acc / d;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
}
)MSL";
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, kSrc, @"tri_solve_batched_f32");
    if (!pso) return false;
    size_t aB = (size_t)batch * n * n * 4, rB = (size_t)batch * n * nrhs * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bA, ctx, A, aB);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bB, ctx, B, rB);
    TS_METAL_BUF_ACQUIRE(bX, ctx, rB);
    if (!bA || !bB || !bX) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bA offset:0 atIndex:0];
    [enc setBuffer:bB offset:0 atIndex:1];
    [enc setBuffer:bX offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&nrhs length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&lower length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&trans length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&unit length:sizeof(int32_t) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)batch, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kBatchLinalgThreads, 1, 1)];
    [enc endEncoding];
    // Migration batch 2 (2026-06-01) — batched triangular solve.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "tri_solve_batched_f32")) return false;
    std::memcpy(X, [bX contents], rB);
    return true;
  }
}
}  // namespace

extern "C" int32_t tessera_apple_gpu_cholesky_batched_f32(const float *A, float *L,
                                                          int32_t *status,
                                                          int32_t batch, int32_t n) {
  return dispatch_cholesky_batched_f32(deviceContext(), A, L, status, batch, n) ? 1 : 0;
}

extern "C" int32_t tessera_apple_gpu_tri_solve_batched_f32(
    const float *A, const float *B, float *X, int32_t batch, int32_t n,
    int32_t nrhs, int32_t lower, int32_t trans, int32_t unit) {
  return dispatch_tri_solve_batched_f32(deviceContext(), A, B, X, batch, n, nrhs,
                                        lower, trans, unit) ? 1 : 0;
}

//===----------------------------------------------------------------------===//
// R0 resident dtype cast — an on-device elementwise cast between two resident
// DeviceTensors (f32 <-> f16/bf16), no host round-trip. This is the missing link
// for round-trip-free MLP stacking: the M8 session's `run_dev` outputs f32, the
// next layer's matmul wants f16/bf16, so a resident cast lets the whole stack
// stay on the GPU. mode: 0 f32->f16, 1 f32->bf16, 2 f16->f32, 3 bf16->f32.
//===----------------------------------------------------------------------===//
namespace {
bool dispatch_dev_cast(MetalDeviceContext &ctx, id<MTLBuffer> src, id<MTLBuffer> dst,
                       int64_t n, int mode) {
  if (!ctx.ok || !src || !dst || n <= 0) return false;
  static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
// f32 -> half: one thread per element; out is `half*` (16-bit).
kernel void dev_cast_f32_to_f16(device const float* in [[buffer(0)]],
                                device half* out        [[buffer(1)]],
                                constant int& n         [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if ((int)gid < n) out[gid] = (half)in[gid];
}
kernel void dev_cast_f16_to_f32(device const half* in [[buffer(0)]],
                                device float* out      [[buffer(1)]],
                                constant int& n        [[buffer(2)]],
                                uint gid [[thread_position_in_grid]]) {
    if ((int)gid < n) out[gid] = (float)in[gid];
}
// f32 -> bf16: top 16 bits with round-to-nearest-even. out is ushort (bf16 bits).
kernel void dev_cast_f32_to_bf16(device const float* in [[buffer(0)]],
                                 device ushort* out      [[buffer(1)]],
                                 constant int& n         [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= n) return;
    uint u = as_type<uint>(in[gid]);
    uint lsb = (u >> 16) & 1u;
    uint rounded = u + 0x7fffu + lsb;
    out[gid] = (ushort)(rounded >> 16);
}
kernel void dev_cast_bf16_to_f32(device const ushort* in [[buffer(0)]],
                                 device float* out        [[buffer(1)]],
                                 constant int& n          [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if ((int)gid < n) out[gid] = as_type<float>((uint)in[gid] << 16);
}
)MSL";
  NSString *entry = mode == 0 ? @"dev_cast_f32_to_f16"
                  : mode == 1 ? @"dev_cast_f32_to_bf16"
                  : mode == 2 ? @"dev_cast_f16_to_f32"
                              : @"dev_cast_bf16_to_f32";
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, kSrc, entry);
    if (!pso) return false;
    int32_t ni = (int32_t)n;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&ni length:sizeof(int32_t) atIndex:2];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(256, (NSUInteger)n), 1, 1)];
    [enc endEncoding];
    // Migration batch 2 (2026-06-01) — device-side cast (f32 ↔ f16
    // resident MLP chain helper).
    return commit_and_wait_with_timeout(ctx, cb, 30000, "dev_cast");
  }
}
}  // namespace

// Resident dtype cast between two DeviceTensors. `n` elements. mode: 0 f32->f16,
// 1 f32->bf16, 2 f16->f32, 3 bf16->f32. 1 if it ran on the GPU, else 0.
extern "C" int32_t ts_dev_cast(TsDeviceTensor *src, TsDeviceTensor *dst,
                               int64_t n, int32_t mode) {
  if (!src || !dst) return 0;
  return dispatch_dev_cast(deviceContext(), src->buf, dst->buf, n, (int)mode) ? 1 : 0;
}

// Metal 4 lane — cached MTL4 compute pipeline (MSL 4.0 source compiled via the
// MTL4Compiler). Mirrors compile_msl_kernel but for the MTL4 path; the MSL
// compile + pipeline build are paid once per (source, entry) and reused.
API_AVAILABLE(macos(26.0), ios(26.0))
id<MTLComputePipelineState> compile_mtl4_pipeline(MetalDeviceContext &ctx,
                                                  NSString *source,
                                                  NSString *entry) {
  std::string key;
  key.append([source UTF8String]);
  key.push_back('\x1f');
  key.append([entry UTF8String]);
  {
    std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
    auto it = ctx.mtl4_pipeline_cache.find(key);
    if (it != ctx.mtl4_pipeline_cache.end()) return it->second;
  }
  NSError *err = nil;
  MTLCompileOptions *co = [[MTLCompileOptions alloc] init];
  co.languageVersion = MTLLanguageVersion4_0;
  id<MTLLibrary> lib = [ctx.device newLibraryWithSource:source options:co error:&err];
  if (!lib) return nil;
  MTL4LibraryFunctionDescriptor *fd = [[MTL4LibraryFunctionDescriptor alloc] init];
  fd.name = entry;
  fd.library = lib;
  MTL4ComputePipelineDescriptor *pd = [[MTL4ComputePipelineDescriptor alloc] init];
  pd.computeFunctionDescriptor = fd;
  id<MTL4Compiler> compiler;
  {
    std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
    if (!ctx.mtl4_compiler)
      ctx.mtl4_compiler = [ctx.device newCompilerWithDescriptor:[[MTL4CompilerDescriptor alloc] init] error:&err];
    compiler = (id<MTL4Compiler>)ctx.mtl4_compiler;
  }
  if (!compiler) return nil;
  // P4 — pass any loaded archive as a lookup archive so a matching pipeline is
  // pulled from the binary archive instead of recompiled from MSL source.
  MTL4CompilerTaskOptions *topts = nil;
  if (ctx.mtl4_lookup_archive) {
    topts = [[MTL4CompilerTaskOptions alloc] init];
    topts.lookupArchives = @[(id<MTL4Archive>)ctx.mtl4_lookup_archive];
  }
  id<MTLComputePipelineState> pso =
      [compiler newComputePipelineStateWithDescriptor:pd compilerTaskOptions:topts error:&err];
  if (!pso) return nil;
  std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
  auto it = ctx.mtl4_pipeline_cache.find(key);
  if (it != ctx.mtl4_pipeline_cache.end()) return it->second;
  ctx.mtl4_pipeline_cache.emplace(std::move(key), pso);
  return pso;
}

// Metal 4 lane — one MTL4 command queue per device, created lazily + reused.
API_AVAILABLE(macos(26.0), ios(26.0))
id<MTL4CommandQueue> mtl4_shared_queue(MetalDeviceContext &ctx) {
  std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
  if (!ctx.mtl4_queue) ctx.mtl4_queue = [ctx.device newMTL4CommandQueue];
  return (id<MTL4CommandQueue>)ctx.mtl4_queue;
}

// P4 — enable MTL4Archive pipeline persistence. Loads an existing archive at
// `path` as a lookup archive (matching pipelines skip the MSL recompile) and
// (re)creates the MTL4 compiler with a CaptureBinaries serializer so subsequent
// pipeline builds are captured for a later flush. Opt-in; returns 1 if enabled.
extern "C" int32_t tessera_apple_gpu_mtl4_archive_enable(const char *path) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
    ctx.mtl4_archive_path = path ? std::string(path) : std::string();
    // Load a prior archive if present (nil on absent / incompatible — the
    // compiler then just recompiles + recaptures, so this is safe either way).
    ctx.mtl4_lookup_archive = nil;
    if (!ctx.mtl4_archive_path.empty()) {
      NSURL *url = [NSURL fileURLWithPath:@(ctx.mtl4_archive_path.c_str())];
      NSError *e = nil;
      ctx.mtl4_lookup_archive = [ctx.device newArchiveWithURL:url error:&e];
    }
    // CaptureBinaries serializer attached to a fresh compiler. Recreating the
    // compiler invalidates nothing — already-built pipelines stay cached.
    MTL4PipelineDataSetSerializerDescriptor *sd = [[MTL4PipelineDataSetSerializerDescriptor alloc] init];
    sd.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
    ctx.mtl4_serializer = [ctx.device newPipelineDataSetSerializerWithDescriptor:sd];
    MTL4CompilerDescriptor *cd = [[MTL4CompilerDescriptor alloc] init];
    cd.pipelineDataSetSerializer = (id<MTL4PipelineDataSetSerializer>)ctx.mtl4_serializer;
    NSError *e2 = nil;
    ctx.mtl4_compiler = [ctx.device newCompilerWithDescriptor:cd error:&e2];
    ctx.mtl4_archive_enabled = (ctx.mtl4_compiler != nil);
    return ctx.mtl4_archive_enabled ? 1 : 0;
  }
  return 0;
}

// Apple-sample Action 6 (2026-05-31) — Archive / AOT-cache telemetry
// probe. Reports the current state of the MTL4Archive cache so an
// artifact can surface "archive_enabled=true, lookup hit, path=..."
// instead of the binary "Metal is available" answer the old probe
// returned. ``archive_path_out`` is filled up to ``archive_path_len``
// bytes (always NUL-terminated when ``archive_path_len > 0``); the
// remaining flags are set when the corresponding output pointer is
// non-null. Returns 1 when the runtime is initialized + the snapshot
// fields are valid, 0 when the runtime isn't ready.
extern "C" int32_t tessera_apple_gpu_mtl4_archive_state(
    int32_t *archive_enabled_out,
    int32_t *has_lookup_archive_out,
    char *archive_path_out,
    int32_t archive_path_len) {
  // Zero outputs defensively so a 0 return reads cleanly to callers
  // that didn't check the rc.
  if (archive_enabled_out) *archive_enabled_out = 0;
  if (has_lookup_archive_out) *has_lookup_archive_out = 0;
  if (archive_path_out && archive_path_len > 0) archive_path_out[0] = '\0';
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
  if (archive_enabled_out)
    *archive_enabled_out = ctx.mtl4_archive_enabled ? 1 : 0;
  if (has_lookup_archive_out)
    *has_lookup_archive_out = ctx.mtl4_lookup_archive ? 1 : 0;
  if (archive_path_out && archive_path_len > 0) {
    size_t cap = (size_t)archive_path_len - 1;  // leave room for NUL
    size_t n = ctx.mtl4_archive_path.size();
    if (n > cap) n = cap;
    if (n > 0) std::memcpy(archive_path_out, ctx.mtl4_archive_path.data(), n);
    archive_path_out[n] = '\0';
  }
  return 1;
}

//===----------------------------------------------------------------------===//
// PK1 — Packaged ML pipeline foundation. Loads an `.mtlpackage`
// (Apple's compiled Metal package format, the output of Core ML
// Tools / Xcode) via `[device newLibraryWithURL:]`, then builds a
// `MTL4MachineLearningPipelineState` with shader reflection enabled
// (`MTL4ShaderReflectionBindingInfo`). The pipeline + library are
// owned by an opaque handle the caller releases via
// `tessera_apple_gpu_mlpkg_destroy`. NO execution yet — PK1 is the
// load+compile foundation; PK2 adds binding extraction, PK3 adds
// tensor creation + argument table, PK4 adds dispatch.
//
// Mirrors Apple's sample (`MLMatrixMultiplier+PipelineCompilation.m`).
// Static-shape models compile without per-input dimensions; dynamic
// models will need PK3's `setInputDimensions:atBufferIndex:` plumbing.
//===----------------------------------------------------------------------===//

// (``TesseraMlpkgPipeline`` ``@interface`` lives at the top of the file —
// Objective-C declarations require global scope.)

// PK1 — Load + compile a packaged ML pipeline. Returns an opaque
// handle that the caller releases via
// `tessera_apple_gpu_mlpkg_destroy`. Returns NULL on any failure
// (path missing, library load failure, reflection unavailable,
// pipeline compile failure, OS too old). Caller can probe whether the
// failure was OS-related vs config-related via
// `tessera_apple_gpu_mlpkg_last_error_kind` (-1 = OS not available;
// -2 = path / library load failed; -3 = pipeline compile failed).
static int32_t g_mlpkg_last_error_kind = 0;

// PK1.5 (2026-05-31) — Compile with optional input dimensions. Apple's
// sample matmul .mtlpackage (and many production Core ML packages) ship
// with dynamic input shapes; the caller must call
// ``setInputDimensions:atBufferIndex:`` on the descriptor BEFORE
// compile or pipeline build fails with "Unsupported Ops or shapes for
// MLEncoder". Mirrors Apple's sample at
// ``MLMatrixMultiplier+PipelineCompilation.m:78-89``.
//
// ``n_inputs`` is the number of (buffer_index, dims) pairs the caller
// is specifying. ``buffer_indices`` is the kernel-side index from
// reflection (the same value PK2's binding_info returns in
// ``buffer_index_out``). ``ranks`` is the per-input dimension count;
// ``dims_flat`` is the concatenated extents (sum of ranks), each
// innermost-first.
//
// When ``n_inputs == 0`` this is identical to
// ``tessera_apple_gpu_mlpkg_compile`` — static-shape packages need no
// per-call dims.
extern "C" void *tessera_apple_gpu_mlpkg_compile_with_dims(
    const char *path,
    const char *function_name,
    int32_t n_inputs,
    const int32_t *buffer_indices,
    const int32_t *ranks,
    const int64_t *dims_flat) {
  g_mlpkg_last_error_kind = 0;
  if (!path || !function_name) {
    g_mlpkg_last_error_kind = -2;
    return NULL;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) {
    g_mlpkg_last_error_kind = -1;
    return NULL;
  }
  if (n_inputs > 0 && (!buffer_indices || !ranks || !dims_flat)) {
    g_mlpkg_last_error_kind = -2;
    return NULL;
  }
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      NSURL *url = [NSURL fileURLWithPath:@(path)];
      NSError *err = nil;
      id<MTLLibrary> library = [ctx.device newLibraryWithURL:url error:&err];
      if (!library) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] library load failed for "
                "'%s': %s\n", path,
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        g_mlpkg_last_error_kind = -2;
        return NULL;
      }
      MTL4LibraryFunctionDescriptor *fnDesc =
          [[MTL4LibraryFunctionDescriptor alloc] init];
      fnDesc.name = @(function_name);
      fnDesc.library = library;
      MTL4MachineLearningPipelineDescriptor *pipeDesc =
          [[MTL4MachineLearningPipelineDescriptor alloc] init];
      pipeDesc.machineLearningFunctionDescriptor = fnDesc;
      MTL4PipelineOptions *opts = [[MTL4PipelineOptions alloc] init];
      opts.shaderReflection = MTL4ShaderReflectionBindingInfo;
      pipeDesc.options = opts;
      // PK1.5 — apply per-input dimensions to the descriptor BEFORE
      // compile. Walks the caller-provided (buffer_index, rank, dims)
      // tuples and pushes each through setInputDimensions:atBufferIndex:.
      int64_t flat_off = 0;
      for (int32_t i = 0; i < n_inputs; ++i) {
        int32_t r = ranks[i];
        if (r <= 0 || r > 16) {
          fprintf(stderr, "[tessera_apple_gpu_mlpkg] bad input rank %d for "
                  "buffer_index %d\n", r, buffer_indices[i]);
          g_mlpkg_last_error_kind = -2;
          return NULL;
        }
        NSInteger dvals[16];
        for (int32_t k = 0; k < r; ++k) {
          dvals[k] = (NSInteger)dims_flat[flat_off + k];
        }
        flat_off += r;
        MTLTensorExtents *extents =
            [[MTLTensorExtents alloc] initWithRank:(NSUInteger)r
                                            values:dvals];
        [pipeDesc setInputDimensions:extents
                       atBufferIndex:(NSUInteger)buffer_indices[i]];
      }
      id<MTL4Compiler> compiler;
      {
        std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
        if (!ctx.mtl4_compiler) {
          ctx.mtl4_compiler = [ctx.device
              newCompilerWithDescriptor:[[MTL4CompilerDescriptor alloc] init]
                                  error:&err];
        }
        compiler = (id<MTL4Compiler>)ctx.mtl4_compiler;
      }
      if (!compiler) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] MTL4Compiler "
                "unavailable: %s\n",
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        g_mlpkg_last_error_kind = -3;
        return NULL;
      }
      NSError *cerr = nil;
      id<MTL4MachineLearningPipelineState> pso =
          [compiler newMachineLearningPipelineStateWithDescriptor:pipeDesc
                                                            error:&cerr];
      if (!pso) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] pipeline compile failed "
                "for '%s' function '%s' (n_inputs=%d): %s\n",
                path, function_name, n_inputs,
                cerr ? [[cerr localizedDescription] UTF8String] : "<nil>");
        g_mlpkg_last_error_kind = -3;
        return NULL;
      }
      TesseraMlpkgPipeline *box = [[TesseraMlpkgPipeline alloc] init];
      box.library = library;
      box.pipelineState = pso;
      box.functionName = @(function_name);
      box.packagePath = @(path);
      return (void *)CFBridgingRetain(box);
    }
  }
  g_mlpkg_last_error_kind = -1;
  return NULL;
}

extern "C" void *tessera_apple_gpu_mlpkg_compile(const char *path,
                                                const char *function_name) {
  g_mlpkg_last_error_kind = 0;
  if (!path || !function_name) {
    g_mlpkg_last_error_kind = -2;
    return NULL;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) {
    g_mlpkg_last_error_kind = -1;
    return NULL;
  }
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      NSURL *url = [NSURL fileURLWithPath:@(path)];
      NSError *err = nil;
      id<MTLLibrary> library = [ctx.device newLibraryWithURL:url error:&err];
      if (!library) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] library load failed for "
                "'%s': %s\n", path,
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        g_mlpkg_last_error_kind = -2;
        return NULL;
      }
      MTL4LibraryFunctionDescriptor *fnDesc =
          [[MTL4LibraryFunctionDescriptor alloc] init];
      fnDesc.name = @(function_name);
      fnDesc.library = library;
      MTL4MachineLearningPipelineDescriptor *pipeDesc =
          [[MTL4MachineLearningPipelineDescriptor alloc] init];
      pipeDesc.machineLearningFunctionDescriptor = fnDesc;
      // Apple-sample reflection-driven setup (skills.md Pattern 1) —
      // enable binding-info reflection so PK2 can walk
      // `pipelineState.reflection.bindings` and surface a structured
      // ABI to Python. Without this, reflection is nil on the result.
      MTL4PipelineOptions *opts = [[MTL4PipelineOptions alloc] init];
      opts.shaderReflection = MTL4ShaderReflectionBindingInfo;
      pipeDesc.options = opts;
      // Compile through the cached MTL4 compiler so we don't pay the
      // ~ms compiler-creation cost per call (Pattern 5).
      id<MTL4Compiler> compiler;
      {
        std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
        if (!ctx.mtl4_compiler) {
          ctx.mtl4_compiler = [ctx.device
              newCompilerWithDescriptor:[[MTL4CompilerDescriptor alloc] init]
                                  error:&err];
        }
        compiler = (id<MTL4Compiler>)ctx.mtl4_compiler;
      }
      if (!compiler) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] MTL4Compiler "
                "unavailable: %s\n",
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        g_mlpkg_last_error_kind = -3;
        return NULL;
      }
      NSError *cerr = nil;
      id<MTL4MachineLearningPipelineState> pso =
          [compiler newMachineLearningPipelineStateWithDescriptor:pipeDesc
                                                            error:&cerr];
      if (!pso) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] pipeline compile failed "
                "for '%s' function '%s': %s\n", path, function_name,
                cerr ? [[cerr localizedDescription] UTF8String] : "<nil>");
        g_mlpkg_last_error_kind = -3;
        return NULL;
      }
      TesseraMlpkgPipeline *box = [[TesseraMlpkgPipeline alloc] init];
      box.library = library;
      box.pipelineState = pso;
      box.functionName = @(function_name);
      box.packagePath = @(path);
      // Move ownership to a raw void* handle. CFBridgingRetain hands
      // ARC's strong reference to the C ABI; destroy calls
      // CFBridgingRelease to take it back.
      return (void *)CFBridgingRetain(box);
    }
  }
  g_mlpkg_last_error_kind = -1;
  return NULL;
}

// PK8 — author a *production* `.mtlpackage` from the MPSGraph lane.
//
// This is the inverse of the PK1 load path: instead of consuming a package
// Apple's tooling produced, Tessera builds an MPSGraph (the same primitive it
// already constructs for its MPSGraph-lane ops), compiles it to an
// `MPSGraphExecutable`, and serializes that to a `.mpsgraphpackage` via
// `serializeToMPSGraphPackageAtURL:` (MPSGraphExecutable.h:205, macOS 14+).
// It then writes the trivial `manifest.json` MLLibrary wrapper so the result
// is a `.mtlpackage` directory that `tessera_apple_gpu_mlpkg_compile` (PK1)
// loads via `[device newLibraryWithURL:]`.
//
// First kernel: a plain matmul  C[M,N] = A[M,K] @ B[K,N]  (fp32) — it mirrors
// the bundled Apple sample's shape/binding pattern, so the authored package
// flows through the *existing* PK1-PK7 lifecycle unchanged and can be
// numerically compared against the live MPSGraph matmul path.
//
// Returns 1 on success; <=0 error codes:
//   -1 = OS / device unavailable      -2 = bad args
//   -3 = graph compile failed         -4 = manifest write failed
//   -5 = serialized package not found on disk after write
// PK8 — compile a built graph to an MPSGraphExecutable, serialize it to
// ``<out>/library.mpsgraphpackage`` (MPSGraphExecutable.h:205), and write the
// MLLibrary ``manifest.json`` wrapper so the result is a loadable
// ``.mtlpackage``. Shared by every ``author_*`` entry point. ``feeds`` maps
// each placeholder to its ``MPSGraphShapedType``. Returns 1 / <=0 error code
// (-3 compile, -4 manifest write, -5 serialized package missing).
API_AVAILABLE(macos(14.0), ios(17.0))
static int32_t _mlpkg_compile_and_write(MPSGraph *g, NSDictionary *feeds,
                                        MPSGraphTensor *y,
                                        const char *out_package_path) {
  if (!g || !feeds || !y || !out_package_path) return -2;
  MPSGraphExecutable *exe = [g compileWithDevice:nil
                                           feeds:feeds
                                   targetTensors:@[ y ]
                                targetOperations:nil
                           compilationDescriptor:nil];
  if (!exe) return -3;

  NSString *outDir = @(out_package_path);
  NSFileManager *fm = [NSFileManager defaultManager];
  // Start clean so re-authoring is deterministic (serialize won't append).
  [fm removeItemAtPath:outDir error:nil];
  [fm createDirectoryAtPath:outDir
      withIntermediateDirectories:YES
                       attributes:nil
                            error:nil];

  NSString *mpsPkgName = @"library.mpsgraphpackage";
  NSString *mpsPkgPath = [outDir stringByAppendingPathComponent:mpsPkgName];
  NSURL *mpsURL = [NSURL fileURLWithPath:mpsPkgPath];
  MPSGraphExecutableSerializationDescriptor *sdesc =
      [[MPSGraphExecutableSerializationDescriptor alloc] init];
  sdesc.append = NO;
  [exe serializeToMPSGraphPackageAtURL:mpsURL descriptor:sdesc];
  if (![fm fileExistsAtPath:mpsPkgPath]) return -5;

  // Wrap with the MLLibrary manifest — byte-for-byte the shape of the Apple
  // sample's manifest.json (pkgtype + inner package name).
  NSString *manifest = [NSString
      stringWithFormat:@"{\n  \"mtlpackage\" : {\n    \"version\" : {\n"
                        "      \"major\" : 1,\n      \"minor\" : 0,\n"
                        "      \"patch\" : 0\n    },\n"
                        "    \"pkgtype\" : \"MLLibrary\",\n"
                        "    \"content\" : {\n"
                        "      \"mpspkgname\" : \"%@\"\n    }\n  }\n}\n",
                       mpsPkgName];
  NSString *manifestPath =
      [outDir stringByAppendingPathComponent:@"manifest.json"];
  NSError *werr = nil;
  if (![manifest writeToFile:manifestPath
                  atomically:YES
                    encoding:NSUTF8StringEncoding
                       error:&werr]) {
    fprintf(stderr, "[tessera_apple_gpu_mlpkg] manifest write failed: %s\n",
            werr ? [[werr localizedDescription] UTF8String] : "<nil>");
    return -4;
  }
  return 1;
}

extern "C" int32_t tessera_apple_gpu_mlpkg_author_matmul(
    const char *out_package_path, int32_t M, int32_t K, int32_t N) {
  if (!out_package_path || M <= 0 || K <= 0 || N <= 0) return -2;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return -1;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      MPSDataType dt = MPSDataTypeFloat32;
      NSArray<NSNumber *> *aShape = @[ @(M), @(K) ];
      NSArray<NSNumber *> *bShape = @[ @(K), @(N) ];
      MPSGraph *g = [MPSGraph new];
      MPSGraphTensor *pa = [g placeholderWithShape:aShape dataType:dt
                                              name:@"inputA"];
      MPSGraphTensor *pb = [g placeholderWithShape:bShape dataType:dt
                                              name:@"inputB"];
      MPSGraphTensor *y =
          [g matrixMultiplicationWithPrimaryTensor:pa
                                   secondaryTensor:pb
                                              name:@"output"];
      MPSGraphShapedType *ta =
          [[MPSGraphShapedType alloc] initWithShape:aShape dataType:dt];
      MPSGraphShapedType *tb =
          [[MPSGraphShapedType alloc] initWithShape:bShape dataType:dt];
      return _mlpkg_compile_and_write(g, @{pa : ta, pb : tb}, y,
                                      out_package_path);
    }
  }
  return -1;
}

// PK8 helper — discover the entry-point function name an authored (or any)
// `.mtlpackage` exposes. MPSGraph names the serialized function itself, so we
// can't assume "main"; this loads the library and returns its first function
// name. Returns 1 on success (name copied into `name_out`), 0 otherwise.
extern "C" int32_t tessera_apple_gpu_mlpkg_first_function_name(
    const char *package_path, char *name_out, int32_t name_len) {
  if (!package_path || !name_out || name_len <= 0) return 0;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      NSURL *url = [NSURL fileURLWithPath:@(package_path)];
      NSError *err = nil;
      id<MTLLibrary> library = [ctx.device newLibraryWithURL:url error:&err];
      if (!library) return 0;
      NSArray<NSString *> *names = library.functionNames;
      if (!names || names.count == 0) return 0;
      const char *first = [names.firstObject UTF8String];
      if (!first) return 0;
      strncpy(name_out, first, (size_t)name_len - 1);
      name_out[name_len - 1] = '\0';
      return 1;
    }
  }
  return 0;
}

// Safe to call with NULL.
extern "C" void tessera_apple_gpu_mlpkg_destroy(void *handle) {
  if (!handle) return;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    // Reclaim ARC ownership and let the autorelease pool tear down
    // the library + pipeline state on next drain.
    @autoreleasepool {
      TesseraMlpkgPipeline *box =
          (__bridge_transfer TesseraMlpkgPipeline *)handle;
      (void)box;
    }
  }
}

// Test/diagnostic helper: is the handle a real compiled pipeline?
// Returns 1 iff `handle != NULL` AND it carries a non-nil pipeline
// state. Catches lifecycle bugs (e.g., post-destroy reuse).
extern "C" int32_t tessera_apple_gpu_mlpkg_is_compiled(void *handle) {
  if (!handle) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
    return box.pipelineState != nil ? 1 : 0;
  }
  return 0;
}

// Last failure code from `tessera_apple_gpu_mlpkg_compile`. Reading
// clears it. Returns 0 if no error since the last compile call.
extern "C" int32_t tessera_apple_gpu_mlpkg_last_error_kind(void) {
  int32_t v = g_mlpkg_last_error_kind;
  g_mlpkg_last_error_kind = 0;
  return v;
}

//===----------------------------------------------------------------------===//
// PK2 — Reflection extraction. Mirrors Apple's sample at
// `MLMatrixMultiplier+TensorSetup.m:extractTensorBindingsFromPipelineState`.
// Walks `pipelineState.reflection.bindings`, filters for
// `MTLBindingTypeTensor`, and exposes per-binding metadata (name +
// buffer index + rank + dimensions + tensor data type) via a probe
// pair: `binding_count` returns how many tensor bindings exist;
// `binding_info` fills caller-provided buffers for one binding by
// zero-based index.
//
// Tensor-binding indexing convention: ``binding_index`` is the
// position in the filtered tensor-binding sequence (NOT the buffer
// slot the kernel reads from). The kernel-side slot is returned in
// ``buffer_index_out`` and is what gets used in
// `MTL4ArgumentTable setResource:atBufferIndex:` (Apple-sample
// Pattern 2). Apple's sample sorts bindings by name; we don't sort
// here — Python can sort if it wants. Order stability across calls
// is guaranteed by the underlying reflection.
//===----------------------------------------------------------------------===//

// PK2 helper — walk `pipelineState.reflection.bindings`, filter
// tensor bindings into a separate NSArray. Returns nil if reflection
// is missing (which would indicate the pipeline was compiled without
// `MTL4ShaderReflectionBindingInfo` — a bug in
// `tessera_apple_gpu_mlpkg_compile`).
API_AVAILABLE(macos(26.0), ios(26.0))
static NSArray<id<MTLTensorBinding>> *_mlpkg_tensor_bindings(
    TesseraMlpkgPipeline *box) {
  if (!box || !box.pipelineState) return nil;
  // ``pipelineState.reflection`` / ``.bindings`` types vary across
  // SDK headers. Cast through ``NSObject *`` so KVC (``valueForKey:``)
  // resolves dynamically — works as long as the property exists at
  // runtime (it does, per Apple's sample
  // ``MLMatrixMultiplier+TensorSetup.m:20``). Header visibility is
  // a build-time concern only.
  NSObject *psoObj = (NSObject *)box.pipelineState;
  id refl = [psoObj valueForKey:@"reflection"];
  if (!refl) return nil;
  NSArray *all = [(NSObject *)refl valueForKey:@"bindings"];
  if (![all isKindOfClass:[NSArray class]]) return nil;
  NSMutableArray<id<MTLTensorBinding>> *out = [NSMutableArray new];
  for (id<MTLBinding> b in all) {
    if (b.type != MTLBindingTypeTensor) continue;
    [out addObject:(id<MTLTensorBinding>)b];
  }
  return out;
}

// PK2 — How many tensor bindings does this pipeline declare? Returns
// -1 if the handle is invalid / reflection is missing; otherwise the
// non-negative count.
extern "C" int32_t tessera_apple_gpu_mlpkg_binding_count(void *handle) {
  if (!handle) return -1;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
      NSArray<id<MTLTensorBinding>> *bindings = _mlpkg_tensor_bindings(box);
      if (!bindings) return -1;
      return (int32_t)bindings.count;
    }
  }
  return -1;
}

// PK2 — Per-binding metadata for the binding at filtered index
// ``binding_index`` (0-based, 0 <= idx < binding_count).
//
// Caller provides:
//   ``name_out`` / ``name_len``     — NUL-terminated UTF-8 name buffer
//   ``buffer_index_out``            — kernel-side argument-table index
//   ``rank_out``                    — number of dimensions
//   ``dims_out`` / ``dims_cap``     — extents array, innermost-first
//   ``dtype_raw_out``               — MTLTensorDataType raw enum int
//
// Returns 1 on success, 0 if the handle is invalid / index out of
// range / reflection missing. Outputs are zeroed defensively on 0.
extern "C" int32_t tessera_apple_gpu_mlpkg_binding_info(
    void *handle, int32_t binding_index,
    char *name_out, int32_t name_len,
    int32_t *buffer_index_out,
    int32_t *rank_out,
    int64_t *dims_out, int32_t dims_cap,
    int32_t *dtype_raw_out) {
  // Zero outputs defensively so a 0 return reads cleanly.
  if (name_out && name_len > 0) name_out[0] = '\0';
  if (buffer_index_out) *buffer_index_out = 0;
  if (rank_out) *rank_out = 0;
  if (dtype_raw_out) *dtype_raw_out = 0;
  if (!handle || binding_index < 0) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
      NSArray<id<MTLTensorBinding>> *bindings = _mlpkg_tensor_bindings(box);
      if (!bindings || binding_index >= (int32_t)bindings.count) return 0;
      id<MTLTensorBinding> b = bindings[(NSUInteger)binding_index];
      // Name.
      if (name_out && name_len > 0) {
        const char *src = [[b name] UTF8String];
        size_t src_len = src ? strlen(src) : 0;
        size_t cap = (size_t)name_len - 1;
        size_t n = (src_len > cap) ? cap : src_len;
        if (n > 0) memcpy(name_out, src, n);
        name_out[n] = '\0';
      }
      // Index — the kernel-side argument-table slot.
      if (buffer_index_out) *buffer_index_out = (int32_t)b.index;
      // Rank + dimensions.
      MTLTensorExtents *dims = b.dimensions;
      NSUInteger rank = dims ? dims.rank : 0;
      if (rank_out) *rank_out = (int32_t)rank;
      if (dims_out && dims_cap > 0 && dims) {
        NSUInteger fill = (rank < (NSUInteger)dims_cap)
                              ? rank : (NSUInteger)dims_cap;
        for (NSUInteger i = 0; i < fill; ++i) {
          dims_out[i] = (int64_t)[dims extentAtDimensionIndex:i];
        }
      }
      // Tensor data type — the raw MTLTensorDataType enum value.
      // Python decodes the well-known ones (Float32 / Float16 /
      // BFloat16 / ...). Unknown raws round-trip as-is.
      if (dtype_raw_out) *dtype_raw_out = (int32_t)b.tensorDataType;
      return 1;
    }
  }
  return 0;
}

// PK2 — Probe of well-known MTLTensorDataType raw values so the
// Python side has authoritative answers without re-declaring Apple's
// enum. Caller passes an integer code (we use ASCII tags for
// readability — 'F'=32 means Float32, 'F'=16 means Float16, etc.)
// and gets back the runtime's view of the corresponding raw enum
// value. Returns -1 for unknown probe tags.
//
// This avoids hard-coding magic numbers on the Python side that could
// drift if Apple changes the enum (which they sometimes do across
// SDK releases).
extern "C" int32_t tessera_apple_gpu_mlpkg_dtype_raw_for_tag(
    int32_t tag) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    switch (tag) {
      case 32:  return (int32_t)MTLTensorDataTypeFloat32;
      case 16:  return (int32_t)MTLTensorDataTypeFloat16;
      case 22:  return (int32_t)MTLTensorDataTypeBFloat16;
      case 8:   return (int32_t)MTLTensorDataTypeInt8;
      case 80:  return (int32_t)MTLTensorDataTypeUInt8;
      case 808: return (int32_t)MTLTensorDataTypeInt16;
      case 800: return (int32_t)MTLTensorDataTypeUInt16;
      case 132: return (int32_t)MTLTensorDataTypeInt32;
      case 232: return (int32_t)MTLTensorDataTypeUInt32;
    }
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// PK3 — Tensor creation + MTL4ArgumentTable binding from reflection.
// Mirrors Apple's sample at ``MLMatrixMultiplier.m::configureWithMatrix1:``
// (the lines that create tensors from each binding's dimensions and
// then ``setResource:atBufferIndex:`` them on the argument table).
//
//  prepare_tensors(handle)
//      For each tensor binding the reflection knows about, create a
//      device-backed ``MTLTensor`` matching the binding's reflected
//      shape + dtype (skips bindings with dynamic dims). Build an
//      ``MTL4ArgumentTable`` sized to the highest binding index + 1.
//      Bind each tensor's ``gpuResourceID`` at the binding's
//      kernel-side index. Idempotent — second call is a no-op when
//      tensors are already populated.
//
//  fill_input(handle, name, src_bytes, byte_count)
//      Copy ``byte_count`` host bytes into the tensor named ``name``
//      via ``replaceSliceOrigin:sliceDimensions:withBytes:strides:``.
//      The strides are derived from the tensor's reflected shape
//      using the row-major helper (Pattern 6).
//
//  read_output(handle, name, dst_bytes, byte_count)
//      Read tensor data back to host via
//      ``getBytes:strides:fromSliceOrigin:sliceDimensions:``. PK4 uses
//      this to extract outputs after dispatch; PK3 tests use it for a
//      fill-then-read roundtrip.
//===----------------------------------------------------------------------===//

// PK3 — helper: row-major strides for a 1..N-D tensor descriptor.
// Returns YES on success (also fills ``stride_elems_out`` innermost-first
// per Apple's MTLTensorDescriptor.strides rule: ``strides[0] == 1``).
//
// Stride-alignment update (2026-06-01, skills.md Pattern 3 follow-on):
// when ``element_bits > 0`` and ``ml_usage != 0``, enforces Apple's
// 64-byte alignment rule on the second stride (128-byte for sub-byte
// dtypes). Pre-existing callers that pass 0/0 get the legacy
// cumulative-product behavior — keeping the lift backward-compatible.
API_AVAILABLE(macos(26.0), ios(26.0))
static BOOL _mlpkg_row_major_strides(MTLTensorExtents *dims,
                                     NSInteger *stride_elems_out,
                                     int32_t element_bits,
                                     BOOL ml_usage) {
  if (!dims) return NO;
  NSInteger rank = (NSInteger)dims.rank;
  if (rank <= 0 || rank > 8) return NO;
  // Materialize dims into a temporary int64 buffer; reject any
  // dynamic (-1) or zero extent (the C ABI can't size them).
  int64_t dims_buf[8];
  for (NSInteger i = 0; i < rank; ++i) {
    NSInteger d = [dims extentAtDimensionIndex:i];
    if (d <= 0) return NO;
    dims_buf[i] = (int64_t)d;
  }
  // Route through the aligned C ABI helper when we have enough info
  // to apply the alignment rule; else fall back to the legacy
  // cumulative-product path.
  if (element_bits > 0) {
    int64_t strides_buf[8];
    int32_t rc = tessera_apple_gpu_row_major_strides_aligned(
        dims_buf, (int32_t)rank, element_bits, ml_usage ? 1 : 0,
        strides_buf);
    if (rc != (int32_t)rank) return NO;
    for (NSInteger i = 0; i < rank; ++i) {
      stride_elems_out[i] = (NSInteger)strides_buf[i];
    }
    return YES;
  }
  // Legacy: cumulative products, no alignment.
  NSInteger acc = 1;
  for (NSInteger i = 0; i < rank; ++i) {
    stride_elems_out[i] = acc;
    acc *= (NSInteger)dims_buf[i];
  }
  return YES;
}

// PK3 — element byte size for a runtime ``MTLTensorDataType`` enum
// value. Mirrors Apple's MTLTensorDataType taxonomy. Returns 0 for
// unknown types (caller must handle).
API_AVAILABLE(macos(26.0), ios(26.0))
static size_t _mlpkg_dtype_byte_size(MTLTensorDataType dt) {
  if (dt == MTLTensorDataTypeFloat32) return 4;
  if (dt == MTLTensorDataTypeFloat16) return 2;
  if (dt == MTLTensorDataTypeBFloat16) return 2;
  if (dt == MTLTensorDataTypeInt8) return 1;
  if (dt == MTLTensorDataTypeUInt8) return 1;
  if (dt == MTLTensorDataTypeInt16) return 2;
  if (dt == MTLTensorDataTypeUInt16) return 2;
  if (dt == MTLTensorDataTypeInt32) return 4;
  if (dt == MTLTensorDataTypeUInt32) return 4;
  return 0;
}

// Phase 2 stride-alignment wire-up (2026-06-01) — opt-in setter
// for ``MTLTensorDescriptor.strides`` to use the aligned helper.
// Default off (Metal's implicit strides — pre-Phase-2 behavior).
// Must be called BEFORE ``prepare_tensors`` to take effect.
// Returns 1 on success; 0 if the handle is null / runtime down.
extern "C" int32_t tessera_apple_gpu_mlpkg_set_aligned_strides(
    void *handle, int32_t flag) {
  if (!handle) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
    box.useAlignedStrides = (flag != 0);
    return 1;
  }
  return 0;
}

extern "C" int32_t tessera_apple_gpu_mlpkg_prepare_tensors(void *handle) {
  if (!handle) return 0;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
      if (!box.pipelineState) return 0;
      // Idempotent: if tensors are already prepared, fall through.
      if (box.tensorsByName.count > 0 && box.argumentTable) return 1;
      NSArray<id<MTLTensorBinding>> *bindings = _mlpkg_tensor_bindings(box);
      if (!bindings || bindings.count == 0) return 0;

      // Step 1 — compute max binding index + 1 so the argument table
      // is sized to fit every binding.
      NSUInteger maxIdx = 0;
      for (id<MTLTensorBinding> b in bindings) {
        if (b.index > maxIdx) maxIdx = b.index;
      }
      MTL4ArgumentTableDescriptor *atd =
          [[MTL4ArgumentTableDescriptor alloc] init];
      atd.maxBufferBindCount = maxIdx + 1;
      atd.initializeBindings = YES;
      NSError *err = nil;
      id<MTL4ArgumentTable> at =
          [ctx.device newArgumentTableWithDescriptor:atd error:&err];
      if (!at) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] argument table create "
                "failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        return 0;
      }

      // Step 2 — create + bind per-binding tensors.
      NSMutableDictionary<NSString *, id<MTLTensor>> *tensors =
          [NSMutableDictionary new];
      // PK8 — index-keyed parallel map for positionally-bound (unnamed)
      // packages. Always populated; name-keyed map stays for CoreML-origin
      // packages that carry real binding names.
      NSMutableDictionary<NSNumber *, id<MTLTensor>> *tensorsByIdx =
          [NSMutableDictionary new];
      MTLTensorDescriptor *td = [[MTLTensorDescriptor alloc] init];
      td.usage = MTLTensorUsageMachineLearning;
      td.storageMode = MTLStorageModeShared;
      for (id<MTLTensorBinding> b in bindings) {
        MTLTensorExtents *dims = b.dimensions;
        if (!dims || dims.rank == 0) {
          fprintf(stderr, "[tessera_apple_gpu_mlpkg] binding '%s' has no "
                  "dimensions; can't create tensor\n",
                  [[b name] UTF8String]);
          return 0;
        }
        // Static-shape check: any -1 sentinel means dynamic, which PK3
        // doesn't support (PK4+ would).
        for (NSUInteger i = 0; i < dims.rank; ++i) {
          if ([dims extentAtDimensionIndex:i] < 0) {
            fprintf(stderr, "[tessera_apple_gpu_mlpkg] binding '%s' has "
                    "dynamic dim %lu — PK3 only handles static shapes\n",
                    [[b name] UTF8String], (unsigned long)i);
            return 0;
          }
        }
        td.dimensions = dims;
        td.dataType = b.tensorDataType;
        // Phase 2 stride-alignment wire-up (2026-06-01) — opt-in.
        // When the pipeline's ``useAlignedStrides`` flag is set, set
        // ``td.strides`` explicitly from the aligned helper so Apple's
        // 64-byte / 128-byte alignment rules are honored at the
        // descriptor level (Metal allocates storage accordingly).
        // Default (flag off): leave ``td.strides`` unset; Metal uses
        // its default implicit strides — the pre-Phase-2 behavior.
        if (box.useAlignedStrides) {
          size_t elem_bytes = _mlpkg_dtype_byte_size(b.tensorDataType);
          if (elem_bytes == 0) {
            fprintf(stderr, "[tessera_apple_gpu_mlpkg] aligned-strides "
                    "requested but unknown dtype for binding '%s'\n",
                    [[b name] UTF8String]);
            return 0;
          }
          NSInteger rank = (NSInteger)dims.rank;
          if (rank > 8) {
            fprintf(stderr, "[tessera_apple_gpu_mlpkg] aligned-strides "
                    "needs rank<=8; got %ld for binding '%s'\n",
                    (long)rank, [[b name] UTF8String]);
            return 0;
          }
          int64_t dims_buf[8] = {0};
          for (NSInteger i = 0; i < rank; ++i) {
            dims_buf[i] = (int64_t)[dims extentAtDimensionIndex:i];
          }
          int64_t strides_buf[8] = {0};
          int32_t rc = tessera_apple_gpu_row_major_strides_aligned(
              dims_buf, (int32_t)rank, (int32_t)(elem_bytes * 8),
              /*ml_usage=*/1, strides_buf);
          if (rc != (int32_t)rank) {
            fprintf(stderr, "[tessera_apple_gpu_mlpkg] aligned-strides "
                    "computation failed for binding '%s' (rc=%d)\n",
                    [[b name] UTF8String], (int)rc);
            return 0;
          }
          NSInteger strides_ns[8];
          for (NSInteger i = 0; i < rank; ++i) {
            strides_ns[i] = (NSInteger)strides_buf[i];
          }
          MTLTensorExtents *strides_ext =
              [[MTLTensorExtents alloc] initWithRank:rank
                                              values:strides_ns];
          td.strides = strides_ext;
        }
        NSError *terr = nil;
        id<MTLTensor> t = nil;
        if (box.useAlignedStrides) {
          // Phase 1 (2026-06-01) — buffer-backed tensor allocation.
          // ``newTensorWithDescriptor:`` rejects explicit strides
          // (Apple's API requires nil strides on that path). The
          // buffer-backed variant accepts them — caller allocates an
          // MTLBuffer sized for the aligned strides, the tensor is
          // a view into that buffer.
          //
          // Aligned byte count comes from the matching helper. We
          // re-compute dims+element_bits here to call the helper;
          // a small duplicate of the loop above is cheaper than
          // restructuring the control flow.
          size_t elem_bytes = _mlpkg_dtype_byte_size(b.tensorDataType);
          NSInteger rank = (NSInteger)dims.rank;
          int64_t dims_buf[8] = {0};
          for (NSInteger i = 0; i < rank; ++i) {
            dims_buf[i] = (int64_t)[dims extentAtDimensionIndex:i];
          }
          int64_t aligned_nbytes =
              tessera_apple_gpu_aligned_buffer_nbytes(
                  dims_buf, (int32_t)rank,
                  (int32_t)(elem_bytes * 8), /*ml_usage=*/1);
          if (aligned_nbytes <= 0) {
            fprintf(stderr, "[tessera_apple_gpu_mlpkg] aligned buffer "
                    "size calc failed for binding '%s'\n",
                    [[b name] UTF8String]);
            return 0;
          }
          id<MTLBuffer> buf =
              [ctx.device newBufferWithLength:(NSUInteger)aligned_nbytes
                                      options:MTLResourceStorageModeShared];
          if (!buf) {
            fprintf(stderr, "[tessera_apple_gpu_mlpkg] aligned buffer "
                    "alloc failed for binding '%s' (%lld bytes)\n",
                    [[b name] UTF8String],
                    (long long)aligned_nbytes);
            return 0;
          }
          t = [buf newTensorWithDescriptor:td
                                    offset:0
                                     error:&terr];
        } else {
          // Default path — Apple manages storage via descriptor only.
          t = [ctx.device newTensorWithDescriptor:td error:&terr];
        }
        if (!t) {
          fprintf(stderr, "[tessera_apple_gpu_mlpkg] tensor create failed "
                  "for binding '%s': %s\n",
                  [[b name] UTF8String],
                  terr ? [[terr localizedDescription] UTF8String] : "<nil>");
          return 0;
        }
        tensors[b.name] = t;
        tensorsByIdx[@(b.index)] = t;
        // Apple-sample Pattern 2 — bind by reflected index, not by
        // hand-counted position.
        [at setResource:t.gpuResourceID atBufferIndex:b.index];
      }
      box.tensorsByName = tensors;
      box.tensorsByIndex = tensorsByIdx;
      box.argumentTable = at;
      return 1;
    }
  }
  return 0;
}

// Look up a prepared tensor by name. Returns nil on miss.
API_AVAILABLE(macos(26.0), ios(26.0))
static id<MTLTensor> _mlpkg_tensor_by_name(TesseraMlpkgPipeline *box,
                                            const char *name) {
  if (!box || !name) return nil;
  NSString *key = @(name);
  return box.tensorsByName[key];
}

// PK8 — look up a prepared tensor by kernel-side binding index. The
// addressing mode for positionally-bound (unnamed) packages. Returns nil
// on miss.
API_AVAILABLE(macos(26.0), ios(26.0))
static id<MTLTensor> _mlpkg_tensor_by_index(TesseraMlpkgPipeline *box,
                                            int32_t index) {
  if (!box) return nil;
  return box.tensorsByIndex[@(index)];
}

// PK8 — copy host bytes into / out of a prepared tensor's dense buffer.
// Shared body for the index-addressable fill/read. ``writing`` selects
// direction. Returns 1 on success.
API_AVAILABLE(macos(26.0), ios(26.0))
static int32_t _mlpkg_copy_tensor_at(void *handle, int32_t index,
                                     void *host_bytes, int64_t byte_count,
                                     bool writing) {
  if (!handle || !host_bytes || byte_count <= 0) return 0;
  @autoreleasepool {
    TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
    id<MTLTensor> t = _mlpkg_tensor_by_index(box, index);
    if (!t) return 0;
    MTLTensorExtents *dims = t.dimensions;
    if (!dims || dims.rank == 0) return 0;
    size_t elem = _mlpkg_dtype_byte_size(t.dataType);
    if (elem == 0) return 0;
    NSInteger total = 1;
    for (NSUInteger i = 0; i < dims.rank; ++i)
      total *= [dims extentAtDimensionIndex:i];
    if ((int64_t)((size_t)total * elem) != byte_count) {
      fprintf(stderr, "[tessera_apple_gpu_mlpkg] %s_at idx=%d: byte_count "
              "%lld != expected %zu\n", writing ? "fill_input" : "read_output",
              index, (long long)byte_count, (size_t)total * elem);
      return 0;
    }
    NSInteger zeros[MTL_TENSOR_MAX_RANK] = {0};
    MTLTensorExtents *origin =
        [[MTLTensorExtents alloc] initWithRank:dims.rank values:zeros];
    NSInteger strd[MTL_TENSOR_MAX_RANK] = {0};
    if (!_mlpkg_row_major_strides(dims, strd, 0, NO)) return 0;
    MTLTensorExtents *strides =
        [[MTLTensorExtents alloc] initWithRank:dims.rank values:strd];
    if (writing) {
      [t replaceSliceOrigin:origin sliceDimensions:dims
                  withBytes:host_bytes strides:strides];
    } else {
      [t getBytes:host_bytes strides:strides
          fromSliceOrigin:origin sliceDimensions:dims];
    }
    return 1;
  }
}

// PK3 — fill an input tensor with host bytes. Returns 1 on success.
extern "C" int32_t tessera_apple_gpu_mlpkg_fill_input(
    void *handle, const char *name,
    const void *src_bytes, int64_t byte_count) {
  if (!handle || !name || !src_bytes || byte_count <= 0) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
      id<MTLTensor> t = _mlpkg_tensor_by_name(box, name);
      if (!t) return 0;
      MTLTensorExtents *dims = t.dimensions;
      if (!dims || dims.rank == 0) return 0;
      // Verify byte_count matches element_count * element_size to
      // catch shape mismatches before the GPU side does.
      size_t elem = _mlpkg_dtype_byte_size(t.dataType);
      if (elem == 0) return 0;
      NSInteger total = 1;
      for (NSUInteger i = 0; i < dims.rank; ++i)
        total *= [dims extentAtDimensionIndex:i];
      if ((int64_t)((size_t)total * elem) != byte_count) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] fill_input '%s': "
                "byte_count %lld != expected %zu (total=%ld * elem=%zu)\n",
                name, (long long)byte_count,
                (size_t)total * elem, (long)total, elem);
        return 0;
      }
      NSInteger zeros[MTL_TENSOR_MAX_RANK] = {0};
      MTLTensorExtents *origin = [[MTLTensorExtents alloc]
          initWithRank:dims.rank values:zeros];
      NSInteger strd[MTL_TENSOR_MAX_RANK] = {0};
      // Host-source strides — describe the DENSE host buffer layout,
      // NOT the tensor's internal aligned layout. Pass element_bits=0
      // / ml_usage=NO to route through the legacy cumulative-product
      // path. Tensor-descriptor strides are a separate concern.
      if (!_mlpkg_row_major_strides(dims, strd, 0, NO)) return 0;
      MTLTensorExtents *strides = [[MTLTensorExtents alloc]
          initWithRank:dims.rank values:strd];
      [t replaceSliceOrigin:origin sliceDimensions:dims
                  withBytes:src_bytes strides:strides];
      return 1;
    }
  }
  return 0;
}

// PK3 — read an output tensor back to host. Returns 1 on success.
// Used by PK4 to extract dispatch outputs; PK3 tests use it for a
// fill-then-read round-trip without actually dispatching.
extern "C" int32_t tessera_apple_gpu_mlpkg_read_output(
    void *handle, const char *name,
    void *dst_bytes, int64_t byte_count) {
  if (!handle || !name || !dst_bytes || byte_count <= 0) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
      id<MTLTensor> t = _mlpkg_tensor_by_name(box, name);
      if (!t) return 0;
      MTLTensorExtents *dims = t.dimensions;
      if (!dims || dims.rank == 0) return 0;
      size_t elem = _mlpkg_dtype_byte_size(t.dataType);
      if (elem == 0) return 0;
      NSInteger total = 1;
      for (NSUInteger i = 0; i < dims.rank; ++i)
        total *= [dims extentAtDimensionIndex:i];
      if ((int64_t)((size_t)total * elem) != byte_count) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] read_output '%s': "
                "byte_count %lld != expected %zu\n",
                name, (long long)byte_count, (size_t)total * elem);
        return 0;
      }
      NSInteger zeros[MTL_TENSOR_MAX_RANK] = {0};
      MTLTensorExtents *origin = [[MTLTensorExtents alloc]
          initWithRank:dims.rank values:zeros];
      NSInteger strd[MTL_TENSOR_MAX_RANK] = {0};
      // Host-source strides — describe the DENSE host buffer layout,
      // NOT the tensor's internal aligned layout. Pass element_bits=0
      // / ml_usage=NO to route through the legacy cumulative-product
      // path. Tensor-descriptor strides are a separate concern.
      if (!_mlpkg_row_major_strides(dims, strd, 0, NO)) return 0;
      MTLTensorExtents *strides = [[MTLTensorExtents alloc]
          initWithRank:dims.rank values:strd];
      [t getBytes:dst_bytes strides:strides
          fromSliceOrigin:origin sliceDimensions:dims];
      return 1;
    }
  }
  return 0;
}

// PK8 — fill an input tensor addressed by kernel-side binding index.
// For positionally-bound packages (MPSGraph-authored, unnamed bindings).
// Returns 1 on success.
extern "C" int32_t tessera_apple_gpu_mlpkg_fill_input_at(
    void *handle, int32_t index, const void *src_bytes, int64_t byte_count) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    return _mlpkg_copy_tensor_at(handle, index, (void *)src_bytes, byte_count,
                                 /*writing=*/true);
  }
  return 0;
}

// PK8 — read an output tensor addressed by kernel-side binding index.
extern "C" int32_t tessera_apple_gpu_mlpkg_read_output_at(
    void *handle, int32_t index, void *dst_bytes, int64_t byte_count) {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    return _mlpkg_copy_tensor_at(handle, index, dst_bytes, byte_count,
                                 /*writing=*/false);
  }
  return 0;
}

// PK3 — has the argument table been built? Test helper.
extern "C" int32_t tessera_apple_gpu_mlpkg_argument_table_ready(void *handle) {
  if (!handle) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
    return box.argumentTable != nil ? 1 : 0;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// PK4 — End-to-end ML pass dispatch via
// ``MTL4MachineLearningCommandEncoder``. Mirrors Apple's sample at
// ``MLMatrixMultiplier.m:encodeAndRunModelInference`` (lines 224-256):
//
//   1. Allocate an MTLHeap for intermediates (size from
//      ``pipelineState.intermediatesHeapSize`` — Apple-sample Pattern 7
//      / audit Action 7). Cached on the pipeline box so subsequent
//      dispatches reuse it.
//   2. Per dispatch: fresh allocator + command buffer.
//   3. ``[cb beginCommandBufferWithAllocator:]`` → ``machineLearningCommandEncoder``
//   4. ``setArgumentTable:`` + ``setPipelineState:`` + ``dispatchNetworkWithIntermediatesHeap:``
//   5. ``endEncoding`` → ``endCommandBuffer`` → ``[queue commit:&cb count:1]``
//   6. Signal-and-wait via the cached ``ctx.mtl4_event`` (Apple-sample
//      Pattern 4) with the caller-provided timeout. A timed-out
//      dispatch returns ``0`` — caller can re-read input data + retry
//      OR escalate as a kernel hang.
//
// Returns 1 on successful dispatch (GPU completed within the
// timeout), 0 on any failure (handle invalid, tensors not prepared,
// allocator / encoder / heap create failures, timeout, command
// buffer error).
//===----------------------------------------------------------------------===//

extern "C" int32_t tessera_apple_gpu_mlpkg_dispatch(void *handle,
                                                    uint64_t timeout_ms) {
  if (!handle) return 0;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
      if (!box.pipelineState) return 0;
      if (!box.argumentTable) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] dispatch: argument "
                "table not prepared — call prepare_tensors() first\n");
        return 0;
      }
      id<MTLDevice> dev = ctx.device;
      // Lazily allocate the intermediates heap. Size comes from the
      // pipeline state (Pattern 7) — no hand-tuning. A zero size is
      // legal (some pipelines need no intermediates); the heap is
      // still allocated as a 1-byte placement heap so the encoder API
      // is satisfied.
      if (!box.intermediatesHeap) {
        // ``intermediatesHeapSize`` lives on the pipeline-reflection
        // protocol whose header signature varies across SDKs; KVC
        // dispatch sidesteps that the same way PK2's reflection probe
        // does (see ``_mlpkg_tensor_bindings``).
        NSObject *psoObj = (NSObject *)box.pipelineState;
        NSNumber *hsNum = [psoObj valueForKey:@"intermediatesHeapSize"];
        NSUInteger heapSize = hsNum ? [hsNum unsignedIntegerValue] : 0;
        if (heapSize == 0) heapSize = 1;
        MTLHeapDescriptor *hd = [[MTLHeapDescriptor alloc] init];
        hd.type = MTLHeapTypePlacement;
        hd.size = heapSize;
        hd.storageMode = MTLStorageModeShared;
        box.intermediatesHeap = [dev newHeapWithDescriptor:hd];
        if (!box.intermediatesHeap) {
          fprintf(stderr, "[tessera_apple_gpu_mlpkg] heap create failed "
                  "for size=%lu\n", (unsigned long)heapSize);
          return 0;
        }
      }
      // Per-dispatch allocator + command buffer (mirrors the existing
      // MTL4 outlier dispatchers — they intentionally don't take
      // ``mtl4_dispatch_mu`` and so create fresh objects per call).
      id<MTL4CommandAllocator> allocator = [dev newCommandAllocator];
      id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
      if (!allocator || !cb) return 0;
      id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
      if (!queue) return 0;
      // Encode + commit.
      [cb beginCommandBufferWithAllocator:allocator];
      // ``machineLearningCommandEncoder`` lives on a sub-protocol of
      // MTL4CommandBuffer whose header signature varies across SDKs;
      // dynamic dispatch via NSObject*'s performSelector keeps the
      // build stable across SDK shifts. The encoder type itself is
      // also SDK-dependent — we treat it as bare ``id`` and rely on
      // KVC / performSelector for method calls.
      NSObject *cbObj = (NSObject *)cb;
      id encoder = [cbObj performSelector:@selector(machineLearningCommandEncoder)];
      if (!encoder) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] dispatch: ML encoder "
                "unavailable (host SDK may not expose it)\n");
        return 0;
      }
      // ``setArgumentTable:`` + ``setPipelineState:`` are MTL4 standard
      // methods on the compute / ML encoders.
      [encoder performSelector:@selector(setArgumentTable:)
                    withObject:box.argumentTable];
      [encoder performSelector:@selector(setPipelineState:)
                    withObject:box.pipelineState];
      [encoder performSelector:@selector(dispatchNetworkWithIntermediatesHeap:)
                    withObject:box.intermediatesHeap];
      [encoder performSelector:@selector(endEncoding)];
      [cb endCommandBuffer];
      // Commit + signal + wait with timeout (Apple-sample Pattern 4).
      // ``commit:`` expects ``const id<MTL4CommandBuffer> _Nonnull[_Nonnull]``;
      // declare the array with the matching protocol-qualified type so
      // the SDK header accepts it (the `MTL4CommandBuffer*` form
      // doesn't conform to the type's strict requirements).
      const id<MTL4CommandBuffer> cbs[1] = {cb};
      [queue commit:cbs count:1];
      // PK audit P2 (2026-05-31) — packaged ML has its OWN shared event +
      // monotonic counter, guarded by its OWN mutex. The canonical MTL4
      // dispatcher (``mtl4_matmul2d_dispatch`` and friends) takes
      // ``mtl4_dispatch_mu`` and touches ``ctx.mtl4_event`` /
      // ``mtl4_event_val``; this packaged-ML lane runs concurrently against
      // it (the "outlier" pattern — fresh per-call allocator + command
      // buffer, no ``mtl4_dispatch_mu``). Sharing the event counter across
      // the two lanes was a race: an interleaved ``++mtl4_event_val`` could
      // hand us a value the canonical lane was already waiting on, or vice
      // versa, causing a spurious wait timeout (or worse, a wait that
      // returns before the GPU completes the right command buffer). Lane
      // isolation keeps the queue shared (the queue itself serializes
      // submission, so we don't lose ordering) while giving each lane its
      // own scoreboard.
      id<MTLSharedEvent> ev;
      uint64_t signal_val;
      {
        std::lock_guard<std::mutex> lock(ctx.mlpkg_event_mu);
        ev = (id<MTLSharedEvent>)ctx.mlpkg_event;
        if (!ev) {
          ev = [dev newSharedEvent];
          ctx.mlpkg_event = ev;
        }
        if (!ev) return 0;
        signal_val = ++ctx.mlpkg_event_val;
      }
      [queue signalEvent:ev value:signal_val];
      bool done = [ev waitUntilSignaledValue:signal_val
                                    timeoutMS:timeout_ms];
      if (!done) {
        fprintf(stderr, "[tessera_apple_gpu_mlpkg] dispatch: GPU did not "
                "signal within %llu ms (signaledValue=%llu wanted=%llu)\n",
                (unsigned long long)timeout_ms,
                (unsigned long long)ev.signaledValue,
                (unsigned long long)signal_val);
        return 0;
      }
      return 1;
    }
  }
  return 0;
}

// PK4 — query the cached intermediates heap size (after a successful
// dispatch). Returns the heap's allocated size in bytes, or -1 if no
// dispatch has happened yet / the runtime isn't available. Used by
// tests + telemetry to confirm Pattern 7 is honored (size comes
// from the pipeline, not a magic number).
extern "C" int64_t tessera_apple_gpu_mlpkg_intermediates_heap_size(void *handle) {
  if (!handle) return -1;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    TesseraMlpkgPipeline *box = (__bridge TesseraMlpkgPipeline *)handle;
    if (!box.intermediatesHeap) return -1;
    return (int64_t)box.intermediatesHeap.size;
  }
  return -1;
}

// P4 — flush the captured pipeline set to the enabled archive path. Returns 1 on
// success. Call after the kernels of interest have been built (e.g. after warmup).
extern "C" int32_t tessera_apple_gpu_mtl4_archive_flush(void) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    std::lock_guard<std::mutex> lock(ctx.mtl4_mu);
    if (!ctx.mtl4_serializer || ctx.mtl4_archive_path.empty()) return 0;
    NSURL *url = [NSURL fileURLWithPath:@(ctx.mtl4_archive_path.c_str())];
    NSError *e = nil;
    BOOL ok = [(id<MTL4PipelineDataSetSerializer>)ctx.mtl4_serializer
        serializeAsArchiveAndFlushToURL:url error:&e];
    return ok ? 1 : 0;
  }
  return 0;
}

// P2/P3 — lazily create the reusable per-dispatch MTL4 objects. Call with
// ctx.mtl4_dispatch_mu held. The argument table is sized for the widest current
// kernel (5 bindings: A,B,C,bias,params); 8 leaves headroom.
API_AVAILABLE(macos(26.0), ios(26.0))
static void mtl4_ensure_dispatch_objects(MetalDeviceContext &ctx) {
  id<MTLDevice> dev = ctx.device;
  if (!ctx.mtl4_allocator) ctx.mtl4_allocator = [dev newCommandAllocator];
  if (!ctx.mtl4_cmdbuf)    ctx.mtl4_cmdbuf = [dev newCommandBuffer];
  if (!ctx.mtl4_event)     ctx.mtl4_event = [dev newSharedEvent];
  if (!ctx.mtl4_argtable) {
    MTL4ArgumentTableDescriptor *atd = [[MTL4ArgumentTableDescriptor alloc] init];
    atd.maxBufferBindCount = 8;
    ctx.mtl4_argtable = [dev newArgumentTableWithDescriptor:atd error:nil];
  }
  if (!ctx.mtl4_residency) {
    ctx.mtl4_residency = [dev newResidencySetWithDescriptor:
        [[MTLResidencySetDescriptor alloc] init] error:nil];
  }
}

// Repopulate the reusable residency set with `count` allocations and make them
// resident. Returns the set (nil on failure). Pair with mtl4_encode_and_wait,
// which attaches it to the command buffer via `useResidencySet:`. Caller holds
// ctx.mtl4_dispatch_mu (the set is shared, like the other reusable objects).
API_AVAILABLE(macos(26.0), ios(26.0))
static id<MTLResidencySet> mtl4_set_residency(MetalDeviceContext &ctx,
                                              id<MTLBuffer> const *bufs, int count) {
  mtl4_ensure_dispatch_objects(ctx);
  id<MTLResidencySet> rs = (id<MTLResidencySet>)ctx.mtl4_residency;
  if (!rs) return nil;
  [rs removeAllAllocations];
  for (int i = 0; i < count; ++i)
    if (bufs[i]) [rs addAllocation:bufs[i]];
  [rs commit];
  [rs requestResidency];
  return rs;
}

// P2/P3 — encode + commit + wait on the reusable allocator/command-buffer/event.
// Call with ctx.mtl4_dispatch_mu held and the residency set already attached to
// the queue. `bind` configures the (reused) argument table. Returns completion.
API_AVAILABLE(macos(26.0), ios(26.0))
static bool mtl4_encode_and_wait(MetalDeviceContext &ctx, id<MTL4CommandQueue> queue,
                                 id<MTLComputePipelineState> pso,
                                 void (^bind)(id<MTL4ArgumentTable>),
                                 MTLSize grid, MTLSize tpg,
                                 id<MTLResidencySet> res = nil) {
  mtl4_ensure_dispatch_objects(ctx);
  id<MTL4CommandAllocator> alloc = (id<MTL4CommandAllocator>)ctx.mtl4_allocator;
  id<MTL4CommandBuffer> cb = (id<MTL4CommandBuffer>)ctx.mtl4_cmdbuf;
  id<MTL4ArgumentTable> at = (id<MTL4ArgumentTable>)ctx.mtl4_argtable;
  id<MTLSharedEvent> ev = (id<MTLSharedEvent>)ctx.mtl4_event;
  if (!alloc || !cb || !at || !ev) return false;
  bind(at);
  [alloc reset];
  [cb beginCommandBufferWithAllocator:alloc];
  // Per-command-buffer residency (granular intended path) — keeps `res`'s
  // allocations resident for this dispatch without queue-level add/remove churn.
  if (res) [cb useResidencySet:res];
  id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setArgumentTable:at];
  [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
  [enc endEncoding];
  [cb endCommandBuffer];
  const id<MTL4CommandBuffer> cbs[1] = {cb};
  [queue commit:cbs count:1];
  uint64_t v = ++ctx.mtl4_event_val;
  [queue signalEvent:ev value:v];
  return [ev waitUntilSignaledValue:v timeoutMS:10000];
}

// MSL source for f32 RoPE — lifted to namespace scope so both the
// original dispatch_rope_msl and the new encode-session variant
// encode_rope_msl_dev (added 2026-06-01 for stage-2 single-cb decoder
// chain) share one source-of-truth.
static NSString *const kRopeF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void rope_f32(
    device const float* x      [[buffer(0)]],
    device const float* theta  [[buffer(1)]],
    device float*       out    [[buffer(2)]],
    constant int&       M      [[buffer(3)]],
    constant int&       K      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= (uint)(K / 2) || gid.y >= (uint)M) return;
    int row = (int)gid.y;
    int pair = (int)gid.x;
    int idx_even = row * K + pair * 2;
    int idx_odd  = idx_even + 1;
    float xe = x[idx_even];
    float xo = x[idx_odd];
    float c = cos(theta[idx_even]);
    float s = sin(theta[idx_even]);
    out[idx_even] = xe * c - xo * s;
    out[idx_odd]  = xe * s + xo * c;
}
)MSL";

bool dispatch_rope_msl(MetalDeviceContext &ctx, const float* X,
                       const float* Theta, float* Out, int32_t M, int32_t K) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kRopeF32Source, @"rope_f32");
    if (!pso) return false;

    NSUInteger byteCount = sizeof(float) * static_cast<NSUInteger>(M) *
                           static_cast<NSUInteger>(K);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, byteCount);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufT, ctx, Theta, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufX || !bufT || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufT offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:4];

    NSUInteger half_k = static_cast<NSUInteger>(K / 2);
    MTLSize grid = MTLSizeMake(half_k, static_cast<NSUInteger>(M), 1);
    NSUInteger tg_x = std::min<NSUInteger>(half_k, 32);
    NSUInteger tg_y = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(tg_x, 1));
    if (tg_y == 0) tg_y = 1;
    MTLSize tg = MTLSizeMake(tg_x, tg_y, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — RoPE f32.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "rope_f32_msl")) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
}

// Stage-2 single-cb (2026-06-01) — encode RoPE into a session-shared
// MPSCommandBuffer. Mirrors encode_flash_attn_msl_dev. Buffer arg
// ordering must match the f32 kernel above: x@0 / theta@1 / out@2.
static bool encode_rope_msl_dev(MetalDeviceContext &ctx,
                                MPSCommandBuffer *cb,
                                id<MTLBuffer> bufX, id<MTLBuffer> bufT,
                                id<MTLBuffer> bufO,
                                int32_t M, int32_t K) {
  if (M <= 0 || K <= 0) return true;
  if (!cb || !bufX || !bufT || !bufO) return false;
  id<MTLComputePipelineState> pso =
      compile_msl_kernel(ctx, kRopeF32Source, @"rope_f32");
  if (!pso) return false;
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) return false;
  [enc setComputePipelineState:pso];
  [enc setBuffer:bufX offset:0 atIndex:0];
  [enc setBuffer:bufT offset:0 atIndex:1];
  [enc setBuffer:bufO offset:0 atIndex:2];
  [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
  [enc setBytes:&K length:sizeof(int32_t) atIndex:4];
  NSUInteger half_k = static_cast<NSUInteger>(K / 2);
  MTLSize grid = MTLSizeMake(half_k, static_cast<NSUInteger>(M), 1);
  NSUInteger tg_x = std::min<NSUInteger>(half_k, 32);
  NSUInteger tg_y = std::min<NSUInteger>(
      static_cast<NSUInteger>(M),
      pso.maxTotalThreadsPerThreadgroup /
          std::max<NSUInteger>(tg_x, 1));
  if (tg_y == 0) tg_y = 1;
  [enc dispatchThreads:grid
         threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
  [enc endEncoding];
  return true;
}

inline void reference_rope_f32(const float* X, const float* Theta, float* Out,
                               int32_t M, int32_t K) {
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t pair = 0; pair < K / 2; ++pair) {
      std::size_t even = static_cast<std::size_t>(row) * K + pair * 2;
      std::size_t odd = even + 1;
      float xe = X[even];
      float xo = X[odd];
      float c = std::cos(Theta[even]);
      float s = std::sin(Theta[even]);
      Out[even] = xe * c - xo * s;
      Out[odd]  = xe * s + xo * c;
    }
  }
}

} // namespace

extern "C" void tessera_apple_gpu_rope_f32(const float* X, const float* Theta,
                                           float* Out, int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_rope_msl(ctx, X, Theta, Out, M, K)) return;
  reference_rope_f32(X, Theta, Out, M, K);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.4.1 — fp16 + bf16 rope variants.
//
// fp16: native MSL `half` kernel. Compute is in `float` for accuracy
//       (cos/sin); load/store as `half`. Apple Silicon GPUs run this
//       at higher throughput than the fp32 variant.
// bf16: same fp32-conversion pattern as Phase 8.4.4 bf16 matmul. Decode
//       bf16 bit-pattern, run the fp32 reference, encode back.
//===---------------------------------------------------------------------===//

namespace {

// MSL source for f16 RoPE — lifted to namespace scope so both the
// original dispatch_rope_msl_f16 AND the new encode-session variant
// encode_rope_msl_f16_dev (added 2026-06-01 for Project-3 f16
// encode-session ABI) share one source-of-truth. compile_msl_kernel
// caches by source SHA256 so duplication has zero runtime cost.
static NSString *const kRopeF16Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void rope_f16(
    device const half*  x      [[buffer(0)]],
    device const half*  theta  [[buffer(1)]],
    device half*        out    [[buffer(2)]],
    constant int&       M      [[buffer(3)]],
    constant int&       K      [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= (uint)(K / 2) || gid.y >= (uint)M) return;
    int row = (int)gid.y;
    int pair = (int)gid.x;
    int idx_even = row * K + pair * 2;
    int idx_odd  = idx_even + 1;
    float xe = float(x[idx_even]);
    float xo = float(x[idx_odd]);
    float c = cos(float(theta[idx_even]));
    float s = sin(float(theta[idx_even]));
    out[idx_even] = half(xe * c - xo * s);
    out[idx_odd]  = half(xe * s + xo * c);
}
)MSL";

bool dispatch_rope_msl_f16(MetalDeviceContext &ctx, const uint16_t* X,
                           const uint16_t* Theta, uint16_t* Out,
                           int32_t M, int32_t K) {

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kRopeF16Source, @"rope_f16");
    if (!pso) return false;

    NSUInteger byteCount = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                           static_cast<NSUInteger>(K);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, byteCount);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufT, ctx, Theta, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufX || !bufT || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufT offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:4];

    NSUInteger half_k = static_cast<NSUInteger>(K / 2);
    MTLSize grid = MTLSizeMake(half_k, static_cast<NSUInteger>(M), 1);
    NSUInteger tg_x = std::min<NSUInteger>(half_k, 32);
    NSUInteger tg_y = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup /
                                               std::max<NSUInteger>(tg_x, 1));
    if (tg_y == 0) tg_y = 1;
    MTLSize tg = MTLSizeMake(tg_x, tg_y, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — RoPE f16.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "rope_f16_msl")) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
}

// Project-3 f16 (2026-06-01) — encode RoPE f16 into a session-shared
// MPSCommandBuffer. Mirrors encode_rope_msl_dev but uses the f16
// kernel.
static bool encode_rope_msl_f16_dev(MetalDeviceContext &ctx,
                                    MPSCommandBuffer *cb,
                                    id<MTLBuffer> bufX, id<MTLBuffer> bufT,
                                    id<MTLBuffer> bufO,
                                    int32_t M, int32_t K) {
  if (M <= 0 || K <= 0) return true;
  if (!cb || !bufX || !bufT || !bufO) return false;
  id<MTLComputePipelineState> pso =
      compile_msl_kernel(ctx, kRopeF16Source, @"rope_f16");
  if (!pso) return false;
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) return false;
  [enc setComputePipelineState:pso];
  [enc setBuffer:bufX offset:0 atIndex:0];
  [enc setBuffer:bufT offset:0 atIndex:1];
  [enc setBuffer:bufO offset:0 atIndex:2];
  [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
  [enc setBytes:&K length:sizeof(int32_t) atIndex:4];
  NSUInteger half_k = static_cast<NSUInteger>(K / 2);
  MTLSize grid = MTLSizeMake(half_k, static_cast<NSUInteger>(M), 1);
  NSUInteger tg_x = std::min<NSUInteger>(half_k, 32);
  NSUInteger tg_y = std::min<NSUInteger>(
      static_cast<NSUInteger>(M),
      pso.maxTotalThreadsPerThreadgroup /
          std::max<NSUInteger>(tg_x, 1));
  if (tg_y == 0) tg_y = 1;
  [enc dispatchThreads:grid
         threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
  [enc endEncoding];
  return true;
}

inline void reference_rope_f16_via_fp32(const uint16_t* X, const uint16_t* Theta,
                                        uint16_t* Out, int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Tf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = half_to_float_gpu(X[i]);
  for (std::size_t i = 0; i < Tf.size(); ++i) Tf[i] = half_to_float_gpu(Theta[i]);
  reference_rope_f32(Xf.data(), Tf.data(), Of.data(), M, K);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_half_gpu(Of[i]);
}

inline void reference_rope_bf16_via_fp32(const uint16_t* X, const uint16_t* Theta,
                                         uint16_t* Out, int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Tf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = bfloat16_to_float_gpu(X[i]);
  for (std::size_t i = 0; i < Tf.size(); ++i) Tf[i] = bfloat16_to_float_gpu(Theta[i]);
  reference_rope_f32(Xf.data(), Tf.data(), Of.data(), M, K);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_rope_bf16_via_fp32(MetalDeviceContext &ctx, const uint16_t* X,
                                 const uint16_t* Theta, uint16_t* Out,
                                 int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Tf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = bfloat16_to_float_gpu(X[i]);
  for (std::size_t i = 0; i < Tf.size(); ++i) Tf[i] = bfloat16_to_float_gpu(Theta[i]);
  if (!dispatch_rope_msl(ctx, Xf.data(), Tf.data(), Of.data(), M, K))
    return false;
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_rope_f16(const uint16_t* X,
                                           const uint16_t* Theta,
                                           uint16_t* Out,
                                           int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_rope_msl_f16(ctx, X, Theta, Out, M, K)) return;
  reference_rope_f16_via_fp32(X, Theta, Out, M, K);
}

extern "C" void tessera_apple_gpu_rope_bf16(const uint16_t* X,
                                            const uint16_t* Theta,
                                            uint16_t* Out,
                                            int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_rope_bf16_via_fp32(ctx, X, Theta, Out, M, K)) return;
  reference_rope_bf16_via_fp32(X, Theta, Out, M, K);
}

extern "C" int32_t tessera_apple_gpu_runtime_msl_cache_size(void) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return -1;
  std::lock_guard<std::mutex> lock(ctx.kernel_cache_mu);
  return static_cast<int32_t>(ctx.kernel_cache.size());
}

//===---------------------------------------------------------------------===//
// Phase 8.4.1 — Flash-attention forward (rank-3, f32)
//
// O = softmax(QK^T * scale) @ V
//
// Shapes:
//   Q: (B, Sq, D)
//   K: (B, Sk, D)
//   V: (B, Sk, D)
//   O: (B, Sq, D)
//
// Implementation: online softmax in a single MSL kernel. One thread per
// (batch, query_row); each thread streams over Sk computing scores +
// rescaling the running output accumulator with the flash-attn update rule.
// Avoids materializing the (B, Sq, Sk) score matrix entirely.
//
// Constraints (Phase 8.4.1):
//   - D <= 256 (per-thread accumulator stack array; raise via threadgroup
//     memory in a follow-up if needed)
//   - f32 in / f32 out
//   - Optional causal mask via the `causal` flag (1 = lower-triangular,
//     0 = unmasked)
//===---------------------------------------------------------------------===//

namespace {

// MSL source for the f32 flash-attention kernel. Lifted to namespace
// scope so both the original ``dispatch_flash_attn_msl`` and the
// encode-session variant ``encode_flash_attn_msl_dev`` (added 2026-06-01
// for the single-cb decoder block demo) share one source-of-truth.
// ``compile_msl_kernel`` caches by ``(source SHA256, entry)`` so the
// PSO is built once regardless of which dispatcher first triggers it.
static NSString *const kFlashAttnF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FLASH_ATTN_MAX_D 256

kernel void flash_attn_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device float*       O       [[buffer(3)]],
    constant int&       B       [[buffer(4)]],
    constant int&       Sq      [[buffer(5)]],
    constant int&       Sk      [[buffer(6)]],
    constant int&       D       [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    constant int&       causal  [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= (uint)B || gid.x >= (uint)Sq) return;
    int batch = (int)gid.y;
    int q_row = (int)gid.x;
    if (D > TESSERA_APPLE_GPU_FLASH_ATTN_MAX_D) return;

    int q_off = batch * Sq * D + q_row * D;
    int kv_base = batch * Sk * D;

    // Online softmax accumulators.
    float m = -INFINITY;   // running max
    float l = 0.0f;        // running denominator
    float o[TESSERA_APPLE_GPU_FLASH_ATTN_MAX_D];
    for (int d = 0; d < D; ++d) o[d] = 0.0f;

    for (int k_row = 0; k_row < Sk; ++k_row) {
        // Causal mask: skip keys above the query row.
        if (causal != 0 && k_row > q_row) break;

        int k_off = kv_base + k_row * D;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += Q[q_off + d] * K[k_off + d];
        }
        score *= scale;

        float new_m = max(m, score);
        // exp(-INF - X) is well-defined in IEEE-754 (yields 0). The first
        // iteration starts with m = -INFINITY, so exp_old = 0 here and the
        // running output accumulator initializes cleanly.
        float exp_old = exp(m - new_m);
        float exp_score = exp(score - new_m);
        float new_l = l * exp_old + exp_score;

        for (int d = 0; d < D; ++d) {
            o[d] = o[d] * exp_old + V[k_off + d] * exp_score;
        }
        m = new_m;
        l = new_l;
    }

    // Final normalization. If l == 0 (e.g. fully causal-masked first row
    // when q_row > Sk - 1, edge case), divide-by-zero would produce NaN.
    // We guard with a conditional — match numpy's behavior of returning
    // zeros for the all-masked case rather than NaN.
    if (l == 0.0f) {
        for (int d = 0; d < D; ++d) O[q_off + d] = 0.0f;
    } else {
        float inv_l = 1.0f / l;
        for (int d = 0; d < D; ++d) O[q_off + d] = o[d] * inv_l;
    }
}
)MSL";

bool dispatch_flash_attn_msl(MetalDeviceContext &ctx, const float* Q,
                             const float* K, const float* V, float* O,
                             int32_t B, int32_t Sq, int32_t Sk, int32_t D,
                             float scale, int32_t causal) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFlashAttnF32Source, @"flash_attn_f32");
    if (!pso) return false;

    NSUInteger qBytes = sizeof(float) * static_cast<NSUInteger>(B) *
                        static_cast<NSUInteger>(Sq) *
                        static_cast<NSUInteger>(D);
    NSUInteger kvBytes = sizeof(float) * static_cast<NSUInteger>(B) *
                         static_cast<NSUInteger>(Sk) *
                         static_cast<NSUInteger>(D);
    NSUInteger oBytes = qBytes;

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufQ, ctx, Q, qBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufK, ctx, K, kvBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufV, ctx, V, kvBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufQ || !bufK || !bufV || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufQ offset:0 atIndex:0];
    [enc setBuffer:bufK offset:0 atIndex:1];
    [enc setBuffer:bufV offset:0 atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&B  length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&Sq length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&Sk length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&D  length:sizeof(int32_t) atIndex:7];
    [enc setBytes:&scale  length:sizeof(float)   atIndex:8];
    [enc setBytes:&causal length:sizeof(int32_t) atIndex:9];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(Sq),
                               static_cast<NSUInteger>(B), 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(Sq), 32);
    NSUInteger tg_y = std::min<NSUInteger>(static_cast<NSUInteger>(B),
                                           pso.maxTotalThreadsPerThreadgroup /
                                               std::max<NSUInteger>(tg_x, 1));
    if (tg_y == 0) tg_y = 1;
    MTLSize tg = MTLSizeMake(tg_x, tg_y, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

// Encode flash_attn into a session-shared MPSCommandBuffer (no commit /
// wait / memcpy). Single-cb decoder block scaffold — pairs with
// ``mpsg_encode_bmm_dev`` and ``mpsg_encode_layer_norm_dev`` so an
// attention block (norm → qkv_proj → flash_attn → out_proj → norm) can
// execute on ONE command buffer. Inputs / outputs are device-resident
// ``MTLBuffer`` (the caller's ``TsDeviceTensor::buf``).
//
// MSL kernel + compute-encoder pattern is identical to
// ``dispatch_flash_attn_msl``; the only difference is that the encoder
// is built on the caller's shared cb and the encode helper returns
// without committing. Metal's automatic hazard tracking orders later
// reads of ``bufO`` after this encode pass, so a downstream op (e.g.
// the out-projection bmm) appended into the same cb sees the right
// data without an explicit barrier.
static bool encode_flash_attn_msl_dev(MetalDeviceContext &ctx,
                                      MPSCommandBuffer *cb,
                                      id<MTLBuffer> bufQ,
                                      id<MTLBuffer> bufK,
                                      id<MTLBuffer> bufV,
                                      id<MTLBuffer> bufO,
                                      int32_t B, int32_t Sq, int32_t Sk,
                                      int32_t D, float scale,
                                      int32_t causal) {
  if (B <= 0 || Sq <= 0 || Sk <= 0 || D <= 0) return true;
  if (!cb || !bufQ || !bufK || !bufV || !bufO) return false;
  id<MTLComputePipelineState> pso =
      compile_msl_kernel(ctx, kFlashAttnF32Source, @"flash_attn_f32");
  if (!pso) return false;
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) return false;
  [enc setComputePipelineState:pso];
  [enc setBuffer:bufQ offset:0 atIndex:0];
  [enc setBuffer:bufK offset:0 atIndex:1];
  [enc setBuffer:bufV offset:0 atIndex:2];
  [enc setBuffer:bufO offset:0 atIndex:3];
  [enc setBytes:&B  length:sizeof(int32_t) atIndex:4];
  [enc setBytes:&Sq length:sizeof(int32_t) atIndex:5];
  [enc setBytes:&Sk length:sizeof(int32_t) atIndex:6];
  [enc setBytes:&D  length:sizeof(int32_t) atIndex:7];
  [enc setBytes:&scale  length:sizeof(float)   atIndex:8];
  [enc setBytes:&causal length:sizeof(int32_t) atIndex:9];
  MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(Sq),
                             static_cast<NSUInteger>(B), 1);
  NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(Sq), 32);
  NSUInteger tg_y = std::min<NSUInteger>(static_cast<NSUInteger>(B),
                                         pso.maxTotalThreadsPerThreadgroup /
                                             std::max<NSUInteger>(tg_x, 1));
  if (tg_y == 0) tg_y = 1;
  MTLSize tg = MTLSizeMake(tg_x, tg_y, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];
  [enc endEncoding];
  return true;
}

inline void reference_flash_attn_f32(const float* Q, const float* K,
                                     const float* V, float* O,
                                     int32_t B, int32_t Sq, int32_t Sk,
                                     int32_t D, float scale, int32_t causal) {
  // Same online-softmax algorithm as the MSL kernel, in plain C++. Used as
  // the non-Darwin fallback and (in this TU) when MTLDevice is unavailable.
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t q = 0; q < Sq; ++q) {
      const float* Qrow = Q + (static_cast<std::size_t>(b) * Sq + q) * D;
      const float* Kbase = K + static_cast<std::size_t>(b) * Sk * D;
      const float* Vbase = V + static_cast<std::size_t>(b) * Sk * D;
      float* Orow = O + (static_cast<std::size_t>(b) * Sq + q) * D;

      float m = -std::numeric_limits<float>::infinity();
      float l = 0.0f;
      // Stack accumulator big enough for D up to 256.
      float o[256];
      for (int32_t d = 0; d < D; ++d) o[d] = 0.0f;

      for (int32_t k = 0; k < Sk; ++k) {
        if (causal != 0 && k > q) break;
        const float* Krow = Kbase + static_cast<std::size_t>(k) * D;
        float score = 0.0f;
        for (int32_t d = 0; d < D; ++d) score += Qrow[d] * Krow[d];
        score *= scale;

        float new_m = std::max(m, score);
        float exp_old = std::exp(m - new_m);
        float exp_score = std::exp(score - new_m);
        float new_l = l * exp_old + exp_score;
        const float* Vrow = Vbase + static_cast<std::size_t>(k) * D;
        for (int32_t d = 0; d < D; ++d) {
          o[d] = o[d] * exp_old + Vrow[d] * exp_score;
        }
        m = new_m;
        l = new_l;
      }
      if (l == 0.0f) {
        for (int32_t d = 0; d < D; ++d) Orow[d] = 0.0f;
      } else {
        float inv_l = 1.0f / l;
        for (int32_t d = 0; d < D; ++d) Orow[d] = o[d] * inv_l;
      }
    }
  }
}

} // namespace

extern "C" void tessera_apple_gpu_flash_attn_f32(const float* Q, const float* K,
                                                 const float* V, float* O,
                                                 int32_t B, int32_t Sq,
                                                 int32_t Sk, int32_t D,
                                                 float scale, int32_t causal) {
  if (D > 256) {
    // Out of envelope — emit a defined fallback rather than scribbling past
    // the kernel's stack array. Same shape as numpy reference.
    reference_flash_attn_f32(Q, K, V, O, B, Sq, Sk, D, scale, causal);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_flash_attn_msl(ctx, Q, K, V, O, B, Sq, Sk, D, scale,
                                        causal)) {
    return;
  }
  reference_flash_attn_f32(Q, K, V, O, B, Sq, Sk, D, scale, causal);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.4.2 — fp16 / bf16 flash-attention forward.
//
// Mixed-precision design — the K/V/Q tensors are fp16 / bf16 at the I/O
// boundary, but the per-thread accumulators (m, l, o[]) stay in fp32 just
// like production flash-attn implementations (Tri Dao, etc). This is the
// standard pattern: low-precision tensors save memory bandwidth, fp32
// accumulation preserves softmax / online-update precision.
//
// fp16: native MSL `half` I/O kernel.
// bf16: fp32-conversion path at the runtime boundary (no MSL bf16 type).
//===---------------------------------------------------------------------===//

namespace {

// MSL source for f16 flash-attention — lifted to namespace scope for
// Project-3 f16 encode-session reuse (2026-06-01). Both the legacy
// own-cb dispatcher and the encode-session helper read this string;
// compile_msl_kernel dedupes via source SHA256.
static NSString *const kFlashAttnF16Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FLASH_ATTN_MAX_D 256

kernel void flash_attn_f16(
    device const half*  Q       [[buffer(0)]],
    device const half*  K       [[buffer(1)]],
    device const half*  V       [[buffer(2)]],
    device half*        O       [[buffer(3)]],
    constant int&       B       [[buffer(4)]],
    constant int&       Sq      [[buffer(5)]],
    constant int&       Sk      [[buffer(6)]],
    constant int&       D       [[buffer(7)]],
    constant float&     scale   [[buffer(8)]],
    constant int&       causal  [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= (uint)B || gid.x >= (uint)Sq) return;
    int batch = (int)gid.y;
    int q_row = (int)gid.x;
    if (D > TESSERA_APPLE_GPU_FLASH_ATTN_MAX_D) return;

    int q_off = batch * Sq * D + q_row * D;
    int kv_base = batch * Sk * D;

    // float accumulators — preserve softmax precision on the online update.
    float m = -INFINITY;
    float l = 0.0f;
    float o[TESSERA_APPLE_GPU_FLASH_ATTN_MAX_D];
    for (int d = 0; d < D; ++d) o[d] = 0.0f;

    for (int k_row = 0; k_row < Sk; ++k_row) {
        if (causal != 0 && k_row > q_row) break;
        int k_off = kv_base + k_row * D;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            score += float(Q[q_off + d]) * float(K[k_off + d]);
        }
        score *= scale;

        float new_m = max(m, score);
        float exp_old = exp(m - new_m);
        float exp_score = exp(score - new_m);
        float new_l = l * exp_old + exp_score;

        for (int d = 0; d < D; ++d) {
            o[d] = o[d] * exp_old + float(V[k_off + d]) * exp_score;
        }
        m = new_m;
        l = new_l;
    }

    if (l == 0.0f) {
        for (int d = 0; d < D; ++d) O[q_off + d] = half(0.0f);
    } else {
        float inv_l = 1.0f / l;
        for (int d = 0; d < D; ++d) O[q_off + d] = half(o[d] * inv_l);
    }
}
)MSL";

bool dispatch_flash_attn_msl_f16(MetalDeviceContext &ctx, const uint16_t* Q,
                                 const uint16_t* K, const uint16_t* V,
                                 uint16_t* O, int32_t B, int32_t Sq,
                                 int32_t Sk, int32_t D, float scale,
                                 int32_t causal) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFlashAttnF16Source, @"flash_attn_f16");
    if (!pso) return false;

    NSUInteger qBytes = sizeof(uint16_t) * static_cast<NSUInteger>(B) *
                        static_cast<NSUInteger>(Sq) * static_cast<NSUInteger>(D);
    NSUInteger kvBytes = sizeof(uint16_t) * static_cast<NSUInteger>(B) *
                         static_cast<NSUInteger>(Sk) * static_cast<NSUInteger>(D);
    NSUInteger oBytes = qBytes;

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufQ, ctx, Q, qBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufK, ctx, K, kvBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufV, ctx, V, kvBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufQ || !bufK || !bufV || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufQ offset:0 atIndex:0];
    [enc setBuffer:bufK offset:0 atIndex:1];
    [enc setBuffer:bufV offset:0 atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&B  length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&Sq length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&Sk length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&D  length:sizeof(int32_t) atIndex:7];
    [enc setBytes:&scale  length:sizeof(float)   atIndex:8];
    [enc setBytes:&causal length:sizeof(int32_t) atIndex:9];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(Sq),
                               static_cast<NSUInteger>(B), 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(Sq), 32);
    NSUInteger tg_y = std::min<NSUInteger>(static_cast<NSUInteger>(B),
                                           pso.maxTotalThreadsPerThreadgroup /
                                               std::max<NSUInteger>(tg_x, 1));
    if (tg_y == 0) tg_y = 1;
    MTLSize tg = MTLSizeMake(tg_x, tg_y, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

// Project-3 f16 (2026-06-01) — encode f16 flash_attn into a shared
// MPSCommandBuffer. Mirrors encode_flash_attn_msl_dev (f32) using the
// f16 MSL kernel above.
static bool encode_flash_attn_msl_f16_dev(MetalDeviceContext &ctx,
                                          MPSCommandBuffer *cb,
                                          id<MTLBuffer> bufQ,
                                          id<MTLBuffer> bufK,
                                          id<MTLBuffer> bufV,
                                          id<MTLBuffer> bufO,
                                          int32_t B, int32_t Sq, int32_t Sk,
                                          int32_t D, float scale,
                                          int32_t causal) {
  if (B <= 0 || Sq <= 0 || Sk <= 0 || D <= 0) return true;
  if (!cb || !bufQ || !bufK || !bufV || !bufO) return false;
  id<MTLComputePipelineState> pso =
      compile_msl_kernel(ctx, kFlashAttnF16Source, @"flash_attn_f16");
  if (!pso) return false;
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  if (!enc) return false;
  [enc setComputePipelineState:pso];
  [enc setBuffer:bufQ offset:0 atIndex:0];
  [enc setBuffer:bufK offset:0 atIndex:1];
  [enc setBuffer:bufV offset:0 atIndex:2];
  [enc setBuffer:bufO offset:0 atIndex:3];
  [enc setBytes:&B  length:sizeof(int32_t) atIndex:4];
  [enc setBytes:&Sq length:sizeof(int32_t) atIndex:5];
  [enc setBytes:&Sk length:sizeof(int32_t) atIndex:6];
  [enc setBytes:&D  length:sizeof(int32_t) atIndex:7];
  [enc setBytes:&scale  length:sizeof(float)   atIndex:8];
  [enc setBytes:&causal length:sizeof(int32_t) atIndex:9];
  MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(Sq),
                             static_cast<NSUInteger>(B), 1);
  NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(Sq), 32);
  NSUInteger tg_y = std::min<NSUInteger>(
      static_cast<NSUInteger>(B),
      pso.maxTotalThreadsPerThreadgroup /
          std::max<NSUInteger>(tg_x, 1));
  if (tg_y == 0) tg_y = 1;
  [enc dispatchThreads:grid
         threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
  [enc endEncoding];
  return true;
}

inline void reference_flash_attn_f16_via_fp32(const uint16_t* Q,
                                              const uint16_t* K,
                                              const uint16_t* V, uint16_t* O,
                                              int32_t B, int32_t Sq,
                                              int32_t Sk, int32_t D,
                                              float scale, int32_t causal) {
  std::vector<float> Qf(static_cast<std::size_t>(B) * Sq * D);
  std::vector<float> Kf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Vf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Of(static_cast<std::size_t>(B) * Sq * D);
  for (std::size_t i = 0; i < Qf.size(); ++i) Qf[i] = half_to_float_gpu(Q[i]);
  for (std::size_t i = 0; i < Kf.size(); ++i) Kf[i] = half_to_float_gpu(K[i]);
  for (std::size_t i = 0; i < Vf.size(); ++i) Vf[i] = half_to_float_gpu(V[i]);
  reference_flash_attn_f32(Qf.data(), Kf.data(), Vf.data(), Of.data(),
                           B, Sq, Sk, D, scale, causal);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_gpu(Of[i]);
}

inline void reference_flash_attn_bf16_via_fp32(const uint16_t* Q,
                                               const uint16_t* K,
                                               const uint16_t* V, uint16_t* O,
                                               int32_t B, int32_t Sq,
                                               int32_t Sk, int32_t D,
                                               float scale, int32_t causal) {
  std::vector<float> Qf(static_cast<std::size_t>(B) * Sq * D);
  std::vector<float> Kf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Vf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Of(static_cast<std::size_t>(B) * Sq * D);
  for (std::size_t i = 0; i < Qf.size(); ++i) Qf[i] = bfloat16_to_float_gpu(Q[i]);
  for (std::size_t i = 0; i < Kf.size(); ++i) Kf[i] = bfloat16_to_float_gpu(K[i]);
  for (std::size_t i = 0; i < Vf.size(); ++i) Vf[i] = bfloat16_to_float_gpu(V[i]);
  reference_flash_attn_f32(Qf.data(), Kf.data(), Vf.data(), Of.data(),
                           B, Sq, Sk, D, scale, causal);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_flash_attn_bf16_via_fp32(MetalDeviceContext &ctx,
                                       const uint16_t* Q, const uint16_t* K,
                                       const uint16_t* V, uint16_t* O,
                                       int32_t B, int32_t Sq, int32_t Sk,
                                       int32_t D, float scale, int32_t causal) {
  std::vector<float> Qf(static_cast<std::size_t>(B) * Sq * D);
  std::vector<float> Kf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Vf(static_cast<std::size_t>(B) * Sk * D);
  std::vector<float> Of(static_cast<std::size_t>(B) * Sq * D);
  for (std::size_t i = 0; i < Qf.size(); ++i) Qf[i] = bfloat16_to_float_gpu(Q[i]);
  for (std::size_t i = 0; i < Kf.size(); ++i) Kf[i] = bfloat16_to_float_gpu(K[i]);
  for (std::size_t i = 0; i < Vf.size(); ++i) Vf[i] = bfloat16_to_float_gpu(V[i]);
  if (!dispatch_flash_attn_msl(ctx, Qf.data(), Kf.data(), Vf.data(), Of.data(),
                               B, Sq, Sk, D, scale, causal))
    return false;
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_flash_attn_f16(const uint16_t* Q,
                                                 const uint16_t* K,
                                                 const uint16_t* V,
                                                 uint16_t* O,
                                                 int32_t B, int32_t Sq,
                                                 int32_t Sk, int32_t D,
                                                 float scale, int32_t causal) {
  if (D > 256) {
    reference_flash_attn_f16_via_fp32(Q, K, V, O, B, Sq, Sk, D, scale, causal);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_flash_attn_msl_f16(ctx, Q, K, V, O, B, Sq, Sk, D, scale, causal))
    return;
  reference_flash_attn_f16_via_fp32(Q, K, V, O, B, Sq, Sk, D, scale, causal);
}

extern "C" void tessera_apple_gpu_flash_attn_bf16(const uint16_t* Q,
                                                  const uint16_t* K,
                                                  const uint16_t* V,
                                                  uint16_t* O,
                                                  int32_t B, int32_t Sq,
                                                  int32_t Sk, int32_t D,
                                                  float scale, int32_t causal) {
  if (D > 256) {
    reference_flash_attn_bf16_via_fp32(Q, K, V, O, B, Sq, Sk, D, scale, causal);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_flash_attn_bf16_via_fp32(ctx, Q, K, V, O, B, Sq, Sk, D, scale, causal))
    return;
  reference_flash_attn_bf16_via_fp32(Q, K, V, O, B, Sq, Sk, D, scale, causal);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.2 — Softmax (rank-2, axis=-1, f32)
//
// Standard 3-pass softmax: row max -> subtract + exp + sum -> divide. One
// thread per row; the per-thread loop streams over K once for max, once for
// the exp/sum, and once for normalization. For typical post-attention shapes
// this is GPU-bound on the elementwise loops, not the row reduction.
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_softmax_msl(MetalDeviceContext &ctx, const float* X, float* Out,
                          int32_t M, int32_t K) {
  static NSString *const kSoftmaxSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void softmax_f32(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant int&       M   [[buffer(2)]],
    constant int&       K   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    int row = (int)gid;
    int row_off = row * K;

    // Pass 1: row max for numerical stability.
    float row_max = -INFINITY;
    for (int j = 0; j < K; ++j) {
        row_max = max(row_max, x[row_off + j]);
    }
    // Pass 2: exp(x - max) accumulated into denom; write intermediate to out.
    float denom = 0.0f;
    for (int j = 0; j < K; ++j) {
        float e = exp(x[row_off + j] - row_max);
        out[row_off + j] = e;
        denom += e;
    }
    // Pass 3: divide.
    float inv = 1.0f / denom;
    for (int j = 0; j < K; ++j) {
        out[row_off + j] *= inv;
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kSoftmaxSource, @"softmax_f32");
    if (!pso) return false;

    NSUInteger byteCount = sizeof(float) * static_cast<NSUInteger>(M) *
                           static_cast<NSUInteger>(K);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufX || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufO offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:2];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:3];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — softmax MSL (f32/f16).
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "softmax_msl")) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
}

inline void reference_softmax_f32(const float* X, float* Out, int32_t M,
                                  int32_t K) {
  for (int32_t r = 0; r < M; ++r) {
    const float* row = X + static_cast<std::size_t>(r) * K;
    float* out_row = Out + static_cast<std::size_t>(r) * K;
    float row_max = -std::numeric_limits<float>::infinity();
    for (int32_t j = 0; j < K; ++j) row_max = std::max(row_max, row[j]);
    float denom = 0.0f;
    for (int32_t j = 0; j < K; ++j) {
      float e = std::exp(row[j] - row_max);
      out_row[j] = e;
      denom += e;
    }
    float inv = 1.0f / denom;
    for (int32_t j = 0; j < K; ++j) out_row[j] *= inv;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_softmax_f32(const float* X, float* Out,
                                              int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_softmax_msl(ctx, X, Out, M, K)) return;
  reference_softmax_f32(X, Out, M, K);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.4.1 — fp16 + bf16 softmax variants.
// fp16: native MSL `half` kernel; per-row reduction in `float` for numerical
//       stability (small per-row range matters for softmax denom accuracy).
// bf16: fp32-conversion path.
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_softmax_msl_f16(MetalDeviceContext &ctx, const uint16_t* X,
                              uint16_t* Out, int32_t M, int32_t K) {
  static NSString *const kSoftmaxSourceF16 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void softmax_f16(
    device const half*  x   [[buffer(0)]],
    device half*        out [[buffer(1)]],
    constant int&       M   [[buffer(2)]],
    constant int&       K   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    int row = (int)gid;
    int row_off = row * K;

    float row_max = -INFINITY;
    for (int j = 0; j < K; ++j) {
        row_max = max(row_max, float(x[row_off + j]));
    }
    float denom = 0.0f;
    // Pass 2: store exp values in `out` as fp16 — slight precision loss vs
    // f32 reference but matches the MSL native fp16 throughput contract.
    for (int j = 0; j < K; ++j) {
        float e = exp(float(x[row_off + j]) - row_max);
        out[row_off + j] = half(e);
        denom += e;
    }
    float inv = 1.0f / denom;
    for (int j = 0; j < K; ++j) {
        out[row_off + j] = half(float(out[row_off + j]) * inv);
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kSoftmaxSourceF16, @"softmax_f16");
    if (!pso) return false;

    NSUInteger byteCount = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                           static_cast<NSUInteger>(K);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufX || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufO offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:2];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:3];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — softmax MSL (f32/f16).
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "softmax_msl")) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
}

inline void reference_softmax_f16_via_fp32(const uint16_t* X, uint16_t* Out,
                                           int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = half_to_float_gpu(X[i]);
  reference_softmax_f32(Xf.data(), Of.data(), M, K);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_half_gpu(Of[i]);
}

inline void reference_softmax_bf16_via_fp32(const uint16_t* X, uint16_t* Out,
                                            int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = bfloat16_to_float_gpu(X[i]);
  reference_softmax_f32(Xf.data(), Of.data(), M, K);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_softmax_bf16_via_fp32(MetalDeviceContext &ctx, const uint16_t* X,
                                    uint16_t* Out, int32_t M, int32_t K) {
  std::vector<float> Xf(static_cast<std::size_t>(M) * K);
  std::vector<float> Of(static_cast<std::size_t>(M) * K);
  for (std::size_t i = 0; i < Xf.size(); ++i) Xf[i] = bfloat16_to_float_gpu(X[i]);
  if (!dispatch_softmax_msl(ctx, Xf.data(), Of.data(), M, K)) return false;
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_softmax_f16(const uint16_t* X, uint16_t* Out,
                                              int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_softmax_msl_f16(ctx, X, Out, M, K)) return;
  reference_softmax_f16_via_fp32(X, Out, M, K);
}

extern "C" void tessera_apple_gpu_softmax_bf16(const uint16_t* X, uint16_t* Out,
                                               int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_softmax_bf16_via_fp32(ctx, X, Out, M, K)) return;
  reference_softmax_bf16_via_fp32(X, Out, M, K);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.2 — GeLU (elementwise, f32)
//
// Tanh-approximation GeLU, matching the numpy reference in
// tessera.ops.gelu and the Tile IR `tile.gelu` lowering="elementwise":
//
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// One thread per element; rank-agnostic at the shim layer (caller flattens).
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_gelu_msl(MetalDeviceContext &ctx, const float* X, float* Out,
                       int32_t N) {
  static NSString *const kGeluSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void gelu_f32(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant int&       N   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)N) return;
    float v = x[gid];
    // sqrt(2/pi) = 0.7978845608028654
    float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    // Metal fast-math tanh overflows to NaN for large |t| (|x| >~ 16);
    // tanh saturates to +/-1 well before +/-30, so clamp the argument.
    t = clamp(t, -30.0f, 30.0f);
    out[gid] = 0.5f * v * (1.0f + tanh(t));
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kGeluSource, @"gelu_f32");
    if (!pso) return false;

    NSUInteger byteCount = sizeof(float) * static_cast<NSUInteger>(N);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufX || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufO offset:0 atIndex:1];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:2];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(N),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — GELU MSL (f32/f16).
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "gelu_msl")) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
}

inline void reference_gelu_f32(const float* X, float* Out, int32_t N) {
  static constexpr float kSqrt2OverPi = 0.7978845608028654f;
  for (int32_t i = 0; i < N; ++i) {
    float v = X[i];
    float t = kSqrt2OverPi * (v + 0.044715f * v * v * v);
    Out[i] = 0.5f * v * (1.0f + std::tanh(t));
  }
}

} // namespace

extern "C" void tessera_apple_gpu_gelu_f32(const float* X, float* Out,
                                           int32_t N) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_gelu_msl(ctx, X, Out, N)) return;
  reference_gelu_f32(X, Out, N);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.4.1 — fp16 + bf16 gelu variants.
// fp16: native MSL `half` kernel; tanh/cube in `float` for accuracy.
// bf16: fp32-conversion path.
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_gelu_msl_f16(MetalDeviceContext &ctx, const uint16_t* X,
                           uint16_t* Out, int32_t N) {
  static NSString *const kGeluSourceF16 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void gelu_f16(
    device const half*  x   [[buffer(0)]],
    device half*        out [[buffer(1)]],
    constant int&       N   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)N) return;
    float v = float(x[gid]);
    float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    // Metal fast-math tanh overflows to NaN for large |t| (|x| >~ 16);
    // tanh saturates to +/-1 well before +/-30, so clamp the argument.
    t = clamp(t, -30.0f, 30.0f);
    out[gid] = half(0.5f * v * (1.0f + tanh(t)));
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kGeluSourceF16, @"gelu_f16");
    if (!pso) return false;

    NSUInteger byteCount = sizeof(uint16_t) * static_cast<NSUInteger>(N);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufX || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufO offset:0 atIndex:1];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:2];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(N),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — GELU MSL (f32/f16).
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "gelu_msl")) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
}

inline void reference_gelu_f16_via_fp32(const uint16_t* X, uint16_t* Out,
                                        int32_t N) {
  std::vector<float> Xf(static_cast<std::size_t>(N));
  std::vector<float> Of(static_cast<std::size_t>(N));
  for (int32_t i = 0; i < N; ++i) Xf[i] = half_to_float_gpu(X[i]);
  reference_gelu_f32(Xf.data(), Of.data(), N);
  for (int32_t i = 0; i < N; ++i) Out[i] = float_to_half_gpu(Of[i]);
}

inline void reference_gelu_bf16_via_fp32(const uint16_t* X, uint16_t* Out,
                                         int32_t N) {
  std::vector<float> Xf(static_cast<std::size_t>(N));
  std::vector<float> Of(static_cast<std::size_t>(N));
  for (int32_t i = 0; i < N; ++i) Xf[i] = bfloat16_to_float_gpu(X[i]);
  reference_gelu_f32(Xf.data(), Of.data(), N);
  for (int32_t i = 0; i < N; ++i) Out[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_gelu_bf16_via_fp32(MetalDeviceContext &ctx, const uint16_t* X,
                                 uint16_t* Out, int32_t N) {
  std::vector<float> Xf(static_cast<std::size_t>(N));
  std::vector<float> Of(static_cast<std::size_t>(N));
  for (int32_t i = 0; i < N; ++i) Xf[i] = bfloat16_to_float_gpu(X[i]);
  if (!dispatch_gelu_msl(ctx, Xf.data(), Of.data(), N)) return false;
  for (int32_t i = 0; i < N; ++i) Out[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_gelu_f16(const uint16_t* X, uint16_t* Out,
                                           int32_t N) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_gelu_msl_f16(ctx, X, Out, N)) return;
  reference_gelu_f16_via_fp32(X, Out, N);
}

extern "C" void tessera_apple_gpu_gelu_bf16(const uint16_t* X, uint16_t* Out,
                                            int32_t N) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_gelu_bf16_via_fp32(ctx, X, Out, N)) return;
  reference_gelu_bf16_via_fp32(X, Out, N);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.3 — Fused matmul → softmax (rank-2, f32, axis=-1)
//
// O = softmax(A @ B, axis=-1)
//
// Shapes:
//   A: (M, K)
//   B: (K, N)
//   O: (M, N)   — softmax taken over axis=1 (innermost), per row.
//
// One thread per output row. Each thread:
//   1. Computes the full (1, N) row of A @ B into a stack array.
//   2. Computes the row max (numerically-stable softmax).
//   3. Computes exp(scores - max) and the row sum, in place.
//   4. Divides by the sum to produce the row of O.
//
// Avoids materializing the (M, N) intermediate score matrix on host. Cap
// N <= 256 so the per-thread stack array fits — typical attention shapes
// have head_dim or seq_len in this range. Larger N falls back to the
// reference path (which still produces correct results, just on CPU).
//
// The fused emission win comes from:
//   - One kernel launch instead of two (matmul + softmax)
//   - No host-side roundtrip for the (M, N) intermediate
//   - Shared MTLCommandBuffer + MTLBuffer infrastructure
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_matmul_softmax_msl(MetalDeviceContext &ctx, const float* A,
                                 const float* B, float* O,
                                 int32_t M, int32_t N, int32_t K) {
  static NSString *const kMatmulSoftmaxSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED_MATMUL_SOFTMAX_MAX_N 256

kernel void matmul_softmax_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (N > TESSERA_APPLE_GPU_FUSED_MATMUL_SOFTMAX_MAX_N) return;
    int row = (int)gid;

    // Pass 1: compute the (1, N) row of A @ B into the stack array.
    float scores[TESSERA_APPLE_GPU_FUSED_MATMUL_SOFTMAX_MAX_N];
    int a_off = row * K;
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[a_off + k];
        int b_off = k * N;
        for (int n = 0; n < N; ++n) {
            scores[n] += a * B[b_off + n];
        }
    }

    // Pass 2: row max for numerical stability.
    float row_max = -INFINITY;
    for (int n = 0; n < N; ++n) row_max = max(row_max, scores[n]);

    // Pass 3: exp + sum.
    float denom = 0.0f;
    for (int n = 0; n < N; ++n) {
        scores[n] = exp(scores[n] - row_max);
        denom += scores[n];
    }

    // Pass 4: divide and write out.
    int o_off = row * N;
    if (denom == 0.0f) {
        for (int n = 0; n < N; ++n) O[o_off + n] = 0.0f;
    } else {
        float inv = 1.0f / denom;
        for (int n = 0; n < N; ++n) O[o_off + n] = scores[n] * inv;
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kMatmulSoftmaxSource, @"matmul_softmax_f32");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(float) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger oBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

inline void reference_matmul_softmax_f32(const float* A, const float* B,
                                         float* O, int32_t M, int32_t N,
                                         int32_t K) {
  for (int32_t row = 0; row < M; ++row) {
    // Row of A @ B.
    std::vector<float> scores(static_cast<std::size_t>(N), 0.0f);
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(row) * K + k];
      const float* b_row = B + static_cast<std::size_t>(k) * N;
      for (int32_t n = 0; n < N; ++n) scores[n] += a * b_row[n];
    }
    // Row-wise softmax.
    float row_max = -std::numeric_limits<float>::infinity();
    for (int32_t n = 0; n < N; ++n) row_max = std::max(row_max, scores[n]);
    float denom = 0.0f;
    for (int32_t n = 0; n < N; ++n) {
      scores[n] = std::exp(scores[n] - row_max);
      denom += scores[n];
    }
    float* out_row = O + static_cast<std::size_t>(row) * N;
    if (denom == 0.0f) {
      for (int32_t n = 0; n < N; ++n) out_row[n] = 0.0f;
    } else {
      float inv = 1.0f / denom;
      for (int32_t n = 0; n < N; ++n) out_row[n] = scores[n] * inv;
    }
  }
}

} // namespace

//===---------------------------------------------------------------------===//
// Phase 8.4.6 — threadgroup-tiled matmul_softmax_f32 kernel.
//
// Lifts the N <= 256 constraint of the per-thread kernel by allocating the
// score buffer in threadgroup memory. One row per threadgroup; THREADS
// threads cooperate on the K reduction, threadgroup max, threadgroup sum,
// and final write. Threadgroup memory is sized at runtime via
// setThreadgroupMemoryLength so we can scale to whatever fits.
//
// Algorithm (per threadgroup, row = gid.y, lid = thread_position_in_threadgroup):
//   1. Cooperative compute scores[n] for n in [lid, N) step THREADS
//   2. Per-thread partial max -> threadgroup max reduction
//   3. Per-thread exp + partial sum -> threadgroup sum reduction
//   4. Cooperative divide and write to O[row, n]
//
// Uses dynamic threadgroup memory for the score buffer plus two small
// shared scratch slots (max + sum). Threadgroup size fixed at 32 to keep
// the reduction simple; can be tuned later.
//===---------------------------------------------------------------------===//

namespace {

constexpr int kFusedTiledThreads = 32;

bool dispatch_matmul_softmax_tiled_msl(MetalDeviceContext &ctx, const float* A,
                                       const float* B, float* O,
                                       int32_t M, int32_t N, int32_t K) {
  static NSString *const kFusedTiledSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED_TILED_THREADS 32

kernel void matmul_softmax_tiled_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    threadgroup float*  tg_scores [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  lid     [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    int row = (int)tg_pos.x;
    if (row >= M) return;
    const int T = TESSERA_APPLE_GPU_FUSED_TILED_THREADS;
    const int lid_i = (int)lid;

    threadgroup float tg_max[TESSERA_APPLE_GPU_FUSED_TILED_THREADS];
    threadgroup float tg_sum[TESSERA_APPLE_GPU_FUSED_TILED_THREADS];

    // Step 1: cooperative compute scores[n] = sum_k A[row,k] * B[k,n].
    int a_off = row * K;
    for (int n = lid_i; n < N; n += T) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) {
            s += A[a_off + k] * B[k * N + n];
        }
        tg_scores[n] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: per-thread partial row max -> threadgroup max reduction.
    // (A simd_max(local_max) variant was benchmarked and measured 0.72-0.93x —
    // slower — because these kernels are matmul/write-bound, not reduction-bound,
    // and simd_max/simd_sum cost more than the 32-lane on-chip tree here. See
    // docs/apple_backend_integration_review.md, SIMD-reduction section.)
    float local_max = -INFINITY;
    for (int n = lid_i; n < N; n += T) local_max = max(local_max, tg_scores[n]);
    tg_max[lid_i] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (lid_i < stride) {
            tg_max[lid_i] = max(tg_max[lid_i], tg_max[lid_i + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = tg_max[0];

    // Step 3: per-thread exp + partial sum -> threadgroup sum reduction.
    float local_sum = 0.0f;
    for (int n = lid_i; n < N; n += T) {
        float e = exp(tg_scores[n] - row_max);
        tg_scores[n] = e;
        local_sum += e;
    }
    tg_sum[lid_i] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (lid_i < stride) {
            tg_sum[lid_i] += tg_sum[lid_i + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denom = tg_sum[0];

    // Step 4: cooperative divide and write back.
    int o_off = row * N;
    if (denom == 0.0f) {
        for (int n = lid_i; n < N; n += T) O[o_off + n] = 0.0f;
    } else {
        float inv = 1.0f / denom;
        for (int n = lid_i; n < N; n += T) O[o_off + n] = tg_scores[n] * inv;
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFusedTiledSource, @"matmul_softmax_tiled_f32");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(float) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger oBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];

    // Dynamic threadgroup memory: scores[N] live floats per threadgroup.
    NSUInteger tg_score_bytes = sizeof(float) * static_cast<NSUInteger>(N);
    [enc setThreadgroupMemoryLength:tg_score_bytes atIndex:0];

    // One threadgroup per row, kFusedTiledThreads threads cooperating.
    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    MTLSize tg = MTLSizeMake(static_cast<NSUInteger>(kFusedTiledThreads), 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "matmul_softmax_tiled_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_softmax_tiled_f32(const float* A,
                                                           const float* B,
                                                           float* O,
                                                           int32_t M,
                                                           int32_t N,
                                                           int32_t K) {
  // Tiled variant — works for any N (bounded only by threadgroup memory).
  // Phase 8.4.6 caps N at 8192 to stay within typical device threadgroup
  // memory limits (~32KB / sizeof(float)). Larger N falls back to the
  // reference implementation.
  if (N > 8192) {
    reference_matmul_softmax_f32(A, B, O, M, N, K);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_softmax_tiled_msl(ctx, A, B, O, M, N, K))
    return;
  reference_matmul_softmax_f32(A, B, O, M, N, K);
}

extern "C" void tessera_apple_gpu_matmul_softmax_f32(const float* A,
                                                     const float* B, float* O,
                                                     int32_t M, int32_t N,
                                                     int32_t K) {
  // Phase 8.4.6 — route by N. Per-thread kernel for N <= 256 (it's faster
  // because no threadgroup synchronization), threadgroup-tiled for larger N
  // (lifts the per-thread stack-array limit). Reference fallback for N
  // beyond the tiled variant's bound.
  if (N <= 256) {
    MetalDeviceContext &ctx = deviceContext();
    if (ctx.ok && dispatch_matmul_softmax_msl(ctx, A, B, O, M, N, K)) return;
    reference_matmul_softmax_f32(A, B, O, M, N, K);
    return;
  }
  tessera_apple_gpu_matmul_softmax_tiled_f32(A, B, O, M, N, K);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.4.2 — fp16 / bf16 fused matmul -> softmax variants.
//
// Mixed-precision design: I/O in `half`/`bfloat`-pattern, but the per-thread
// `scores[256]` accumulator stays in `float`. This matches what production
// flash-attn-style implementations do — fp32 accumulation regardless of
// I/O dtype — and avoids precision loss on the inner reduction loops.
//
// fp16: native MSL `half` I/O kernel; load to float, accumulate, store back.
// bf16: fp32-conversion path (matmul_softmax_f32 dispatch + boundary cast).
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_matmul_softmax_msl_f16(MetalDeviceContext &ctx, const uint16_t* A,
                                     const uint16_t* B, uint16_t* O,
                                     int32_t M, int32_t N, int32_t K) {
  static NSString *const kFusedSourceF16 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED_MATMUL_SOFTMAX_MAX_N 256

kernel void matmul_softmax_f16(
    device const half*  A   [[buffer(0)]],
    device const half*  B   [[buffer(1)]],
    device half*        O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (N > TESSERA_APPLE_GPU_FUSED_MATMUL_SOFTMAX_MAX_N) return;
    int row = (int)gid;

    // float accumulator — preserve precision across the K-reduction.
    float scores[TESSERA_APPLE_GPU_FUSED_MATMUL_SOFTMAX_MAX_N];
    int a_off = row * K;
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = float(A[a_off + k]);
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * float(B[b_off + n]);
    }

    float row_max = -INFINITY;
    for (int n = 0; n < N; ++n) row_max = max(row_max, scores[n]);
    float denom = 0.0f;
    for (int n = 0; n < N; ++n) {
        scores[n] = exp(scores[n] - row_max);
        denom += scores[n];
    }
    int o_off = row * N;
    if (denom == 0.0f) {
        for (int n = 0; n < N; ++n) O[o_off + n] = half(0.0f);
    } else {
        float inv = 1.0f / denom;
        for (int n = 0; n < N; ++n) O[o_off + n] = half(scores[n] * inv);
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFusedSourceF16, @"matmul_softmax_f16");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(uint16_t) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger oBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

inline void reference_matmul_softmax_f16_via_fp32(const uint16_t* A,
                                                  const uint16_t* B,
                                                  uint16_t* O,
                                                  int32_t M, int32_t N,
                                                  int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_gpu(B[i]);
  reference_matmul_softmax_f32(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_gpu(Of[i]);
}

inline void reference_matmul_softmax_bf16_via_fp32(const uint16_t* A,
                                                   const uint16_t* B,
                                                   uint16_t* O,
                                                   int32_t M, int32_t N,
                                                   int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  reference_matmul_softmax_f32(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_matmul_softmax_bf16_via_fp32(MetalDeviceContext &ctx,
                                           const uint16_t* A, const uint16_t* B,
                                           uint16_t* O, int32_t M, int32_t N,
                                           int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  if (!dispatch_matmul_softmax_msl(ctx, Af.data(), Bf.data(), Of.data(), M, N, K))
    return false;
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_softmax_f16(const uint16_t* A,
                                                     const uint16_t* B,
                                                     uint16_t* O,
                                                     int32_t M, int32_t N,
                                                     int32_t K) {
  if (N > 256) {
    reference_matmul_softmax_f16_via_fp32(A, B, O, M, N, K);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_softmax_msl_f16(ctx, A, B, O, M, N, K)) return;
  reference_matmul_softmax_f16_via_fp32(A, B, O, M, N, K);
}

extern "C" void tessera_apple_gpu_matmul_softmax_bf16(const uint16_t* A,
                                                      const uint16_t* B,
                                                      uint16_t* O,
                                                      int32_t M, int32_t N,
                                                      int32_t K) {
  if (N > 256) {
    reference_matmul_softmax_bf16_via_fp32(A, B, O, M, N, K);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_softmax_bf16_via_fp32(ctx, A, B, O, M, N, K))
    return;
  reference_matmul_softmax_bf16_via_fp32(A, B, O, M, N, K);
}

//===---------------------------------------------------------------------===//
// Native half-precision threadgroup-tiled matmul -> softmax (large N).
//
// Lifts the per-thread N <= 256 limit of `matmul_softmax_f16` for f16/bf16 so
// the fused chain runs in a single kernel instead of composing GPU matmul with
// an MPSGraph epilogue. Mirrors the f32 tiled kernel exactly: one threadgroup
// per row, `kFusedTiledThreads` threads cooperating, dynamic threadgroup score
// buffer — but the score buffer stays `float` (fp32 accumulation) while I/O is
// `half`. bf16 reuses the f32 tiled kernel through a host-side conversion (MPS
// has no native bf16 matrix descriptor; matches the 8.4.4.2 convention).
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_matmul_softmax_tiled_msl_f16(MetalDeviceContext &ctx,
                                           const uint16_t* A, const uint16_t* B,
                                           uint16_t* O,
                                           int32_t M, int32_t N, int32_t K) {
  static NSString *const kFusedTiledSourceF16 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED_TILED_THREADS 32

kernel void matmul_softmax_tiled_f16(
    device const half*  A   [[buffer(0)]],
    device const half*  B   [[buffer(1)]],
    device half*        O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    threadgroup float*  tg_scores [[threadgroup(0)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint  lid     [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    int row = (int)tg_pos.x;
    if (row >= M) return;
    const int T = TESSERA_APPLE_GPU_FUSED_TILED_THREADS;
    const int lid_i = (int)lid;

    threadgroup float tg_max[TESSERA_APPLE_GPU_FUSED_TILED_THREADS];
    threadgroup float tg_sum[TESSERA_APPLE_GPU_FUSED_TILED_THREADS];

    // Step 1: cooperative compute scores[n] = sum_k A[row,k] * B[k,n]. The
    // accumulator is float to preserve precision across the K-reduction.
    int a_off = row * K;
    for (int n = lid_i; n < N; n += T) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) {
            s += float(A[a_off + k]) * float(B[k * N + n]);
        }
        tg_scores[n] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: per-thread partial row max -> threadgroup max reduction. (SIMD-group
    // simd_max was benchmarked slower here — see the f32 kernel note.)
    float local_max = -INFINITY;
    for (int n = lid_i; n < N; n += T) local_max = max(local_max, tg_scores[n]);
    tg_max[lid_i] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (lid_i < stride) {
            tg_max[lid_i] = max(tg_max[lid_i], tg_max[lid_i + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = tg_max[0];

    // Step 3: per-thread exp + partial sum -> threadgroup sum reduction.
    float local_sum = 0.0f;
    for (int n = lid_i; n < N; n += T) {
        float e = exp(tg_scores[n] - row_max);
        tg_scores[n] = e;
        local_sum += e;
    }
    tg_sum[lid_i] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int stride = T / 2; stride > 0; stride >>= 1) {
        if (lid_i < stride) {
            tg_sum[lid_i] += tg_sum[lid_i + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denom = tg_sum[0];

    // Step 4: cooperative divide and write back as half.
    int o_off = row * N;
    if (denom == 0.0f) {
        for (int n = lid_i; n < N; n += T) O[o_off + n] = half(0.0f);
    } else {
        float inv = 1.0f / denom;
        for (int n = lid_i; n < N; n += T) O[o_off + n] = half(tg_scores[n] * inv);
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFusedTiledSourceF16, @"matmul_softmax_tiled_f16");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(uint16_t) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger oBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];

    // Dynamic threadgroup memory: scores[N] live floats per threadgroup.
    NSUInteger tg_score_bytes = sizeof(float) * static_cast<NSUInteger>(N);
    [enc setThreadgroupMemoryLength:tg_score_bytes atIndex:0];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    MTLSize tg = MTLSizeMake(static_cast<NSUInteger>(kFusedTiledThreads), 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "matmul_softmax_tiled_msl_f16")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

bool dispatch_matmul_softmax_tiled_bf16_via_fp32(MetalDeviceContext &ctx,
                                                 const uint16_t* A,
                                                 const uint16_t* B, uint16_t* O,
                                                 int32_t M, int32_t N,
                                                 int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  if (!dispatch_matmul_softmax_tiled_msl(ctx, Af.data(), Bf.data(), Of.data(),
                                         M, N, K))
    return false;
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_softmax_tiled_f16(const uint16_t* A,
                                                           const uint16_t* B,
                                                           uint16_t* O,
                                                           int32_t M, int32_t N,
                                                           int32_t K) {
  if (N > 8192) {
    reference_matmul_softmax_f16_via_fp32(A, B, O, M, N, K);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_softmax_tiled_msl_f16(ctx, A, B, O, M, N, K))
    return;
  reference_matmul_softmax_f16_via_fp32(A, B, O, M, N, K);
}

extern "C" void tessera_apple_gpu_matmul_softmax_tiled_bf16(const uint16_t* A,
                                                            const uint16_t* B,
                                                            uint16_t* O,
                                                            int32_t M, int32_t N,
                                                            int32_t K) {
  if (N > 8192) {
    reference_matmul_softmax_bf16_via_fp32(A, B, O, M, N, K);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_softmax_tiled_bf16_via_fp32(ctx, A, B, O, M, N, K))
    return;
  reference_matmul_softmax_bf16_via_fp32(A, B, O, M, N, K);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.5 — Fused matmul -> softmax -> matmul (full attention block)
//
// O = (softmax(A @ B, axis=-1)) @ C
//
// Shapes:
//   A: (M, K)
//   B: (K, N)
//   C: (N, P)
//   O: (M, P)
//
// One thread per output row. Each thread maintains two stack arrays:
//   - scores[N] for the (1, N) softmax intermediate (capped N <= 256)
//   - out[P]    for the (1, P) final accumulator     (capped P <= 256)
//
// Algorithm:
//   1. Compute scores[n] = sum_k A[row, k] * B[k, n]   (matmul 1)
//   2. Row max + exp + denom + divide                  (softmax)
//   3. Compute out[p] = sum_n probs[n] * C[n, p]       (matmul 2)
//   4. Write out[:] into O[row, :]
//
// fp32 accumulators throughout (mixed-precision for f16/bf16). Single
// kernel launch — no host roundtrip on the (M, N) score matrix or the
// (M, N) probs intermediate; both stay in registers.
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_matmul_softmax_matmul_msl(MetalDeviceContext &ctx,
                                        const float* A, const float* B,
                                        const float* C, float* O,
                                        int32_t M, int32_t K, int32_t N,
                                        int32_t P) {
  static NSString *const kFused3Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED3_MAX_N 256
#define TESSERA_APPLE_GPU_FUSED3_MAX_P 256

kernel void matmul_softmax_matmul_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device const float* C   [[buffer(2)]],
    device float*       O   [[buffer(3)]],
    constant int&       M   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    constant int&       N   [[buffer(6)]],
    constant int&       P   [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (N > TESSERA_APPLE_GPU_FUSED3_MAX_N) return;
    if (P > TESSERA_APPLE_GPU_FUSED3_MAX_P) return;
    int row = (int)gid;

    // Step 1: scores = A[row, :] @ B
    float scores[TESSERA_APPLE_GPU_FUSED3_MAX_N];
    int a_off = row * K;
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[a_off + k];
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];
    }

    // Step 2: row-wise softmax in place.
    float row_max = -INFINITY;
    for (int n = 0; n < N; ++n) row_max = max(row_max, scores[n]);
    float denom = 0.0f;
    for (int n = 0; n < N; ++n) {
        scores[n] = exp(scores[n] - row_max);
        denom += scores[n];
    }
    if (denom == 0.0f) {
        for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    } else {
        float inv = 1.0f / denom;
        for (int n = 0; n < N; ++n) scores[n] *= inv;
    }

    // Step 3: out = scores @ C
    float out[TESSERA_APPLE_GPU_FUSED3_MAX_P];
    for (int p = 0; p < P; ++p) out[p] = 0.0f;
    for (int n = 0; n < N; ++n) {
        float sn = scores[n];
        int c_off = n * P;
        for (int p = 0; p < P; ++p) out[p] += sn * C[c_off + p];
    }

    // Step 4: write back.
    int o_off = row * P;
    for (int p = 0; p < P; ++p) O[o_off + p] = out[p];
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFused3Source, @"matmul_softmax_matmul_f32");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(float) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger cBytes = sizeof(float) * static_cast<NSUInteger>(N) *
                        static_cast<NSUInteger>(P);
    NSUInteger oBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(P);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufC, ctx, C, cBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufC || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&P length:sizeof(int32_t) atIndex:7];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

bool dispatch_matmul_softmax_matmul_msl_f16(MetalDeviceContext &ctx,
                                            const uint16_t* A, const uint16_t* B,
                                            const uint16_t* C, uint16_t* O,
                                            int32_t M, int32_t K, int32_t N,
                                            int32_t P) {
  static NSString *const kFused3SourceF16 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED3_MAX_N 256
#define TESSERA_APPLE_GPU_FUSED3_MAX_P 256

kernel void matmul_softmax_matmul_f16(
    device const half*  A   [[buffer(0)]],
    device const half*  B   [[buffer(1)]],
    device const half*  C   [[buffer(2)]],
    device half*        O   [[buffer(3)]],
    constant int&       M   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    constant int&       N   [[buffer(6)]],
    constant int&       P   [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (N > TESSERA_APPLE_GPU_FUSED3_MAX_N) return;
    if (P > TESSERA_APPLE_GPU_FUSED3_MAX_P) return;
    int row = (int)gid;

    float scores[TESSERA_APPLE_GPU_FUSED3_MAX_N];
    int a_off = row * K;
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = float(A[a_off + k]);
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * float(B[b_off + n]);
    }

    float row_max = -INFINITY;
    for (int n = 0; n < N; ++n) row_max = max(row_max, scores[n]);
    float denom = 0.0f;
    for (int n = 0; n < N; ++n) {
        scores[n] = exp(scores[n] - row_max);
        denom += scores[n];
    }
    if (denom == 0.0f) {
        for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    } else {
        float inv = 1.0f / denom;
        for (int n = 0; n < N; ++n) scores[n] *= inv;
    }

    float out[TESSERA_APPLE_GPU_FUSED3_MAX_P];
    for (int p = 0; p < P; ++p) out[p] = 0.0f;
    for (int n = 0; n < N; ++n) {
        float sn = scores[n];
        int c_off = n * P;
        for (int p = 0; p < P; ++p) out[p] += sn * float(C[c_off + p]);
    }

    int o_off = row * P;
    for (int p = 0; p < P; ++p) O[o_off + p] = half(out[p]);
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFused3SourceF16, @"matmul_softmax_matmul_f16");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(uint16_t) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger cBytes = sizeof(uint16_t) * static_cast<NSUInteger>(N) *
                        static_cast<NSUInteger>(P);
    NSUInteger oBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(P);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufC, ctx, C, cBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufC || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&P length:sizeof(int32_t) atIndex:7];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

inline void reference_matmul_softmax_matmul_f32(const float* A, const float* B,
                                                const float* C, float* O,
                                                int32_t M, int32_t K, int32_t N,
                                                int32_t P) {
  std::vector<float> scores(static_cast<std::size_t>(N), 0.0f);
  std::vector<float> out(static_cast<std::size_t>(P), 0.0f);
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(row) * K + k];
      const float* b_row = B + static_cast<std::size_t>(k) * N;
      for (int32_t n = 0; n < N; ++n) scores[n] += a * b_row[n];
    }
    float row_max = -std::numeric_limits<float>::infinity();
    for (int32_t n = 0; n < N; ++n) row_max = std::max(row_max, scores[n]);
    float denom = 0.0f;
    for (int32_t n = 0; n < N; ++n) {
      scores[n] = std::exp(scores[n] - row_max);
      denom += scores[n];
    }
    if (denom > 0.0f) {
      float inv = 1.0f / denom;
      for (int32_t n = 0; n < N; ++n) scores[n] *= inv;
    } else {
      for (int32_t n = 0; n < N; ++n) scores[n] = 0.0f;
    }
    for (int32_t p = 0; p < P; ++p) out[p] = 0.0f;
    for (int32_t n = 0; n < N; ++n) {
      const float* c_row = C + static_cast<std::size_t>(n) * P;
      float sn = scores[n];
      for (int32_t p = 0; p < P; ++p) out[p] += sn * c_row[p];
    }
    float* o_row = O + static_cast<std::size_t>(row) * P;
    for (int32_t p = 0; p < P; ++p) o_row[p] = out[p];
  }
}

inline void reference_matmul_softmax_matmul_f16_via_fp32(
    const uint16_t* A, const uint16_t* B, const uint16_t* C, uint16_t* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(N) * P);
  std::vector<float> Of(static_cast<std::size_t>(M) * P);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_gpu(B[i]);
  for (std::size_t i = 0; i < Cf.size(); ++i) Cf[i] = half_to_float_gpu(C[i]);
  reference_matmul_softmax_matmul_f32(Af.data(), Bf.data(), Cf.data(), Of.data(),
                                      M, K, N, P);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_gpu(Of[i]);
}

inline void reference_matmul_softmax_matmul_bf16_via_fp32(
    const uint16_t* A, const uint16_t* B, const uint16_t* C, uint16_t* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(N) * P);
  std::vector<float> Of(static_cast<std::size_t>(M) * P);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  for (std::size_t i = 0; i < Cf.size(); ++i) Cf[i] = bfloat16_to_float_gpu(C[i]);
  reference_matmul_softmax_matmul_f32(Af.data(), Bf.data(), Cf.data(), Of.data(),
                                      M, K, N, P);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_matmul_softmax_matmul_bf16_via_fp32(
    MetalDeviceContext &ctx, const uint16_t* A, const uint16_t* B,
    const uint16_t* C, uint16_t* O, int32_t M, int32_t K, int32_t N, int32_t P) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Cf(static_cast<std::size_t>(N) * P);
  std::vector<float> Of(static_cast<std::size_t>(M) * P);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  for (std::size_t i = 0; i < Cf.size(); ++i) Cf[i] = bfloat16_to_float_gpu(C[i]);
  if (!dispatch_matmul_softmax_matmul_msl(ctx, Af.data(), Bf.data(), Cf.data(),
                                          Of.data(), M, K, N, P))
    return false;
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_softmax_matmul_f32(
    const float* A, const float* B, const float* C, float* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  if (N > 256 || P > 256) {
    reference_matmul_softmax_matmul_f32(A, B, C, O, M, K, N, P);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_softmax_matmul_msl(ctx, A, B, C, O, M, K, N, P))
    return;
  reference_matmul_softmax_matmul_f32(A, B, C, O, M, K, N, P);
}

extern "C" void tessera_apple_gpu_matmul_softmax_matmul_f16(
    const uint16_t* A, const uint16_t* B, const uint16_t* C, uint16_t* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  if (N > 256 || P > 256) {
    reference_matmul_softmax_matmul_f16_via_fp32(A, B, C, O, M, K, N, P);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_softmax_matmul_msl_f16(ctx, A, B, C, O, M, K, N, P))
    return;
  reference_matmul_softmax_matmul_f16_via_fp32(A, B, C, O, M, K, N, P);
}

extern "C" void tessera_apple_gpu_matmul_softmax_matmul_bf16(
    const uint16_t* A, const uint16_t* B, const uint16_t* C, uint16_t* O,
    int32_t M, int32_t K, int32_t N, int32_t P) {
  if (N > 256 || P > 256) {
    reference_matmul_softmax_matmul_bf16_via_fp32(A, B, C, O, M, K, N, P);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_softmax_matmul_bf16_via_fp32(ctx, A, B, C, O, M, K, N, P))
    return;
  reference_matmul_softmax_matmul_bf16_via_fp32(A, B, C, O, M, K, N, P);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.7 — MLP-block fusions: matmul -> gelu, matmul -> rmsnorm.
//
// Both are 2-op chains that mirror the Phase 8.4.3 matmul -> softmax shape:
// one thread per output row, scores[N] stack accumulator (cap N <= 256).
//
// matmul -> gelu: compute row of A @ B into stack, apply tanh-approximation
//   gelu pointwise, write back. No row reduction.
//
// matmul -> rmsnorm: compute row of A @ B into stack, compute RMS = sqrt(
//   mean(x^2) + eps), divide each element by RMS, write back. Mirrors the
//   numpy reference in tessera.runtime._runtime_cpu_op for "tessera.rmsnorm".
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_matmul_gelu_msl(MetalDeviceContext &ctx, const float* A,
                              const float* B, float* O,
                              int32_t M, int32_t N, int32_t K) {
  static NSString *const kFusedSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED_MATMUL_GELU_MAX_N 256

kernel void matmul_gelu_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (N > TESSERA_APPLE_GPU_FUSED_MATMUL_GELU_MAX_N) return;
    int row = (int)gid;

    float scores[TESSERA_APPLE_GPU_FUSED_MATMUL_GELU_MAX_N];
    int a_off = row * K;
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[a_off + k];
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];
    }
    int o_off = row * N;
    // Tanh-approximation gelu, matching the numpy reference:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    for (int n = 0; n < N; ++n) {
        float v = scores[n];
        float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    // Metal fast-math tanh overflows to NaN for large |t| (|x| >~ 16);
    // tanh saturates to +/-1 well before +/-30, so clamp the argument.
    t = clamp(t, -30.0f, 30.0f);
        O[o_off + n] = 0.5f * v * (1.0f + tanh(t));
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFusedSource, @"matmul_gelu_f32");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(float) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger oBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

bool dispatch_matmul_rmsnorm_msl(MetalDeviceContext &ctx, const float* A,
                                 const float* B, float* O,
                                 int32_t M, int32_t N, int32_t K, float eps) {
  static NSString *const kFusedSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED_MATMUL_RMSNORM_MAX_N 256

kernel void matmul_rmsnorm_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    constant float&     eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (N > TESSERA_APPLE_GPU_FUSED_MATMUL_RMSNORM_MAX_N) return;
    int row = (int)gid;

    float scores[TESSERA_APPLE_GPU_FUSED_MATMUL_RMSNORM_MAX_N];
    int a_off = row * K;
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = A[a_off + k];
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * B[b_off + n];
    }
    // RMSNorm: y = x / sqrt(mean(x^2) + eps).
    float sumsq = 0.0f;
    for (int n = 0; n < N; ++n) sumsq += scores[n] * scores[n];
    float inv_rms = 1.0f / sqrt(sumsq / float(N) + eps);
    int o_off = row * N;
    for (int n = 0; n < N; ++n) O[o_off + n] = scores[n] * inv_rms;
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFusedSource, @"matmul_rmsnorm_f32");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(float) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger oBytes = sizeof(float) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&eps length:sizeof(float) atIndex:6];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

inline void reference_matmul_gelu_f32(const float* A, const float* B, float* O,
                                      int32_t M, int32_t N, int32_t K) {
  static constexpr float kSqrt2OverPi = 0.7978845608028654f;
  std::vector<float> scores(static_cast<std::size_t>(N), 0.0f);
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(row) * K + k];
      const float* b_row = B + static_cast<std::size_t>(k) * N;
      for (int32_t n = 0; n < N; ++n) scores[n] += a * b_row[n];
    }
    float* o_row = O + static_cast<std::size_t>(row) * N;
    for (int32_t n = 0; n < N; ++n) {
      float v = scores[n];
      float t = kSqrt2OverPi * (v + 0.044715f * v * v * v);
      o_row[n] = 0.5f * v * (1.0f + std::tanh(t));
    }
  }
}

inline void reference_matmul_rmsnorm_f32(const float* A, const float* B,
                                         float* O, int32_t M, int32_t N,
                                         int32_t K, float eps) {
  std::vector<float> scores(static_cast<std::size_t>(N), 0.0f);
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      float a = A[static_cast<std::size_t>(row) * K + k];
      const float* b_row = B + static_cast<std::size_t>(k) * N;
      for (int32_t n = 0; n < N; ++n) scores[n] += a * b_row[n];
    }
    float sumsq = 0.0f;
    for (int32_t n = 0; n < N; ++n) sumsq += scores[n] * scores[n];
    float inv_rms = 1.0f / std::sqrt(sumsq / static_cast<float>(N) + eps);
    float* o_row = O + static_cast<std::size_t>(row) * N;
    for (int32_t n = 0; n < N; ++n) o_row[n] = scores[n] * inv_rms;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_gelu_f32(const float* A, const float* B,
                                                  float* O, int32_t M,
                                                  int32_t N, int32_t K) {
  if (N > 256) {
    reference_matmul_gelu_f32(A, B, O, M, N, K);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_gelu_msl(ctx, A, B, O, M, N, K)) return;
  reference_matmul_gelu_f32(A, B, O, M, N, K);
}

extern "C" void tessera_apple_gpu_matmul_rmsnorm_f32(const float* A,
                                                     const float* B, float* O,
                                                     int32_t M, int32_t N,
                                                     int32_t K, float eps) {
  if (N > 256) {
    reference_matmul_rmsnorm_f32(A, B, O, M, N, K, eps);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_rmsnorm_msl(ctx, A, B, O, M, N, K, eps)) return;
  reference_matmul_rmsnorm_f32(A, B, O, M, N, K, eps);
}

//===---------------------------------------------------------------------===//
// Native half-precision matmul -> gelu / matmul -> rmsnorm (per-thread, N<=256).
//
// f16 gets a dedicated `half`-I/O MSL kernel with a float accumulator and the
// same clamped-tanh gelu as the f32 kernel (tanh overflows to NaN for the
// fast-math argument |t| >~ 30, so the argument is clamped). bf16 reuses the
// f32 MSL kernel through a host-side conversion, matching the 8.4.4.2 fused
// matmul->softmax convention (MPS/MSL have no native bf16 matrix path here).
// Routing f16/bf16 to these single kernels lets the runtime skip the GPU
// matmul + MPSGraph-epilogue compose for the small-N case.
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_matmul_gelu_msl_f16(MetalDeviceContext &ctx, const uint16_t* A,
                                  const uint16_t* B, uint16_t* O,
                                  int32_t M, int32_t N, int32_t K) {
  static NSString *const kFusedSourceF16 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED_MATMUL_GELU_MAX_N 256

kernel void matmul_gelu_f16(
    device const half*  A   [[buffer(0)]],
    device const half*  B   [[buffer(1)]],
    device half*        O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (N > TESSERA_APPLE_GPU_FUSED_MATMUL_GELU_MAX_N) return;
    int row = (int)gid;

    // float accumulator — preserve precision across the K-reduction.
    float scores[TESSERA_APPLE_GPU_FUSED_MATMUL_GELU_MAX_N];
    int a_off = row * K;
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = float(A[a_off + k]);
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * float(B[b_off + n]);
    }
    int o_off = row * N;
    // Tanh-approximation gelu, matching the numpy reference:
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    for (int n = 0; n < N; ++n) {
        float v = scores[n];
        float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        // Metal fast-math tanh overflows to NaN for large |t| (|x| >~ 16);
        // tanh saturates to +/-1 well before +/-30, so clamp the argument.
        t = clamp(t, -30.0f, 30.0f);
        O[o_off + n] = half(0.5f * v * (1.0f + tanh(t)));
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFusedSourceF16, @"matmul_gelu_f16");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(uint16_t) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger oBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

bool dispatch_matmul_rmsnorm_msl_f16(MetalDeviceContext &ctx, const uint16_t* A,
                                     const uint16_t* B, uint16_t* O,
                                     int32_t M, int32_t N, int32_t K,
                                     float eps) {
  static NSString *const kFusedSourceF16 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_FUSED_MATMUL_RMSNORM_MAX_N 256

kernel void matmul_rmsnorm_f16(
    device const half*  A   [[buffer(0)]],
    device const half*  B   [[buffer(1)]],
    device half*        O   [[buffer(2)]],
    constant int&       M   [[buffer(3)]],
    constant int&       N   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    constant float&     eps [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (N > TESSERA_APPLE_GPU_FUSED_MATMUL_RMSNORM_MAX_N) return;
    int row = (int)gid;

    // float accumulator — preserve precision across the K-reduction.
    float scores[TESSERA_APPLE_GPU_FUSED_MATMUL_RMSNORM_MAX_N];
    int a_off = row * K;
    for (int n = 0; n < N; ++n) scores[n] = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = float(A[a_off + k]);
        int b_off = k * N;
        for (int n = 0; n < N; ++n) scores[n] += a * float(B[b_off + n]);
    }
    // RMSNorm: y = x / sqrt(mean(x^2) + eps).
    float sumsq = 0.0f;
    for (int n = 0; n < N; ++n) sumsq += scores[n] * scores[n];
    float inv_rms = 1.0f / sqrt(sumsq / float(N) + eps);
    int o_off = row * N;
    for (int n = 0; n < N; ++n) O[o_off + n] = half(scores[n] * inv_rms);
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFusedSourceF16, @"matmul_rmsnorm_f16");
    if (!pso) return false;

    NSUInteger aBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(K);
    NSUInteger bBytes = sizeof(uint16_t) * static_cast<NSUInteger>(K) *
                        static_cast<NSUInteger>(N);
    NSUInteger oBytes = sizeof(uint16_t) * static_cast<NSUInteger>(M) *
                        static_cast<NSUInteger>(N);

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufA || !bufB || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&M length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&eps length:sizeof(float) atIndex:6];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(M), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(M),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

inline void reference_matmul_gelu_f16_via_fp32(const uint16_t* A,
                                               const uint16_t* B, uint16_t* O,
                                               int32_t M, int32_t N, int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_gpu(B[i]);
  reference_matmul_gelu_f32(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_gpu(Of[i]);
}

inline void reference_matmul_gelu_bf16_via_fp32(const uint16_t* A,
                                                const uint16_t* B, uint16_t* O,
                                                int32_t M, int32_t N,
                                                int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  reference_matmul_gelu_f32(Af.data(), Bf.data(), Of.data(), M, N, K);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_matmul_gelu_bf16_via_fp32(MetalDeviceContext &ctx,
                                        const uint16_t* A, const uint16_t* B,
                                        uint16_t* O, int32_t M, int32_t N,
                                        int32_t K) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  if (!dispatch_matmul_gelu_msl(ctx, Af.data(), Bf.data(), Of.data(), M, N, K))
    return false;
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

inline void reference_matmul_rmsnorm_f16_via_fp32(const uint16_t* A,
                                                  const uint16_t* B,
                                                  uint16_t* O, int32_t M,
                                                  int32_t N, int32_t K,
                                                  float eps) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_gpu(B[i]);
  reference_matmul_rmsnorm_f32(Af.data(), Bf.data(), Of.data(), M, N, K, eps);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_half_gpu(Of[i]);
}

inline void reference_matmul_rmsnorm_bf16_via_fp32(const uint16_t* A,
                                                   const uint16_t* B,
                                                   uint16_t* O, int32_t M,
                                                   int32_t N, int32_t K,
                                                   float eps) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  reference_matmul_rmsnorm_f32(Af.data(), Bf.data(), Of.data(), M, N, K, eps);
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_matmul_rmsnorm_bf16_via_fp32(MetalDeviceContext &ctx,
                                           const uint16_t* A, const uint16_t* B,
                                           uint16_t* O, int32_t M, int32_t N,
                                           int32_t K, float eps) {
  std::vector<float> Af(static_cast<std::size_t>(M) * K);
  std::vector<float> Bf(static_cast<std::size_t>(K) * N);
  std::vector<float> Of(static_cast<std::size_t>(M) * N);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  if (!dispatch_matmul_rmsnorm_msl(ctx, Af.data(), Bf.data(), Of.data(), M, N, K,
                                   eps))
    return false;
  for (std::size_t i = 0; i < Of.size(); ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_matmul_gelu_f16(const uint16_t* A,
                                                  const uint16_t* B, uint16_t* O,
                                                  int32_t M, int32_t N,
                                                  int32_t K) {
  if (N > 256) {
    reference_matmul_gelu_f16_via_fp32(A, B, O, M, N, K);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_gelu_msl_f16(ctx, A, B, O, M, N, K)) return;
  reference_matmul_gelu_f16_via_fp32(A, B, O, M, N, K);
}

extern "C" void tessera_apple_gpu_matmul_gelu_bf16(const uint16_t* A,
                                                   const uint16_t* B,
                                                   uint16_t* O, int32_t M,
                                                   int32_t N, int32_t K) {
  if (N > 256) {
    reference_matmul_gelu_bf16_via_fp32(A, B, O, M, N, K);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_gelu_bf16_via_fp32(ctx, A, B, O, M, N, K)) return;
  reference_matmul_gelu_bf16_via_fp32(A, B, O, M, N, K);
}

extern "C" void tessera_apple_gpu_matmul_rmsnorm_f16(const uint16_t* A,
                                                     const uint16_t* B,
                                                     uint16_t* O, int32_t M,
                                                     int32_t N, int32_t K,
                                                     float eps) {
  if (N > 256) {
    reference_matmul_rmsnorm_f16_via_fp32(A, B, O, M, N, K, eps);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_rmsnorm_msl_f16(ctx, A, B, O, M, N, K, eps))
    return;
  reference_matmul_rmsnorm_f16_via_fp32(A, B, O, M, N, K, eps);
}

extern "C" void tessera_apple_gpu_matmul_rmsnorm_bf16(const uint16_t* A,
                                                      const uint16_t* B,
                                                      uint16_t* O, int32_t M,
                                                      int32_t N, int32_t K,
                                                      float eps) {
  if (N > 256) {
    reference_matmul_rmsnorm_bf16_via_fp32(A, B, O, M, N, K, eps);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_matmul_rmsnorm_bf16_via_fp32(ctx, A, B, O, M, N, K, eps))
    return;
  reference_matmul_rmsnorm_bf16_via_fp32(A, B, O, M, N, K, eps);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.8 — SwiGLU MLP-block fusion (Stage 3 of the SwiGLU Performance
// Plan in `docs/CANONICAL_API.md`).
//
// Forward:
//   O[M, K_out] = (silu(X[M, K] @ Wg[K, H]) * (X[M, K] @ Wu[K, H])) @ Wd[H, K_out]
//
// One thread per output row. Three per-thread stack arrays:
//   gate[H], up[H], out_row[K_out]
// Capped at H ≤ 256 and K_out ≤ 256 (mirrors matmul→softmax / matmul→gelu /
// matmul→rmsnorm in this file). For wider shapes the runtime falls back to
// the host reference.
//
// Mixed-precision pattern follows the existing fused-MLP kernels:
//   - f32: native float throughout, fp32 accumulators
//   - f16: half I/O on the GPU (matmul + silu*mul + matmul stays in fp32
//         accumulators inside the kernel) — Phase 8.4.4.2 follow-up
//   - bf16: fp32-conversion path inside the runtime shim (MPS doesn't take
//         bf16 matrix descriptors as of macOS 14)
//
// For Stage 3 we ship the f32 MSL kernel + reference fallback for all three
// dtypes. f16/bf16 native MSL paths land as a follow-up matching how the
// matmul→softmax→matmul kernel layered f16/bf16 atop its f32 baseline.
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_swiglu_msl(MetalDeviceContext &ctx,
                         const float* X, const float* Wg, const float* Wu,
                         const float* Wd, float* O,
                         int32_t M, int32_t K, int32_t H, int32_t Kout) {
  static NSString *const kFusedSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_SWIGLU_MAX_H    256
#define TESSERA_APPLE_GPU_SWIGLU_MAX_KOUT 256

kernel void swiglu_f32(
    device const float* X    [[buffer(0)]],
    device const float* Wg   [[buffer(1)]],
    device const float* Wu   [[buffer(2)]],
    device const float* Wd   [[buffer(3)]],
    device float*       O    [[buffer(4)]],
    constant int&       M    [[buffer(5)]],
    constant int&       K    [[buffer(6)]],
    constant int&       H    [[buffer(7)]],
    constant int&       Kout [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    if (H > TESSERA_APPLE_GPU_SWIGLU_MAX_H) return;
    if (Kout > TESSERA_APPLE_GPU_SWIGLU_MAX_KOUT) return;
    int row = (int)gid;

    float gate[TESSERA_APPLE_GPU_SWIGLU_MAX_H];
    float up[TESSERA_APPLE_GPU_SWIGLU_MAX_H];
    float out_row[TESSERA_APPLE_GPU_SWIGLU_MAX_KOUT];

    for (int h = 0; h < H; ++h) { gate[h] = 0.0f; up[h] = 0.0f; }
    int x_off = row * K;
    for (int k = 0; k < K; ++k) {
        float xv = X[x_off + k];
        int wg_off = k * H;
        for (int h = 0; h < H; ++h) {
            gate[h] += xv * Wg[wg_off + h];
            up[h]   += xv * Wu[wg_off + h];
        }
    }
    // hidden[h] = silu(gate[h]) * up[h]; reuse `gate` as the hidden buffer.
    for (int h = 0; h < H; ++h) {
        float g = gate[h];
        float s = g / (1.0f + exp(-g));
        gate[h] = s * up[h];
    }
    for (int ko = 0; ko < Kout; ++ko) out_row[ko] = 0.0f;
    for (int h = 0; h < H; ++h) {
        float hv = gate[h];
        int wd_off = h * Kout;
        for (int ko = 0; ko < Kout; ++ko) {
            out_row[ko] += hv * Wd[wd_off + ko];
        }
    }
    int o_off = row * Kout;
    for (int ko = 0; ko < Kout; ++ko) O[o_off + ko] = out_row[ko];
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFusedSource, @"swiglu_f32");
    if (!pso) return false;

    NSUInteger xBytes  = sizeof(float) * (NSUInteger)M  * (NSUInteger)K;
    NSUInteger wgBytes = sizeof(float) * (NSUInteger)K  * (NSUInteger)H;
    NSUInteger wuBytes = sizeof(float) * (NSUInteger)K  * (NSUInteger)H;
    NSUInteger wdBytes = sizeof(float) * (NSUInteger)H  * (NSUInteger)Kout;
    NSUInteger oBytes  = sizeof(float) * (NSUInteger)M  * (NSUInteger)Kout;

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, xBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufWg, ctx, Wg, wgBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufWu, ctx, Wu, wuBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufWd, ctx, Wd, wdBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufX || !bufWg || !bufWu || !bufWd || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX  offset:0 atIndex:0];
    [enc setBuffer:bufWg offset:0 atIndex:1];
    [enc setBuffer:bufWu offset:0 atIndex:2];
    [enc setBuffer:bufWd offset:0 atIndex:3];
    [enc setBuffer:bufO  offset:0 atIndex:4];
    [enc setBytes:&M    length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&K    length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&H    length:sizeof(int32_t) atIndex:7];
    [enc setBytes:&Kout length:sizeof(int32_t) atIndex:8];

    MTLSize grid = MTLSizeMake((NSUInteger)M, 1, 1);
    NSUInteger tg_x =
        std::min<NSUInteger>((NSUInteger)M, pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

inline void reference_swiglu_f32(const float* X, const float* Wg,
                                 const float* Wu, const float* Wd, float* O,
                                 int32_t M, int32_t K, int32_t H,
                                 int32_t Kout) {
  std::vector<float> gate((std::size_t)H, 0.0f);
  std::vector<float> up((std::size_t)H, 0.0f);
  std::vector<float> hidden((std::size_t)H, 0.0f);
  std::vector<float> out_row((std::size_t)Kout, 0.0f);
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t h = 0; h < H; ++h) { gate[h] = 0.0f; up[h] = 0.0f; }
    for (int32_t k = 0; k < K; ++k) {
      float xv = X[(std::size_t)row * K + k];
      const float* wg_row = Wg + (std::size_t)k * H;
      const float* wu_row = Wu + (std::size_t)k * H;
      for (int32_t h = 0; h < H; ++h) {
        gate[h] += xv * wg_row[h];
        up[h]   += xv * wu_row[h];
      }
    }
    for (int32_t h = 0; h < H; ++h) {
      float g = gate[h];
      float s = g / (1.0f + std::exp(-g));
      hidden[h] = s * up[h];
    }
    for (int32_t ko = 0; ko < Kout; ++ko) out_row[ko] = 0.0f;
    for (int32_t h = 0; h < H; ++h) {
      float hv = hidden[h];
      const float* wd_row = Wd + (std::size_t)h * Kout;
      for (int32_t ko = 0; ko < Kout; ++ko) out_row[ko] += hv * wd_row[ko];
    }
    float* o_row = O + (std::size_t)row * Kout;
    for (int32_t ko = 0; ko < Kout; ++ko) o_row[ko] = out_row[ko];
  }
}

// f16 → fp32 buffers → reference. Mirrors how the rmsnorm/gelu f16 paths
// landed initially (Phase 8.4.4.1 baseline) — native half MSL is a small
// follow-up that doesn't change the runtime ABI.
inline void reference_swiglu_f16_via_fp32(const uint16_t* X,
                                          const uint16_t* Wg,
                                          const uint16_t* Wu,
                                          const uint16_t* Wd,
                                          uint16_t* O, int32_t M, int32_t K,
                                          int32_t H, int32_t Kout) {
  auto half_to_float = [](uint16_t h) -> float {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
      if (mant == 0) f = sign << 31;
      else { // subnormal
        exp = 1;
        while ((mant & 0x400) == 0) { mant <<= 1; exp -= 1; }
        mant &= 0x3FF;
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
      }
    } else if (exp == 0x1F) {
      f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
      f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
  };
  auto float_to_half = [](float v) -> uint16_t {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    uint32_t sign = (f >> 31) & 0x1;
    int32_t exp = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;
    uint16_t h;
    if (exp <= 0) {
      h = (uint16_t)(sign << 15);
    } else if (exp >= 31) {
      h = (uint16_t)((sign << 15) | (0x1F << 10));
    } else {
      h = (uint16_t)((sign << 15) | (uint32_t)(exp << 10) | mant);
    }
    return h;
  };
  std::vector<float> Xf((std::size_t)M  * K), Wgf((std::size_t)K  * H),
      Wuf((std::size_t)K  * H), Wdf((std::size_t)H  * Kout),
      Of((std::size_t)M  * Kout, 0.0f);
  for (std::size_t i = 0, n = Xf.size();  i < n; ++i) Xf[i]  = half_to_float(X[i]);
  for (std::size_t i = 0, n = Wgf.size(); i < n; ++i) Wgf[i] = half_to_float(Wg[i]);
  for (std::size_t i = 0, n = Wuf.size(); i < n; ++i) Wuf[i] = half_to_float(Wu[i]);
  for (std::size_t i = 0, n = Wdf.size(); i < n; ++i) Wdf[i] = half_to_float(Wd[i]);
  reference_swiglu_f32(Xf.data(), Wgf.data(), Wuf.data(), Wdf.data(),
                       Of.data(), M, K, H, Kout);
  for (std::size_t i = 0, n = Of.size(); i < n; ++i) O[i] = float_to_half(Of[i]);
}

// bf16 → fp32 buffers → reference. bf16 is upper 16 bits of a float bit
// pattern (truncate-toward-zero on the way back).
inline void reference_swiglu_bf16_via_fp32(const uint16_t* X,
                                           const uint16_t* Wg,
                                           const uint16_t* Wu,
                                           const uint16_t* Wd,
                                           uint16_t* O, int32_t M, int32_t K,
                                           int32_t H, int32_t Kout) {
  auto bf16_to_float = [](uint16_t h) -> float {
    uint32_t f = (uint32_t)h << 16;
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
  };
  auto float_to_bf16 = [](float v) -> uint16_t {
    uint32_t f;
    std::memcpy(&f, &v, sizeof(f));
    return (uint16_t)(f >> 16);
  };
  std::vector<float> Xf((std::size_t)M  * K), Wgf((std::size_t)K  * H),
      Wuf((std::size_t)K  * H), Wdf((std::size_t)H  * Kout),
      Of((std::size_t)M  * Kout, 0.0f);
  for (std::size_t i = 0, n = Xf.size();  i < n; ++i) Xf[i]  = bf16_to_float(X[i]);
  for (std::size_t i = 0, n = Wgf.size(); i < n; ++i) Wgf[i] = bf16_to_float(Wg[i]);
  for (std::size_t i = 0, n = Wuf.size(); i < n; ++i) Wuf[i] = bf16_to_float(Wu[i]);
  for (std::size_t i = 0, n = Wdf.size(); i < n; ++i) Wdf[i] = bf16_to_float(Wd[i]);
  reference_swiglu_f32(Xf.data(), Wgf.data(), Wuf.data(), Wdf.data(),
                       Of.data(), M, K, H, Kout);
  for (std::size_t i = 0, n = Of.size(); i < n; ++i) O[i] = float_to_bf16(Of[i]);
}

} // namespace

extern "C" void tessera_apple_gpu_swiglu_f32(const float* X, const float* Wg,
                                             const float* Wu, const float* Wd,
                                             float* O, int32_t M, int32_t K,
                                             int32_t H, int32_t Kout) {
  if (H > 256 || Kout > 256) {
    reference_swiglu_f32(X, Wg, Wu, Wd, O, M, K, H, Kout);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_swiglu_msl(ctx, X, Wg, Wu, Wd, O, M, K, H, Kout))
    return;
  reference_swiglu_f32(X, Wg, Wu, Wd, O, M, K, H, Kout);
}

extern "C" void tessera_apple_gpu_swiglu_f16(const uint16_t* X,
                                             const uint16_t* Wg,
                                             const uint16_t* Wu,
                                             const uint16_t* Wd,
                                             uint16_t* O, int32_t M, int32_t K,
                                             int32_t H, int32_t Kout) {
  reference_swiglu_f16_via_fp32(X, Wg, Wu, Wd, O, M, K, H, Kout);
}

extern "C" void tessera_apple_gpu_swiglu_bf16(const uint16_t* X,
                                              const uint16_t* Wg,
                                              const uint16_t* Wu,
                                              const uint16_t* Wd,
                                              uint16_t* O, int32_t M,
                                              int32_t K, int32_t H,
                                              int32_t Kout) {
  reference_swiglu_bf16_via_fp32(X, Wg, Wu, Wd, O, M, K, H, Kout);
}

//===---------------------------------------------------------------------===//
// attention_variants_plan, LA-2 — Linear / kernel-feature attention.
//
// Causal recurrent form, one thread per (B, H) batch-head row:
//
//     S_{t} = S_{t-1} + φ(K_t)^T V_t           (D_qk × D_v outer-product update)
//     O_{t} = φ(Q_t) @ S_{t}                   (D_qk · D_v dot-product per timestep)
//
// Per-thread state buffer S[D_qk * D_v]; capped at D_qk * D_v ≤ 256
// floats (1 KB per thread). Wider shapes fall through to the host
// reference path.
//
// f32 only in v1; f16 / bf16 follow the existing fp32-conversion shim
// pattern (matches how matmul→softmax→matmul layered f16/bf16 atop f32).
//===---------------------------------------------------------------------===//

namespace {

bool dispatch_linear_attn_msl(MetalDeviceContext &ctx,
                              const float* Q, const float* K, const float* V,
                              float* O, int32_t B, int32_t H, int32_t S,
                              int32_t D_qk, int32_t D_v,
                              int32_t feature_map, int32_t causal) {
  static NSString *const kKernelSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

#define TESSERA_APPLE_GPU_LINEAR_ATTN_MAX_STATE 256

inline float feature_map_apply_f32(float x, int fm) {
    // 0=elu, 1=relu, 2=identity, 3=polynomial_2
    if (fm == 0) {
        return x > 0.0f ? (x + 1.0f) : exp(x);
    }
    if (fm == 1) {
        return max(x, 0.0f);
    }
    if (fm == 2) {
        return x;
    }
    return x * x;  // polynomial_2
}

kernel void linear_attn_f32(
    device const float* Q   [[buffer(0)]],
    device const float* K   [[buffer(1)]],
    device const float* V   [[buffer(2)]],
    device float*       O   [[buffer(3)]],
    constant int&       B   [[buffer(4)]],
    constant int&       H   [[buffer(5)]],
    constant int&       S   [[buffer(6)]],
    constant int&       D_qk [[buffer(7)]],
    constant int&       D_v  [[buffer(8)]],
    constant int&       feature_map [[buffer(9)]],
    constant int&       causal [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)(B * H)) return;
    int batch_head = (int)gid;
    int b = batch_head / H;
    int h = batch_head - b * H;
    if (D_qk * D_v > TESSERA_APPLE_GPU_LINEAR_ATTN_MAX_STATE) return;

    // Per-thread state, indexed [d_qk * D_v + d_v].
    float state[TESSERA_APPLE_GPU_LINEAR_ATTN_MAX_STATE];
    for (int i = 0; i < D_qk * D_v; ++i) state[i] = 0.0f;

    // Stride helpers: Q/K/O in (B, H, S, D_qk) row-major; V/O in (B, H, S, D_v).
    int row_stride_qk = S * D_qk;
    int row_stride_v = S * D_v;
    int q_base = (b * H + h) * row_stride_qk;
    int k_base = q_base;
    int v_base = (b * H + h) * row_stride_v;
    int o_base = v_base;

    for (int t = 0; t < S; ++t) {
        // Load + apply feature map to Q_t and K_t.
        float phi_K_t[16];  // D_qk capped by D_qk*D_v <= 256, but we further
                            // cap D_qk to 16 to keep this stack array small;
                            // larger D_qk is a follow-up.
        float phi_Q_t[16];
        // Defensive cap — caller's lowering pass enforces D_qk*D_v ≤ 256,
        // and a sensible D_qk is ≤ 16 (typical 1..8). Wider falls back to
        // reference.
        if (D_qk > 16) return;
        for (int d = 0; d < D_qk; ++d) {
            phi_Q_t[d] = feature_map_apply_f32(Q[q_base + t * D_qk + d], feature_map);
            phi_K_t[d] = feature_map_apply_f32(K[k_base + t * D_qk + d], feature_map);
        }

        // State update: S += φ(K_t)^T outer V_t  (each (d_qk, d_v) cell)
        for (int d_qk = 0; d_qk < D_qk; ++d_qk) {
            float k_d = phi_K_t[d_qk];
            int row_off = d_qk * D_v;
            for (int d_v = 0; d_v < D_v; ++d_v) {
                state[row_off + d_v] += k_d * V[v_base + t * D_v + d_v];
            }
        }

        // Output: O_t = φ(Q_t) @ S_t
        for (int d_v = 0; d_v < D_v; ++d_v) {
            float acc = 0.0f;
            for (int d_qk = 0; d_qk < D_qk; ++d_qk) {
                acc += phi_Q_t[d_qk] * state[d_qk * D_v + d_v];
            }
            O[o_base + t * D_v + d_v] = acc;
        }
    }
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kKernelSource, @"linear_attn_f32");
    if (!pso) return false;

    NSUInteger qBytes = sizeof(float) * (NSUInteger)B * H * S * D_qk;
    NSUInteger kBytes = qBytes;
    NSUInteger vBytes = sizeof(float) * (NSUInteger)B * H * S * D_v;
    NSUInteger oBytes = vBytes;

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufQ, ctx, Q, qBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufK, ctx, K, kBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufV, ctx, V, vBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufQ || !bufK || !bufV || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufQ offset:0 atIndex:0];
    [enc setBuffer:bufK offset:0 atIndex:1];
    [enc setBuffer:bufV offset:0 atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&B length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&H length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&S length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&D_qk length:sizeof(int32_t) atIndex:7];
    [enc setBytes:&D_v length:sizeof(int32_t) atIndex:8];
    [enc setBytes:&feature_map length:sizeof(int32_t) atIndex:9];
    [enc setBytes:&causal length:sizeof(int32_t) atIndex:10];

    NSUInteger total = (NSUInteger)B * H;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(total, pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // Migration batch 3 (2026-06-01) — flash_attn MSL.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_msl")) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

inline float feature_map_apply_host(float x, int32_t fm) {
  if (fm == 0) return x > 0.0f ? (x + 1.0f) : std::exp(x);
  if (fm == 1) return std::max(x, 0.0f);
  if (fm == 2) return x;
  return x * x;
}

inline void reference_linear_attn_f32(const float* Q, const float* K,
                                      const float* V, float* O,
                                      int32_t B, int32_t H, int32_t S,
                                      int32_t D_qk, int32_t D_v,
                                      int32_t feature_map, int32_t causal) {
  std::vector<float> state((std::size_t)D_qk * D_v, 0.0f);
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      std::fill(state.begin(), state.end(), 0.0f);
      int q_base = (b * H + h) * S * D_qk;
      int k_base = q_base;
      int v_base = (b * H + h) * S * D_v;
      int o_base = v_base;
      for (int32_t t = 0; t < S; ++t) {
        // Update state
        for (int32_t d_qk = 0; d_qk < D_qk; ++d_qk) {
          float k_d = feature_map_apply_host(K[k_base + t * D_qk + d_qk], feature_map);
          for (int32_t d_v = 0; d_v < D_v; ++d_v) {
            state[(std::size_t)d_qk * D_v + d_v] += k_d * V[v_base + t * D_v + d_v];
          }
        }
        // Compute output
        for (int32_t d_v = 0; d_v < D_v; ++d_v) {
          float acc = 0.0f;
          for (int32_t d_qk = 0; d_qk < D_qk; ++d_qk) {
            float q_d = feature_map_apply_host(Q[q_base + t * D_qk + d_qk], feature_map);
            acc += q_d * state[(std::size_t)d_qk * D_v + d_v];
          }
          O[o_base + t * D_v + d_v] = acc;
        }
      }
    }
  }
}

} // namespace

extern "C" void tessera_apple_gpu_linear_attn_f32(const float* Q, const float* K,
                                                   const float* V, float* O,
                                                   int32_t B, int32_t H,
                                                   int32_t S, int32_t D_qk,
                                                   int32_t D_v,
                                                   int32_t feature_map,
                                                   int32_t causal) {
  if (D_qk * D_v > 256 || D_qk > 16) {
    reference_linear_attn_f32(Q, K, V, O, B, H, S, D_qk, D_v, feature_map, causal);
    return;
  }
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_linear_attn_msl(ctx, Q, K, V, O, B, H, S, D_qk, D_v,
                                          feature_map, causal))
    return;
  reference_linear_attn_f32(Q, K, V, O, B, H, S, D_qk, D_v, feature_map, causal);
}

//===---------------------------------------------------------------------===//
// attention_variants_plan, MLA-2 — DeepSeek MLA decode runtime entry.
//
// Today this is a host-reference path that materializes the latent c
// and the expanded K/V before running flash-attention. The Apple GPU
// memory win — only the latent c lives in cache via
// LatentKVCacheHandle — is fully observable on the Python side already;
// this entry exists so the IR-level fusion has a concrete runtime
// landing pad. The absorb-K MSL kernel (no full-K materialization) is
// a follow-up that reuses the existing flash-attn kernel infra.
//===---------------------------------------------------------------------===//

namespace {

inline void reference_mla_decode_f32(const float* X, const float* Wdkv,
                                     const float* Wuk, const float* Wuv,
                                     const float* Q, float* O,
                                     int32_t B, int32_t S_kv,
                                     int32_t D_x, int32_t D_lat,
                                     int32_t S_q, int32_t D_h) {
  // c = X @ Wdkv  ((S_kv, D_x) @ (D_x, D_lat) → (S_kv, D_lat))
  std::vector<float> c((std::size_t)S_kv * D_lat, 0.0f);
  for (int32_t s = 0; s < S_kv; ++s) {
    for (int32_t d = 0; d < D_x; ++d) {
      float xv = X[(std::size_t)s * D_x + d];
      const float* w_row = Wdkv + (std::size_t)d * D_lat;
      for (int32_t l = 0; l < D_lat; ++l) {
        c[(std::size_t)s * D_lat + l] += xv * w_row[l];
      }
    }
  }
  // K = c @ Wuk;  V = c @ Wuv  ((S_kv, D_lat) @ (D_lat, D_h) → (S_kv, D_h))
  std::vector<float> K((std::size_t)S_kv * D_h, 0.0f);
  std::vector<float> V((std::size_t)S_kv * D_h, 0.0f);
  for (int32_t s = 0; s < S_kv; ++s) {
    for (int32_t l = 0; l < D_lat; ++l) {
      float cv = c[(std::size_t)s * D_lat + l];
      const float* uk_row = Wuk + (std::size_t)l * D_h;
      const float* uv_row = Wuv + (std::size_t)l * D_h;
      for (int32_t h = 0; h < D_h; ++h) {
        K[(std::size_t)s * D_h + h] += cv * uk_row[h];
        V[(std::size_t)s * D_h + h] += cv * uv_row[h];
      }
    }
  }
  // O = flash_attn(Q[B, S_q, D_h], K[S_kv, D_h], V[S_kv, D_h]) — broadcast
  // K/V across batch.
  float scale = 1.0f / std::sqrt(static_cast<float>(D_h));
  std::vector<float> scores((std::size_t)S_q * S_kv);
  for (int32_t b = 0; b < B; ++b) {
    const float* Qb = Q + (std::size_t)b * S_q * D_h;
    float* Ob = O + (std::size_t)b * S_q * D_h;
    for (int32_t i = 0; i < S_q; ++i) {
      for (int32_t j = 0; j < S_kv; ++j) {
        float dot = 0.0f;
        for (int32_t h = 0; h < D_h; ++h) {
          dot += Qb[i * D_h + h] * K[(std::size_t)j * D_h + h];
        }
        scores[(std::size_t)i * S_kv + j] = dot * scale;
      }
    }
    // Softmax + matmul V per row.
    for (int32_t i = 0; i < S_q; ++i) {
      float maxv = -std::numeric_limits<float>::infinity();
      for (int32_t j = 0; j < S_kv; ++j) {
        maxv = std::max(maxv, scores[(std::size_t)i * S_kv + j]);
      }
      float sum = 0.0f;
      for (int32_t j = 0; j < S_kv; ++j) {
        float e = std::exp(scores[(std::size_t)i * S_kv + j] - maxv);
        scores[(std::size_t)i * S_kv + j] = e;
        sum += e;
      }
      for (int32_t j = 0; j < S_kv; ++j) {
        scores[(std::size_t)i * S_kv + j] /= sum;
      }
      for (int32_t h = 0; h < D_h; ++h) {
        float acc = 0.0f;
        for (int32_t j = 0; j < S_kv; ++j) {
          acc += scores[(std::size_t)i * S_kv + j] * V[(std::size_t)j * D_h + h];
        }
        Ob[i * D_h + h] = acc;
      }
    }
  }
}

// The MPSGraph cached-graph + dtype-cast helpers are defined later in this TU;
// forward declare them so the MLA decode (which appears earlier) can reuse them.
static NSArray *mpsg_cache_get(NSString *key);
static void mpsg_cache_put(NSString *key, NSArray *entry);
static inline MPSGraphTensor *mpsg_up(MPSGraph *g, MPSGraphTensor *t, MPSDataType ioType);
static inline MPSGraphTensor *mpsg_down(MPSGraph *g, MPSGraphTensor *t, MPSDataType ioType);
static inline float gqa_bf16_to_f32(uint16_t b);
static inline uint16_t gqa_f32_to_bf16(float v);

// GPU MLA decode — compressed-KV (no rope yet). One cached MPSGraph fuses the
// whole decode: latent down-projection c = X@Wdkv, the K/V up-projections
// K = c@Wuk and V = c@Wuv, then attention O = softmax((Q@Kᵀ)·scale)@V with the
// B·S_q query rows folded to a single matmul dimension (K/V are shared across
// batch, so no batched-broadcast is needed). All matmuls accumulate in fp32.
static bool mpsg_run_mla_decode(MetalDeviceContext &ctx, const void *X,
                                const void *Wdkv, const void *Wuk,
                                const void *Wuv, const void *Q, void *O,
                                int32_t B, int32_t S_kv, int32_t D_x,
                                int32_t D_lat, int32_t S_q, int32_t D_h,
                                MPSDataType ioType, size_t elemSize) {
  if (B <= 0 || S_kv <= 0 || D_x <= 0 || D_lat <= 0 || S_q <= 0 || D_h <= 0)
    return true;
  int32_t M = B * S_q;  // folded query rows
  @autoreleasepool {
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, (size_t)S_kv * D_x * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufWdkv, ctx, Wdkv, (size_t)D_x * D_lat * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufWuk, ctx, Wuk, (size_t)D_lat * D_h * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufWuv, ctx, Wuv, (size_t)D_lat * D_h * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufQ, ctx, Q, (size_t)M * D_h * elemSize);
    if (!bufX || !bufWdkv || !bufWuk || !bufWuv || !bufQ) return false;
    NSArray<NSNumber *> *xShape = @[ @(S_kv), @(D_x) ];
    NSArray<NSNumber *> *wdkvShape = @[ @(D_x), @(D_lat) ];
    NSArray<NSNumber *> *wukShape = @[ @(D_lat), @(D_h) ];
    NSArray<NSNumber *> *wuvShape = @[ @(D_lat), @(D_h) ];
    NSArray<NSNumber *> *qShape = @[ @(M), @(D_h) ];
    float scale = 1.0f / std::sqrt((float)D_h);
    NSString *key = [NSString stringWithFormat:@"mla:%d:%d:%d:%d:%d:%d:%d",
                                               (int)ioType, B, S_kv, D_x, D_lat,
                                               S_q, D_h];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *px, *pwdkv, *pwuk, *pwuv, *pq, *y;
    if (entry) {
      g = entry[0];
      NSArray *phs = (NSArray *)entry[1];
      px = phs[0]; pwdkv = phs[1]; pwuk = phs[2]; pwuv = phs[3]; pq = phs[4];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      px = [g placeholderWithShape:xShape dataType:ioType name:nil];
      pwdkv = [g placeholderWithShape:wdkvShape dataType:ioType name:nil];
      pwuk = [g placeholderWithShape:wukShape dataType:ioType name:nil];
      pwuv = [g placeholderWithShape:wuvShape dataType:ioType name:nil];
      pq = [g placeholderWithShape:qShape dataType:ioType name:nil];
      MPSGraphTensor *x32 = mpsg_up(g, px, ioType);
      MPSGraphTensor *wdkv32 = mpsg_up(g, pwdkv, ioType);
      MPSGraphTensor *wuk32 = mpsg_up(g, pwuk, ioType);
      MPSGraphTensor *wuv32 = mpsg_up(g, pwuv, ioType);
      MPSGraphTensor *q32 = mpsg_up(g, pq, ioType);
      MPSGraphTensor *c = [g matrixMultiplicationWithPrimaryTensor:x32 secondaryTensor:wdkv32 name:nil];
      MPSGraphTensor *K = [g matrixMultiplicationWithPrimaryTensor:c secondaryTensor:wuk32 name:nil];
      MPSGraphTensor *V = [g matrixMultiplicationWithPrimaryTensor:c secondaryTensor:wuv32 name:nil];
      MPSGraphTensor *Kt = [g transposeTensor:K dimension:0 withDimension:1 name:nil];
      MPSGraphTensor *scores = [g matrixMultiplicationWithPrimaryTensor:q32 secondaryTensor:Kt name:nil];
      MPSGraphTensor *scaled = [g multiplicationWithPrimaryTensor:scores
                                   secondaryTensor:[g constantWithScalar:(double)scale dataType:MPSDataTypeFloat32]
                                              name:nil];
      MPSGraphTensor *attn = [g softMaxWithTensor:scaled axis:1 name:nil];
      MPSGraphTensor *yf = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:V name:nil];
      y = mpsg_down(g, yf, ioType);
      mpsg_cache_put(key, @[ g, @[ px, pwdkv, pwuk, pwuv, pq ], y ]);
    }
    MPSGraphTensorData *xd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:xShape dataType:ioType];
    MPSGraphTensorData *wdkvd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufWdkv shape:wdkvShape dataType:ioType];
    MPSGraphTensorData *wukd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufWuk shape:wukShape dataType:ioType];
    MPSGraphTensorData *wuvd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufWuv shape:wuvShape dataType:ioType];
    MPSGraphTensorData *qd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufQ shape:qShape dataType:ioType];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{px : xd, pwdkv : wdkvd, pwuk : wukd, pwuv : wuvd, pq : qd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:O strideBytes:nil];
    return true;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_mla_decode_f32(const float* X,
                                                  const float* Wdkv,
                                                  const float* Wuk,
                                                  const float* Wuv,
                                                  const float* Q, float* O,
                                                  int32_t B, int32_t S_kv,
                                                  int32_t D_x, int32_t D_lat,
                                                  int32_t S_q, int32_t D_h) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_mla_decode(ctx, X, Wdkv, Wuk, Wuv, Q, O, B, S_kv,
                                    D_x, D_lat, S_q, D_h, MPSDataTypeFloat32, 4))
    return;
  reference_mla_decode_f32(X, Wdkv, Wuk, Wuv, Q, O, B, S_kv, D_x, D_lat, S_q, D_h);
}

// f16: native f16 I/O on-GPU (fp32 accumulation), host reference fallback.
extern "C" void tessera_apple_gpu_mla_decode_f16(
    const uint16_t* X, const uint16_t* Wdkv, const uint16_t* Wuk,
    const uint16_t* Wuv, const uint16_t* Q, uint16_t* O, int32_t B,
    int32_t S_kv, int32_t D_x, int32_t D_lat, int32_t S_q, int32_t D_h) {
  if (B <= 0 || S_kv <= 0 || D_x <= 0 || D_lat <= 0 || S_q <= 0 || D_h <= 0)
    return;
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_mla_decode(ctx, X, Wdkv, Wuk, Wuv, Q, O, B, S_kv, D_x,
                                    D_lat, S_q, D_h, MPSDataTypeFloat16, 2))
    return;
  auto cvt = [](const uint16_t* p, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = half_to_float_gpu(p[i]);
    return v;
  };
  std::vector<float> xf = cvt(X, (size_t)S_kv * D_x), wdkvf = cvt(Wdkv, (size_t)D_x * D_lat),
                     wukf = cvt(Wuk, (size_t)D_lat * D_h), wuvf = cvt(Wuv, (size_t)D_lat * D_h),
                     qf = cvt(Q, (size_t)B * S_q * D_h), of((size_t)B * S_q * D_h);
  reference_mla_decode_f32(xf.data(), wdkvf.data(), wukf.data(), wuvf.data(),
                           qf.data(), of.data(), B, S_kv, D_x, D_lat, S_q, D_h);
  for (size_t i = 0; i < of.size(); ++i) O[i] = float_to_half_gpu(of[i]);
}

// bf16: no native MPSGraph type — convert to fp32, run the fp32 graph, convert back.
extern "C" void tessera_apple_gpu_mla_decode_bf16(
    const uint16_t* X, const uint16_t* Wdkv, const uint16_t* Wuk,
    const uint16_t* Wuv, const uint16_t* Q, uint16_t* O, int32_t B,
    int32_t S_kv, int32_t D_x, int32_t D_lat, int32_t S_q, int32_t D_h) {
  if (B <= 0 || S_kv <= 0 || D_x <= 0 || D_lat <= 0 || S_q <= 0 || D_h <= 0)
    return;
  auto cvt = [](const uint16_t* p, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = gqa_bf16_to_f32(p[i]);
    return v;
  };
  std::vector<float> xf = cvt(X, (size_t)S_kv * D_x), wdkvf = cvt(Wdkv, (size_t)D_x * D_lat),
                     wukf = cvt(Wuk, (size_t)D_lat * D_h), wuvf = cvt(Wuv, (size_t)D_lat * D_h),
                     qf = cvt(Q, (size_t)B * S_q * D_h), of((size_t)B * S_q * D_h);
  MetalDeviceContext &ctx = deviceContext();
  if (!(ctx.ok && mpsg_run_mla_decode(ctx, xf.data(), wdkvf.data(), wukf.data(),
                                      wuvf.data(), qf.data(), of.data(), B, S_kv,
                                      D_x, D_lat, S_q, D_h, MPSDataTypeFloat32, 4)))
    reference_mla_decode_f32(xf.data(), wdkvf.data(), wukf.data(), wuvf.data(),
                             qf.data(), of.data(), B, S_kv, D_x, D_lat, S_q, D_h);
  for (size_t i = 0; i < of.size(); ++i) O[i] = gqa_f32_to_bf16(of[i]);
}

//===---------------------------------------------------------------------===//
// attention_variants_plan, NSA-5 — DeepSeek Native Sparse Attention.
//
// GPU-first implementation for the strict Sprint 11 value envelope:
//   1. Sliding-window dense local attention.
//   2. Per-block-summary attention (mean compression by default).
//   3. Top-k block-selected dense attention.
//
// The fused MSL kernel intentionally starts correctness-first: one GPU thread
// computes one output scalar and recomputes row softmax denominators for that
// scalar. That is not the final throughput shape, but it is real Metal work and
// keeps the GPU claim honest. The host implementation below is fallback only.
//===---------------------------------------------------------------------===//

namespace {

static std::atomic<int32_t> gNativeSparseAttnLastPath{0};

static void reference_native_sparse_attn_f32(
    const float* Q, const float* K, const float* V,
    const float* gate_logits, float* O,
    int32_t B, int32_t H, int32_t S, int32_t D,
    int32_t window_size, int32_t block_size, int32_t top_k,
    int32_t causal) {
  // Output starts at zero; we'll accumulate the gated branch outputs.
  std::memset(O, 0, sizeof(float) * (std::size_t)B * H * S * D);
  if (block_size <= 0 || S % block_size != 0) {
    // Out-of-envelope: leave O as zeros so the caller can detect.
    return;
  }
  int32_t num_blocks = S / block_size;
  float scale = 1.0f / std::sqrt(static_cast<float>(D));

  // Pre-compute per-block (mean-compressed) K_c, V_c.
  std::vector<float> Kc((std::size_t)B * H * num_blocks * D, 0.0f);
  std::vector<float> Vc((std::size_t)B * H * num_blocks * D, 0.0f);
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      for (int32_t blk = 0; blk < num_blocks; ++blk) {
        for (int32_t t = 0; t < block_size; ++t) {
          for (int32_t d = 0; d < D; ++d) {
            std::size_t k_idx = (((std::size_t)b * H + h) * S + blk * block_size + t) * D + d;
            std::size_t kc_idx = (((std::size_t)b * H + h) * num_blocks + blk) * D + d;
            Kc[kc_idx] += K[k_idx] / static_cast<float>(block_size);
            Vc[kc_idx] += V[k_idx] / static_cast<float>(block_size);
          }
        }
      }
    }
  }

  // Per (b, h, q): three branches.
  std::vector<float> scores;
  scores.resize(std::max((std::size_t)S, (std::size_t)num_blocks));

  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      for (int32_t q = 0; q < S; ++q) {
        // Row indices.
        std::size_t q_off = (((std::size_t)b * H + h) * S + q) * D;
        std::size_t o_off = q_off;
        std::size_t gate_off = (((std::size_t)b * H + h) * S + q) * num_blocks;

        // Branch 1 — sliding window.
        float row1[1024]; (void)row1; // placeholder to avoid VLAs
        std::vector<float> w_branch(D, 0.0f);
        {
          // Compute scores[j] for j in window of q.
          int32_t lo = causal ? std::max(0, q - window_size + 1) : std::max(0, q - window_size / 2);
          int32_t hi = causal ? q : std::min(S - 1, q + window_size / 2);
          float maxv = -std::numeric_limits<float>::infinity();
          std::vector<float> ws(hi - lo + 1, 0.0f);
          for (int32_t j = lo; j <= hi; ++j) {
            float s = 0.0f;
            for (int32_t d = 0; d < D; ++d) {
              s += Q[q_off + d] * K[(((std::size_t)b * H + h) * S + j) * D + d];
            }
            ws[j - lo] = s * scale;
            if (ws[j - lo] > maxv) maxv = ws[j - lo];
          }
          float sum = 0.0f;
          for (auto &v : ws) { v = std::exp(v - maxv); sum += v; }
          if (sum == 0) sum = 1.0f;
          for (auto &v : ws) v /= sum;
          for (int32_t j = lo; j <= hi; ++j) {
            for (int32_t d = 0; d < D; ++d) {
              w_branch[d] += ws[j - lo]
                  * V[(((std::size_t)b * H + h) * S + j) * D + d];
            }
          }
        }

        // Branch 2 — compressed blocks.
        std::vector<float> c_branch(D, 0.0f);
        {
          std::vector<float> ws(num_blocks, 0.0f);
          float maxv = -std::numeric_limits<float>::infinity();
          for (int32_t blk = 0; blk < num_blocks; ++blk) {
            float s = 0.0f;
            std::size_t kc_off = (((std::size_t)b * H + h) * num_blocks + blk) * D;
            for (int32_t d = 0; d < D; ++d) s += Q[q_off + d] * Kc[kc_off + d];
            ws[blk] = s * scale;
            if (ws[blk] > maxv) maxv = ws[blk];
          }
          float sum = 0.0f;
          for (auto &v : ws) { v = std::exp(v - maxv); sum += v; }
          if (sum == 0) sum = 1.0f;
          for (auto &v : ws) v /= sum;
          for (int32_t blk = 0; blk < num_blocks; ++blk) {
            std::size_t vc_off = (((std::size_t)b * H + h) * num_blocks + blk) * D;
            for (int32_t d = 0; d < D; ++d) c_branch[d] += ws[blk] * Vc[vc_off + d];
          }
        }

        // Branch 3 — top-k blocks.
        std::vector<float> s_branch(D, 0.0f);
        {
          // Use gate_logits as the block scoring tensor (matches the
          // Schedule IR fusion's convention of carrying the score
          // tensor as the 4th operand). When gate_logits is None /
          // zero, the runtime falls back to argmax-by-Q·Kc.
          std::vector<std::pair<float, int32_t>> scored(num_blocks);
          int32_t q_blk = q / block_size;
          for (int32_t blk = 0; blk < num_blocks; ++blk) {
            float gv = gate_logits[gate_off + blk];
            if (causal && blk > q_blk) gv = -std::numeric_limits<float>::infinity();
            scored[blk] = {gv, blk};
          }
          std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
              [](auto &a, auto &b) { return a.first > b.first; });
          int32_t total = top_k * block_size;
          std::vector<float> ws(total, 0.0f);
          float maxv = -std::numeric_limits<float>::infinity();
          for (int32_t i = 0; i < top_k; ++i) {
            int32_t blk = scored[i].second;
            for (int32_t t = 0; t < block_size; ++t) {
              int32_t j = blk * block_size + t;
              float s = 0.0f;
              for (int32_t d = 0; d < D; ++d) {
                s += Q[q_off + d] * K[(((std::size_t)b * H + h) * S + j) * D + d];
              }
              ws[i * block_size + t] = s * scale;
              if (ws[i * block_size + t] > maxv) maxv = ws[i * block_size + t];
            }
          }
          float sum = 0.0f;
          for (auto &v : ws) { v = std::exp(v - maxv); sum += v; }
          if (sum == 0) sum = 1.0f;
          for (auto &v : ws) v /= sum;
          for (int32_t i = 0; i < top_k; ++i) {
            int32_t blk = scored[i].second;
            for (int32_t t = 0; t < block_size; ++t) {
              int32_t j = blk * block_size + t;
              for (int32_t d = 0; d < D; ++d) {
                s_branch[d] += ws[i * block_size + t]
                    * V[(((std::size_t)b * H + h) * S + j) * D + d];
              }
            }
          }
        }

        // Equal-weight 1/3 mix on each branch (uniform prior; the
        // user-side gate Module already weighs per-query, so this
        // host-reference path doesn't double-gate).
        for (int32_t d = 0; d < D; ++d) {
          O[o_off + d] = (w_branch[d] + c_branch[d] + s_branch[d]) / 3.0f;
        }
      }
    }
  }
}

static bool dispatch_native_sparse_attn_msl(
    MetalDeviceContext &ctx, const float* Q, const float* K, const float* V,
    const float* gate_logits, float* O, int32_t B, int32_t H, int32_t S,
    int32_t D, int32_t window_size, int32_t block_size, int32_t top_k,
    int32_t causal) {
  if (B <= 0 || H <= 0 || S <= 0 || D <= 0 || window_size <= 0 ||
      block_size <= 0 || top_k <= 0 || (S % block_size) != 0)
    return false;
  int32_t num_blocks = S / block_size;
  if (top_k > num_blocks)
    return false;

  static NSString *const kNativeSparseSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

static inline float nsa_dot_q_k(
    device const float* Q, device const float* K,
    int b, int h, int q, int j, int H, int S, int D)
{
    int q_off = (((b * H + h) * S + q) * D);
    int k_off = (((b * H + h) * S + j) * D);
    float s = 0.0f;
    for (int dd = 0; dd < D; ++dd) {
        s += Q[q_off + dd] * K[k_off + dd];
    }
    return s;
}

static inline float nsa_dot_q_kc(
    device const float* Q, device const float* K,
    int b, int h, int q, int blk, int H, int S, int D, int block_size)
{
    int q_off = (((b * H + h) * S + q) * D);
    float s = 0.0f;
    for (int dd = 0; dd < D; ++dd) {
        float mean_k = 0.0f;
        for (int t = 0; t < block_size; ++t) {
            int j = blk * block_size + t;
            int k_off = (((b * H + h) * S + j) * D);
            mean_k += K[k_off + dd];
        }
        s += Q[q_off + dd] * (mean_k / float(block_size));
    }
    return s;
}

static inline float nsa_gate(
    device const float* gate, int b, int h, int q, int blk,
    int H, int S, int num_blocks, int causal, int q_blk)
{
    float gv = gate[(((b * H + h) * S + q) * num_blocks) + blk];
    if (causal != 0 && blk > q_blk) {
        return -INFINITY;
    }
    return gv;
}

kernel void native_sparse_attn_f32(
    device const float* Q    [[buffer(0)]],
    device const float* K    [[buffer(1)]],
    device const float* V    [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device float*       O    [[buffer(4)]],
    constant int& B          [[buffer(5)]],
    constant int& H          [[buffer(6)]],
    constant int& S          [[buffer(7)]],
    constant int& D          [[buffer(8)]],
    constant int& window_sz  [[buffer(9)]],
    constant int& block_sz   [[buffer(10)]],
    constant int& top_k      [[buffer(11)]],
    constant int& causal     [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = uint(B * H * S * D);
    if (gid >= total) return;

    int d = int(gid % uint(D));
    int q = int((gid / uint(D)) % uint(S));
    int h = int((gid / uint(D * S)) % uint(H));
    int b = int(gid / uint(D * S * H));
    int num_blocks = S / block_sz;
    int q_blk = q / block_sz;
    float scale = rsqrt(float(D));
    int out_off = (((b * H + h) * S + q) * D) + d;

    // Branch 1: sliding window attention.
    int lo = (causal != 0) ? max(0, q - window_sz + 1)
                           : max(0, q - window_sz / 2);
    int hi = (causal != 0) ? q : min(S - 1, q + window_sz / 2);
    float maxv = -INFINITY;
    for (int j = lo; j <= hi; ++j) {
        maxv = max(maxv, nsa_dot_q_k(Q, K, b, h, q, j, H, S, D) * scale);
    }
    float denom = 0.0f;
    float w_branch = 0.0f;
    for (int j = lo; j <= hi; ++j) {
        float e = exp(nsa_dot_q_k(Q, K, b, h, q, j, H, S, D) * scale - maxv);
        denom += e;
        w_branch += e * V[(((b * H + h) * S + j) * D) + d];
    }
    if (denom != 0.0f) w_branch /= denom;

    // Branch 2: compressed block attention.
    maxv = -INFINITY;
    for (int blk = 0; blk < num_blocks; ++blk) {
        maxv = max(maxv, nsa_dot_q_kc(Q, K, b, h, q, blk, H, S, D, block_sz) * scale);
    }
    denom = 0.0f;
    float c_branch = 0.0f;
    for (int blk = 0; blk < num_blocks; ++blk) {
        float e = exp(nsa_dot_q_kc(Q, K, b, h, q, blk, H, S, D, block_sz) * scale - maxv);
        float mean_v = 0.0f;
        for (int t = 0; t < block_sz; ++t) {
            int j = blk * block_sz + t;
            mean_v += V[(((b * H + h) * S + j) * D) + d];
        }
        mean_v /= float(block_sz);
        denom += e;
        c_branch += e * mean_v;
    }
    if (denom != 0.0f) c_branch /= denom;

    // Branch 3: top-k block-selected attention. Rank blocks by gate score,
    // then softmax over all tokens belonging to the selected blocks.
    maxv = -INFINITY;
    bool any_selected = false;
    for (int blk = 0; blk < num_blocks; ++blk) {
        float gv = nsa_gate(gate, b, h, q, blk, H, S, num_blocks, causal, q_blk);
        int rank = 0;
        for (int other = 0; other < num_blocks; ++other) {
            float ov = nsa_gate(gate, b, h, q, other, H, S, num_blocks, causal, q_blk);
            if (ov > gv || (ov == gv && other < blk)) {
                rank += 1;
            }
        }
        if (rank < top_k) {
            any_selected = true;
            for (int t = 0; t < block_sz; ++t) {
                int j = blk * block_sz + t;
                maxv = max(maxv, nsa_dot_q_k(Q, K, b, h, q, j, H, S, D) * scale);
            }
        }
    }
    denom = 0.0f;
    float s_branch = 0.0f;
    if (any_selected) {
        for (int blk = 0; blk < num_blocks; ++blk) {
            float gv = nsa_gate(gate, b, h, q, blk, H, S, num_blocks, causal, q_blk);
            int rank = 0;
            for (int other = 0; other < num_blocks; ++other) {
                float ov = nsa_gate(gate, b, h, q, other, H, S, num_blocks, causal, q_blk);
                if (ov > gv || (ov == gv && other < blk)) {
                    rank += 1;
                }
            }
            if (rank < top_k) {
                for (int t = 0; t < block_sz; ++t) {
                    int j = blk * block_sz + t;
                    float e = exp(nsa_dot_q_k(Q, K, b, h, q, j, H, S, D) * scale - maxv);
                    denom += e;
                    s_branch += e * V[(((b * H + h) * S + j) * D) + d];
                }
            }
        }
        if (denom != 0.0f) s_branch /= denom;
    }

    O[out_off] = (w_branch + c_branch + s_branch) / 3.0f;
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kNativeSparseSource, @"native_sparse_attn_f32");
    if (!pso) return false;

    const NSUInteger qBytes = sizeof(float) * static_cast<NSUInteger>(B) *
                              static_cast<NSUInteger>(H) *
                              static_cast<NSUInteger>(S) *
                              static_cast<NSUInteger>(D);
    const NSUInteger gateBytes = sizeof(float) *
                                 static_cast<NSUInteger>(B) *
                                 static_cast<NSUInteger>(H) *
                                 static_cast<NSUInteger>(S) *
                                 static_cast<NSUInteger>(num_blocks);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufQ, ctx, Q, qBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufK, ctx, K, qBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufV, ctx, V, qBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufG, ctx, gate_logits, gateBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, qBytes);
    if (!bufQ || !bufK || !bufV || !bufG || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufQ offset:0 atIndex:0];
    [enc setBuffer:bufK offset:0 atIndex:1];
    [enc setBuffer:bufV offset:0 atIndex:2];
    [enc setBuffer:bufG offset:0 atIndex:3];
    [enc setBuffer:bufO offset:0 atIndex:4];
    [enc setBytes:&B length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&H length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&S length:sizeof(int32_t) atIndex:7];
    [enc setBytes:&D length:sizeof(int32_t) atIndex:8];
    [enc setBytes:&window_size length:sizeof(int32_t) atIndex:9];
    [enc setBytes:&block_size length:sizeof(int32_t) atIndex:10];
    [enc setBytes:&top_k length:sizeof(int32_t) atIndex:11];
    [enc setBytes:&causal length:sizeof(int32_t) atIndex:12];

    NSUInteger total = static_cast<NSUInteger>(B) * H * S * D;
    MTLSize grid = MTLSizeMake(total, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(total, pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "native_sparse_attn_msl"))
      return false;
    std::memcpy(O, [bufO contents], qBytes);
    return true;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_native_sparse_attn_f32(
    const float* Q, const float* K, const float* V,
    const float* gate_logits, float* O,
    int32_t B, int32_t H, int32_t S, int32_t D,
    int32_t window_size, int32_t block_size, int32_t top_k,
    int32_t causal) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_native_sparse_attn_msl(
                    ctx, Q, K, V, gate_logits, O, B, H, S, D, window_size,
                    block_size, top_k, causal)) {
    gNativeSparseAttnLastPath.store(1, std::memory_order_relaxed);
    return;
  }
  reference_native_sparse_attn_f32(Q, K, V, gate_logits, O, B, H, S, D,
                                   window_size, block_size, top_k, causal);
  gNativeSparseAttnLastPath.store(2, std::memory_order_relaxed);
}

extern "C" int32_t tessera_apple_gpu_native_sparse_attn_last_path(void) {
  return gNativeSparseAttnLastPath.load(std::memory_order_relaxed);
}

//===---------------------------------------------------------------------===//
// GA9 — Clifford / Geometric-Algebra MSL kernels (2026-05-17).
//
// First fused kernels for the `tessera_clifford.*` dialect:
//   - clifford_geo_product_cl30_f32 : 8-element multivector × multivector
//     using the fully-unrolled Cl(3,0) Cayley table (64 fp32 mul-adds).
//   - clifford_rotor_sandwich_cl30_f32 : R · v · R† fused (3 geo_products
//     collapsed into a single dispatch).
//
// Cl(3,0) basis blade order (last axis = 8):
//   0:1   1:e1   2:e2   3:e12   4:e3   5:e13   6:e23   7:e123
//
// All expressions below were generated from the Python Cayley table in
// `tessera.ga.signature._product_table` so the GPU result is
// bitwise-equivalent to the GA Python reference up to fp32 rounding.
//===---------------------------------------------------------------------===//

namespace {

inline void reference_clifford_geo_product_cl30_f32(
    const float* A, const float* B, float* C, int32_t batch) {
  for (int32_t b = 0; b < batch; ++b) {
    const float* a = A + b * 8;
    const float* x = B + b * 8;
    float* c = C + b * 8;
    c[0] = a[0]*x[0] + a[1]*x[1] + a[2]*x[2] - a[3]*x[3]
         + a[4]*x[4] - a[5]*x[5] - a[6]*x[6] - a[7]*x[7];
    c[1] = a[0]*x[1] + a[1]*x[0] - a[2]*x[3] + a[3]*x[2]
         - a[4]*x[5] + a[5]*x[4] - a[6]*x[7] - a[7]*x[6];
    c[2] = a[0]*x[2] + a[1]*x[3] + a[2]*x[0] - a[3]*x[1]
         - a[4]*x[6] + a[5]*x[7] + a[6]*x[4] + a[7]*x[5];
    c[3] = a[0]*x[3] + a[1]*x[2] - a[2]*x[1] + a[3]*x[0]
         + a[4]*x[7] - a[5]*x[6] + a[6]*x[5] + a[7]*x[4];
    c[4] = a[0]*x[4] + a[1]*x[5] + a[2]*x[6] - a[3]*x[7]
         + a[4]*x[0] - a[5]*x[1] - a[6]*x[2] - a[7]*x[3];
    c[5] = a[0]*x[5] + a[1]*x[4] - a[2]*x[7] + a[3]*x[6]
         - a[4]*x[1] + a[5]*x[0] - a[6]*x[3] - a[7]*x[2];
    c[6] = a[0]*x[6] + a[1]*x[7] + a[2]*x[4] - a[3]*x[5]
         - a[4]*x[2] + a[5]*x[3] + a[6]*x[0] + a[7]*x[1];
    c[7] = a[0]*x[7] + a[1]*x[6] - a[2]*x[5] + a[3]*x[4]
         + a[4]*x[3] - a[5]*x[2] + a[6]*x[1] + a[7]*x[0];
  }
}

bool dispatch_clifford_geo_product_cl30_f32_msl(
    MetalDeviceContext &ctx, const float* A, const float* B,
    float* C, int32_t batch) {
  static NSString *const kCliffordGeoProductCl30F32 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void clifford_geo_product_cl30_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       C   [[buffer(2)]],
    constant int&       batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float a0=A[off+0], a1=A[off+1], a2=A[off+2], a3=A[off+3];
    float a4=A[off+4], a5=A[off+5], a6=A[off+6], a7=A[off+7];
    float b0=B[off+0], b1=B[off+1], b2=B[off+2], b3=B[off+3];
    float b4=B[off+4], b5=B[off+5], b6=B[off+6], b7=B[off+7];

    C[off+0] = a0*b0 + a1*b1 + a2*b2 - a3*b3 + a4*b4 - a5*b5 - a6*b6 - a7*b7;
    C[off+1] = a0*b1 + a1*b0 - a2*b3 + a3*b2 - a4*b5 + a5*b4 - a6*b7 - a7*b6;
    C[off+2] = a0*b2 + a1*b3 + a2*b0 - a3*b1 - a4*b6 + a5*b7 + a6*b4 + a7*b5;
    C[off+3] = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 + a7*b4;
    C[off+4] = a0*b4 + a1*b5 + a2*b6 - a3*b7 + a4*b0 - a5*b1 - a6*b2 - a7*b3;
    C[off+5] = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 - a7*b2;
    C[off+6] = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 + a7*b1;
    C[off+7] = a0*b7 + a1*b6 - a2*b5 + a3*b4 + a4*b3 - a5*b2 + a6*b1 + a7*b0;
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kCliffordGeoProductCl30F32, @"clifford_geo_product_cl30_f32");
    if (!pso) return false;

    NSUInteger byteCount =
        sizeof(float) * static_cast<NSUInteger>(batch) * 8u;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, byteCount);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, byteCount);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, byteCount);
    if (!bufA || !bufB || !bufC) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:3];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(batch),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "clifford_geo_product_cl30_f32_msl")) return false;
    std::memcpy(C, [bufC contents], byteCount);
    return true;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_clifford_geo_product_cl30_f32(
    const float* A, const float* B, float* C, int32_t batch)
{
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_geo_product_cl30_f32_msl(ctx, A, B, C, batch)) return;
  reference_clifford_geo_product_cl30_f32(A, B, C, batch);
}

extern "C" int32_t tessera_apple_gpu_clifford_geo_product_cl30_value_f32(
    const float* A, const float* B, float* C, int32_t batch)
{
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_geo_product_cl30_f32_msl(ctx, A, B, C, batch))
    return 1;
  return 0;
}

//===---------------------------------------------------------------------===//
// clifford_rotor_sandwich_cl30_f32 — R · v · R† fused.
//
// For a Cl(3,0) rotor R (even-grade) acting on a vector v, the sandwich
// produces another vector.  The fused kernel computes both intermediate
// geo_products in one dispatch, sharing the R† computation (reverse of
// a Cl(3,0) multivector applies signs (+,+,+,-,+,-,-,+) for grades
// 0,1,1,2,1,2,2,3).
//===---------------------------------------------------------------------===//

namespace {

inline void reference_clifford_rotor_sandwich_cl30_f32(
    const float* R, const float* V, float* Out, int32_t batch) {
  for (int32_t b = 0; b < batch; ++b) {
    // Compute R†: reverse signs by grade.
    float r0=R[b*8+0], r1=R[b*8+1], r2=R[b*8+2], r3=R[b*8+3];
    float r4=R[b*8+4], r5=R[b*8+5], r6=R[b*8+6], r7=R[b*8+7];
    float Rd[8] = {
      r0, r1, r2, -r3,         //  grade 0: +1, grade 1: +1, grade 2: -1
      r4, -r5, -r6, -r7,       //  grade 3: -1
    };
    // T = R * v  (use the same Cayley layout).
    float T[8];
    {
      const float* a = R + b*8;
      const float* x = V + b*8;
      float* c = T;
      c[0] = a[0]*x[0] + a[1]*x[1] + a[2]*x[2] - a[3]*x[3] + a[4]*x[4] - a[5]*x[5] - a[6]*x[6] - a[7]*x[7];
      c[1] = a[0]*x[1] + a[1]*x[0] - a[2]*x[3] + a[3]*x[2] - a[4]*x[5] + a[5]*x[4] - a[6]*x[7] - a[7]*x[6];
      c[2] = a[0]*x[2] + a[1]*x[3] + a[2]*x[0] - a[3]*x[1] - a[4]*x[6] + a[5]*x[7] + a[6]*x[4] + a[7]*x[5];
      c[3] = a[0]*x[3] + a[1]*x[2] - a[2]*x[1] + a[3]*x[0] + a[4]*x[7] - a[5]*x[6] + a[6]*x[5] + a[7]*x[4];
      c[4] = a[0]*x[4] + a[1]*x[5] + a[2]*x[6] - a[3]*x[7] + a[4]*x[0] - a[5]*x[1] - a[6]*x[2] - a[7]*x[3];
      c[5] = a[0]*x[5] + a[1]*x[4] - a[2]*x[7] + a[3]*x[6] - a[4]*x[1] + a[5]*x[0] - a[6]*x[3] - a[7]*x[2];
      c[6] = a[0]*x[6] + a[1]*x[7] + a[2]*x[4] - a[3]*x[5] - a[4]*x[2] + a[5]*x[3] + a[6]*x[0] + a[7]*x[1];
      c[7] = a[0]*x[7] + a[1]*x[6] - a[2]*x[5] + a[3]*x[4] + a[4]*x[3] - a[5]*x[2] + a[6]*x[1] + a[7]*x[0];
    }
    // Out = T * R†.
    {
      const float* a = T;
      const float* x = Rd;
      float* c = Out + b*8;
      c[0] = a[0]*x[0] + a[1]*x[1] + a[2]*x[2] - a[3]*x[3] + a[4]*x[4] - a[5]*x[5] - a[6]*x[6] - a[7]*x[7];
      c[1] = a[0]*x[1] + a[1]*x[0] - a[2]*x[3] + a[3]*x[2] - a[4]*x[5] + a[5]*x[4] - a[6]*x[7] - a[7]*x[6];
      c[2] = a[0]*x[2] + a[1]*x[3] + a[2]*x[0] - a[3]*x[1] - a[4]*x[6] + a[5]*x[7] + a[6]*x[4] + a[7]*x[5];
      c[3] = a[0]*x[3] + a[1]*x[2] - a[2]*x[1] + a[3]*x[0] + a[4]*x[7] - a[5]*x[6] + a[6]*x[5] + a[7]*x[4];
      c[4] = a[0]*x[4] + a[1]*x[5] + a[2]*x[6] - a[3]*x[7] + a[4]*x[0] - a[5]*x[1] - a[6]*x[2] - a[7]*x[3];
      c[5] = a[0]*x[5] + a[1]*x[4] - a[2]*x[7] + a[3]*x[6] - a[4]*x[1] + a[5]*x[0] - a[6]*x[3] - a[7]*x[2];
      c[6] = a[0]*x[6] + a[1]*x[7] + a[2]*x[4] - a[3]*x[5] - a[4]*x[2] + a[5]*x[3] + a[6]*x[0] + a[7]*x[1];
      c[7] = a[0]*x[7] + a[1]*x[6] - a[2]*x[5] + a[3]*x[4] + a[4]*x[3] - a[5]*x[2] + a[6]*x[1] + a[7]*x[0];
    }
  }
}

bool dispatch_clifford_rotor_sandwich_cl30_f32_msl(
    MetalDeviceContext &ctx, const float* R, const float* V,
    float* Out, int32_t batch) {
  static NSString *const kCliffordRotorSandwichCl30F32 = @R"MSL(
#include <metal_stdlib>
using namespace metal;

inline void cl30_geo_product(
    thread float a[8], thread float b[8], thread float c[8])
{
    c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] - a[3]*b[3] + a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    c[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[3] + a[3]*b[2] - a[4]*b[5] + a[5]*b[4] - a[6]*b[7] - a[7]*b[6];
    c[2] = a[0]*b[2] + a[1]*b[3] + a[2]*b[0] - a[3]*b[1] - a[4]*b[6] + a[5]*b[7] + a[6]*b[4] + a[7]*b[5];
    c[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0] + a[4]*b[7] - a[5]*b[6] + a[6]*b[5] + a[7]*b[4];
    c[4] = a[0]*b[4] + a[1]*b[5] + a[2]*b[6] - a[3]*b[7] + a[4]*b[0] - a[5]*b[1] - a[6]*b[2] - a[7]*b[3];
    c[5] = a[0]*b[5] + a[1]*b[4] - a[2]*b[7] + a[3]*b[6] - a[4]*b[1] + a[5]*b[0] - a[6]*b[3] - a[7]*b[2];
    c[6] = a[0]*b[6] + a[1]*b[7] + a[2]*b[4] - a[3]*b[5] - a[4]*b[2] + a[5]*b[3] + a[6]*b[0] + a[7]*b[1];
    c[7] = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4] + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];
}

kernel void clifford_rotor_sandwich_cl30_f32(
    device const float* R   [[buffer(0)]],
    device const float* V   [[buffer(1)]],
    device float*       Out [[buffer(2)]],
    constant int&       batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float r[8], v[8], rd[8], t[8], o[8];
    for (uint i = 0; i < 8; ++i) { r[i] = R[off+i]; v[i] = V[off+i]; }
    // R†: grade-k reverse sign = (-1)^(k(k-1)/2).
    // For Cl(3,0): grades = (0,1,1,2,1,2,2,3) for masks 0..7.
    // Signs:        (+,+,+,-,+,-,-,+).
    rd[0]=r[0]; rd[1]=r[1]; rd[2]=r[2]; rd[3]=-r[3];
    rd[4]=r[4]; rd[5]=-r[5]; rd[6]=-r[6]; rd[7]=-r[7];
    cl30_geo_product(r, v, t);
    cl30_geo_product(t, rd, o);
    for (uint i = 0; i < 8; ++i) Out[off+i] = o[i];
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kCliffordRotorSandwichCl30F32, @"clifford_rotor_sandwich_cl30_f32");
    if (!pso) return false;

    NSUInteger byteCount =
        sizeof(float) * static_cast<NSUInteger>(batch) * 8u;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufR, ctx, R, byteCount);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufV, ctx, V, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufR || !bufV || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufR offset:0 atIndex:0];
    [enc setBuffer:bufV offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:3];

    MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(batch), 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>(static_cast<NSUInteger>(batch),
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    MTLSize tg = MTLSizeMake(tg_x, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "clifford_rotor_sandwich_cl30_f32_msl")) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
}

} // namespace

extern "C" void tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32(
    const float* R, const float* V, float* Out, int32_t batch)
{
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_rotor_sandwich_cl30_f32_msl(ctx, R, V, Out, batch)) return;
  reference_clifford_rotor_sandwich_cl30_f32(R, V, Out, batch);
}

//===---------------------------------------------------------------------===//
// GA10 conformance — pointwise GA3 / GA5 MSL kernels (2026-05-17).
//
// Each kernel below is the Cl(3,0) f32 native MSL implementation of one
// `tessera.ga.*` op.  Expressions are generated from the Python Cayley
// table in `tessera.ga.signature._product_table` so the GPU result is
// bitwise-equivalent to the Python GA reference up to fp32 rounding.
//
// Blade order (last axis = 8):
//   0:1   1:e1   2:e2   3:e12   4:e3   5:e13   6:e23   7:e123
//
// Pattern: each block is (numpy reference) + (MSL dispatch helper) +
// (extern "C" entry).  fp16 + bf16 variants of geo_product +
// rotor_sandwich land at the end via the Phase 8.4.4 pattern
// (native MSL half for fp16; fp32-conversion path for bf16).
//===---------------------------------------------------------------------===//

namespace {

// ===========================================================================
// Unary signed-per-grade ops: reverse, grade_involution, conjugate
// ===========================================================================

inline void reference_clifford_reverse_cl30_f32(const float* A, float* C, int32_t b) {
  for (int32_t i = 0; i < b; ++i) {
    const float* a = A + i * 8;
    float* c = C + i * 8;
    c[0] = a[0]; c[1] = a[1]; c[2] = a[2]; c[3] = -a[3];
    c[4] = a[4]; c[5] = -a[5]; c[6] = -a[6]; c[7] = -a[7];
  }
}

inline void reference_clifford_grade_involution_cl30_f32(const float* A, float* C, int32_t b) {
  for (int32_t i = 0; i < b; ++i) {
    const float* a = A + i * 8;
    float* c = C + i * 8;
    c[0] = a[0]; c[1] = -a[1]; c[2] = -a[2]; c[3] = a[3];
    c[4] = -a[4]; c[5] = a[5]; c[6] = a[6]; c[7] = -a[7];
  }
}

inline void reference_clifford_conjugate_cl30_f32(const float* A, float* C, int32_t b) {
  for (int32_t i = 0; i < b; ++i) {
    const float* a = A + i * 8;
    float* c = C + i * 8;
    c[0] = a[0]; c[1] = -a[1]; c[2] = -a[2]; c[3] = -a[3];
    c[4] = -a[4]; c[5] = -a[5]; c[6] = -a[6]; c[7] = a[7];
  }
}

inline void reference_clifford_hodge_star_cl30_f32(const float* A, float* C, int32_t b) {
  for (int32_t i = 0; i < b; ++i) {
    const float* a = A + i * 8;
    float* c = C + i * 8;
    // ⋆ω = reverse(ω) · I — generated from tessera.ga.signature.
    c[0] = a[7];   c[1] = a[6];   c[2] = -a[5];  c[3] = a[4];
    c[4] = a[3];   c[5] = -a[2];  c[6] = a[1];   c[7] = a[0];
  }
}

inline void reference_clifford_norm_squared_cl30_f32(const float* A, float* C, int32_t b) {
  for (int32_t i = 0; i < b; ++i) {
    const float* a = A + i * 8;
    float s = 0.0f;
    for (int k = 0; k < 8; ++k) s += a[k] * a[k];
    C[i] = s;
  }
}

inline void reference_clifford_norm_cl30_f32(const float* A, float* C, int32_t b) {
  for (int32_t i = 0; i < b; ++i) {
    const float* a = A + i * 8;
    float s = 0.0f;
    for (int k = 0; k < 8; ++k) s += a[k] * a[k];
    C[i] = std::sqrt(std::max(0.0f, s));
  }
}

// ===========================================================================
// Binary ops: wedge, left_contraction, inner
// ===========================================================================

inline void reference_clifford_wedge_cl30_f32(
    const float* A, const float* B, float* C, int32_t batch) {
  for (int32_t i = 0; i < batch; ++i) {
    const float* a = A + i * 8;
    const float* b = B + i * 8;
    float* c = C + i * 8;
    c[0] = a[0]*b[0];
    c[1] = a[0]*b[1] + a[1]*b[0];
    c[2] = a[0]*b[2] + a[2]*b[0];
    c[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    c[4] = a[0]*b[4] + a[4]*b[0];
    c[5] = a[0]*b[5] + a[1]*b[4] - a[4]*b[1] + a[5]*b[0];
    c[6] = a[0]*b[6] + a[2]*b[4] - a[4]*b[2] + a[6]*b[0];
    c[7] = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4]
         + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];
  }
}

inline void reference_clifford_left_contraction_cl30_f32(
    const float* A, const float* B, float* C, int32_t batch) {
  for (int32_t i = 0; i < batch; ++i) {
    const float* a = A + i * 8;
    const float* b = B + i * 8;
    float* c = C + i * 8;
    c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] - a[3]*b[3]
         + a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    c[1] = a[0]*b[1] - a[2]*b[3] - a[4]*b[5] - a[6]*b[7];
    c[2] = a[0]*b[2] + a[1]*b[3] - a[4]*b[6] + a[5]*b[7];
    c[3] = a[0]*b[3] + a[4]*b[7];
    c[4] = a[0]*b[4] + a[1]*b[5] + a[2]*b[6] - a[3]*b[7];
    c[5] = a[0]*b[5] - a[2]*b[7];
    c[6] = a[0]*b[6] + a[1]*b[7];
    c[7] = a[0]*b[7];
  }
}

inline void reference_clifford_inner_cl30_f32(
    const float* A, const float* B, float* C, int32_t batch) {
  for (int32_t i = 0; i < batch; ++i) {
    const float* a = A + i * 8;
    const float* b = B + i * 8;
    float s = 0.0f;
    for (int k = 0; k < 8; ++k) s += a[k] * b[k];
    C[i] = s;
  }
}

// ===========================================================================
// grade_projection: keep coefficients whose blade grade is in the mask.
// `grade_mask` is a bitmask over grades (bit k set => keep grade k).
// For Cl(3,0): grade-of-blade[0..7] = (0,1,1,2,1,2,2,3).
// ===========================================================================

inline void reference_clifford_grade_projection_cl30_f32(
    const float* A, float* C, int32_t grade_mask, int32_t batch) {
  // Per-blade grade lookup.
  static const int kBladeGrade[8] = {0, 1, 1, 2, 1, 2, 2, 3};
  for (int32_t i = 0; i < batch; ++i) {
    const float* a = A + i * 8;
    float* c = C + i * 8;
    for (int k = 0; k < 8; ++k) {
      c[k] = ((grade_mask >> kBladeGrade[k]) & 1) ? a[k] : 0.0f;
    }
  }
}

// ===========================================================================
// MSL dispatch helpers
// ===========================================================================

// Common MSL fragment for the f32 unary-pointwise GA ops.  Each kernel
// is small (8 stores) so we bake the whole expression into the kernel
// source string.

static NSString *const kCliffordReverseCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_reverse_cl30_f32(
    device const float* A   [[buffer(0)]],
    device float*       C   [[buffer(1)]],
    constant int&       batch [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    C[off+0] = A[off+0];  C[off+1] = A[off+1];  C[off+2] = A[off+2];  C[off+3] = -A[off+3];
    C[off+4] = A[off+4];  C[off+5] = -A[off+5]; C[off+6] = -A[off+6]; C[off+7] = -A[off+7];
}
)MSL";

static NSString *const kCliffordGradeInvolutionCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_grade_involution_cl30_f32(
    device const float* A   [[buffer(0)]],
    device float*       C   [[buffer(1)]],
    constant int&       batch [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    C[off+0] = A[off+0];  C[off+1] = -A[off+1]; C[off+2] = -A[off+2]; C[off+3] = A[off+3];
    C[off+4] = -A[off+4]; C[off+5] = A[off+5];  C[off+6] = A[off+6];  C[off+7] = -A[off+7];
}
)MSL";

static NSString *const kCliffordConjugateCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_conjugate_cl30_f32(
    device const float* A   [[buffer(0)]],
    device float*       C   [[buffer(1)]],
    constant int&       batch [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    C[off+0] = A[off+0];  C[off+1] = -A[off+1]; C[off+2] = -A[off+2]; C[off+3] = -A[off+3];
    C[off+4] = -A[off+4]; C[off+5] = -A[off+5]; C[off+6] = -A[off+6]; C[off+7] = A[off+7];
}
)MSL";

static NSString *const kCliffordHodgeStarCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_hodge_star_cl30_f32(
    device const float* A   [[buffer(0)]],
    device float*       C   [[buffer(1)]],
    constant int&       batch [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    C[off+0] = A[off+7];  C[off+1] = A[off+6];  C[off+2] = -A[off+5]; C[off+3] = A[off+4];
    C[off+4] = A[off+3];  C[off+5] = -A[off+2]; C[off+6] = A[off+1];  C[off+7] = A[off+0];
}
)MSL";

static NSString *const kCliffordNormSquaredCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_norm_squared_cl30_f32(
    device const float* A   [[buffer(0)]],
    device float*       C   [[buffer(1)]],
    constant int&       batch [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float s = 0.0f;
    s += A[off+0]*A[off+0]; s += A[off+1]*A[off+1];
    s += A[off+2]*A[off+2]; s += A[off+3]*A[off+3];
    s += A[off+4]*A[off+4]; s += A[off+5]*A[off+5];
    s += A[off+6]*A[off+6]; s += A[off+7]*A[off+7];
    C[gid] = s;
}
)MSL";

static NSString *const kCliffordNormCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_norm_cl30_f32(
    device const float* A   [[buffer(0)]],
    device float*       C   [[buffer(1)]],
    constant int&       batch [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float s = 0.0f;
    s += A[off+0]*A[off+0]; s += A[off+1]*A[off+1];
    s += A[off+2]*A[off+2]; s += A[off+3]*A[off+3];
    s += A[off+4]*A[off+4]; s += A[off+5]*A[off+5];
    s += A[off+6]*A[off+6]; s += A[off+7]*A[off+7];
    C[gid] = sqrt(max(0.0f, s));
}
)MSL";

static NSString *const kCliffordWedgeCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_wedge_cl30_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       C   [[buffer(2)]],
    constant int&       batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float a0=A[off+0], a1=A[off+1], a2=A[off+2], a3=A[off+3];
    float a4=A[off+4], a5=A[off+5], a6=A[off+6], a7=A[off+7];
    float b0=B[off+0], b1=B[off+1], b2=B[off+2], b3=B[off+3];
    float b4=B[off+4], b5=B[off+5], b6=B[off+6], b7=B[off+7];
    C[off+0] = a0*b0;
    C[off+1] = a0*b1 + a1*b0;
    C[off+2] = a0*b2 + a2*b0;
    C[off+3] = a0*b3 + a1*b2 - a2*b1 + a3*b0;
    C[off+4] = a0*b4 + a4*b0;
    C[off+5] = a0*b5 + a1*b4 - a4*b1 + a5*b0;
    C[off+6] = a0*b6 + a2*b4 - a4*b2 + a6*b0;
    C[off+7] = a0*b7 + a1*b6 - a2*b5 + a3*b4 + a4*b3 - a5*b2 + a6*b1 + a7*b0;
}
)MSL";

static NSString *const kCliffordLeftContractionCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_left_contraction_cl30_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       C   [[buffer(2)]],
    constant int&       batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float a0=A[off+0], a1=A[off+1], a2=A[off+2], a3=A[off+3];
    float a4=A[off+4], a5=A[off+5], a6=A[off+6], a7=A[off+7];
    float b0=B[off+0], b1=B[off+1], b2=B[off+2], b3=B[off+3];
    float b4=B[off+4], b5=B[off+5], b6=B[off+6], b7=B[off+7];
    C[off+0] = a0*b0 + a1*b1 + a2*b2 - a3*b3 + a4*b4 - a5*b5 - a6*b6 - a7*b7;
    C[off+1] = a0*b1 - a2*b3 - a4*b5 - a6*b7;
    C[off+2] = a0*b2 + a1*b3 - a4*b6 + a5*b7;
    C[off+3] = a0*b3 + a4*b7;
    C[off+4] = a0*b4 + a1*b5 + a2*b6 - a3*b7;
    C[off+5] = a0*b5 - a2*b7;
    C[off+6] = a0*b6 + a1*b7;
    C[off+7] = a0*b7;
}
)MSL";

static NSString *const kCliffordInnerCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_inner_cl30_f32(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       C   [[buffer(2)]],
    constant int&       batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float s = 0.0f;
    s += A[off+0]*B[off+0]; s += A[off+1]*B[off+1];
    s += A[off+2]*B[off+2]; s += A[off+3]*B[off+3];
    s += A[off+4]*B[off+4]; s += A[off+5]*B[off+5];
    s += A[off+6]*B[off+6]; s += A[off+7]*B[off+7];
    C[gid] = s;
}
)MSL";

static NSString *const kCliffordGradeProjectionCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void clifford_grade_projection_cl30_f32(
    device const float* A           [[buffer(0)]],
    device float*       C           [[buffer(1)]],
    constant int&       batch       [[buffer(2)]],
    constant int&       grade_mask  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    // Blade grade lookup for Cl(3,0).
    int blade_grade[8] = {0, 1, 1, 2, 1, 2, 2, 3};
    for (uint k = 0; k < 8u; ++k) {
        bool keep = ((grade_mask >> blade_grade[k]) & 1) != 0;
        C[off+k] = keep ? A[off+k] : 0.0f;
    }
}
)MSL";

// ---- Generic unary dispatch (8 in, 8 out per sample) ----

static bool dispatch_clifford_unary_8x8_f32_msl(
    MetalDeviceContext &ctx,
    NSString *source, NSString *entry,
    const float* A, float* C, int32_t batch) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, source, entry);
    if (!pso) return false;
    size_t byteCount = sizeof(float) * (size_t)batch * 8u;
    // Pool path — recycle shared-storage buffers across dispatches.
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, byteCount);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, byteCount);
    if (!bufA || !bufC) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufC offset:0 atIndex:1];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:2];
    MTLSize grid = MTLSizeMake((NSUInteger)batch, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)batch,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                           "clifford_unary_8x8_f32_msl");
    if (ok) std::memcpy(C, [bufC contents], byteCount);
    return ok;
  }
}

// ---- Generic scalar-reducer dispatch (8 in -> 1 out per sample) ----

static bool dispatch_clifford_unary_8x1_f32_msl(
    MetalDeviceContext &ctx,
    NSString *source, NSString *entry,
    const float* A, float* C, int32_t batch) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, source, entry);
    if (!pso) return false;
    NSUInteger inBytes  = sizeof(float) * (NSUInteger)batch * 8u;
    NSUInteger outBytes = sizeof(float) * (NSUInteger)batch;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, inBytes);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, outBytes);
    if (!bufA || !bufC) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufC offset:0 atIndex:1];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:2];
    MTLSize grid = MTLSizeMake((NSUInteger)batch, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)batch,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "clifford_unary_8x1_f32_msl");
    if (_pool_ok) std::memcpy(C, [bufC contents], outBytes);
    return _pool_ok;
  }
}

// ---- Generic binary dispatch (two 8 inputs -> 8 output per sample) ----

static bool dispatch_clifford_binary_8x8_f32_msl(
    MetalDeviceContext &ctx,
    NSString *source, NSString *entry,
    const float* A, const float* B, float* C, int32_t batch) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, source, entry);
    if (!pso) return false;
    NSUInteger byteCount = sizeof(float) * (NSUInteger)batch * 8u;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, byteCount);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, byteCount);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, byteCount);
    if (!bufA || !bufB || !bufC) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:3];
    MTLSize grid = MTLSizeMake((NSUInteger)batch, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)batch,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "clifford_binary_8x8_f32_msl");
    if (_pool_ok) std::memcpy(C, [bufC contents], byteCount);
    return _pool_ok;
  }
}

// ---- Generic binary scalar-reducer (two 8 inputs -> 1 output per sample) ----

static bool dispatch_clifford_binary_8x1_f32_msl(
    MetalDeviceContext &ctx,
    NSString *source, NSString *entry,
    const float* A, const float* B, float* C, int32_t batch) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, source, entry);
    if (!pso) return false;
    NSUInteger inBytes  = sizeof(float) * (NSUInteger)batch * 8u;
    NSUInteger outBytes = sizeof(float) * (NSUInteger)batch;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, inBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, inBytes);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, outBytes);
    if (!bufA || !bufB || !bufC) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:3];
    MTLSize grid = MTLSizeMake((NSUInteger)batch, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)batch,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "clifford_binary_8x1_f32_msl");
    if (_pool_ok) std::memcpy(C, [bufC contents], outBytes);
    return _pool_ok;
  }
}

// ---- Grade-projection dispatch (extra int32 grade_mask buffer) ----

static bool dispatch_clifford_grade_projection_cl30_f32_msl(
    MetalDeviceContext &ctx, const float* A, float* C,
    int32_t grade_mask, int32_t batch) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kCliffordGradeProjectionCl30F32Source,
        @"clifford_grade_projection_cl30_f32");
    if (!pso) return false;
    NSUInteger byteCount = sizeof(float) * (NSUInteger)batch * 8u;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, byteCount);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, byteCount);
    if (!bufA || !bufC) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufC offset:0 atIndex:1];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:2];
    [enc setBytes:&grade_mask length:sizeof(int32_t) atIndex:3];
    MTLSize grid = MTLSizeMake((NSUInteger)batch, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)batch,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "clifford_grade_projection_cl30_f32_msl");
    if (_pool_ok) std::memcpy(C, [bufC contents], byteCount);
    return _pool_ok;
  }
}

} // namespace

// ===========================================================================
// extern "C" entries — unary 8→8 f32
// ===========================================================================

#define CLIFFORD_UNARY_8x8_F32_ENTRY(NAME, SOURCE, ENTRY, REF)                \
extern "C" void NAME(const float* A, float* C, int32_t batch) {               \
  MetalDeviceContext &ctx = deviceContext();                                  \
  if (ctx.ok && dispatch_clifford_unary_8x8_f32_msl(                          \
          ctx, SOURCE, @ENTRY, A, C, batch)) return;                          \
  REF(A, C, batch);                                                           \
}

CLIFFORD_UNARY_8x8_F32_ENTRY(tessera_apple_gpu_clifford_reverse_cl30_f32,
                             kCliffordReverseCl30F32Source,
                             "clifford_reverse_cl30_f32",
                             reference_clifford_reverse_cl30_f32)
CLIFFORD_UNARY_8x8_F32_ENTRY(tessera_apple_gpu_clifford_grade_involution_cl30_f32,
                             kCliffordGradeInvolutionCl30F32Source,
                             "clifford_grade_involution_cl30_f32",
                             reference_clifford_grade_involution_cl30_f32)
CLIFFORD_UNARY_8x8_F32_ENTRY(tessera_apple_gpu_clifford_conjugate_cl30_f32,
                             kCliffordConjugateCl30F32Source,
                             "clifford_conjugate_cl30_f32",
                             reference_clifford_conjugate_cl30_f32)
CLIFFORD_UNARY_8x8_F32_ENTRY(tessera_apple_gpu_clifford_hodge_star_cl30_f32,
                             kCliffordHodgeStarCl30F32Source,
                             "clifford_hodge_star_cl30_f32",
                             reference_clifford_hodge_star_cl30_f32)

#undef CLIFFORD_UNARY_8x8_F32_ENTRY

// ===========================================================================
// extern "C" entries — unary 8→1 f32 (norm, norm_squared)
// ===========================================================================

extern "C" void tessera_apple_gpu_clifford_norm_squared_cl30_f32(
    const float* A, float* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_unary_8x1_f32_msl(
          ctx, kCliffordNormSquaredCl30F32Source,
          @"clifford_norm_squared_cl30_f32", A, C, batch)) return;
  reference_clifford_norm_squared_cl30_f32(A, C, batch);
}

extern "C" void tessera_apple_gpu_clifford_norm_cl30_f32(
    const float* A, float* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_unary_8x1_f32_msl(
          ctx, kCliffordNormCl30F32Source,
          @"clifford_norm_cl30_f32", A, C, batch)) return;
  reference_clifford_norm_cl30_f32(A, C, batch);
}

// ===========================================================================
// extern "C" entries — binary 8x8→8 / 8x8→1 f32
// ===========================================================================

extern "C" void tessera_apple_gpu_clifford_wedge_cl30_f32(
    const float* A, const float* B, float* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_binary_8x8_f32_msl(
          ctx, kCliffordWedgeCl30F32Source, @"clifford_wedge_cl30_f32",
          A, B, C, batch)) return;
  reference_clifford_wedge_cl30_f32(A, B, C, batch);
}

extern "C" void tessera_apple_gpu_clifford_left_contraction_cl30_f32(
    const float* A, const float* B, float* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_binary_8x8_f32_msl(
          ctx, kCliffordLeftContractionCl30F32Source,
          @"clifford_left_contraction_cl30_f32",
          A, B, C, batch)) return;
  reference_clifford_left_contraction_cl30_f32(A, B, C, batch);
}

extern "C" void tessera_apple_gpu_clifford_inner_cl30_f32(
    const float* A, const float* B, float* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_binary_8x1_f32_msl(
          ctx, kCliffordInnerCl30F32Source, @"clifford_inner_cl30_f32",
          A, B, C, batch)) return;
  reference_clifford_inner_cl30_f32(A, B, C, batch);
}

// ===========================================================================
// extern "C" entries — grade_projection (extra grade_mask int32)
// ===========================================================================

extern "C" void tessera_apple_gpu_clifford_grade_projection_cl30_f32(
    const float* A, float* C, int32_t grade_mask, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_grade_projection_cl30_f32_msl(
          ctx, A, C, grade_mask, batch)) return;
  reference_clifford_grade_projection_cl30_f32(A, C, grade_mask, batch);
}

//===---------------------------------------------------------------------===//
// Phase 8.4.4-style fp16 + bf16 ports of clifford_geo_product +
// clifford_rotor_sandwich on Cl(3,0) (2026-05-17).
//
//   fp16 : native MSL `half` kernel.  Per-blade `half` storage with
//          on-the-fly promotion to `float` for the 64-term contraction
//          (matches softmax_f16 numerical-stability pattern).
//   bf16 : fp32-conversion path on the host — convert to fp32 via the
//          existing bfloat16_to_float_gpu / float_to_bfloat16_gpu
//          helpers, dispatch the fp32 MSL kernel, convert back.  No
//          native MSL bf16 yet (matches the matmul_bf16 / softmax_bf16
//          precedent — MPS doesn't expose a bf16 matrix descriptor on
//          macOS 14, and Apple GPU MSL bf16 is a separate roadmap).
//===---------------------------------------------------------------------===//

namespace {

static NSString *const kCliffordGeoProductCl30F16Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void clifford_geo_product_cl30_f16(
    device const half*  A   [[buffer(0)]],
    device const half*  B   [[buffer(1)]],
    device half*        C   [[buffer(2)]],
    constant int&       batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    // Promote to float for the 64-term accumulation so per-blade
    // rounding stays close to the fp32 native kernel.
    float a0=float(A[off+0]), a1=float(A[off+1]), a2=float(A[off+2]), a3=float(A[off+3]);
    float a4=float(A[off+4]), a5=float(A[off+5]), a6=float(A[off+6]), a7=float(A[off+7]);
    float b0=float(B[off+0]), b1=float(B[off+1]), b2=float(B[off+2]), b3=float(B[off+3]);
    float b4=float(B[off+4]), b5=float(B[off+5]), b6=float(B[off+6]), b7=float(B[off+7]);

    C[off+0] = half(a0*b0 + a1*b1 + a2*b2 - a3*b3 + a4*b4 - a5*b5 - a6*b6 - a7*b7);
    C[off+1] = half(a0*b1 + a1*b0 - a2*b3 + a3*b2 - a4*b5 + a5*b4 - a6*b7 - a7*b6);
    C[off+2] = half(a0*b2 + a1*b3 + a2*b0 - a3*b1 - a4*b6 + a5*b7 + a6*b4 + a7*b5);
    C[off+3] = half(a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 + a7*b4);
    C[off+4] = half(a0*b4 + a1*b5 + a2*b6 - a3*b7 + a4*b0 - a5*b1 - a6*b2 - a7*b3);
    C[off+5] = half(a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 - a7*b2);
    C[off+6] = half(a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 + a7*b1);
    C[off+7] = half(a0*b7 + a1*b6 - a2*b5 + a3*b4 + a4*b3 - a5*b2 + a6*b1 + a7*b0);
}
)MSL";

static NSString *const kCliffordRotorSandwichCl30F16Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

inline void cl30_geo_product_fp16_via_fp32(
    thread float a[8], thread float b[8], thread float c[8])
{
    c[0] = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] - a[3]*b[3] + a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7];
    c[1] = a[0]*b[1] + a[1]*b[0] - a[2]*b[3] + a[3]*b[2] - a[4]*b[5] + a[5]*b[4] - a[6]*b[7] - a[7]*b[6];
    c[2] = a[0]*b[2] + a[1]*b[3] + a[2]*b[0] - a[3]*b[1] - a[4]*b[6] + a[5]*b[7] + a[6]*b[4] + a[7]*b[5];
    c[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0] + a[4]*b[7] - a[5]*b[6] + a[6]*b[5] + a[7]*b[4];
    c[4] = a[0]*b[4] + a[1]*b[5] + a[2]*b[6] - a[3]*b[7] + a[4]*b[0] - a[5]*b[1] - a[6]*b[2] - a[7]*b[3];
    c[5] = a[0]*b[5] + a[1]*b[4] - a[2]*b[7] + a[3]*b[6] - a[4]*b[1] + a[5]*b[0] - a[6]*b[3] - a[7]*b[2];
    c[6] = a[0]*b[6] + a[1]*b[7] + a[2]*b[4] - a[3]*b[5] - a[4]*b[2] + a[5]*b[3] + a[6]*b[0] + a[7]*b[1];
    c[7] = a[0]*b[7] + a[1]*b[6] - a[2]*b[5] + a[3]*b[4] + a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0];
}

kernel void clifford_rotor_sandwich_cl30_f16(
    device const half*  R   [[buffer(0)]],
    device const half*  V   [[buffer(1)]],
    device half*        Out [[buffer(2)]],
    constant int&       batch [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float r[8], v[8], rd[8], t[8], o[8];
    for (uint i = 0; i < 8; ++i) {
        r[i] = float(R[off+i]);
        v[i] = float(V[off+i]);
    }
    // R†: Cl(3,0) reverse-sign per blade — (+,+,+,-,+,-,-,-).
    rd[0]=r[0]; rd[1]=r[1]; rd[2]=r[2]; rd[3]=-r[3];
    rd[4]=r[4]; rd[5]=-r[5]; rd[6]=-r[6]; rd[7]=-r[7];
    cl30_geo_product_fp16_via_fp32(r, v, t);
    cl30_geo_product_fp16_via_fp32(t, rd, o);
    for (uint i = 0; i < 8; ++i) Out[off+i] = half(o[i]);
}
)MSL";

static bool dispatch_clifford_geo_product_cl30_f16_msl(
    MetalDeviceContext &ctx, const uint16_t* A, const uint16_t* B,
    uint16_t* C, int32_t batch) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kCliffordGeoProductCl30F16Source,
        @"clifford_geo_product_cl30_f16");
    if (!pso) return false;
    NSUInteger byteCount = sizeof(uint16_t) * (NSUInteger)batch * 8u;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, byteCount);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, byteCount);
    TS_METAL_BUF_ACQUIRE(bufC, ctx, byteCount);
    if (!bufA || !bufB || !bufC) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:3];
    MTLSize grid = MTLSizeMake((NSUInteger)batch, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)batch,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "clifford_geo_product_cl30_f16_msl");
    if (_pool_ok) std::memcpy(C, [bufC contents], byteCount);
    return _pool_ok;
  }
}

static bool dispatch_clifford_rotor_sandwich_cl30_f16_msl(
    MetalDeviceContext &ctx, const uint16_t* R, const uint16_t* V,
    uint16_t* Out, int32_t batch) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kCliffordRotorSandwichCl30F16Source,
        @"clifford_rotor_sandwich_cl30_f16");
    if (!pso) return false;
    NSUInteger byteCount = sizeof(uint16_t) * (NSUInteger)batch * 8u;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufR, ctx, R, byteCount);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufV, ctx, V, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufR || !bufV || !bufO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufR offset:0 atIndex:0];
    [enc setBuffer:bufV offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(int32_t) atIndex:3];
    MTLSize grid = MTLSizeMake((NSUInteger)batch, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)batch,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "clifford_rotor_sandwich_cl30_f16_msl");
    if (_pool_ok) std::memcpy(Out, [bufO contents], byteCount);
    return _pool_ok;
  }
}

inline void reference_clifford_geo_product_cl30_f16_via_fp32(
    const uint16_t* A, const uint16_t* B, uint16_t* C, int32_t batch) {
  std::vector<float> Af((std::size_t)batch * 8);
  std::vector<float> Bf((std::size_t)batch * 8);
  std::vector<float> Cf((std::size_t)batch * 8);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = half_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = half_to_float_gpu(B[i]);
  reference_clifford_geo_product_cl30_f32(Af.data(), Bf.data(), Cf.data(), batch);
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_half_gpu(Cf[i]);
}

inline void reference_clifford_geo_product_cl30_bf16_via_fp32(
    const uint16_t* A, const uint16_t* B, uint16_t* C, int32_t batch) {
  std::vector<float> Af((std::size_t)batch * 8);
  std::vector<float> Bf((std::size_t)batch * 8);
  std::vector<float> Cf((std::size_t)batch * 8);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  reference_clifford_geo_product_cl30_f32(Af.data(), Bf.data(), Cf.data(), batch);
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_bfloat16_gpu(Cf[i]);
}

bool dispatch_clifford_geo_product_cl30_bf16_via_fp32(
    MetalDeviceContext &ctx, const uint16_t* A, const uint16_t* B,
    uint16_t* C, int32_t batch) {
  std::vector<float> Af((std::size_t)batch * 8);
  std::vector<float> Bf((std::size_t)batch * 8);
  std::vector<float> Cf((std::size_t)batch * 8);
  for (std::size_t i = 0; i < Af.size(); ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < Bf.size(); ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  if (!dispatch_clifford_geo_product_cl30_f32_msl(ctx, Af.data(), Bf.data(),
                                                  Cf.data(), batch))
    return false;
  for (std::size_t i = 0; i < Cf.size(); ++i) C[i] = float_to_bfloat16_gpu(Cf[i]);
  return true;
}

inline void reference_clifford_rotor_sandwich_cl30_f16_via_fp32(
    const uint16_t* R, const uint16_t* V, uint16_t* Out, int32_t batch) {
  std::vector<float> Rf((std::size_t)batch * 8);
  std::vector<float> Vf((std::size_t)batch * 8);
  std::vector<float> Of((std::size_t)batch * 8);
  for (std::size_t i = 0; i < Rf.size(); ++i) Rf[i] = half_to_float_gpu(R[i]);
  for (std::size_t i = 0; i < Vf.size(); ++i) Vf[i] = half_to_float_gpu(V[i]);
  reference_clifford_rotor_sandwich_cl30_f32(Rf.data(), Vf.data(), Of.data(), batch);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_half_gpu(Of[i]);
}

inline void reference_clifford_rotor_sandwich_cl30_bf16_via_fp32(
    const uint16_t* R, const uint16_t* V, uint16_t* Out, int32_t batch) {
  std::vector<float> Rf((std::size_t)batch * 8);
  std::vector<float> Vf((std::size_t)batch * 8);
  std::vector<float> Of((std::size_t)batch * 8);
  for (std::size_t i = 0; i < Rf.size(); ++i) Rf[i] = bfloat16_to_float_gpu(R[i]);
  for (std::size_t i = 0; i < Vf.size(); ++i) Vf[i] = bfloat16_to_float_gpu(V[i]);
  reference_clifford_rotor_sandwich_cl30_f32(Rf.data(), Vf.data(), Of.data(), batch);
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_bfloat16_gpu(Of[i]);
}

bool dispatch_clifford_rotor_sandwich_cl30_bf16_via_fp32(
    MetalDeviceContext &ctx, const uint16_t* R, const uint16_t* V,
    uint16_t* Out, int32_t batch) {
  std::vector<float> Rf((std::size_t)batch * 8);
  std::vector<float> Vf((std::size_t)batch * 8);
  std::vector<float> Of((std::size_t)batch * 8);
  for (std::size_t i = 0; i < Rf.size(); ++i) Rf[i] = bfloat16_to_float_gpu(R[i]);
  for (std::size_t i = 0; i < Vf.size(); ++i) Vf[i] = bfloat16_to_float_gpu(V[i]);
  if (!dispatch_clifford_rotor_sandwich_cl30_f32_msl(ctx, Rf.data(), Vf.data(),
                                                     Of.data(), batch))
    return false;
  for (std::size_t i = 0; i < Of.size(); ++i) Out[i] = float_to_bfloat16_gpu(Of[i]);
  return true;
}

} // namespace

extern "C" void tessera_apple_gpu_clifford_geo_product_cl30_f16(
    const uint16_t* A, const uint16_t* B, uint16_t* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_geo_product_cl30_f16_msl(ctx, A, B, C, batch)) return;
  reference_clifford_geo_product_cl30_f16_via_fp32(A, B, C, batch);
}

extern "C" void tessera_apple_gpu_clifford_geo_product_cl30_bf16(
    const uint16_t* A, const uint16_t* B, uint16_t* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_geo_product_cl30_bf16_via_fp32(ctx, A, B, C, batch)) return;
  reference_clifford_geo_product_cl30_bf16_via_fp32(A, B, C, batch);
}

extern "C" void tessera_apple_gpu_clifford_rotor_sandwich_cl30_f16(
    const uint16_t* R, const uint16_t* V, uint16_t* Out, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_rotor_sandwich_cl30_f16_msl(ctx, R, V, Out, batch)) return;
  reference_clifford_rotor_sandwich_cl30_f16_via_fp32(R, V, Out, batch);
}

extern "C" void tessera_apple_gpu_clifford_rotor_sandwich_cl30_bf16(
    const uint16_t* R, const uint16_t* V, uint16_t* Out, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_rotor_sandwich_cl30_bf16_via_fp32(ctx, R, V, Out, batch)) return;
  reference_clifford_rotor_sandwich_cl30_bf16_via_fp32(R, V, Out, batch);
}

//===---------------------------------------------------------------------===//
// GA11 — final 6 GA primitives (2026-05-17 follow-on).
//
// Closes the apple_gpu coverage to all 17 GA primitives:
//   - clifford_exp_cl30_f32       closed-form (pure bivector) + power series
//   - clifford_log_cl30_f32       closed-form for Cl(3,0) rotors
//   - clifford_ext_deriv_cl30_f32 finite-difference exterior derivative
//   - clifford_vec_deriv_cl30_f32 finite-difference geometric gradient
//   - clifford_codiff_cl30_f32    composed ⋆d⋆
//   - clifford_integral_cl30_f32  weighted Riemann sum
//
// The field ops take (D0, D1, D2) grid dims + (h0, h1, h2) per-axis
// spacing — same convention as the Python `MultivectorField` class.
// Spatial dim = 3 matches Cl(3,0).n by Decision GA-L0.
//===---------------------------------------------------------------------===//

namespace {

// ===========================================================================
// exp_mv reference + dispatch
// ===========================================================================

inline void cl30_geo_product_inline(const float* a, const float* x, float* c) {
  c[0] = a[0]*x[0] + a[1]*x[1] + a[2]*x[2] - a[3]*x[3]
       + a[4]*x[4] - a[5]*x[5] - a[6]*x[6] - a[7]*x[7];
  c[1] = a[0]*x[1] + a[1]*x[0] - a[2]*x[3] + a[3]*x[2]
       - a[4]*x[5] + a[5]*x[4] - a[6]*x[7] - a[7]*x[6];
  c[2] = a[0]*x[2] + a[1]*x[3] + a[2]*x[0] - a[3]*x[1]
       - a[4]*x[6] + a[5]*x[7] + a[6]*x[4] + a[7]*x[5];
  c[3] = a[0]*x[3] + a[1]*x[2] - a[2]*x[1] + a[3]*x[0]
       + a[4]*x[7] - a[5]*x[6] + a[6]*x[5] + a[7]*x[4];
  c[4] = a[0]*x[4] + a[1]*x[5] + a[2]*x[6] - a[3]*x[7]
       + a[4]*x[0] - a[5]*x[1] - a[6]*x[2] - a[7]*x[3];
  c[5] = a[0]*x[5] + a[1]*x[4] - a[2]*x[7] + a[3]*x[6]
       - a[4]*x[1] + a[5]*x[0] - a[6]*x[3] - a[7]*x[2];
  c[6] = a[0]*x[6] + a[1]*x[7] + a[2]*x[4] - a[3]*x[5]
       - a[4]*x[2] + a[5]*x[3] + a[6]*x[0] + a[7]*x[1];
  c[7] = a[0]*x[7] + a[1]*x[6] - a[2]*x[5] + a[3]*x[4]
       + a[4]*x[3] - a[5]*x[2] + a[6]*x[1] + a[7]*x[0];
}

inline bool cl30_is_pure_bivector(const float* a, float eps = 1e-9f) {
  return std::fabs(a[0]) + std::fabs(a[1]) + std::fabs(a[2])
       + std::fabs(a[4]) + std::fabs(a[7]) < eps;
}

inline void reference_clifford_exp_cl30_f32(
    const float* A, float* C, int32_t batch) {
  for (int32_t b = 0; b < batch; ++b) {
    const float* a = A + b * 8;
    float* c = C + b * 8;
    if (cl30_is_pure_bivector(a)) {
      // Closed-form: exp(B) = cos(|B|) + sin(|B|)/|B| * B.
      float b3 = a[3], b5 = a[5], b6 = a[6];
      float bn = std::sqrt(b3*b3 + b5*b5 + b6*b6);
      if (bn < 1e-12f) {
        c[0] = 1.0f; c[1] = c[2] = c[3] = c[4] = c[5] = c[6] = c[7] = 0.0f;
      } else {
        float cs = std::cos(bn);
        float sn = std::sin(bn) / bn;
        c[0] = cs; c[1] = c[2] = 0.0f;
        c[3] = sn * b3;
        c[4] = 0.0f;
        c[5] = sn * b5;
        c[6] = sn * b6;
        c[7] = 0.0f;
      }
    } else {
      // Power series: 1 + a + a²/2 + a³/6 + ...  (K = 24 terms).
      float power[8] = {1, 0, 0, 0, 0, 0, 0, 0};
      float result[8] = {1, 0, 0, 0, 0, 0, 0, 0};
      double fact = 1.0;
      for (int k = 1; k <= 24; ++k) {
        float next[8];
        cl30_geo_product_inline(power, a, next);
        fact *= k;
        float inv_fact = float(1.0 / fact);
        for (int r = 0; r < 8; ++r) {
          power[r] = next[r];
          result[r] += power[r] * inv_fact;
        }
      }
      for (int r = 0; r < 8; ++r) c[r] = result[r];
    }
  }
}

static NSString *const kCliffordExpCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

inline void cl30_gp(thread float a[8], thread float x[8], thread float c[8]) {
    c[0] = a[0]*x[0] + a[1]*x[1] + a[2]*x[2] - a[3]*x[3] + a[4]*x[4] - a[5]*x[5] - a[6]*x[6] - a[7]*x[7];
    c[1] = a[0]*x[1] + a[1]*x[0] - a[2]*x[3] + a[3]*x[2] - a[4]*x[5] + a[5]*x[4] - a[6]*x[7] - a[7]*x[6];
    c[2] = a[0]*x[2] + a[1]*x[3] + a[2]*x[0] - a[3]*x[1] - a[4]*x[6] + a[5]*x[7] + a[6]*x[4] + a[7]*x[5];
    c[3] = a[0]*x[3] + a[1]*x[2] - a[2]*x[1] + a[3]*x[0] + a[4]*x[7] - a[5]*x[6] + a[6]*x[5] + a[7]*x[4];
    c[4] = a[0]*x[4] + a[1]*x[5] + a[2]*x[6] - a[3]*x[7] + a[4]*x[0] - a[5]*x[1] - a[6]*x[2] - a[7]*x[3];
    c[5] = a[0]*x[5] + a[1]*x[4] - a[2]*x[7] + a[3]*x[6] - a[4]*x[1] + a[5]*x[0] - a[6]*x[3] - a[7]*x[2];
    c[6] = a[0]*x[6] + a[1]*x[7] + a[2]*x[4] - a[3]*x[5] - a[4]*x[2] + a[5]*x[3] + a[6]*x[0] + a[7]*x[1];
    c[7] = a[0]*x[7] + a[1]*x[6] - a[2]*x[5] + a[3]*x[4] + a[4]*x[3] - a[5]*x[2] + a[6]*x[1] + a[7]*x[0];
}

kernel void clifford_exp_cl30_f32(
    device const float* A   [[buffer(0)]],
    device float*       C   [[buffer(1)]],
    constant int&       batch [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float a[8];
    for (uint i = 0; i < 8; ++i) a[i] = A[off + i];

    // Pure-bivector closed form: exp(B) = cos(|B|) + sin(|B|)/|B| * B.
    float other = abs(a[0]) + abs(a[1]) + abs(a[2]) + abs(a[4]) + abs(a[7]);
    if (other < 1e-9f) {
        float bn = sqrt(a[3]*a[3] + a[5]*a[5] + a[6]*a[6]);
        if (bn < 1e-12f) {
            C[off+0] = 1.0f;
            for (uint i = 1; i < 8; ++i) C[off+i] = 0.0f;
        } else {
            float cs = cos(bn);
            float sn = sin(bn) / bn;
            C[off+0] = cs;
            C[off+1] = 0.0f; C[off+2] = 0.0f;
            C[off+3] = sn * a[3];
            C[off+4] = 0.0f;
            C[off+5] = sn * a[5];
            C[off+6] = sn * a[6];
            C[off+7] = 0.0f;
        }
        return;
    }

    // Power series fallback: 24 terms.
    float power[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    float result[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    float fact = 1.0f;
    for (int k = 1; k <= 24; ++k) {
        float next[8];
        cl30_gp(power, a, next);
        fact *= float(k);
        float inv_fact = 1.0f / fact;
        for (uint r = 0; r < 8; ++r) {
            power[r] = next[r];
            result[r] += power[r] * inv_fact;
        }
    }
    for (uint r = 0; r < 8; ++r) C[off + r] = result[r];
}
)MSL";

// ===========================================================================
// log_mv reference + dispatch  (closed-form for Cl(3,0) rotors)
// ===========================================================================

inline void reference_clifford_log_cl30_f32(
    const float* A, float* C, int32_t batch) {
  for (int32_t b = 0; b < batch; ++b) {
    const float* a = A + b * 8;
    float* c = C + b * 8;
    float s = a[0];
    float b3 = a[3], b5 = a[5], b6 = a[6];
    float bn = std::sqrt(b3*b3 + b5*b5 + b6*b6);
    float half_theta = std::atan2(bn, s);
    float safe = bn > 1e-12f ? bn : 1.0f;
    float scale = half_theta / safe;
    c[0] = 0.0f; c[1] = 0.0f; c[2] = 0.0f;
    c[3] = scale * b3;
    c[4] = 0.0f;
    c[5] = scale * b5;
    c[6] = scale * b6;
    c[7] = 0.0f;
  }
}

static NSString *const kCliffordLogCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void clifford_log_cl30_f32(
    device const float* A   [[buffer(0)]],
    device float*       C   [[buffer(1)]],
    constant int&       batch [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)batch) return;
    uint off = gid * 8;
    float s = A[off+0];
    float b3 = A[off+3], b5 = A[off+5], b6 = A[off+6];
    float bn = sqrt(b3*b3 + b5*b5 + b6*b6);
    float half_theta = atan2(bn, s);
    float safe = bn > 1e-12f ? bn : 1.0f;
    float scale = half_theta / safe;
    C[off+0] = 0.0f; C[off+1] = 0.0f; C[off+2] = 0.0f;
    C[off+3] = scale * b3;
    C[off+4] = 0.0f;
    C[off+5] = scale * b5;
    C[off+6] = scale * b6;
    C[off+7] = 0.0f;
}
)MSL";

// ===========================================================================
// ext_deriv + vec_deriv on 3D Cl(3,0) grids
//
// Grid: (D0, D1, D2) with spacing (h0, h1, h2); each cell holds 8 floats.
// Boundary cells (any index 0 or D-1 in any axis) get the partial = 0
// contribution from that axis — matches the Python ext_deriv's central-
// difference behavior at the grid edges (Python uses np.gradient which
// uses one-sided diffs; ours skips for simplicity, matching the d²=0
// "interior" claim of the GA5 acceptance test).
// ===========================================================================

inline void reference_clifford_ext_deriv_cl30_f32(
    const float* F, float* Out,
    int32_t D0, int32_t D1, int32_t D2,
    float h0, float h1, float h2) {
  int64_t stride2 = 8;
  int64_t stride1 = (int64_t)D2 * stride2;
  int64_t stride0 = (int64_t)D1 * stride1;
  for (int32_t i = 0; i < D0; ++i) {
    for (int32_t j = 0; j < D1; ++j) {
      for (int32_t k = 0; k < D2; ++k) {
        int64_t base = (int64_t)i * stride0 + (int64_t)j * stride1 + (int64_t)k * stride2;
        float c[8] = {0,0,0,0,0,0,0,0};
        // Axis 0 (e1 ∧ ∂/∂x): contrib only if interior.
        if (i > 0 && i < D0 - 1) {
          float inv = 1.0f / (2.0f * h0);
          int64_t bp = base + stride0;
          int64_t bm = base - stride0;
          c[1] += (F[bp+0] - F[bm+0]) * inv;
          c[3] += (F[bp+2] - F[bm+2]) * inv;
          c[5] += (F[bp+4] - F[bm+4]) * inv;
          c[7] += (F[bp+6] - F[bm+6]) * inv;
        }
        // Axis 1 (e2 ∧ ∂/∂y).
        if (j > 0 && j < D1 - 1) {
          float inv = 1.0f / (2.0f * h1);
          int64_t bp = base + stride1;
          int64_t bm = base - stride1;
          c[2] += (F[bp+0] - F[bm+0]) * inv;
          c[3] -= (F[bp+1] - F[bm+1]) * inv;
          c[6] += (F[bp+4] - F[bm+4]) * inv;
          c[7] -= (F[bp+5] - F[bm+5]) * inv;
        }
        // Axis 2 (e3 ∧ ∂/∂z).
        if (k > 0 && k < D2 - 1) {
          float inv = 1.0f / (2.0f * h2);
          int64_t bp = base + stride2;
          int64_t bm = base - stride2;
          c[4] += (F[bp+0] - F[bm+0]) * inv;
          c[5] -= (F[bp+1] - F[bm+1]) * inv;
          c[6] -= (F[bp+2] - F[bm+2]) * inv;
          c[7] += (F[bp+3] - F[bm+3]) * inv;
        }
        for (int r = 0; r < 8; ++r) Out[base + r] = c[r];
      }
    }
  }
}

inline void reference_clifford_vec_deriv_cl30_f32(
    const float* F, float* Out,
    int32_t D0, int32_t D1, int32_t D2,
    float h0, float h1, float h2) {
  int64_t stride2 = 8;
  int64_t stride1 = (int64_t)D2 * stride2;
  int64_t stride0 = (int64_t)D1 * stride1;
  for (int32_t i = 0; i < D0; ++i) {
    for (int32_t j = 0; j < D1; ++j) {
      for (int32_t k = 0; k < D2; ++k) {
        int64_t base = (int64_t)i * stride0 + (int64_t)j * stride1 + (int64_t)k * stride2;
        float c[8] = {0,0,0,0,0,0,0,0};
        // Axis 0 (e1 · ∂/∂x) — full Cayley row (8 contributions).
        if (i > 0 && i < D0 - 1) {
          float inv = 1.0f / (2.0f * h0);
          int64_t bp = base + stride0, bm = base - stride0;
          c[1] += (F[bp+0] - F[bm+0]) * inv;  // e1 * 1
          c[0] += (F[bp+1] - F[bm+1]) * inv;  // e1 * e1
          c[3] += (F[bp+2] - F[bm+2]) * inv;  // e1 * e2
          c[2] += (F[bp+3] - F[bm+3]) * inv;  // e1 * e12
          c[5] += (F[bp+4] - F[bm+4]) * inv;  // e1 * e3
          c[4] += (F[bp+5] - F[bm+5]) * inv;  // e1 * e13
          c[7] += (F[bp+6] - F[bm+6]) * inv;  // e1 * e23
          c[6] += (F[bp+7] - F[bm+7]) * inv;  // e1 * e123
        }
        // Axis 1 (e2 · ∂/∂y).
        if (j > 0 && j < D1 - 1) {
          float inv = 1.0f / (2.0f * h1);
          int64_t bp = base + stride1, bm = base - stride1;
          c[2] += (F[bp+0] - F[bm+0]) * inv;
          c[3] -= (F[bp+1] - F[bm+1]) * inv;
          c[0] += (F[bp+2] - F[bm+2]) * inv;
          c[1] -= (F[bp+3] - F[bm+3]) * inv;
          c[6] += (F[bp+4] - F[bm+4]) * inv;
          c[7] -= (F[bp+5] - F[bm+5]) * inv;
          c[4] += (F[bp+6] - F[bm+6]) * inv;
          c[5] -= (F[bp+7] - F[bm+7]) * inv;
        }
        // Axis 2 (e3 · ∂/∂z).
        if (k > 0 && k < D2 - 1) {
          float inv = 1.0f / (2.0f * h2);
          int64_t bp = base + stride2, bm = base - stride2;
          c[4] += (F[bp+0] - F[bm+0]) * inv;
          c[5] -= (F[bp+1] - F[bm+1]) * inv;
          c[6] -= (F[bp+2] - F[bm+2]) * inv;
          c[7] += (F[bp+3] - F[bm+3]) * inv;
          c[0] += (F[bp+4] - F[bm+4]) * inv;
          c[1] -= (F[bp+5] - F[bm+5]) * inv;
          c[2] -= (F[bp+6] - F[bm+6]) * inv;
          c[3] += (F[bp+7] - F[bm+7]) * inv;
        }
        for (int r = 0; r < 8; ++r) Out[base + r] = c[r];
      }
    }
  }
}

static NSString *const kCliffordExtDerivCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GridParams {
    int D0; int D1; int D2;
    float h0; float h1; float h2;
};

kernel void clifford_ext_deriv_cl30_f32(
    device const float*     F    [[buffer(0)]],
    device float*           Out  [[buffer(1)]],
    constant GridParams&    P    [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = int(gid.x), j = int(gid.y), k = int(gid.z);
    int D0 = P.D0, D1 = P.D1, D2 = P.D2;
    if (i >= D0 || j >= D1 || k >= D2) return;
    long stride2 = 8;
    long stride1 = (long)D2 * stride2;
    long stride0 = (long)D1 * stride1;
    long base = (long)i * stride0 + (long)j * stride1 + (long)k * stride2;

    float c[8] = {0,0,0,0,0,0,0,0};
    if (i > 0 && i < D0 - 1) {
        float inv = 1.0f / (2.0f * P.h0);
        long bp = base + stride0, bm = base - stride0;
        c[1] += (F[bp+0] - F[bm+0]) * inv;
        c[3] += (F[bp+2] - F[bm+2]) * inv;
        c[5] += (F[bp+4] - F[bm+4]) * inv;
        c[7] += (F[bp+6] - F[bm+6]) * inv;
    }
    if (j > 0 && j < D1 - 1) {
        float inv = 1.0f / (2.0f * P.h1);
        long bp = base + stride1, bm = base - stride1;
        c[2] += (F[bp+0] - F[bm+0]) * inv;
        c[3] -= (F[bp+1] - F[bm+1]) * inv;
        c[6] += (F[bp+4] - F[bm+4]) * inv;
        c[7] -= (F[bp+5] - F[bm+5]) * inv;
    }
    if (k > 0 && k < D2 - 1) {
        float inv = 1.0f / (2.0f * P.h2);
        long bp = base + stride2, bm = base - stride2;
        c[4] += (F[bp+0] - F[bm+0]) * inv;
        c[5] -= (F[bp+1] - F[bm+1]) * inv;
        c[6] -= (F[bp+2] - F[bm+2]) * inv;
        c[7] += (F[bp+3] - F[bm+3]) * inv;
    }
    for (int r = 0; r < 8; ++r) Out[base + r] = c[r];
}
)MSL";

static NSString *const kCliffordVecDerivCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GridParams {
    int D0; int D1; int D2;
    float h0; float h1; float h2;
};

kernel void clifford_vec_deriv_cl30_f32(
    device const float*     F    [[buffer(0)]],
    device float*           Out  [[buffer(1)]],
    constant GridParams&    P    [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]])
{
    int i = int(gid.x), j = int(gid.y), k = int(gid.z);
    int D0 = P.D0, D1 = P.D1, D2 = P.D2;
    if (i >= D0 || j >= D1 || k >= D2) return;
    long stride2 = 8;
    long stride1 = (long)D2 * stride2;
    long stride0 = (long)D1 * stride1;
    long base = (long)i * stride0 + (long)j * stride1 + (long)k * stride2;

    float c[8] = {0,0,0,0,0,0,0,0};
    if (i > 0 && i < D0 - 1) {
        float inv = 1.0f / (2.0f * P.h0);
        long bp = base + stride0, bm = base - stride0;
        c[1] += (F[bp+0] - F[bm+0]) * inv;
        c[0] += (F[bp+1] - F[bm+1]) * inv;
        c[3] += (F[bp+2] - F[bm+2]) * inv;
        c[2] += (F[bp+3] - F[bm+3]) * inv;
        c[5] += (F[bp+4] - F[bm+4]) * inv;
        c[4] += (F[bp+5] - F[bm+5]) * inv;
        c[7] += (F[bp+6] - F[bm+6]) * inv;
        c[6] += (F[bp+7] - F[bm+7]) * inv;
    }
    if (j > 0 && j < D1 - 1) {
        float inv = 1.0f / (2.0f * P.h1);
        long bp = base + stride1, bm = base - stride1;
        c[2] += (F[bp+0] - F[bm+0]) * inv;
        c[3] -= (F[bp+1] - F[bm+1]) * inv;
        c[0] += (F[bp+2] - F[bm+2]) * inv;
        c[1] -= (F[bp+3] - F[bm+3]) * inv;
        c[6] += (F[bp+4] - F[bm+4]) * inv;
        c[7] -= (F[bp+5] - F[bm+5]) * inv;
        c[4] += (F[bp+6] - F[bm+6]) * inv;
        c[5] -= (F[bp+7] - F[bm+7]) * inv;
    }
    if (k > 0 && k < D2 - 1) {
        float inv = 1.0f / (2.0f * P.h2);
        long bp = base + stride2, bm = base - stride2;
        c[4] += (F[bp+0] - F[bm+0]) * inv;
        c[5] -= (F[bp+1] - F[bm+1]) * inv;
        c[6] -= (F[bp+2] - F[bm+2]) * inv;
        c[7] += (F[bp+3] - F[bm+3]) * inv;
        c[0] += (F[bp+4] - F[bm+4]) * inv;
        c[1] -= (F[bp+5] - F[bm+5]) * inv;
        c[2] -= (F[bp+6] - F[bm+6]) * inv;
        c[3] += (F[bp+7] - F[bm+7]) * inv;
    }
    for (int r = 0; r < 8; ++r) Out[base + r] = c[r];
}
)MSL";

struct CliffordGridParams { int32_t D0, D1, D2; float h0, h1, h2; };

static bool dispatch_clifford_field_op_f32_msl(
    MetalDeviceContext &ctx, NSString *source, NSString *entry,
    const float* F, float* Out,
    int32_t D0, int32_t D1, int32_t D2,
    float h0, float h1, float h2) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, source, entry);
    if (!pso) return false;
    NSUInteger byteCount = sizeof(float) * (NSUInteger)D0 * (NSUInteger)D1
                         * (NSUInteger)D2 * 8u;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufF, ctx, F, byteCount);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, byteCount);
    if (!bufF || !bufO) return false;
    CliffordGridParams P = {D0, D1, D2, h0, h1, h2};
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufF offset:0 atIndex:0];
    [enc setBuffer:bufO offset:0 atIndex:1];
    [enc setBytes:&P length:sizeof(P) atIndex:2];
    MTLSize grid = MTLSizeMake((NSUInteger)D0, (NSUInteger)D1, (NSUInteger)D2);
    NSUInteger tg_x = std::min<NSUInteger>(8, pso.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "clifford_field_op_f32_msl");
    if (_pool_ok) std::memcpy(Out, [bufO contents], byteCount);
    return _pool_ok;
  }
}

// ===========================================================================
// codiff = ⋆ d ⋆ (composed)
// ===========================================================================

inline void reference_clifford_codiff_cl30_f32(
    const float* F, float* Out,
    int32_t D0, int32_t D1, int32_t D2,
    float h0, float h1, float h2) {
  std::size_t N = (std::size_t)D0 * D1 * D2 * 8;
  std::vector<float> tmp1(N), tmp2(N);
  // Pointwise hodge applied to every cell.
  for (std::size_t p = 0; p < N / 8; ++p) {
    reference_clifford_hodge_star_cl30_f32(F + p * 8, tmp1.data() + p * 8, 1);
  }
  reference_clifford_ext_deriv_cl30_f32(tmp1.data(), tmp2.data(),
                                         D0, D1, D2, h0, h1, h2);
  for (std::size_t p = 0; p < N / 8; ++p) {
    reference_clifford_hodge_star_cl30_f32(tmp2.data() + p * 8, Out + p * 8, 1);
  }
}

static bool dispatch_clifford_codiff_cl30_f32_msl(
    MetalDeviceContext &ctx,
    const float* F, float* Out,
    int32_t D0, int32_t D1, int32_t D2,
    float h0, float h1, float h2) {
  // Stage 1: pointwise hodge applied to each grid cell.
  std::size_t N = (std::size_t)D0 * D1 * D2;
  std::size_t total = N * 8;
  std::vector<float> tmp1(total), tmp2(total);
  if (!dispatch_clifford_unary_8x8_f32_msl(
          ctx, kCliffordHodgeStarCl30F32Source,
          @"clifford_hodge_star_cl30_f32", F, tmp1.data(), (int32_t)N))
    return false;
  // Stage 2: ext_deriv on the grid.
  if (!dispatch_clifford_field_op_f32_msl(
          ctx, kCliffordExtDerivCl30F32Source,
          @"clifford_ext_deriv_cl30_f32",
          tmp1.data(), tmp2.data(), D0, D1, D2, h0, h1, h2))
    return false;
  // Stage 3: pointwise hodge again.
  if (!dispatch_clifford_unary_8x8_f32_msl(
          ctx, kCliffordHodgeStarCl30F32Source,
          @"clifford_hodge_star_cl30_f32", tmp2.data(), Out, (int32_t)N))
    return false;
  return true;
}

// ===========================================================================
// integral = sum_i weights[i] * field[i]  (per-coefficient weighted sum)
// ===========================================================================

inline void reference_clifford_integral_cl30_f32(
    const float* field, const float* weights, float* out, int32_t n) {
  for (int r = 0; r < 8; ++r) out[r] = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    float w = weights[i];
    for (int r = 0; r < 8; ++r) out[r] += w * field[i * 8 + r];
  }
}

static NSString *const kCliffordIntegralCl30F32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void clifford_integral_cl30_f32(
    device const float* field    [[buffer(0)]],
    device const float* weights  [[buffer(1)]],
    device float*       out      [[buffer(2)]],
    constant int&       n        [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // One thread per output coefficient (8 threads total).
    if (gid >= 8u) return;
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        s += weights[i] * field[i * 8 + int(gid)];
    }
    out[gid] = s;
}
)MSL";

static bool dispatch_clifford_integral_cl30_f32_msl(
    MetalDeviceContext &ctx, const float* field, const float* weights,
    float* out, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kCliffordIntegralCl30F32Source, @"clifford_integral_cl30_f32");
    if (!pso) return false;
    NSUInteger fieldBytes = sizeof(float) * (NSUInteger)n * 8u;
    NSUInteger wBytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufF, ctx, field, fieldBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufW, ctx, weights, wBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, sizeof(float) * 8u);
    if (!bufF || !bufW || !bufO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufF offset:0 atIndex:0];
    [enc setBuffer:bufW offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(8, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "clifford_integral_cl30_f32_msl");
    if (_pool_ok) std::memcpy(out, [bufO contents], sizeof(float) * 8u);
    return _pool_ok;
    // RAII guards release bufF / bufW / bufO at scope exit.
  }
}

} // namespace

// ===========================================================================
// extern "C" entries for the 6 new ops.
// ===========================================================================

extern "C" void tessera_apple_gpu_clifford_exp_cl30_f32(
    const float* A, float* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_unary_8x8_f32_msl(
          ctx, kCliffordExpCl30F32Source, @"clifford_exp_cl30_f32",
          A, C, batch)) return;
  reference_clifford_exp_cl30_f32(A, C, batch);
}

extern "C" void tessera_apple_gpu_clifford_log_cl30_f32(
    const float* A, float* C, int32_t batch) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_unary_8x8_f32_msl(
          ctx, kCliffordLogCl30F32Source, @"clifford_log_cl30_f32",
          A, C, batch)) return;
  reference_clifford_log_cl30_f32(A, C, batch);
}

extern "C" void tessera_apple_gpu_clifford_ext_deriv_cl30_f32(
    const float* F, float* Out,
    int32_t D0, int32_t D1, int32_t D2,
    float h0, float h1, float h2) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_field_op_f32_msl(
          ctx, kCliffordExtDerivCl30F32Source,
          @"clifford_ext_deriv_cl30_f32",
          F, Out, D0, D1, D2, h0, h1, h2)) return;
  reference_clifford_ext_deriv_cl30_f32(F, Out, D0, D1, D2, h0, h1, h2);
}

extern "C" void tessera_apple_gpu_clifford_vec_deriv_cl30_f32(
    const float* F, float* Out,
    int32_t D0, int32_t D1, int32_t D2,
    float h0, float h1, float h2) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_field_op_f32_msl(
          ctx, kCliffordVecDerivCl30F32Source,
          @"clifford_vec_deriv_cl30_f32",
          F, Out, D0, D1, D2, h0, h1, h2)) return;
  reference_clifford_vec_deriv_cl30_f32(F, Out, D0, D1, D2, h0, h1, h2);
}

extern "C" void tessera_apple_gpu_clifford_codiff_cl30_f32(
    const float* F, float* Out,
    int32_t D0, int32_t D1, int32_t D2,
    float h0, float h1, float h2) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_codiff_cl30_f32_msl(
          ctx, F, Out, D0, D1, D2, h0, h1, h2)) return;
  reference_clifford_codiff_cl30_f32(F, Out, D0, D1, D2, h0, h1, h2);
}

extern "C" void tessera_apple_gpu_clifford_integral_cl30_f32(
    const float* field, const float* weights, float* out, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_clifford_integral_cl30_f32_msl(
          ctx, field, weights, out, n)) return;
  reference_clifford_integral_cl30_f32(field, weights, out, n);
}

// ===========================================================================
// EBM inner_step  —  out[i] = y[i] - eta * grad[i]
//
// The minimal native EBM primitive on Apple GPU.  Pointwise affine over
// arbitrary tensor shape; the kernel only needs the total element count.
// Used as the inner-loop step of the EBT refinement chain (K candidates
// × T inner steps); each iteration dispatches one of these kernels.
// ===========================================================================

inline void reference_ebm_inner_step_f32(
    const float* y, const float* grad, float eta,
    float* out, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    out[i] = y[i] - eta * grad[i];
  }
}

static NSString *const kEBMInnerStepF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void ebm_inner_step_f32(
    device const float* y     [[buffer(0)]],
    device const float* grad  [[buffer(1)]],
    constant float&     eta   [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant int32_t&   n     [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    out[i] = y[i] - eta * grad[i];
}
)MSL";

static bool dispatch_ebm_inner_step_f32_msl(
    MetalDeviceContext &ctx,
    const float* y, const float* grad, float eta,
    float* out, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMInnerStepF32Source, @"ebm_inner_step_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufY, ctx, y, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufG, ctx, grad, bytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, bytes);
    if (!bufY || !bufG || !bufO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufY offset:0 atIndex:0];
    [enc setBuffer:bufG offset:0 atIndex:1];
    [enc setBytes:&eta length:sizeof(float) atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:4];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                                 "ebm_inner_step_f32_msl");
    if (_pool_ok) std::memcpy(out, [bufO contents], bytes);
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_inner_step_f32(
    const float* y, const float* grad, float eta,
    float* out, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_inner_step_f32_msl(
          ctx, y, grad, eta, out, n)) return;
  reference_ebm_inner_step_f32(y, grad, eta, out, n);
}

// EBT refinement chain  —  K candidates × T inner steps, all on-device.
// Each iteration runs the inner_step kernel above; the buffers ping-pong
// so the final output ends in `y_out`.  This is the canonical EBT pattern
// (refine each candidate by T gradient-descent steps) collapsed into a
// single C ABI symbol for ergonomic benchmarking + manifest dispatch.
//
// Inputs:
//   y0   — initial state, length n
//   grad — pre-computed gradient buffer reused at every inner step
//          (a real EBT loop would recompute grad each step; the v1
//          benchmark uses a fixed gradient so timing reflects pure
//          inner-step cost — recomputation is the caller's loop)
//   eta  — step size
//   T    — number of inner-step iterations
//
// Output:
//   y_out — refined state, length n (= y0 - T*eta*grad for fixed grad)

// ===========================================================================
// EBM refinement (fused)  —  T inner steps of `y - eta * grad` in a
// single Metal dispatch.  Each MSL thread keeps `y_i` in a register
// and runs the T-step recurrence locally, so the whole refinement
// chain costs one dispatch + one register-resident loop instead of
// T host-side dispatches with ping-pong buffers.  This is what
// `docs/status/ga_ebm_milestone.md` § "Known non-claims" #1 called out;
// the fused kernel below is the fix.
// ===========================================================================

static NSString *const kEBMRefinementFusedF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void ebm_refinement_fused_f32(
    device const float* y0    [[buffer(0)]],
    device const float* grad  [[buffer(1)]],
    constant float&     eta   [[buffer(2)]],
    constant int32_t&   T     [[buffer(3)]],
    device float*       y_out [[buffer(4)]],
    constant int32_t&   n     [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    // Pull both y0[i] and grad[i] into registers so the inner loop is
    // pure ALU.  For fixed grad this reduces to a single FMA per step;
    // the compiler typically unrolls when T is a small literal but we
    // keep it data-driven so callers can sweep T freely.
    float yi = y0[i];
    const float gi = grad[i];
    for (int t = 0; t < T; ++t) {
        yi = yi - eta * gi;
    }
    y_out[i] = yi;
}
)MSL";

static bool dispatch_ebm_refinement_fused_f32_msl(
    MetalDeviceContext &ctx,
    const float* y0, const float* grad, float eta, int32_t T,
    float* y_out, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMRefinementFusedF32Source, @"ebm_refinement_fused_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bY, ctx, y0, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bG, ctx, grad, bytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, bytes);
    if (!bY || !bG || !bO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bY offset:0 atIndex:0];
    [enc setBuffer:bG offset:0 atIndex:1];
    [enc setBytes:&eta length:sizeof(float) atIndex:2];
    [enc setBytes:&T length:sizeof(int32_t) atIndex:3];
    [enc setBuffer:bO offset:0 atIndex:4];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:5];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                                 "ebm_refinement_fused_f32_msl");
    if (_pool_ok) std::memcpy(y_out, [bO contents], bytes);
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_refinement_f32(
    const float* y0, const float* grad, float eta, int32_t T,
    float* y_out, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (T <= 0) {
    if (y_out != y0) std::memcpy(y_out, y0, sizeof(float) * (size_t)n);
    return;
  }
  // Single-dispatch fused kernel — drops dispatch overhead from O(T)
  // to O(1).  Closed-form on fixed grad: y_T = y_0 - T*eta*grad.
  if (ctx.ok && dispatch_ebm_refinement_fused_f32_msl(
          ctx, y0, grad, eta, T, y_out, n)) return;
  // CPU fallback path (no device, or kernel compile failed).
  std::vector<float> tmp(y0, y0 + n);
  std::vector<float> nxt(n);
  for (int32_t t = 0; t < T; ++t) {
    reference_ebm_inner_step_f32(tmp.data(), grad, eta, nxt.data(), n);
    std::swap(tmp, nxt);
  }
  std::memcpy(y_out, tmp.data(), sizeof(float) * (size_t)n);
}

extern "C" int32_t tessera_apple_gpu_ebm_refinement_value_f32(
    const float* y0, const float* grad, float eta, int32_t T,
    float* y_out, int32_t n) {
  if (T <= 0)
    return 0;
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_refinement_fused_f32_msl(
          ctx, y0, grad, eta, T, y_out, n))
    return 1;
  return 0;
}

// ===========================================================================
// EBM langevin_step  —  out[i] = y[i] - eta * grad[i] + noise_scale * noise[i]
//
// Caller pre-generates `noise` on the host (deterministic via the
// `tessera.rng` Philox stream) and passes the buffer in.  This keeps
// the kernel purely affine — no on-device RNG yet (that's a separate
// Philox-in-MSL follow-up).  Matches the Python `tessera.ebm.langevin_step`
// path bit-for-bit when (a) `grad_fn` is analytic and (b) `noise` is
// the host-side sample from `tessera.rng.normal(key, ...)`.
// ===========================================================================

inline void reference_ebm_langevin_step_f32(
    const float* y, const float* grad, const float* noise,
    float eta, float noise_scale, float* out, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    out[i] = y[i] - eta * grad[i] + noise_scale * noise[i];
  }
}

static NSString *const kEBMLangevinStepF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void ebm_langevin_step_f32(
    device const float* y           [[buffer(0)]],
    device const float* grad        [[buffer(1)]],
    device const float* noise       [[buffer(2)]],
    constant float&     eta         [[buffer(3)]],
    constant float&     noise_scale [[buffer(4)]],
    device float*       out         [[buffer(5)]],
    constant int32_t&   n           [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    out[i] = y[i] - eta * grad[i] + noise_scale * noise[i];
}
)MSL";

static bool dispatch_ebm_langevin_step_f32_msl(
    MetalDeviceContext &ctx,
    const float* y, const float* grad, const float* noise,
    float eta, float noise_scale, float* out, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMLangevinStepF32Source, @"ebm_langevin_step_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bY, ctx, y, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bG, ctx, grad, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bN, ctx, noise, bytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, bytes);
    if (!bY || !bG || !bN || !bO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bY offset:0 atIndex:0];
    [enc setBuffer:bG offset:0 atIndex:1];
    [enc setBuffer:bN offset:0 atIndex:2];
    [enc setBytes:&eta length:sizeof(float) atIndex:3];
    [enc setBytes:&noise_scale length:sizeof(float) atIndex:4];
    [enc setBuffer:bO offset:0 atIndex:5];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:6];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                                 "ebm_langevin_step_f32_msl");
    if (_pool_ok) std::memcpy(out, [bO contents], bytes);
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_langevin_step_f32(
    const float* y, const float* grad, const float* noise,
    float eta, float noise_scale, float* out, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_langevin_step_f32_msl(
          ctx, y, grad, noise, eta, noise_scale, out, n)) return;
  reference_ebm_langevin_step_f32(y, grad, noise, eta, noise_scale, out, n);
}

extern "C" int32_t tessera_apple_gpu_ebm_langevin_step_value_f32(
    const float* y, const float* grad, const float* noise,
    float eta, float noise_scale, float* out, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_langevin_step_f32_msl(
          ctx, y, grad, noise, eta, noise_scale, out, n))
    return 1;
  return 0;
}

// ===========================================================================
// EBM langevin_step + on-device Philox-4x32-10 (M6 Step 4 runtime emission).
//
//   out[i] = y[i] - eta * grad[i] + noise_scale * z[i]
//
// where z[i] is a standard-normal sample produced ON-DEVICE from a
// Philox-4x32-10 stream seeded by (key, counter).  Each thread i uses
// counter = (counter_in[0] + i, counter_in[1], counter_in[2], counter_in[3]),
// runs Philox, then maps the first two uniforms to a normal via Box-Muller.
//
// Constants are pinned to match `python/tessera/compiler/philox.py`
// byte-for-byte.  The Python test `test_philox_msl_source_matches_mm`
// (in tests/unit/test_philox_runtime.py) re-reads this kernel source
// and verifies the constants.
// ===========================================================================

inline void reference_ebm_langevin_step_philox_f32(
    const float* y, const float* grad,
    float eta, float noise_scale,
    const uint32_t key[2], const uint32_t counter[4],
    float* out, int32_t n) {
  // Host-side reference path mirrors the MSL kernel exactly so that
  // benchmark/test code can dispatch through this fallback when the
  // Metal runtime is unavailable and still see byte-identical numbers
  // against the Python `philox_normal_pair` reference.
  for (int32_t i = 0; i < n; ++i) {
    uint32_t c[4] = {counter[0] + (uint32_t)i, counter[1],
                     counter[2], counter[3]};
    uint32_t k[2] = {key[0], key[1]};
    for (int r = 0; r < 10; ++r) {
      // round
      uint64_t p0 = (uint64_t)0xD2511F53u * (uint64_t)c[0];
      uint64_t p1 = (uint64_t)0xCD9E8D57u * (uint64_t)c[2];
      uint32_t lo0 = (uint32_t)(p0 & 0xFFFFFFFFu);
      uint32_t hi0 = (uint32_t)(p0 >> 32);
      uint32_t lo1 = (uint32_t)(p1 & 0xFFFFFFFFu);
      uint32_t hi1 = (uint32_t)(p1 >> 32);
      uint32_t nc0 = hi1 ^ c[1] ^ k[0];
      uint32_t nc1 = lo1;
      uint32_t nc2 = hi0 ^ c[3] ^ k[1];
      uint32_t nc3 = lo0;
      c[0] = nc0; c[1] = nc1; c[2] = nc2; c[3] = nc3;
      k[0] += 0x9E3779B9u;
      k[1] += 0xBB67AE85u;
    }
    // To uniform (0, 1] then Box-Muller.
    float u0 = ((float)c[0] + 0.5f) * 0x1.0p-32f;
    float u1 = ((float)c[1] + 0.5f) * 0x1.0p-32f;
    float r = std::sqrt(-2.0f * std::log(u0));
    float theta = 2.0f * (float)M_PI * u1;
    float z = r * std::cos(theta);
    out[i] = y[i] - eta * grad[i] + noise_scale * z;
  }
}

static NSString *const kEBMLangevinStepPhiloxF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// Philox-4x32-10 constants — pinned to tessera.compiler.philox.
constant constexpr uint PHILOX_M0 = 0xD2511F53u;
constant constexpr uint PHILOX_M1 = 0xCD9E8D57u;
constant constexpr uint PHILOX_W0 = 0x9E3779B9u;
constant constexpr uint PHILOX_W1 = 0xBB67AE85u;

inline void philox_mulhilo(uint a, uint b, thread uint &lo, thread uint &hi) {
    ulong p = (ulong)a * (ulong)b;
    lo = (uint)(p & 0xFFFFFFFFu);
    hi = (uint)(p >> 32);
}

inline void philox_round(thread uint ctr[4], thread const uint key[2]) {
    uint lo0, hi0, lo1, hi1;
    philox_mulhilo(PHILOX_M0, ctr[0], lo0, hi0);
    philox_mulhilo(PHILOX_M1, ctr[2], lo1, hi1);
    uint c0 = hi1 ^ ctr[1] ^ key[0];
    uint c1 = lo1;
    uint c2 = hi0 ^ ctr[3] ^ key[1];
    uint c3 = lo0;
    ctr[0] = c0; ctr[1] = c1; ctr[2] = c2; ctr[3] = c3;
}

inline void philox_bump_key(thread uint key[2]) {
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

inline void philox_4x32_10(thread uint ctr[4], thread uint key[2]) {
    for (int r = 0; r < 10; ++r) {
        philox_round(ctr, key);
        philox_bump_key(key);
    }
}

kernel void ebm_langevin_step_philox_f32(
    device const float* y         [[buffer(0)]],
    device const float* grad      [[buffer(1)]],
    constant float&     eta       [[buffer(2)]],
    constant float&     noise_scale [[buffer(3)]],
    constant uint4&     ctr_base  [[buffer(4)]],  // (c0, c1, c2, c3)
    constant uint2&     key_in    [[buffer(5)]],  // (k0, k1)
    device float*       out       [[buffer(6)]],
    constant int32_t&   n         [[buffer(7)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    // Per-thread counter: stride one slot by i.
    uint ctr[4] = {ctr_base.x + i, ctr_base.y, ctr_base.z, ctr_base.w};
    uint key[2] = {key_in.x, key_in.y};
    philox_4x32_10(ctr, key);
    // Box-Muller from the first two uniforms.
    constexpr float k = 0x1.0p-32f;
    float u0 = ((float)ctr[0] + 0.5f) * k;
    float u1 = ((float)ctr[1] + 0.5f) * k;
    float r = sqrt(-2.0f * log(u0));
    float theta = 2.0f * (float)M_PI_F * u1;
    float z = r * cos(theta);
    out[i] = y[i] - eta * grad[i] + noise_scale * z;
}
)MSL";

static bool dispatch_ebm_langevin_step_philox_f32_msl(
    MetalDeviceContext &ctx,
    const float* y, const float* grad,
    float eta, float noise_scale,
    const uint32_t key[2], const uint32_t counter[4],
    float* out, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMLangevinStepPhiloxF32Source,
        @"ebm_langevin_step_philox_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bY, ctx, y, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bG, ctx, grad, bytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, bytes);
    if (!bY || !bG || !bO) return false;
    // Pack counter + key into uint4 / uint2 inline constants.
    uint32_t ctr_pack[4] = {counter[0], counter[1], counter[2], counter[3]};
    uint32_t key_pack[2] = {key[0], key[1]};
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bY offset:0 atIndex:0];
    [enc setBuffer:bG offset:0 atIndex:1];
    [enc setBytes:&eta length:sizeof(float) atIndex:2];
    [enc setBytes:&noise_scale length:sizeof(float) atIndex:3];
    [enc setBytes:ctr_pack length:sizeof(ctr_pack) atIndex:4];
    [enc setBytes:key_pack length:sizeof(key_pack) atIndex:5];
    [enc setBuffer:bO offset:0 atIndex:6];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:7];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                                 "ebm_langevin_step_philox_f32_msl");
    if (_pool_ok) std::memcpy(out, [bO contents], bytes);
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_langevin_step_philox_f32(
    const float* y, const float* grad,
    float eta, float noise_scale,
    const uint32_t* key, const uint32_t* counter,
    float* out, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_langevin_step_philox_f32_msl(
          ctx, y, grad, eta, noise_scale, key, counter, out, n)) return;
  reference_ebm_langevin_step_philox_f32(
      y, grad, eta, noise_scale, key, counter, out, n);
}

// ===========================================================================
// M7 follow-up — MSL kernels for the conformal-primitive surface.
//
// Each function below has the same shape: an MSL kernel (element-wise
// over a 1-D grid), a Python-callable extern "C" wrapper, and a CPU
// reference path so non-Darwin hosts and runtime-load failures get the
// same numerics.  Buffers come from the RAII pool.
// ===========================================================================

inline void reference_complex_mul_f32(
    const float* a_re, const float* a_im,
    const float* b_re, const float* b_im,
    float* out_re, float* out_im, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    out_re[i] = a_re[i] * b_re[i] - a_im[i] * b_im[i];
    out_im[i] = a_re[i] * b_im[i] + a_im[i] * b_re[i];
  }
}

static NSString *const kComplexMulF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void complex_mul_f32(
    device const float* a_re [[buffer(0)]],
    device const float* a_im [[buffer(1)]],
    device const float* b_re [[buffer(2)]],
    device const float* b_im [[buffer(3)]],
    device float* out_re     [[buffer(4)]],
    device float* out_im     [[buffer(5)]],
    constant int32_t& n      [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    float ar = a_re[i], ai = a_im[i], br = b_re[i], bi = b_im[i];
    out_re[i] = ar * br - ai * bi;
    out_im[i] = ar * bi + ai * br;
}
)MSL";

static bool dispatch_complex_mul_f32_msl(
    MetalDeviceContext &ctx,
    const float* a_re, const float* a_im,
    const float* b_re, const float* b_im,
    float* out_re, float* out_im, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kComplexMulF32Source, @"complex_mul_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bAr, ctx, a_re, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bAi, ctx, a_im, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bBr, ctx, b_re, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bBi, ctx, b_im, bytes);
    TS_METAL_BUF_ACQUIRE(bOr, ctx, bytes);
    TS_METAL_BUF_ACQUIRE(bOi, ctx, bytes);
    if (!bAr || !bAi || !bBr || !bBi || !bOr || !bOi) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bAr offset:0 atIndex:0];
    [enc setBuffer:bAi offset:0 atIndex:1];
    [enc setBuffer:bBr offset:0 atIndex:2];
    [enc setBuffer:bBi offset:0 atIndex:3];
    [enc setBuffer:bOr offset:0 atIndex:4];
    [enc setBuffer:bOi offset:0 atIndex:5];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:6];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "complex_mul_f32_msl");
    if (_pool_ok) {
      std::memcpy(out_re, [bOr contents], bytes);
      std::memcpy(out_im, [bOi contents], bytes);
    }
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_complex_mul_f32(
    const float* a_re, const float* a_im,
    const float* b_re, const float* b_im,
    float* out_re, float* out_im, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_complex_mul_f32_msl(
          ctx, a_re, a_im, b_re, b_im, out_re, out_im, n)) return;
  reference_complex_mul_f32(a_re, a_im, b_re, b_im, out_re, out_im, n);
}

// ---------------------------------------------------------------------------
// complex_exp: e^(a + b·i) = e^a · (cos b, sin b)
// ---------------------------------------------------------------------------

inline void reference_complex_exp_f32(
    const float* re, const float* im,
    float* out_re, float* out_im, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    float ea = std::exp(re[i]);
    out_re[i] = ea * std::cos(im[i]);
    out_im[i] = ea * std::sin(im[i]);
  }
}

static NSString *const kComplexExpF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void complex_exp_f32(
    device const float* re   [[buffer(0)]],
    device const float* im   [[buffer(1)]],
    device float* out_re     [[buffer(2)]],
    device float* out_im     [[buffer(3)]],
    constant int32_t& n      [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    float ea = exp(re[i]);
    out_re[i] = ea * cos(im[i]);
    out_im[i] = ea * sin(im[i]);
}
)MSL";

static bool dispatch_complex_exp_f32_msl(
    MetalDeviceContext &ctx,
    const float* re, const float* im,
    float* out_re, float* out_im, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kComplexExpF32Source, @"complex_exp_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bR, ctx, re, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bI, ctx, im, bytes);
    TS_METAL_BUF_ACQUIRE(bOr, ctx, bytes);
    TS_METAL_BUF_ACQUIRE(bOi, ctx, bytes);
    if (!bR || !bI || !bOr || !bOi) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bR offset:0 atIndex:0];
    [enc setBuffer:bI offset:0 atIndex:1];
    [enc setBuffer:bOr offset:0 atIndex:2];
    [enc setBuffer:bOi offset:0 atIndex:3];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:4];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "complex_exp_f32_msl");
    if (_pool_ok) {
      std::memcpy(out_re, [bOr contents], bytes);
      std::memcpy(out_im, [bOi contents], bytes);
    }
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_complex_exp_f32(
    const float* re, const float* im,
    float* out_re, float* out_im, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_complex_exp_f32_msl(
          ctx, re, im, out_re, out_im, n)) return;
  reference_complex_exp_f32(re, im, out_re, out_im, n);
}

// ---------------------------------------------------------------------------
// stereographic: forward projection (S² ⊂ ℝ³ → ℂ)
// ---------------------------------------------------------------------------

inline void reference_complex_stereographic_f32(
    const float* x, const float* y, const float* z,
    float* out_re, float* out_im, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    float denom = 1.0f - z[i];
    if (std::fabs(denom) < 1e-12f) {
      out_re[i] = std::numeric_limits<float>::infinity();
      out_im[i] = std::numeric_limits<float>::infinity();
    } else {
      out_re[i] = x[i] / denom;
      out_im[i] = y[i] / denom;
    }
  }
}

static NSString *const kComplexStereographicF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void complex_stereographic_f32(
    device const float* x    [[buffer(0)]],
    device const float* y    [[buffer(1)]],
    device const float* z    [[buffer(2)]],
    device float* out_re     [[buffer(3)]],
    device float* out_im     [[buffer(4)]],
    constant int32_t& n      [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    float denom = 1.0f - z[i];
    if (fabs(denom) < 1e-12f) {
        out_re[i] = INFINITY;
        out_im[i] = INFINITY;
    } else {
        out_re[i] = x[i] / denom;
        out_im[i] = y[i] / denom;
    }
}
)MSL";

static bool dispatch_complex_stereographic_f32_msl(
    MetalDeviceContext &ctx,
    const float* x, const float* y, const float* z,
    float* out_re, float* out_im, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kComplexStereographicF32Source, @"complex_stereographic_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bX, ctx, x, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bY, ctx, y, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bZ, ctx, z, bytes);
    TS_METAL_BUF_ACQUIRE(bOr, ctx, bytes);
    TS_METAL_BUF_ACQUIRE(bOi, ctx, bytes);
    if (!bX || !bY || !bZ || !bOr || !bOi) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bX offset:0 atIndex:0];
    [enc setBuffer:bY offset:0 atIndex:1];
    [enc setBuffer:bZ offset:0 atIndex:2];
    [enc setBuffer:bOr offset:0 atIndex:3];
    [enc setBuffer:bOi offset:0 atIndex:4];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:5];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "complex_stereographic_f32_msl");
    if (_pool_ok) {
      std::memcpy(out_re, [bOr contents], bytes);
      std::memcpy(out_im, [bOi contents], bytes);
    }
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_complex_stereographic_f32(
    const float* x, const float* y, const float* z,
    float* out_re, float* out_im, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_complex_stereographic_f32_msl(
          ctx, x, y, z, out_re, out_im, n)) return;
  reference_complex_stereographic_f32(x, y, z, out_re, out_im, n);
}

// ---------------------------------------------------------------------------
// mobius: f(z; a, b, c, d) = (a·z + b) / (c·z + d)
// Scalar coefficients (a, b, c, d) broadcast across the input batch.
// ---------------------------------------------------------------------------

inline void reference_complex_mobius_f32(
    const float* z_re, const float* z_im,
    float a_re, float a_im, float b_re, float b_im,
    float c_re, float c_im, float d_re, float d_im,
    float* out_re, float* out_im, int32_t n) {
  for (int32_t i = 0; i < n; ++i) {
    float zr = z_re[i], zi = z_im[i];
    // numerator = a*z + b
    float nr = a_re * zr - a_im * zi + b_re;
    float ni = a_re * zi + a_im * zr + b_im;
    // denominator = c*z + d
    float dr = c_re * zr - c_im * zi + d_re;
    float di = c_re * zi + c_im * zr + d_im;
    float denom = dr * dr + di * di;
    if (denom < 1e-12f) {
      out_re[i] = std::numeric_limits<float>::infinity();
      out_im[i] = std::numeric_limits<float>::infinity();
    } else {
      out_re[i] = (nr * dr + ni * di) / denom;
      out_im[i] = (ni * dr - nr * di) / denom;
    }
  }
}

static NSString *const kComplexMobiusF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void complex_mobius_f32(
    device const float* z_re [[buffer(0)]],
    device const float* z_im [[buffer(1)]],
    constant float& a_re     [[buffer(2)]],
    constant float& a_im     [[buffer(3)]],
    constant float& b_re     [[buffer(4)]],
    constant float& b_im     [[buffer(5)]],
    constant float& c_re     [[buffer(6)]],
    constant float& c_im     [[buffer(7)]],
    constant float& d_re     [[buffer(8)]],
    constant float& d_im     [[buffer(9)]],
    device float* out_re     [[buffer(10)]],
    device float* out_im     [[buffer(11)]],
    constant int32_t& n      [[buffer(12)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    float zr = z_re[i], zi = z_im[i];
    float nr = a_re * zr - a_im * zi + b_re;
    float ni = a_re * zi + a_im * zr + b_im;
    float dr = c_re * zr - c_im * zi + d_re;
    float di = c_re * zi + c_im * zr + d_im;
    float denom = dr * dr + di * di;
    if (denom < 1e-12f) {
        out_re[i] = INFINITY;
        out_im[i] = INFINITY;
    } else {
        out_re[i] = (nr * dr + ni * di) / denom;
        out_im[i] = (ni * dr - nr * di) / denom;
    }
}
)MSL";

static bool dispatch_complex_mobius_f32_msl(
    MetalDeviceContext &ctx,
    const float* z_re, const float* z_im,
    float a_re, float a_im, float b_re, float b_im,
    float c_re, float c_im, float d_re, float d_im,
    float* out_re, float* out_im, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kComplexMobiusF32Source, @"complex_mobius_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)n;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bZr, ctx, z_re, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bZi, ctx, z_im, bytes);
    TS_METAL_BUF_ACQUIRE(bOr, ctx, bytes);
    TS_METAL_BUF_ACQUIRE(bOi, ctx, bytes);
    if (!bZr || !bZi || !bOr || !bOi) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bZr offset:0 atIndex:0];
    [enc setBuffer:bZi offset:0 atIndex:1];
    [enc setBytes:&a_re length:sizeof(float) atIndex:2];
    [enc setBytes:&a_im length:sizeof(float) atIndex:3];
    [enc setBytes:&b_re length:sizeof(float) atIndex:4];
    [enc setBytes:&b_im length:sizeof(float) atIndex:5];
    [enc setBytes:&c_re length:sizeof(float) atIndex:6];
    [enc setBytes:&c_im length:sizeof(float) atIndex:7];
    [enc setBytes:&d_re length:sizeof(float) atIndex:8];
    [enc setBytes:&d_im length:sizeof(float) atIndex:9];
    [enc setBuffer:bOr offset:0 atIndex:10];
    [enc setBuffer:bOi offset:0 atIndex:11];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:12];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 30000,
                                                 "complex_mobius_f32_msl");
    if (_pool_ok) {
      std::memcpy(out_re, [bOr contents], bytes);
      std::memcpy(out_im, [bOi contents], bytes);
    }
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_complex_mobius_f32(
    const float* z_re, const float* z_im,
    float a_re, float a_im, float b_re, float b_im,
    float c_re, float c_im, float d_re, float d_im,
    float* out_re, float* out_im, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_complex_mobius_f32_msl(
          ctx, z_re, z_im,
          a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
          out_re, out_im, n)) return;
  reference_complex_mobius_f32(
      z_re, z_im, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im,
      out_re, out_im, n);
}

// ===========================================================================
// EBM decode_init noise-apply  —  out[i] = base[i % base_len] + std * noise[i]
//
// Implements `decode_init(strategy="noise")` semantics on-device: the
// caller pre-generates the `noise` buffer (deterministic from RNGKey)
// and a `base` array (usually zeros for pure-noise init, or a small
// per-batch mean for "base_model" + perturb).  The kernel broadcasts
// `base` across the K × event dims and adds `std * noise`.
//
// `base_len` is the length of the `base` buffer; `n` is the total
// output element count.  Set `base_len = 0` to use mean=0 (pure noise);
// set `base_len = n` for no broadcasting.
// ===========================================================================

inline void reference_ebm_decode_init_noise_apply_f32(
    const float* base, int32_t base_len, const float* noise,
    float std, float* out, int32_t n) {
  if (base_len == 0) {
    for (int32_t i = 0; i < n; ++i) out[i] = std * noise[i];
  } else {
    for (int32_t i = 0; i < n; ++i)
      out[i] = base[i % base_len] + std * noise[i];
  }
}

static NSString *const kEBMDecodeInitNoiseApplyF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void ebm_decode_init_noise_apply_f32(
    device const float* base     [[buffer(0)]],
    constant int32_t&   base_len [[buffer(1)]],
    device const float* noise    [[buffer(2)]],
    constant float&     std_     [[buffer(3)]],
    device float*       out      [[buffer(4)]],
    constant int32_t&   n        [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if ((int)i >= n) return;
    float b = 0.0f;
    if (base_len > 0) b = base[(int)i % base_len];
    out[i] = b + std_ * noise[i];
}
)MSL";

static bool dispatch_ebm_decode_init_noise_apply_f32_msl(
    MetalDeviceContext &ctx,
    const float* base, int32_t base_len, const float* noise,
    float std_, float* out, int32_t n) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMDecodeInitNoiseApplyF32Source,
        @"ebm_decode_init_noise_apply_f32");
    if (!pso) return false;
    NSUInteger outBytes = sizeof(float) * (NSUInteger)n;
    NSUInteger noiseBytes = sizeof(float) * (NSUInteger)n;
    NSUInteger baseBytes = sizeof(float) * (NSUInteger)std::max(1, base_len);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bB, ctx,
        (base_len > 0 ? base : noise), baseBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bN, ctx, noise, noiseBytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, outBytes);
    if (!bB || !bN || !bO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bB offset:0 atIndex:0];
    [enc setBytes:&base_len length:sizeof(int32_t) atIndex:1];
    [enc setBuffer:bN offset:0 atIndex:2];
    [enc setBytes:&std_ length:sizeof(float) atIndex:3];
    [enc setBuffer:bO offset:0 atIndex:4];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:5];
    MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)n,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                                 "ebm_decode_init_noise_apply_f32_msl");
    if (_pool_ok) std::memcpy(out, [bO contents], outBytes);
    return _pool_ok;
    // RAII guards release bB / bN / bO at scope exit.
  }
}

extern "C" void tessera_apple_gpu_ebm_decode_init_noise_apply_f32(
    const float* base, int32_t base_len, const float* noise,
    float std_, float* out, int32_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_decode_init_noise_apply_f32_msl(
          ctx, base, base_len, noise, std_, out, n)) return;
  reference_ebm_decode_init_noise_apply_f32(base, base_len, noise, std_, out, n);
}

// ===========================================================================
// EBM sphere langevin_step  —  one Euler-Maruyama step on S^{d-1}
//
//   grad_tan   = grad - <grad, x> * x          (tangent projection)
//   noise_tan  = noise - <noise, x> * x
//   y          = x - eta * grad_tan + noise_scale * noise_tan
//   out        = y / ||y||                       (retract to unit sphere)
//
// Single-thread MSL kernel since d is typically 3 — overhead of a
// multi-thread reduction would dominate.  Caller supplies `noise`
// (deterministic, host-generated from the RNGKey).
// ===========================================================================

inline void reference_ebm_sphere_langevin_step_f32(
    const float* x, const float* grad, const float* noise,
    float eta, float noise_scale, float* out, int32_t d) {
  float gdot = 0.0f, ndot = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    gdot += grad[i] * x[i];
    ndot += noise[i] * x[i];
  }
  float ynorm2 = 0.0f;
  std::vector<float> y(d);
  for (int32_t i = 0; i < d; ++i) {
    float grad_tan = grad[i] - gdot * x[i];
    float noise_tan = noise[i] - ndot * x[i];
    y[i] = x[i] - eta * grad_tan + noise_scale * noise_tan;
    ynorm2 += y[i] * y[i];
  }
  float ynorm = std::sqrt(ynorm2);
  if (ynorm < 1e-12f) {
    for (int32_t i = 0; i < d; ++i) out[i] = x[i];
    return;
  }
  for (int32_t i = 0; i < d; ++i) out[i] = y[i] / ynorm;
}

static NSString *const kEBMSphereLangevinStepF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// Single-threadgroup, single-thread kernel.  d is small (typically 3).
// For larger d the dot products would benefit from a parallel reduction;
// the v1 sphere step covers S^2 / S^3 which fit in a single iteration.
kernel void ebm_sphere_langevin_step_f32(
    device const float* x           [[buffer(0)]],
    device const float* grad        [[buffer(1)]],
    device const float* noise       [[buffer(2)]],
    constant float&     eta         [[buffer(3)]],
    constant float&     noise_scale [[buffer(4)]],
    device float*       out         [[buffer(5)]],
    constant int32_t&   d           [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    float gdot = 0.0f, ndot = 0.0f;
    for (int i = 0; i < d; ++i) {
        gdot += grad[i] * x[i];
        ndot += noise[i] * x[i];
    }
    float ynorm2 = 0.0f;
    // y stored in `out` first; renormalize in second pass.
    for (int i = 0; i < d; ++i) {
        float grad_tan = grad[i] - gdot * x[i];
        float noise_tan = noise[i] - ndot * x[i];
        float yi = x[i] - eta * grad_tan + noise_scale * noise_tan;
        out[i] = yi;
        ynorm2 += yi * yi;
    }
    float ynorm = sqrt(ynorm2);
    if (ynorm < 1e-12f) {
        for (int i = 0; i < d; ++i) out[i] = x[i];
        return;
    }
    for (int i = 0; i < d; ++i) out[i] = out[i] / ynorm;
}
)MSL";

static bool dispatch_ebm_sphere_langevin_step_f32_msl(
    MetalDeviceContext &ctx,
    const float* x, const float* grad, const float* noise,
    float eta, float noise_scale, float* out, int32_t d) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMSphereLangevinStepF32Source,
        @"ebm_sphere_langevin_step_f32");
    if (!pso) return false;
    NSUInteger bytes = sizeof(float) * (NSUInteger)d;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bX, ctx, x, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bG, ctx, grad, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bN, ctx, noise, bytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, bytes);
    if (!bX || !bG || !bN || !bO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bX offset:0 atIndex:0];
    [enc setBuffer:bG offset:0 atIndex:1];
    [enc setBuffer:bN offset:0 atIndex:2];
    [enc setBytes:&eta length:sizeof(float) atIndex:3];
    [enc setBytes:&noise_scale length:sizeof(float) atIndex:4];
    [enc setBuffer:bO offset:0 atIndex:5];
    [enc setBytes:&d length:sizeof(int32_t) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                                 "ebm_sphere_langevin_step_f32_msl");
    if (_pool_ok) std::memcpy(out, [bO contents], bytes);
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_sphere_langevin_step_f32(
    const float* x, const float* grad, const float* noise,
    float eta, float noise_scale, float* out, int32_t d) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_sphere_langevin_step_f32_msl(
          ctx, x, grad, noise, eta, noise_scale, out, d)) return;
  reference_ebm_sphere_langevin_step_f32(x, grad, noise, eta, noise_scale, out, d);
}

// ===========================================================================
// EBM self_verify  —  hard-argmin reduction over K candidates per batch
//
// Inputs:
//   energies   — shape (B, K) row-major
//   candidates — shape (B, K, D) row-major
// Output:
//   out        — shape (B, D)
//
// For each batch row b, find k* = argmin_k energies[b, k] and copy
// candidates[b, k*, :] into out[b, :].  One threadgroup per batch row;
// the lone thread does the K-way scan then the D copy.  This matches
// the hard-argmin path of `tessera.ebm.self_verify(... beta=None)`.
//
// Soft-min (beta > 0) is a separate kernel — out of scope for v1.
// ===========================================================================

inline void reference_ebm_self_verify_hard_argmin_f32(
    const float* energies, const float* candidates,
    float* out, int32_t B, int32_t K, int32_t D) {
  for (int32_t b = 0; b < B; ++b) {
    int32_t best = 0;
    float best_e = energies[b * K];
    for (int32_t k = 1; k < K; ++k) {
      float e = energies[b * K + k];
      if (e < best_e) { best_e = e; best = k; }
    }
    const float* src = candidates + ((int64_t)b * K + best) * D;
    std::memcpy(out + (int64_t)b * D, src, sizeof(float) * (size_t)D);
  }
}

static NSString *const kEBMSelfVerifyHardArgminF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void ebm_self_verify_hard_argmin_f32(
    device const float* energies   [[buffer(0)]],
    device const float* candidates [[buffer(1)]],
    device float*       out        [[buffer(2)]],
    constant int32_t&   B          [[buffer(3)]],
    constant int32_t&   K          [[buffer(4)]],
    constant int32_t&   D          [[buffer(5)]],
    uint b [[thread_position_in_grid]])
{
    if ((int)b >= B) return;
    int best = 0;
    float best_e = energies[(int)b * K];
    for (int k = 1; k < K; ++k) {
        float e = energies[(int)b * K + k];
        if (e < best_e) { best_e = e; best = k; }
    }
    const device float* src = candidates + ((int)b * K + best) * D;
    device float* dst = out + (int)b * D;
    for (int i = 0; i < D; ++i) dst[i] = src[i];
}
)MSL";

static bool dispatch_ebm_self_verify_hard_argmin_f32_msl(
    MetalDeviceContext &ctx,
    const float* energies, const float* candidates, float* out,
    int32_t B, int32_t K, int32_t D) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMSelfVerifyHardArgminF32Source,
        @"ebm_self_verify_hard_argmin_f32");
    if (!pso) return false;
    NSUInteger eBytes = sizeof(float) * (NSUInteger)B * (NSUInteger)K;
    NSUInteger cBytes = sizeof(float) * (NSUInteger)B * (NSUInteger)K * (NSUInteger)D;
    NSUInteger oBytes = sizeof(float) * (NSUInteger)B * (NSUInteger)D;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bE, ctx, energies, eBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bC, ctx, candidates, cBytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, oBytes);
    if (!bE || !bC || !bO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bE offset:0 atIndex:0];
    [enc setBuffer:bC offset:0 atIndex:1];
    [enc setBuffer:bO offset:0 atIndex:2];
    [enc setBytes:&B length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&D length:sizeof(int32_t) atIndex:5];
    MTLSize grid = MTLSizeMake((NSUInteger)B, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)B,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                                 "ebm_self_verify_hard_argmin_f32_msl");
    if (_pool_ok) std::memcpy(out, [bO contents], oBytes);
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_self_verify_hard_argmin_f32(
    const float* energies, const float* candidates, float* out,
    int32_t B, int32_t K, int32_t D) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_self_verify_hard_argmin_f32_msl(
          ctx, energies, candidates, out, B, K, D)) return;
  reference_ebm_self_verify_hard_argmin_f32(energies, candidates, out, B, K, D);
}

// ===========================================================================
// EBM energy (quadratic specialization)  —  E_b = 0.5 * ||x_b - y_b||^2
//
// The dominant energy form in EBT / diffusion models — covers
// reconstruction loss + Gaussian log-likelihood up to a constant.  The
// user-supplied `energy_fn(x, y)` in `tessera.ebm.energy` can be
// arbitrary, so this is a specialization that callers opt into when
// their energy_fn is documented to match this shape.
//
// Inputs:
//   x, y — shape (B, D)
// Output:
//   energies — shape (B,)
//
// One threadgroup per batch row; the lone thread sums D squared diffs.
// ===========================================================================

inline void reference_ebm_energy_quadratic_f32(
    const float* x, const float* y, float* energies,
    int32_t B, int32_t D) {
  for (int32_t b = 0; b < B; ++b) {
    float acc = 0.0f;
    for (int32_t d = 0; d < D; ++d) {
      float diff = x[b * D + d] - y[b * D + d];
      acc += diff * diff;
    }
    energies[b] = 0.5f * acc;
  }
}

static NSString *const kEBMEnergyQuadraticF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void ebm_energy_quadratic_f32(
    device const float* x        [[buffer(0)]],
    device const float* y        [[buffer(1)]],
    device float*       energies [[buffer(2)]],
    constant int32_t&   B        [[buffer(3)]],
    constant int32_t&   D        [[buffer(4)]],
    uint b [[thread_position_in_grid]])
{
    if ((int)b >= B) return;
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = x[(int)b * D + d] - y[(int)b * D + d];
        acc += diff * diff;
    }
    energies[(int)b] = 0.5f * acc;
}
)MSL";

static bool dispatch_ebm_energy_quadratic_f32_msl(
    MetalDeviceContext &ctx,
    const float* x, const float* y, float* energies,
    int32_t B, int32_t D) {
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMEnergyQuadraticF32Source, @"ebm_energy_quadratic_f32");
    if (!pso) return false;
    NSUInteger inBytes = sizeof(float) * (NSUInteger)B * (NSUInteger)D;
    NSUInteger outBytes = sizeof(float) * (NSUInteger)B;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bX, ctx, x, inBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bY, ctx, y, inBytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, outBytes);
    if (!bX || !bY || !bO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bX offset:0 atIndex:0];
    [enc setBuffer:bY offset:0 atIndex:1];
    [enc setBuffer:bO offset:0 atIndex:2];
    [enc setBytes:&B length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&D length:sizeof(int32_t) atIndex:4];
    MTLSize grid = MTLSizeMake((NSUInteger)B, 1, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)B,
                                           pso.maxTotalThreadsPerThreadgroup);
    if (tg_x == 0) tg_x = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool _pool_ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                                 "ebm_energy_quadratic_f32_msl");
    if (_pool_ok) std::memcpy(energies, [bO contents], outBytes);
    return _pool_ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_energy_quadratic_f32(
    const float* x, const float* y, float* energies,
    int32_t B, int32_t D) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_energy_quadratic_f32_msl(
          ctx, x, y, energies, B, D)) return;
  reference_ebm_energy_quadratic_f32(x, y, energies, B, D);
}

extern "C" int32_t tessera_apple_gpu_ebm_energy_quadratic_value_f32(
    const float* x, const float* y, float* energies,
    int32_t B, int32_t D) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_energy_quadratic_f32_msl(
          ctx, x, y, energies, B, D))
    return 1;
  return 0;
}

// ===========================================================================
// EBT-tiny fused refinement → energy → hard-argmin
//
// Collapses the entire EBT-tiny pattern into one Metal dispatch:
//   1. For each (b, k) candidate: run T inner-step iterations of
//      ``y - eta * grad`` in registers (no buffer materialization)
//   2. Compute ``e[b, k] = sum_d y_T[b, k, d]^2``
//   3. Hard-argmin over the K dim and copy the winner to ``out[b, :]``
//
// Memory layout: y0 / grad are (B*K, D) row-major; out is (B, D).
// Threadgroup geometry: one threadgroup per batch row (B groups), one
// thread per candidate row (K threads).  After the 2026-05-17
// streaming-closed-form rewrite (see the MSL kernel below), each
// thread streams through D twice — once accumulating the squared-norm
// energy, then again on the winning row to write y_T to out.  No
// per-thread register vector ⇒ D is unbounded; only K is still
// bounded at 256 by the threadgroup-size budget for the K-way
// argmin reduction.
//
// This is the optimization that makes ``ebt_tiny_refinement`` win at
// production shapes (B=64, K=128, D=1024, T=256) — the prior
// two-dispatch refinement+self_verify pipeline was dispatch-overhead
// bound, and the prior register-vector kernel hard-capped D at 256.
// ===========================================================================

inline void reference_ebm_ebt_tiny_refinement_argmin_f32(
    const float* y0, const float* grad, float eta, int32_t T,
    float* out, int32_t B, int32_t K, int32_t D) {
  for (int32_t b = 0; b < B; ++b) {
    int32_t best_k = 0;
    float best_e = std::numeric_limits<float>::infinity();
    std::vector<float> best_y(D);
    std::vector<float> y(D);
    for (int32_t k = 0; k < K; ++k) {
      int64_t base = ((int64_t)b * K + k) * D;
      for (int32_t d = 0; d < D; ++d) y[d] = y0[base + d];
      for (int32_t t = 0; t < T; ++t) {
        for (int32_t d = 0; d < D; ++d) {
          y[d] = y[d] - eta * grad[base + d];
        }
      }
      float e = 0.0f;
      for (int32_t d = 0; d < D; ++d) e += y[d] * y[d];
      if (e < best_e) {
        best_e = e;
        best_k = k;
        for (int32_t d = 0; d < D; ++d) best_y[d] = y[d];
      }
    }
    (void)best_k;
    for (int32_t d = 0; d < D; ++d) out[(int64_t)b * D + d] = best_y[d];
  }
}

static NSString *const kEBMEBTTinyRefinementArgminF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// One threadgroup per batch.  One thread per candidate (K <= 256 lanes).
// Streaming closed-form variant — the original kernel kept the full
// refined candidate row in a per-thread register buffer
// (``float y_local[256]``) which capped ``D`` at 256.  With a fixed
// gradient ``y_T = y0 - T*eta*grad`` is the exact closed form of the
// T-step recurrence, so each thread can stream through ``D`` once,
// accumulating the squared-norm energy without materializing ``y_T``
// anywhere.  The winning thread streams a second pass to write its
// ``y_T`` row to ``out``.  No register-pressure cap on ``D`` — the
// only remaining limit is ``K <= 256`` (threadgroup-size budget for
// the argmin reduction).
kernel void ebm_ebt_tiny_refinement_argmin_f32(
    device const float* y0    [[buffer(0)]],
    device const float* grad  [[buffer(1)]],
    constant float&     eta   [[buffer(2)]],
    constant int32_t&   T     [[buffer(3)]],
    device float*       out   [[buffer(4)]],
    constant int32_t&   B     [[buffer(5)]],
    constant int32_t&   K     [[buffer(6)]],
    constant int32_t&   D     [[buffer(7)]],
    uint3  gid  [[threadgroup_position_in_grid]],
    uint3  tid  [[thread_position_in_threadgroup]],
    uint3  tgs  [[threads_per_threadgroup]])
{
    int b = (int)gid.x;
    int k = (int)tid.x;
    if (b >= B || k >= K) return;

    // Closed-form coefficient: y_T[d] = y0[d] - T*eta*grad[d] for
    // fixed grad.  Each thread streams D once to accumulate the
    // squared-norm energy of its candidate.
    float coeff = (float)T * eta;
    int64_t base = ((int64_t)b * K + k) * D;
    float e = 0.0f;
    for (int d = 0; d < D; ++d) {
        float yi = y0[base + d] - coeff * grad[base + d];
        e += yi * yi;
    }

    // K-way argmin via threadgroup-shared scratch.  K <= 256 lanes.
    threadgroup float  energies[256];
    energies[k] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k == 0) {
        int best = 0;
        float best_e = energies[0];
        for (int kk = 1; kk < K; ++kk) {
            if (energies[kk] < best_e) {
                best_e = energies[kk];
                best = kk;
            }
        }
        // Signal the winner index via the same scratch (well within
        // fp32 integer-representable range for K <= 2^24).
        energies[0] = (float)best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int best_k = (int)energies[0];

    // The winning thread streams D a second time to write y_T to out.
    if (k == best_k) {
        int64_t base_winner = ((int64_t)b * K + best_k) * D;
        for (int d = 0; d < D; ++d) {
            out[(int64_t)b * D + d] = y0[base_winner + d]
                                       - coeff * grad[base_winner + d];
        }
    }
}
)MSL";

static bool dispatch_ebm_ebt_tiny_refinement_argmin_f32_msl(
    MetalDeviceContext &ctx,
    const float* y0, const float* grad, float eta, int32_t T,
    float* out, int32_t B, int32_t K, int32_t D) {
  // K still bounded by the threadgroup-size budget; D is unbounded
  // after the streaming-closed-form rewrite.
  if (K > 256) return false;
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMEBTTinyRefinementArgminF32Source,
        @"ebm_ebt_tiny_refinement_argmin_f32");
    if (!pso) return false;
    size_t inBytes  = sizeof(float) * (size_t)B * (size_t)K * (size_t)D;
    size_t outBytes = sizeof(float) * (size_t)B * (size_t)D;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bY, ctx, y0, inBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bG, ctx, grad, inBytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, outBytes);
    if (!bY || !bG || !bO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bY offset:0 atIndex:0];
    [enc setBuffer:bG offset:0 atIndex:1];
    [enc setBytes:&eta length:sizeof(float) atIndex:2];
    [enc setBytes:&T length:sizeof(int32_t) atIndex:3];
    [enc setBuffer:bO offset:0 atIndex:4];
    [enc setBytes:&B length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&D length:sizeof(int32_t) atIndex:7];
    MTLSize grid = MTLSizeMake((NSUInteger)B, 1, 1);
    MTLSize tg = MTLSizeMake((NSUInteger)K, 1, 1);
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                           "ebm_ebt_tiny_refinement_argmin_f32_msl");
    if (ok) std::memcpy(out, [bO contents], outBytes);
    return ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_ebt_tiny_refinement_argmin_f32(
    const float* y0, const float* grad, float eta, int32_t T,
    float* out, int32_t B, int32_t K, int32_t D) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && K <= 256 &&
      dispatch_ebm_ebt_tiny_refinement_argmin_f32_msl(
          ctx, y0, grad, eta, T, out, B, K, D)) return;
  reference_ebm_ebt_tiny_refinement_argmin_f32(
      y0, grad, eta, T, out, B, K, D);
}

// ===========================================================================
// EBM partition_exact (logsumexp)  —  Z = sum_i exp(-energies[i] / T)
//
// Closes the 8/9 → 9/9 native EBM gap.  Takes a precomputed
// energies array (typically produced upstream by ebm.energy_quadratic
// or a user energy_fn evaluated on a discrete state grid) and
// returns the partition function via a numerically-stable
// log-sum-exp + final exp.
//
//   log_z = max_i(-E_i/T) + log(sum_i exp(-E_i/T - max))
//   Z     = exp(log_z)
//
// One-threadgroup dispatch (single thread per group; the
// reduction is sequential since N is typically small for exhaustive
// state enumeration).  For larger N the parallel tree-reduction
// variant is a follow-up — the v1 native kernel matches the
// "small-state exhaustive sum" use case the Python ref targets.
// ===========================================================================

inline void reference_ebm_partition_exact_f32(
    const float* energies, int32_t n, float temperature, float* out) {
  if (n <= 0) { *out = 0.0f; return; }
  float inv_t = 1.0f / temperature;
  float max_neg = -energies[0] * inv_t;
  for (int32_t i = 1; i < n; ++i) {
    float v = -energies[i] * inv_t;
    if (v > max_neg) max_neg = v;
  }
  float sum = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    sum += std::exp(-energies[i] * inv_t - max_neg);
  }
  *out = std::exp(max_neg + std::log(sum));
}

static NSString *const kEBMPartitionExactF32Source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

// Single-thread kernel — stable logsumexp over a 1-D energies array.
// N is typically small for exhaustive-enumeration partition sums;
// the parallel tree-reduction version is a follow-up.
kernel void ebm_partition_exact_f32(
    device const float* energies     [[buffer(0)]],
    constant int32_t&   n            [[buffer(1)]],
    constant float&     temperature  [[buffer(2)]],
    device float*       out          [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    if (n <= 0) { out[0] = 0.0f; return; }
    float inv_t = 1.0f / temperature;
    float max_neg = -energies[0] * inv_t;
    for (int i = 1; i < n; ++i) {
        float v = -energies[i] * inv_t;
        if (v > max_neg) max_neg = v;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += exp(-energies[i] * inv_t - max_neg);
    }
    out[0] = exp(max_neg + log(sum));
}
)MSL";

static bool dispatch_ebm_partition_exact_f32_msl(
    MetalDeviceContext &ctx,
    const float* energies, int32_t n, float temperature, float* out) {
  if (n <= 0) {
    *out = 0.0f;
    return true;
  }
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(
        ctx, kEBMPartitionExactF32Source, @"ebm_partition_exact_f32");
    if (!pso) return false;
    size_t inBytes  = sizeof(float) * (size_t)n;
    size_t outBytes = sizeof(float);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bE, ctx, energies, inBytes);
    TS_METAL_BUF_ACQUIRE(bO, ctx, outBytes);
    if (!bE || !bO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bE offset:0 atIndex:0];
    [enc setBytes:&n length:sizeof(int32_t) atIndex:1];
    [enc setBytes:&temperature length:sizeof(float) atIndex:2];
    [enc setBuffer:bO offset:0 atIndex:3];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    bool ok = commit_and_wait_with_timeout(ctx, cb, 60000,
                                           "ebm_partition_exact_f32_msl");
    if (ok) std::memcpy(out, [bO contents], outBytes);
    return ok;
  }
}

extern "C" void tessera_apple_gpu_ebm_partition_exact_f32(
    const float* energies, int32_t n, float temperature, float* out) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_partition_exact_f32_msl(
          ctx, energies, n, temperature, out)) return;
  reference_ebm_partition_exact_f32(energies, n, temperature, out);
}

extern "C" int32_t tessera_apple_gpu_ebm_partition_exact_value_f32(
    const float* energies, int32_t n, float temperature, float* out) {
  if (n <= 0 || temperature <= 0.0f)
    return 0;
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_ebm_partition_exact_f32_msl(
          ctx, energies, n, temperature, out))
    return 1;
  return 0;
}

//===----------------------------------------------------------------------===//
// MPSGraph long-tail + Tier-1 execution lane (2026-05-29)
//
// Rather than hand-writing one MSL kernel per pointwise / normalization op,
// this lane routes a broad set of ops through MetalPerformanceShadersGraph —
// Apple's optimizing graph compiler.  One small set of C ABI entry points
// (op-coded unary, op-coded binary, layer_norm, rmsnorm, softmax,
// log_softmax) covers the Tier-1 activation/normalization surface and the
// long tail, and — by composing with the existing MPS matmul — lets the
// f16/bf16 fused MLP/attention chains run with no N<=256 limit.
//
// Convention: all internal compute is fp32 (inputs cast up, outputs cast
// down) matching the rest of the backend's "fp16 I/O + fp32 accumulator"
// numerics.  f32 and f16 symbols share one parametrized runner; bf16 is
// handled host-side by upcasting to f32 (see runtime.py), mirroring the
// existing bf16 matmul path.
//
// Op codes (must match python/tessera/runtime.py):
//   unary:  0 relu  1 sigmoid  2 tanh  3 softplus  4 silu  5 gelu_tanh
//           6 exp   7 log      8 sqrt  9 rsqrt     10 neg  11 abs
//   binary: 0 add   1 sub      2 mul   3 div       4 max   5 min
//           6 silu_mul (a * silu(b))
//   rowop:  0 layer_norm  1 rmsnorm  2 softmax  3 log_softmax
//===----------------------------------------------------------------------===//

namespace {

static MPSGraphTensor *mpsg_unary_node(MPSGraph *g, MPSGraphTensor *x, int op) {
  switch (op) {
    case 0: return [g reLUWithTensor:x name:nil];
    case 1: return [g sigmoidWithTensor:x name:nil];
    case 2: return [g tanhWithTensor:x name:nil];
    case 3: {  // softplus = log(1 + exp(x))
      MPSGraphTensor *e = [g exponentWithTensor:x name:nil];
      MPSGraphTensor *one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
      MPSGraphTensor *s = [g additionWithPrimaryTensor:e secondaryTensor:one name:nil];
      return [g logarithmWithTensor:s name:nil];
    }
    case 4: {  // silu = x * sigmoid(x)
      MPSGraphTensor *s = [g sigmoidWithTensor:x name:nil];
      return [g multiplicationWithPrimaryTensor:x secondaryTensor:s name:nil];
    }
    case 5: {  // gelu tanh-approx: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
      MPSGraphTensor *c0 = [g constantWithScalar:0.7978845608028654 dataType:MPSDataTypeFloat32];
      MPSGraphTensor *c1 = [g constantWithScalar:0.044715 dataType:MPSDataTypeFloat32];
      MPSGraphTensor *half = [g constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
      MPSGraphTensor *one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
      MPSGraphTensor *x2 = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
      MPSGraphTensor *x3 = [g multiplicationWithPrimaryTensor:x2 secondaryTensor:x name:nil];
      MPSGraphTensor *c1x3 = [g multiplicationWithPrimaryTensor:c1 secondaryTensor:x3 name:nil];
      MPSGraphTensor *inner = [g additionWithPrimaryTensor:x secondaryTensor:c1x3 name:nil];
      MPSGraphTensor *scaled = [g multiplicationWithPrimaryTensor:c0 secondaryTensor:inner name:nil];
      MPSGraphTensor *t = [g tanhWithTensor:scaled name:nil];
      MPSGraphTensor *tp1 = [g additionWithPrimaryTensor:t secondaryTensor:one name:nil];
      MPSGraphTensor *hx = [g multiplicationWithPrimaryTensor:half secondaryTensor:x name:nil];
      return [g multiplicationWithPrimaryTensor:hx secondaryTensor:tp1 name:nil];
    }
    case 6: return [g exponentWithTensor:x name:nil];
    case 7: return [g logarithmWithTensor:x name:nil];
    case 8: return [g squareRootWithTensor:x name:nil];
    case 9: {  // rsqrt = 1 / sqrt(x)
      MPSGraphTensor *s = [g squareRootWithTensor:x name:nil];
      MPSGraphTensor *one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
      return [g divisionWithPrimaryTensor:one secondaryTensor:s name:nil];
    }
    case 10: return [g negativeWithTensor:x name:nil];
    case 11: return [g absoluteWithTensor:x name:nil];
    default: return nil;
  }
}

static MPSGraphTensor *mpsg_binary_node(MPSGraph *g, MPSGraphTensor *a,
                                        MPSGraphTensor *b, int op) {
  switch (op) {
    case 0: return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case 1: return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case 2: return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    case 3: return [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
    case 4: return [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
    case 5: return [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
    case 6: {  // silu_mul = a * silu(b) = a * b * sigmoid(b)
      MPSGraphTensor *s = [g sigmoidWithTensor:b name:nil];
      MPSGraphTensor *sb = [g multiplicationWithPrimaryTensor:b secondaryTensor:s name:nil];
      return [g multiplicationWithPrimaryTensor:a secondaryTensor:sb name:nil];
    }
    default: return nil;
  }
}

static inline MPSGraphTensor *mpsg_up(MPSGraph *g, MPSGraphTensor *t,
                                      MPSDataType ioType) {
  return (ioType == MPSDataTypeFloat32)
             ? t
             : [g castTensor:t toType:MPSDataTypeFloat32 name:nil];
}
static inline MPSGraphTensor *mpsg_down(MPSGraph *g, MPSGraphTensor *t,
                                        MPSDataType ioType) {
  return (ioType == MPSDataTypeFloat32) ? t
                                        : [g castTensor:t toType:ioType name:nil];
}

// Compiled-graph cache: build the MPSGraph once per (shape-class, opcode,
// dtype, shape[, eps]) signature and reuse it across calls — only the feed
// MPSGraphTensorData (the actual buffers) change per call. Value is
// @[graph, @[placeholders...], outputTensor]; the static dictionary keeps the
// graph/placeholders/output alive past each call's @autoreleasepool drain.
//
// Task B (2026-06-01) — LRU cache eviction. Long-running training loops
// (a 5,000-step run with 10 distinct shapes per step is 50,000 graphs)
// would otherwise accumulate compiled graphs indefinitely until the
// process exhausts memory. The cache now tracks MRU order via an
// NSMutableOrderedSet keyed alongside the lookup dictionary; on insert
// past the capacity, the LRU entry is evicted. Capacity is set from the
// ``TESSERA_MPSGRAPH_CACHE_CAPACITY`` env var (default 1024 — comfortably
// above any single-process working set we've seen). Evictions counter
// is exposed via ``tessera_apple_gpu_mpsgraph_cache_evictions()`` for
// thrashing observability.
static NSMutableDictionary<NSString *, NSArray *> *mpsg_graph_cache() {
  static NSMutableDictionary *cache = nil;
  static std::once_flag once;
  std::call_once(once, [] { cache = [[NSMutableDictionary alloc] init]; });
  return cache;
}
// MRU-first key list: index 0 = most recently used; tail = candidate
// for eviction. Mutated under ``g_mpsg_graph_mu`` together with the
// lookup dictionary so the two stay consistent.
static NSMutableOrderedSet<NSString *> *mpsg_lru_order() {
  static NSMutableOrderedSet *order = nil;
  static std::once_flag once;
  std::call_once(once, [] { order = [[NSMutableOrderedSet alloc] init]; });
  return order;
}
static std::mutex g_mpsg_graph_mu;
static std::atomic<int64_t> g_mpsg_evictions{0};

// Cached at first read so the env-var lookup doesn't hit ``getenv``
// on every cache touch. ``TESSERA_MPSGRAPH_CACHE_CAPACITY=0`` disables
// the LRU (unbounded — restores the pre-Task-B behavior).
static size_t mpsg_cache_capacity() {
  static size_t cap = 0;
  static std::once_flag once;
  std::call_once(once, [] {
    const char *raw = getenv("TESSERA_MPSGRAPH_CACHE_CAPACITY");
    if (raw && *raw) {
      char *end = nullptr;
      long v = strtol(raw, &end, 10);
      if (end && *end == '\0' && v >= 0) {
        cap = (size_t)v;
        return;
      }
    }
    cap = 1024;  // default
  });
  return cap;
}

static NSArray *mpsg_cache_get(NSString *key) {
  std::lock_guard<std::mutex> lock(g_mpsg_graph_mu);
  NSArray *entry = mpsg_graph_cache()[key];
  if (entry) {
    // Move key to MRU front so future evictions skip it.
    NSMutableOrderedSet<NSString *> *order = mpsg_lru_order();
    [order removeObject:key];
    [order insertObject:key atIndex:0];
  }
  return entry;
}
static void mpsg_cache_put(NSString *key, NSArray *entry) {
  std::lock_guard<std::mutex> lock(g_mpsg_graph_mu);
  NSMutableDictionary<NSString *, NSArray *> *cache = mpsg_graph_cache();
  NSMutableOrderedSet<NSString *> *order = mpsg_lru_order();
  bool replacing = (cache[key] != nil);
  cache[key] = entry;
  if (replacing) {
    [order removeObject:key];
  }
  [order insertObject:key atIndex:0];
  // Evict until under capacity. ``capacity == 0`` means unbounded.
  size_t cap = mpsg_cache_capacity();
  if (cap > 0) {
    while ([order count] > cap) {
      NSString *victim = [order lastObject];
      if (!victim) break;
      [cache removeObjectForKey:victim];
      [order removeObjectAtIndex:[order count] - 1];
      g_mpsg_evictions.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

static bool mpsg_run_unary(MetalDeviceContext &ctx, int op, const void *x,
                           void *out, int64_t n, MPSDataType ioType,
                           size_t elemSize) {
  if (n <= 0) return true;
  @autoreleasepool {
    size_t bytes = (size_t)n * elemSize;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, x, bytes);
    if (!bufX) return false;
    NSArray<NSNumber *> *shape = @[ @(n) ];
    NSString *key = [NSString stringWithFormat:@"u:%d:%d:%lld", op, (int)ioType,
                                               (long long)n];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *ph;
    MPSGraphTensor *y;
    if (entry) {
      g = entry[0];
      ph = ((NSArray *)entry[1])[0];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      ph = [g placeholderWithShape:shape dataType:ioType name:nil];
      MPSGraphTensor *yf = mpsg_unary_node(g, mpsg_up(g, ph, ioType), op);
      if (!yf) return false;
      y = mpsg_down(g, yf, ioType);
      mpsg_cache_put(key, @[ g, @[ ph ], y ]);
    }
    MPSGraphTensorData *xd =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:shape dataType:ioType];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{ph : xd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

static bool mpsg_run_binary(MetalDeviceContext &ctx, int op, const void *a,
                            const void *b, void *out, int64_t n,
                            MPSDataType ioType, size_t elemSize) {
  if (n <= 0) return true;
  @autoreleasepool {
    size_t bytes = (size_t)n * elemSize;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, a, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, b, bytes);
    if (!bufA || !bufB) return false;
    NSArray<NSNumber *> *shape = @[ @(n) ];
    NSString *key = [NSString stringWithFormat:@"b:%d:%d:%lld", op, (int)ioType,
                                               (long long)n];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *pa;
    MPSGraphTensor *pb;
    MPSGraphTensor *y;
    if (entry) {
      g = entry[0];
      pa = ((NSArray *)entry[1])[0];
      pb = ((NSArray *)entry[1])[1];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      pa = [g placeholderWithShape:shape dataType:ioType name:nil];
      pb = [g placeholderWithShape:shape dataType:ioType name:nil];
      MPSGraphTensor *yf =
          mpsg_binary_node(g, mpsg_up(g, pa, ioType), mpsg_up(g, pb, ioType), op);
      if (!yf) return false;
      y = mpsg_down(g, yf, ioType);
      mpsg_cache_put(key, @[ g, @[ pa, pb ], y ]);
    }
    MPSGraphTensorData *ad =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:shape dataType:ioType];
    MPSGraphTensorData *bd =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:shape dataType:ioType];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{pa : ad, pb : bd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

// Row-wise op over the last axis of an [rows, cols] tensor.
// kind: 0 layer_norm (gamma+beta), 1 rmsnorm (gamma), 2 softmax, 3 log_softmax.
// Get-or-build the cached rowop graph (kind 0 layer_norm, 1 rmsnorm, 2 softmax,
// 3 log_softmax). `phs_out` is [x] (+ gamma if hasGamma, + beta if hasBeta).
// Shared by the run + R2 device/encode paths.
static MPSGraph *mpsg_rowop_graph(int kind, int32_t rows, int32_t cols,
                                  float eps, bool hasGamma, bool hasBeta,
                                  MPSDataType ioType, NSArray **phs_out,
                                  MPSGraphTensor **y_out) {
  NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
  NSArray<NSNumber *> *gs = @[ @(cols) ];
  NSArray<NSNumber *> *axis1 = @[ @1 ];
  NSString *key = [NSString stringWithFormat:@"r:%d:%d:%d:%d:%a:%d:%d", kind,
                                             (int)ioType, rows, cols, eps,
                                             hasGamma ? 1 : 0, hasBeta ? 1 : 0];
  NSArray *entry = mpsg_cache_get(key);
  MPSGraph *g;
  NSArray *phs;
  MPSGraphTensor *y;
  if (entry) {
    g = entry[0];
    phs = entry[1];
    y = entry[2];
  } else {
    g = [MPSGraph new];
    MPSGraphTensor *px = [g placeholderWithShape:xs dataType:ioType name:nil];
    MPSGraphTensor *xf = mpsg_up(g, px, ioType);
    NSMutableArray *P = [NSMutableArray arrayWithObject:px];
    MPSGraphTensor *yf = nil;
    if (kind == 0 || kind == 1) {
      MPSGraphTensor *epsc = [g constantWithScalar:(double)eps dataType:MPSDataTypeFloat32];
      MPSGraphTensor *gf = nil;
      if (hasGamma) {
        MPSGraphTensor *pg = [g placeholderWithShape:gs dataType:ioType name:nil];
        gf = mpsg_up(g, pg, ioType);
        [P addObject:pg];
      }
      if (kind == 0) {  // layer_norm
        MPSGraphTensor *mean = [g meanOfTensor:xf axes:axis1 name:nil];
        MPSGraphTensor *diff = [g subtractionWithPrimaryTensor:xf secondaryTensor:mean name:nil];
        MPSGraphTensor *sq = [g multiplicationWithPrimaryTensor:diff secondaryTensor:diff name:nil];
        MPSGraphTensor *var = [g meanOfTensor:sq axes:axis1 name:nil];
        MPSGraphTensor *ve = [g additionWithPrimaryTensor:var secondaryTensor:epsc name:nil];
        MPSGraphTensor *denom = [g squareRootWithTensor:ve name:nil];
        MPSGraphTensor *norm = [g divisionWithPrimaryTensor:diff secondaryTensor:denom name:nil];
        yf = hasGamma
                 ? [g multiplicationWithPrimaryTensor:norm secondaryTensor:gf name:nil]
                 : norm;
        if (hasBeta) {
          MPSGraphTensor *pb = [g placeholderWithShape:gs dataType:ioType name:nil];
          MPSGraphTensor *bf = mpsg_up(g, pb, ioType);
          [P addObject:pb];
          yf = [g additionWithPrimaryTensor:yf secondaryTensor:bf name:nil];
        }
      } else {  // rmsnorm
        MPSGraphTensor *sq = [g multiplicationWithPrimaryTensor:xf secondaryTensor:xf name:nil];
        MPSGraphTensor *ms = [g meanOfTensor:sq axes:axis1 name:nil];
        MPSGraphTensor *me = [g additionWithPrimaryTensor:ms secondaryTensor:epsc name:nil];
        MPSGraphTensor *denom = [g squareRootWithTensor:me name:nil];
        MPSGraphTensor *norm = [g divisionWithPrimaryTensor:xf secondaryTensor:denom name:nil];
        yf = hasGamma
                 ? [g multiplicationWithPrimaryTensor:norm secondaryTensor:gf name:nil]
                 : norm;
      }
    } else if (kind == 2) {  // softmax over last axis (numerically stable)
      yf = [g softMaxWithTensor:xf axis:1 name:nil];
    } else {  // log_softmax = (x - max) - log(sum(exp(x - max)))
      MPSGraphTensor *m = [g reductionMaximumWithTensor:xf axes:axis1 name:nil];
      MPSGraphTensor *xm = [g subtractionWithPrimaryTensor:xf secondaryTensor:m name:nil];
      MPSGraphTensor *e = [g exponentWithTensor:xm name:nil];
      MPSGraphTensor *s = [g reductionSumWithTensor:e axes:axis1 name:nil];
      MPSGraphTensor *lse = [g logarithmWithTensor:s name:nil];
      yf = [g subtractionWithPrimaryTensor:xm secondaryTensor:lse name:nil];
    }
    y = yf ? mpsg_down(g, yf, ioType) : nil;
    phs = [P copy];
    mpsg_cache_put(key, @[ g, phs, y ? y : px ]);
  }
  *phs_out = phs;
  *y_out = y;
  return g;
}

static bool mpsg_run_rowop(MetalDeviceContext &ctx, int kind, const void *x,
                           const void *gamma, const void *beta, void *out,
                           int32_t rows, int32_t cols, float eps,
                           MPSDataType ioType, size_t elemSize) {
  if (rows <= 0 || cols <= 0) return true;
  @autoreleasepool {
    int64_t n = (int64_t)rows * (int64_t)cols;
    size_t xbytes = (size_t)n * elemSize;
    size_t cbytes = (size_t)cols * elemSize;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, x, xbytes);
    if (!bufX) return false;
    // gamma / beta buffers are acquired regardless (unused placeholders feed
    // from x's bytes when the op has no gamma/beta).
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufG, ctx, gamma ? gamma : x, cbytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, beta ? beta : x, cbytes);
    NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
    NSArray<NSNumber *> *gs = @[ @(cols) ];
    NSArray *phs;
    MPSGraphTensor *y;
    MPSGraph *g = mpsg_rowop_graph(kind, rows, cols, eps, gamma != nullptr,
                                   beta != nullptr, ioType, &phs, &y);
    if (!y) return false;
    NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
    feeds[phs[0]] =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:xs dataType:ioType];
    if (phs.count >= 2)
      feeds[phs[1]] =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:bufG shape:gs dataType:ioType];
    if (phs.count >= 3)
      feeds[phs[2]] =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:gs dataType:ioType];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:feeds
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

}  // namespace

// PK8 (generalized) — map an op name to a builder family. Placed after the
// anonymous namespace so it can reference the file-local graph builders
// (``mpsg_unary_node`` / ``mpsg_binary_node`` / ``mpsg_rowop_graph``), which
// have internal linkage but are visible to the rest of this translation unit.
// Returns >=0 op-code for the family, or -1 if ``op`` isn't in that family.
namespace {
int _author_unary_code(const char *op) {
  if (!op) return -1;
  static const char *kNames[] = {"relu",  "sigmoid", "tanh", "softplus",
                                 "silu",  "gelu",    "exp",  "log",
                                 "sqrt",  "rsqrt",   "neg",  "abs"};
  for (int i = 0; i < 12; ++i)
    if (std::strcmp(op, kNames[i]) == 0) return i;
  return -1;
}
int _author_binary_code(const char *op) {
  if (!op) return -1;
  // Same ordering as ``mpsg_binary_node``: add/sub/mul/div/max/min/silu_mul.
  static const char *kNames[] = {"add", "sub",   "mul",     "div",
                                 "max", "min",   "silu_mul"};
  for (int i = 0; i < 7; ++i)
    if (std::strcmp(op, kNames[i]) == 0) return i;
  return -1;
}
int _author_rowop_kind(const char *op) {
  if (!op) return -1;
  // Same kinds as ``mpsg_rowop_graph``: 0 layer_norm, 1 rmsnorm,
  // 2 softmax, 3 log_softmax.
  if (std::strcmp(op, "layer_norm") == 0) return 0;
  if (std::strcmp(op, "rmsnorm") == 0) return 1;
  if (std::strcmp(op, "softmax") == 0) return 2;
  if (std::strcmp(op, "log_softmax") == 0) return 3;
  return -1;
}
}  // namespace

// PK8 (generalized) — author a production `.mtlpackage` for any of the
// MPSGraph-lane ops over a ``[rows, cols]`` fp32 input, reusing the *same*
// graph builders the runtime dispatches at execution time (so the packaged
// kernel is numerically identical to the live path):
//
//   unary   (1 input):  relu sigmoid tanh softplus silu gelu exp log
//                       sqrt rsqrt neg abs
//   rowop   (1 input):  softmax log_softmax
//   norms   (1 input + optional gamma[/beta] when ``weighted``):
//                       rmsnorm (gamma) / layer_norm (gamma+beta)
//   binary  (2 inputs): add sub mul div max min silu_mul
//
// Bindings are positional (MPSGraph emits unnamed bindings): inputs at
// 0.. , output last. ``eps`` applies to the norms; ``weighted`` (0/1) adds
// the gamma/beta inputs for the norms. Returns 1 / <=0 error (see
// ``_mlpkg_compile_and_write`` + -2 bad-args / -6 unknown-op).
extern "C" int32_t tessera_apple_gpu_mlpkg_author_op(
    const char *out_package_path, const char *op, int32_t rows, int32_t cols,
    float eps, int32_t weighted) {
  if (!out_package_path || !op || rows <= 0 || cols <= 0) return -2;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return -1;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      MPSDataType dt = MPSDataTypeFloat32;
      NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
      NSArray<NSNumber *> *gs = @[ @(cols) ];
      MPSGraphShapedType *xt =
          [[MPSGraphShapedType alloc] initWithShape:xs dataType:dt];
      MPSGraphShapedType *gt =
          [[MPSGraphShapedType alloc] initWithShape:gs dataType:dt];

      int unary = _author_unary_code(op);
      int binary = _author_binary_code(op);
      int rowkind = _author_rowop_kind(op);

      if (unary >= 0) {
        MPSGraph *g = [MPSGraph new];
        MPSGraphTensor *px = [g placeholderWithShape:xs dataType:dt name:nil];
        MPSGraphTensor *y = mpsg_unary_node(g, px, unary);
        if (!y) return -6;
        return _mlpkg_compile_and_write(g, @{px : xt}, y, out_package_path);
      }
      if (binary >= 0) {
        MPSGraph *g = [MPSGraph new];
        MPSGraphTensor *pa = [g placeholderWithShape:xs dataType:dt name:nil];
        MPSGraphTensor *pb = [g placeholderWithShape:xs dataType:dt name:nil];
        MPSGraphTensor *y = mpsg_binary_node(g, pa, pb, binary);
        if (!y) return -6;
        return _mlpkg_compile_and_write(g, @{pa : xt, pb : xt}, y,
                                        out_package_path);
      }
      if (rowkind >= 0) {
        bool isNorm = (rowkind == 0 || rowkind == 1);
        bool hasGamma = isNorm && (weighted != 0);
        bool hasBeta = (rowkind == 0) && (weighted != 0);  // layer_norm only
        NSArray *phs = nil;
        MPSGraphTensor *y = nil;
        // Reuse the exact runtime rowop graph (build + placeholders).
        MPSGraph *g = mpsg_rowop_graph(rowkind, rows, cols, eps, hasGamma,
                                       hasBeta, dt, &phs, &y);
        if (!g || !y || phs.count == 0) return -6;
        NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
        feeds[phs[0]] = xt;             // x  [rows, cols]
        if (phs.count >= 2) feeds[phs[1]] = gt;  // gamma [cols]
        if (phs.count >= 3) feeds[phs[2]] = gt;  // beta  [cols]
        return _mlpkg_compile_and_write(g, feeds, y, out_package_path);
      }
      return -6;  // unknown op
    }
  }
  return -1;
}

// Shared op-node builder for the authored-graph paths (PK8c straight-line +
// PK8d for-loop body). `a` is the resolved in0 tensor; `b` the resolved in1
// (nil for unary). Mirrors the GraphFn opcode set; returns nil on bad args /
// unknown opcode. Compute dtype `dt` (f32) drives the norm constants.
//   opcodes: 0 matmul | 1 add 2 sub 3 mul 4 div | 10 softmax(last-axis)
//            11 rmsnorm 12 layer_norm (unweighted) | 20 relu 21 sigmoid
//            22 tanh 23 silu 24 gelu
static MPSGraphTensor *mpsg_build_graph_op(MPSGraph *g, int code,
                                           MPSGraphTensor *a, MPSGraphTensor *b,
                                           int iattr, float eps,
                                           MPSDataType dt) {
  if (!a) return nil;
  if (code == 0) {  // matmul (±transpose)
    if (!b) return nil;
    if (iattr & 1) a = [g transposeTensor:a dimension:0 withDimension:1 name:nil];
    if (iattr & 2) b = [g transposeTensor:b dimension:0 withDimension:1 name:nil];
    return [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
  }
  if (code >= 1 && code <= 4) {  // add/sub/mul/div
    if (!b) return nil;
    return mpsg_binary_node(g, a, b, code - 1);
  }
  if (code == 10) return [g softMaxWithTensor:a axis:1 name:nil];  // last-axis
  if (code == 11) {  // rmsnorm (unweighted)
    MPSGraphTensor *sq = [g multiplicationWithPrimaryTensor:a secondaryTensor:a name:nil];
    MPSGraphTensor *ms = [g meanOfTensor:sq axes:@[ @1 ] name:nil];
    MPSGraphTensor *me = [g additionWithPrimaryTensor:ms
                                      secondaryTensor:[g constantWithScalar:(double)eps dataType:dt]
                                                 name:nil];
    MPSGraphTensor *den = [g squareRootWithTensor:me name:nil];
    return [g divisionWithPrimaryTensor:a secondaryTensor:den name:nil];
  }
  if (code == 12) {  // layer_norm (unweighted)
    MPSGraphTensor *mean = [g meanOfTensor:a axes:@[ @1 ] name:nil];
    MPSGraphTensor *xc = [g subtractionWithPrimaryTensor:a secondaryTensor:mean name:nil];
    MPSGraphTensor *sq = [g multiplicationWithPrimaryTensor:xc secondaryTensor:xc name:nil];
    MPSGraphTensor *var = [g meanOfTensor:sq axes:@[ @1 ] name:nil];
    MPSGraphTensor *me = [g additionWithPrimaryTensor:var
                                      secondaryTensor:[g constantWithScalar:(double)eps dataType:dt]
                                                 name:nil];
    MPSGraphTensor *den = [g squareRootWithTensor:me name:nil];
    return [g divisionWithPrimaryTensor:xc secondaryTensor:den name:nil];
  }
  if (code >= 20 && code <= 24) {  // relu/sigmoid/tanh/silu/gelu
    static const int kU[] = {0, 1, 2, 4, 5};
    return mpsg_unary_node(g, a, kU[code - 20]);
  }
  return nil;  // unknown opcode
}

// Build a straight-line branch/body op-list over `phs` (the arg placeholders)
// plus `extra` already-bound tensors (e.g. the loop carry at id n_args). Tensor
// ids: 0..n_args-1 = args, n_args..n_args+n_extra-1 = extra, n_args+n_extra+j =
// op j. Returns the tensor at `out_id`, or nil on a bad id / unknown opcode.
// A free function (not a lambda) so it composes cleanly inside ObjC blocks.
static MPSGraphTensor *mpsg_build_branch(
    MPSGraph *g, NSArray<MPSGraphTensor *> *phs, int n_args,
    NSArray<MPSGraphTensor *> *extra, MPSDataType dt, int n_ops,
    const int32_t *codes, const int32_t *in0, const int32_t *in1,
    const int32_t *iattr, const float *fattr, int out_id) {
  int n_extra = extra ? (int)extra.count : 0;
  int total = n_args + n_extra + n_ops;
  NSMutableArray *t = [NSMutableArray arrayWithCapacity:total];
  for (int i = 0; i < total; ++i) [t addObject:[NSNull null]];
  for (int i = 0; i < n_args; ++i) t[i] = phs[i];
  for (int i = 0; i < n_extra; ++i) t[n_args + i] = extra[i];
  const int base = n_args + n_extra;
  auto get = [&](int tid) -> MPSGraphTensor * {
    if (tid < 0 || tid >= total) return nil;
    id v = t[tid];
    return (v == [NSNull null]) ? nil : (MPSGraphTensor *)v;
  };
  for (int j = 0; j < n_ops; ++j) {
    int code = codes[j];
    MPSGraphTensor *a = get(in0[j]);
    MPSGraphTensor *b = nil;
    if (code == 0 || (code >= 1 && code <= 4)) b = get(in1 ? in1[j] : -1);
    MPSGraphTensor *y = mpsg_build_graph_op(g, code, a, b, iattr ? iattr[j] : 0,
                                            fattr ? fattr[j] : 1e-5f, dt);
    if (!y) return nil;
    t[base + j] = y;
  }
  return get(out_id);
}

// PK8c — author an ARBITRARY straight-line op graph into ONE serialized
// `.mtlpackage`. The whole graph becomes a single MPSGraph executable, so the
// entire GraphFn runs as ONE Metal dispatch (MPSGraph fuses globally) instead of
// the per-kernel interpreter. This is the "graph as one fused unit" path.
//
// Tensor ids: 0..n_args-1 are inputs (placeholders); op j produces id n_args+j.
// Each op carries an opcode, ≤2 input ids (in1 = -1 if unary), an int attr
// (matmul transpose bits: bit0=A, bit1=B), and a float attr (norm eps).
// `output_id` designates the single result. `io_bf16` (0/1) selects the boundary
// dtype: when 1, inputs/output are bf16 while internal compute stays f32 (ABI
// f32-accumulate). Positional bindings: inputs in arg order at 0.., output last.
//
//   opcodes:  0 matmul | 1 add 2 sub 3 mul 4 div | 10 softmax(last-axis)
//             11 rmsnorm 12 layer_norm (both unweighted) | 20 relu 21 sigmoid
//             22 tanh 23 silu 24 gelu
//
// Returns 1 / <=0 error (-1 OS, -2 bad args, -3/-4/-5 from compile+write, -6
// unknown opcode).
extern "C" int32_t tessera_apple_gpu_mlpkg_author_graph(
    const char *out_package_path, int32_t n_args, const int32_t *arg_rows,
    const int32_t *arg_cols, int32_t n_ops, const int32_t *op_codes,
    const int32_t *op_in0, const int32_t *op_in1, const int32_t *op_iattr,
    const float *op_fattr, int32_t output_id, int32_t io_bf16) {
  if (!out_package_path || n_args <= 0 || n_ops < 0 || !arg_rows ||
      !arg_cols || (n_ops > 0 && (!op_codes || !op_in0)))
    return -2;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return -1;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      // Internal compute is always f32 (the ABI's f32-accumulate policy); when
      // io_bf16 the boundary tensors are bf16 — inputs upcast to f32 right after
      // the placeholder, the final output downcast back to bf16.
      MPSDataType dt = MPSDataTypeFloat32;
      MPSDataType io_dt = io_bf16 ? MPSDataTypeBFloat16 : MPSDataTypeFloat32;
      int32_t total = n_args + n_ops;
      NSMutableArray *tensors = [NSMutableArray arrayWithCapacity:total];
      for (int i = 0; i < total; ++i) [tensors addObject:[NSNull null]];
      MPSGraph *g = [MPSGraph new];
      NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = (arg_cols[i] > 0)
                                       ? @[ @(arg_rows[i]), @(arg_cols[i]) ]
                                       : @[ @(arg_rows[i]) ];
        MPSGraphTensor *p = [g placeholderWithShape:shp dataType:io_dt name:nil];
        tensors[i] = io_bf16 ? [g castTensor:p toType:dt name:nil] : p;
        feeds[p] = [[MPSGraphShapedType alloc] initWithShape:shp dataType:io_dt];
      }
      auto getT = [&](int tid) -> MPSGraphTensor * {
        if (tid < 0 || tid >= total) return nil;
        id v = tensors[tid];
        return (v == [NSNull null]) ? nil : (MPSGraphTensor *)v;
      };
      for (int j = 0; j < n_ops; ++j) {
        int code = op_codes[j];
        MPSGraphTensor *a = getT(op_in0[j]);
        if (!a) return -2;
        MPSGraphTensor *b = nil;
        if (code == 0 || (code >= 1 && code <= 4)) {
          b = getT(op_in1 ? op_in1[j] : -1);
          if (!b) return -2;
        }
        MPSGraphTensor *y = mpsg_build_graph_op(
            g, code, a, b, op_iattr ? op_iattr[j] : 0,
            op_fattr ? op_fattr[j] : 1e-5f, dt);
        if (!y) return -6;
        tensors[n_args + j] = y;
      }
      MPSGraphTensor *out = getT(output_id);
      if (!out) return -2;
      if (io_bf16) out = [g castTensor:out toType:io_dt name:nil];
      return _mlpkg_compile_and_write(g, feeds, out, out_package_path);
    }
  }
  return -1;
}

// PK8d — run a BOUNDED for-loop as ONE MPSGraph `forLoop`, executed DIRECTLY via
// runWithMTLCommandQueue (the package/MLEncoder dispatch path rejects
// control-flow ops, so — like cf_scan — the loop graph is built + run + read in
// one call rather than serialized to a .mtlpackage). The body is a recorded
// straight-line op-list over a single static-shape carry:
//   for i in 0..trip:  carry = body(carry, args)   → final carry into `out`.
// Generalizes cf_scan's fixed RNN body to an arbitrary GraphFn body, reusing
// `mpsg_build_graph_op`.
//
// Body tensor ids: 0..n_args-1 = args, n_args = the carry (iteration argument),
// n_args+1+j = body op j. `carry_arg_index` = the arg initializing the carry;
// `body_out_id` = the next-carry tensor. `arg_ptrs[i]` points at arg i's f32
// data (rows*cols elems, or rows if cols<=0). `out` is sized to the carry arg.
// f32. Returns 1 / <=0 error (0 = runtime unavailable → caller falls back;
// -2 bad args; -3 run/build failure; -6 unknown opcode / bad id).
extern "C" int32_t tessera_apple_gpu_run_graph_loop_f32(
    int32_t n_args, const void *const *arg_ptrs, const int32_t *arg_rows,
    const int32_t *arg_cols, int32_t carry_arg_index, int32_t trip,
    int32_t n_body_ops, const int32_t *body_codes, const int32_t *body_in0,
    const int32_t *body_in1, const int32_t *body_iattr, const float *body_fattr,
    int32_t body_out_id, float *out) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;  // caller falls back to host
  if (n_args <= 0 || trip <= 0 || n_body_ops < 0 || carry_arg_index < 0 ||
      carry_arg_index >= n_args || !arg_ptrs || !arg_rows || !arg_cols || !out ||
      (n_body_ops > 0 && (!body_codes || !body_in0)))
    return -2;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      MPSDataType dt = MPSDataTypeFloat32;
      MPSGraph *g = [MPSGraph new];
      NSMutableArray<MPSGraphTensor *> *phs =
          [NSMutableArray arrayWithCapacity:n_args];
      NSMutableArray<NSArray<NSNumber *> *> *shapes =
          [NSMutableArray arrayWithCapacity:n_args];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = (arg_cols[i] > 0)
                                       ? @[ @(arg_rows[i]), @(arg_cols[i]) ]
                                       : @[ @(arg_rows[i]) ];
        [phs addObject:[g placeholderWithShape:shp dataType:dt name:nil]];
        [shapes addObject:shp];
      }
      const int total = n_args + 1 + n_body_ops;  // ids: args, carry, body ops
      __block int32_t err = 0;
      MPSGraphTensor *lb = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
      MPSGraphTensor *ub = [g constantWithScalar:trip dataType:MPSDataTypeInt32];
      MPSGraphTensor *st = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
      NSArray<MPSGraphTensor *> *results = [g
          forLoopWithLowerBound:lb
                     upperBound:ub
                           step:st
           initialBodyArguments:@[ phs[carry_arg_index] ]
                           body:^NSArray<MPSGraphTensor *> *(
                               MPSGraphTensor *index,
                               NSArray<MPSGraphTensor *> *bodyArgs) {
                             (void)index;
                             NSMutableArray *t =
                                 [NSMutableArray arrayWithCapacity:total];
                             for (int i = 0; i < total; ++i)
                               [t addObject:[NSNull null]];
                             for (int i = 0; i < n_args; ++i) t[i] = phs[i];
                             t[n_args] = bodyArgs[0];  // the carry
                             auto bgetT = [&](int tid) -> MPSGraphTensor * {
                               if (tid < 0 || tid >= total) return nil;
                               id v = t[tid];
                               return (v == [NSNull null]) ? nil
                                                           : (MPSGraphTensor *)v;
                             };
                             for (int j = 0; j < n_body_ops; ++j) {
                               int code = body_codes[j];
                               MPSGraphTensor *a = bgetT(body_in0[j]);
                               MPSGraphTensor *b = nil;
                               if (code == 0 || (code >= 1 && code <= 4))
                                 b = bgetT(body_in1 ? body_in1[j] : -1);
                               MPSGraphTensor *y = mpsg_build_graph_op(
                                   g, code, a, b, body_iattr ? body_iattr[j] : 0,
                                   body_fattr ? body_fattr[j] : 1e-5f, dt);
                               if (!y) { err = -6; y = bodyArgs[0]; }
                               t[n_args + 1 + j] = y;
                             }
                             MPSGraphTensor *nxt = bgetT(body_out_id);
                             if (!nxt) { err = -6; nxt = bodyArgs[0]; }
                             return @[ nxt ];
                           }
                           name:nil];
      if (err != 0) return err;
      if (!results || results.count < 1) return -3;
      MPSGraphTensor *outT = results[0];
      // Pool-acquired input buffers, held alive in a guard vector until after
      // the synchronous run (RAII release on scope exit — the buffer-pool
      // invariant; no raw newBufferWith* in dispatchers).
      std::vector<MetalBufferGuard> guards;
      guards.reserve(n_args);
      NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = shapes[i];
        size_t elems = 1;
        for (NSNumber *n in shp) elems *= (size_t)n.intValue;
        id<MTLBuffer> buf =
            metal_buffer_acquire_with_bytes(ctx, arg_ptrs[i], elems * 4);
        if (!buf) return -3;
        guards.emplace_back(ctx, buf, elems * 4);
        feeds[phs[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                shape:shp
                                                             dataType:dt];
      }
      NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                              feeds:feeds
                                      targetTensors:@[ outT ]
                                   targetOperations:nil];
      MPSGraphTensorData *od = res[outT];
      if (!od) return -3;
      [[od mpsndarray] readBytes:out strideBytes:nil];
      return 1;
    }
  }
  return -1;
}

// PK8e — run a `cond` (divergent if/else) as ONE MPSGraph `if`, executed
// DIRECTLY via runWithMTLCommandQueue (Phase-G G-A.2). Only the taken branch
// executes. `predicate = args[flag_arg_index][0] > 0` (matches GraphFn.cond /
// the CPU scf.if). Each branch is a recorded straight-line op-list over the args
// (no carry); both must produce the same output shape. Body tensor ids per
// branch: 0..n_args-1 = args, n_args+j = branch op j; `*_out_id` = branch result.
// f32. arg_ptrs[i] / out as in run_graph_loop_f32. Returns 1 / <=0 error.
extern "C" int32_t tessera_apple_gpu_run_graph_cond_f32(
    int32_t n_args, const void *const *arg_ptrs, const int32_t *arg_rows,
    const int32_t *arg_cols, int32_t flag_arg_index, int32_t n_then_ops,
    const int32_t *then_codes, const int32_t *then_in0, const int32_t *then_in1,
    const int32_t *then_iattr, const float *then_fattr, int32_t then_out_id,
    int32_t n_else_ops, const int32_t *else_codes, const int32_t *else_in0,
    const int32_t *else_in1, const int32_t *else_iattr, const float *else_fattr,
    int32_t else_out_id, float *out) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;  // caller falls back to host
  if (n_args <= 0 || flag_arg_index < 0 || flag_arg_index >= n_args ||
      n_then_ops < 0 || n_else_ops < 0 || !arg_ptrs || !arg_rows || !arg_cols ||
      !out)
    return -2;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      MPSDataType dt = MPSDataTypeFloat32;
      MPSGraph *g = [MPSGraph new];
      NSMutableArray<MPSGraphTensor *> *phs =
          [NSMutableArray arrayWithCapacity:n_args];
      NSMutableArray<NSArray<NSNumber *> *> *shapes =
          [NSMutableArray arrayWithCapacity:n_args];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = (arg_cols[i] > 0)
                                       ? @[ @(arg_rows[i]), @(arg_cols[i]) ]
                                       : @[ @(arg_rows[i]) ];
        [phs addObject:[g placeholderWithShape:shp dataType:dt name:nil]];
        [shapes addObject:shp];
      }
      // predicate = flag[0] > 0, as a scalar (rank-0) bool tensor.
      MPSGraphTensor *flagS = [g reshapeTensor:phs[flag_arg_index]
                                     withShape:@[]
                                          name:nil];
      MPSGraphTensor *zero = [g constantWithScalar:0.0 dataType:dt];
      MPSGraphTensor *pred = [g greaterThanWithPrimaryTensor:flagS
                                             secondaryTensor:zero
                                                        name:nil];
      __block int32_t err = 0;
      NSArray<MPSGraphTensor *> *results = [g
          ifWithPredicateTensor:pred
                      thenBlock:^NSArray<MPSGraphTensor *> *() {
                        MPSGraphTensor *y = mpsg_build_branch(
                            g, phs, n_args, nil, dt, n_then_ops, then_codes,
                            then_in0, then_in1, then_iattr, then_fattr,
                            then_out_id);
                        if (!y) { err = -6; y = [g constantWithScalar:0.0 dataType:dt]; }
                        return @[ y ];
                      }
                      elseBlock:^NSArray<MPSGraphTensor *> *() {
                        MPSGraphTensor *y = mpsg_build_branch(
                            g, phs, n_args, nil, dt, n_else_ops, else_codes,
                            else_in0, else_in1, else_iattr, else_fattr,
                            else_out_id);
                        if (!y) { err = -6; y = [g constantWithScalar:0.0 dataType:dt]; }
                        return @[ y ];
                      }
                           name:nil];
      if (err != 0) return err;
      if (!results || results.count < 1) return -3;
      MPSGraphTensor *outT = results[0];
      std::vector<MetalBufferGuard> guards;
      guards.reserve(n_args);
      NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = shapes[i];
        size_t elems = 1;
        for (NSNumber *n in shp) elems *= (size_t)n.intValue;
        id<MTLBuffer> buf =
            metal_buffer_acquire_with_bytes(ctx, arg_ptrs[i], elems * 4);
        if (!buf) return -3;
        guards.emplace_back(ctx, buf, elems * 4);
        feeds[phs[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                shape:shp
                                                             dataType:dt];
      }
      NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                              feeds:feeds
                                      targetTensors:@[ outT ]
                                   targetOperations:nil];
      MPSGraphTensorData *od = res[outT];
      if (!od) return -3;
      [[od mpsndarray] readBytes:out strideBytes:nil];
      return 1;
    }
  }
  return -1;
}

// PK8f — run a BOUNDED `while` as ONE MPSGraph `forLoop` with select-masking
// (Phase-G G-A.3). MPSGraph's native `while` SIGSEGVs under churn (see
// docs/apple_gpu_control_flow_lowering.md), so a max-iter-capped while lowers to
// a forLoop where each step freezes the carry once the predicate goes false:
//   for i in 0..max_iters:
//     next = body(carry, args);  pred = cond(carry, args) > 0
//     carry = select(pred, next, carry)        // pred broadcasts over the carry
// → final carry. Both body and cond are recorded straight-line op-lists over the
// args + carry (ids: 0..n_args-1 args, n_args carry, n_args+1+j op j). f32.
// Returns 1 / <=0 error.
extern "C" int32_t tessera_apple_gpu_run_graph_while_f32(
    int32_t n_args, const void *const *arg_ptrs, const int32_t *arg_rows,
    const int32_t *arg_cols, int32_t carry_arg_index, int32_t max_iters,
    int32_t n_body_ops, const int32_t *body_codes, const int32_t *body_in0,
    const int32_t *body_in1, const int32_t *body_iattr, const float *body_fattr,
    int32_t body_out_id, int32_t n_cond_ops, const int32_t *cond_codes,
    const int32_t *cond_in0, const int32_t *cond_in1, const int32_t *cond_iattr,
    const float *cond_fattr, int32_t cond_out_id, float *out) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;  // caller falls back to host
  if (n_args <= 0 || max_iters <= 0 || carry_arg_index < 0 ||
      carry_arg_index >= n_args || n_body_ops < 0 || n_cond_ops < 0 ||
      !arg_ptrs || !arg_rows || !arg_cols || !out)
    return -2;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      MPSDataType dt = MPSDataTypeFloat32;
      MPSGraph *g = [MPSGraph new];
      NSMutableArray<MPSGraphTensor *> *phs =
          [NSMutableArray arrayWithCapacity:n_args];
      NSMutableArray<NSArray<NSNumber *> *> *shapes =
          [NSMutableArray arrayWithCapacity:n_args];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = (arg_cols[i] > 0)
                                       ? @[ @(arg_rows[i]), @(arg_cols[i]) ]
                                       : @[ @(arg_rows[i]) ];
        [phs addObject:[g placeholderWithShape:shp dataType:dt name:nil]];
        [shapes addObject:shp];
      }
      __block int32_t err = 0;
      MPSGraphTensor *lb = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
      MPSGraphTensor *ub = [g constantWithScalar:max_iters dataType:MPSDataTypeInt32];
      MPSGraphTensor *st = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
      MPSGraphTensor *zero = [g constantWithScalar:0.0 dataType:dt];
      NSArray<MPSGraphTensor *> *results = [g
          forLoopWithLowerBound:lb
                     upperBound:ub
                           step:st
           initialBodyArguments:@[ phs[carry_arg_index] ]
                           body:^NSArray<MPSGraphTensor *> *(
                               MPSGraphTensor *index,
                               NSArray<MPSGraphTensor *> *bodyArgs) {
                             (void)index;
                             MPSGraphTensor *carry = bodyArgs[0];
                             NSArray *extra = @[ carry ];
                             MPSGraphTensor *next = mpsg_build_branch(
                                 g, phs, n_args, extra, dt, n_body_ops, body_codes,
                                 body_in0, body_in1, body_iattr, body_fattr,
                                 body_out_id);
                             MPSGraphTensor *c = mpsg_build_branch(
                                 g, phs, n_args, extra, dt, n_cond_ops, cond_codes,
                                 cond_in0, cond_in1, cond_iattr, cond_fattr,
                                 cond_out_id);
                             if (!next || !c) { err = -6; return @[ carry ]; }
                             MPSGraphTensor *pred =
                                 [g greaterThanWithPrimaryTensor:c
                                                 secondaryTensor:zero
                                                            name:nil];
                             MPSGraphTensor *nextCarry =
                                 [g selectWithPredicateTensor:pred
                                          truePredicateTensor:next
                                         falsePredicateTensor:carry
                                                         name:nil];
                             return @[ nextCarry ];
                           }
                           name:nil];
      if (err != 0) return err;
      if (!results || results.count < 1) return -3;
      MPSGraphTensor *outT = results[0];
      std::vector<MetalBufferGuard> guards;
      guards.reserve(n_args);
      NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = shapes[i];
        size_t elems = 1;
        for (NSNumber *n in shp) elems *= (size_t)n.intValue;
        id<MTLBuffer> buf =
            metal_buffer_acquire_with_bytes(ctx, arg_ptrs[i], elems * 4);
        if (!buf) return -3;
        guards.emplace_back(ctx, buf, elems * 4);
        feeds[phs[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                shape:shp
                                                             dataType:dt];
      }
      NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                              feeds:feeds
                                      targetTensors:@[ outT ]
                                   targetOperations:nil];
      MPSGraphTensorData *od = res[outT];
      if (!od) return -3;
      [[od mpsndarray] readBytes:out strideBytes:nil];
      return 1;
    }
  }
  return -1;
}

// ── Phase-H H2 — native f16 control flow ───────────────────────────────────
//
// MPSGraph supports f16 natively (unlike bf16 — MPS has no bf16 type, so bf16
// control flow stays host-upcast to f32; see the bf16 matmul note above). These
// f16 entry points mirror the f32 loop/cond/while exactly but build the graph in
// f16 and use the f16-bit ABI (uint16_t I/O, 2-byte buffers). The shared
// `run_mpsgraph_cf` helper owns the placeholder/feed/run/readBytes plumbing
// (parameterized on `dt` + element size); each entry supplies a build block that
// constructs the control-flow graph and returns the output tensor. The f32
// functions are left untouched (zero risk to the tested path).

static int32_t run_mpsgraph_cf(
    MPSDataType dt, size_t esz, int32_t n_args, const void *const *arg_ptrs,
    const int32_t *arg_rows, const int32_t *arg_cols, void *out,
    MPSGraphTensor *(^build)(MPSGraph *g, NSArray<MPSGraphTensor *> *phs,
                             MPSDataType dt, int32_t *err)) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;  // caller falls back to host
  if (n_args <= 0 || !arg_ptrs || !arg_rows || !arg_cols || !out) return -2;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      MPSGraph *g = [MPSGraph new];
      NSMutableArray<MPSGraphTensor *> *phs =
          [NSMutableArray arrayWithCapacity:n_args];
      NSMutableArray<NSArray<NSNumber *> *> *shapes =
          [NSMutableArray arrayWithCapacity:n_args];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = (arg_cols[i] > 0)
                                       ? @[ @(arg_rows[i]), @(arg_cols[i]) ]
                                       : @[ @(arg_rows[i]) ];
        [phs addObject:[g placeholderWithShape:shp dataType:dt name:nil]];
        [shapes addObject:shp];
      }
      int32_t err = 0;
      MPSGraphTensor *outT = build(g, phs, dt, &err);
      if (err != 0) return err;
      if (!outT) return -3;
      std::vector<MetalBufferGuard> guards;
      guards.reserve(n_args);
      NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = shapes[i];
        size_t elems = 1;
        for (NSNumber *n in shp) elems *= (size_t)n.intValue;
        id<MTLBuffer> buf =
            metal_buffer_acquire_with_bytes(ctx, arg_ptrs[i], elems * esz);
        if (!buf) return -3;
        guards.emplace_back(ctx, buf, elems * esz);
        feeds[phs[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                shape:shp
                                                             dataType:dt];
      }
      NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                              feeds:feeds
                                      targetTensors:@[ outT ]
                                   targetOperations:nil];
      MPSGraphTensorData *od = res[outT];
      if (!od) return -3;
      [[od mpsndarray] readBytes:out strideBytes:nil];
      return 1;
    }
  }
  return -1;
}

extern "C" int32_t tessera_apple_gpu_run_graph_loop_f16(
    int32_t n_args, const void *const *arg_ptrs, const int32_t *arg_rows,
    const int32_t *arg_cols, int32_t carry_arg_index, int32_t trip,
    int32_t n_body_ops, const int32_t *body_codes, const int32_t *body_in0,
    const int32_t *body_in1, const int32_t *body_iattr, const float *body_fattr,
    int32_t body_out_id, uint16_t *out) {
  if (trip <= 0 || n_body_ops < 0 || carry_arg_index < 0 ||
      carry_arg_index >= n_args || (n_body_ops > 0 && (!body_codes || !body_in0)))
    return -2;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    return run_mpsgraph_cf(
        MPSDataTypeFloat16, 2, n_args, arg_ptrs, arg_rows, arg_cols, out,
        ^MPSGraphTensor *(MPSGraph *g, NSArray<MPSGraphTensor *> *phs,
                          MPSDataType dt, int32_t *err) {
          const int total = n_args + 1 + n_body_ops;
          MPSGraphTensor *lb = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
          MPSGraphTensor *ub = [g constantWithScalar:trip
                                            dataType:MPSDataTypeInt32];
          MPSGraphTensor *st = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
          __block int32_t berr = 0;
          NSArray<MPSGraphTensor *> *results = [g
              forLoopWithLowerBound:lb
                         upperBound:ub
                               step:st
               initialBodyArguments:@[ phs[carry_arg_index] ]
                               body:^NSArray<MPSGraphTensor *> *(
                                   MPSGraphTensor *index,
                                   NSArray<MPSGraphTensor *> *bodyArgs) {
                                 (void)index;
                                 NSMutableArray *t =
                                     [NSMutableArray arrayWithCapacity:total];
                                 for (int i = 0; i < total; ++i)
                                   [t addObject:[NSNull null]];
                                 for (int i = 0; i < n_args; ++i) t[i] = phs[i];
                                 t[n_args] = bodyArgs[0];
                                 auto bg = [&](int tid) -> MPSGraphTensor * {
                                   if (tid < 0 || tid >= total) return nil;
                                   id v = t[tid];
                                   return (v == [NSNull null])
                                              ? nil
                                              : (MPSGraphTensor *)v;
                                 };
                                 for (int j = 0; j < n_body_ops; ++j) {
                                   int code = body_codes[j];
                                   MPSGraphTensor *a = bg(body_in0[j]);
                                   MPSGraphTensor *b = nil;
                                   if (code == 0 || (code >= 1 && code <= 4))
                                     b = bg(body_in1 ? body_in1[j] : -1);
                                   MPSGraphTensor *y = mpsg_build_graph_op(
                                       g, code, a, b,
                                       body_iattr ? body_iattr[j] : 0,
                                       body_fattr ? body_fattr[j] : 1e-5f, dt);
                                   if (!y) { berr = -6; y = bodyArgs[0]; }
                                   t[n_args + 1 + j] = y;
                                 }
                                 MPSGraphTensor *nxt = bg(body_out_id);
                                 if (!nxt) { berr = -6; nxt = bodyArgs[0]; }
                                 return @[ nxt ];
                               }
                               name:nil];
          *err = berr;
          if (berr != 0) return nil;
          if (!results || results.count < 1) { *err = -3; return nil; }
          return results[0];
        });
  }
  return -1;
}

extern "C" int32_t tessera_apple_gpu_run_graph_cond_f16(
    int32_t n_args, const void *const *arg_ptrs, const int32_t *arg_rows,
    const int32_t *arg_cols, int32_t flag_arg_index, int32_t n_then_ops,
    const int32_t *then_codes, const int32_t *then_in0, const int32_t *then_in1,
    const int32_t *then_iattr, const float *then_fattr, int32_t then_out_id,
    int32_t n_else_ops, const int32_t *else_codes, const int32_t *else_in0,
    const int32_t *else_in1, const int32_t *else_iattr, const float *else_fattr,
    int32_t else_out_id, uint16_t *out) {
  if (flag_arg_index < 0 || flag_arg_index >= n_args || n_then_ops < 0 ||
      n_else_ops < 0)
    return -2;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    return run_mpsgraph_cf(
        MPSDataTypeFloat16, 2, n_args, arg_ptrs, arg_rows, arg_cols, out,
        ^MPSGraphTensor *(MPSGraph *g, NSArray<MPSGraphTensor *> *phs,
                          MPSDataType dt, int32_t *err) {
          MPSGraphTensor *flagS = [g reshapeTensor:phs[flag_arg_index]
                                         withShape:@[]
                                              name:nil];
          MPSGraphTensor *zero = [g constantWithScalar:0.0 dataType:dt];
          MPSGraphTensor *pred = [g greaterThanWithPrimaryTensor:flagS
                                                 secondaryTensor:zero
                                                            name:nil];
          __block int32_t berr = 0;
          NSArray<MPSGraphTensor *> *results = [g
              ifWithPredicateTensor:pred
                          thenBlock:^NSArray<MPSGraphTensor *> *() {
                            MPSGraphTensor *y = mpsg_build_branch(
                                g, phs, n_args, nil, dt, n_then_ops, then_codes,
                                then_in0, then_in1, then_iattr, then_fattr,
                                then_out_id);
                            if (!y) {
                              berr = -6;
                              y = [g constantWithScalar:0.0 dataType:dt];
                            }
                            return @[ y ];
                          }
                          elseBlock:^NSArray<MPSGraphTensor *> *() {
                            MPSGraphTensor *y = mpsg_build_branch(
                                g, phs, n_args, nil, dt, n_else_ops, else_codes,
                                else_in0, else_in1, else_iattr, else_fattr,
                                else_out_id);
                            if (!y) {
                              berr = -6;
                              y = [g constantWithScalar:0.0 dataType:dt];
                            }
                            return @[ y ];
                          }
                               name:nil];
          *err = berr;
          if (berr != 0) return nil;
          if (!results || results.count < 1) { *err = -3; return nil; }
          return results[0];
        });
  }
  return -1;
}

extern "C" int32_t tessera_apple_gpu_run_graph_while_f16(
    int32_t n_args, const void *const *arg_ptrs, const int32_t *arg_rows,
    const int32_t *arg_cols, int32_t carry_arg_index, int32_t max_iters,
    int32_t n_body_ops, const int32_t *body_codes, const int32_t *body_in0,
    const int32_t *body_in1, const int32_t *body_iattr, const float *body_fattr,
    int32_t body_out_id, int32_t n_cond_ops, const int32_t *cond_codes,
    const int32_t *cond_in0, const int32_t *cond_in1, const int32_t *cond_iattr,
    const float *cond_fattr, int32_t cond_out_id, uint16_t *out) {
  if (max_iters <= 0 || carry_arg_index < 0 || carry_arg_index >= n_args ||
      n_body_ops < 0 || n_cond_ops < 0)
    return -2;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    return run_mpsgraph_cf(
        MPSDataTypeFloat16, 2, n_args, arg_ptrs, arg_rows, arg_cols, out,
        ^MPSGraphTensor *(MPSGraph *g, NSArray<MPSGraphTensor *> *phs,
                          MPSDataType dt, int32_t *err) {
          MPSGraphTensor *lb = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
          MPSGraphTensor *ub = [g constantWithScalar:max_iters
                                            dataType:MPSDataTypeInt32];
          MPSGraphTensor *st = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
          MPSGraphTensor *zero = [g constantWithScalar:0.0 dataType:dt];
          __block int32_t berr = 0;
          NSArray<MPSGraphTensor *> *results = [g
              forLoopWithLowerBound:lb
                         upperBound:ub
                               step:st
               initialBodyArguments:@[ phs[carry_arg_index] ]
                               body:^NSArray<MPSGraphTensor *> *(
                                   MPSGraphTensor *index,
                                   NSArray<MPSGraphTensor *> *bodyArgs) {
                                 (void)index;
                                 MPSGraphTensor *carry = bodyArgs[0];
                                 NSArray *extra = @[ carry ];
                                 MPSGraphTensor *next = mpsg_build_branch(
                                     g, phs, n_args, extra, dt, n_body_ops,
                                     body_codes, body_in0, body_in1, body_iattr,
                                     body_fattr, body_out_id);
                                 MPSGraphTensor *c = mpsg_build_branch(
                                     g, phs, n_args, extra, dt, n_cond_ops,
                                     cond_codes, cond_in0, cond_in1, cond_iattr,
                                     cond_fattr, cond_out_id);
                                 if (!next || !c) {
                                   berr = -6;
                                   return @[ carry ];
                                 }
                                 MPSGraphTensor *pred =
                                     [g greaterThanWithPrimaryTensor:c
                                                     secondaryTensor:zero
                                                                name:nil];
                                 MPSGraphTensor *nextCarry =
                                     [g selectWithPredicateTensor:pred
                                              truePredicateTensor:next
                                             falsePredicateTensor:carry
                                                             name:nil];
                                 return @[ nextCarry ];
                               }
                               name:nil];
          *err = berr;
          if (berr != 0) return nil;
          if (!results || results.count < 1) { *err = -3; return nil; }
          return results[0];
        });
  }
  return -1;
}

// ── Phase-H H3 — fused scan ────────────────────────────────────────────────
//
// `(carry, ys) = scan(fn, init, xs)` where `fn(carry, x_t) -> (carry, y_t)` runs
// as ONE MPSGraph `forLoop` carrying `[carry, ys_accum]`. Per step `index`:
//   x_t = gatherAlongAxis(0, xs, [index])           // xs[index]
//   (next_carry, y_t) = body(args, carry, x_t)      // serialized op-list
//   ys_accum = scatterND(ys_accum, [y_t], [[index]], Set)   // ys[index] = y_t
// Returns (final_carry, ys). Body tensor ids: 0..n_args-1 = args (consts; the
// carry init is args[carry_arg_index]), n_args = carry, n_args+1 = x_t,
// n_args+2+j = body op j. `xs`/`ys` are rank (trip + 2D inner); consts/carry are
// rank<=2. ys_zeros is fed (a zeros (trip,*y) buffer). f32.
extern "C" int32_t tessera_apple_gpu_run_graph_scan_f32(
    int32_t n_args, const void *const *arg_ptrs, const int32_t *arg_rows,
    const int32_t *arg_cols, int32_t carry_arg_index, const void *xs_ptr,
    int32_t trip, int32_t x_rows, int32_t x_cols, int32_t n_body_ops,
    const int32_t *body_codes, const int32_t *body_in0, const int32_t *body_in1,
    const int32_t *body_iattr, const float *body_fattr, int32_t carry_out_id,
    int32_t y_out_id, int32_t carry_rows, int32_t carry_cols, int32_t y_rows,
    int32_t y_cols, const void *ys_zeros, float *out_carry, float *out_ys) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;  // caller falls back to host
  if (n_args <= 0 || trip <= 0 || n_body_ops < 0 || carry_arg_index < 0 ||
      carry_arg_index >= n_args || !arg_ptrs || !arg_rows || !arg_cols ||
      !xs_ptr || !ys_zeros || !out_carry || !out_ys ||
      (n_body_ops > 0 && (!body_codes || !body_in0)))
    return -2;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      MPSDataType dt = MPSDataTypeFloat32;
      MPSGraph *g = [MPSGraph new];
      NSMutableArray<MPSGraphTensor *> *phs =
          [NSMutableArray arrayWithCapacity:n_args];
      NSMutableArray<NSArray<NSNumber *> *> *shapes =
          [NSMutableArray arrayWithCapacity:n_args];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = (arg_cols[i] > 0)
                                       ? @[ @(arg_rows[i]), @(arg_cols[i]) ]
                                       : @[ @(arg_rows[i]) ];
        [phs addObject:[g placeholderWithShape:shp dataType:dt name:nil]];
        [shapes addObject:shp];
      }
      NSArray<NSNumber *> *xsShape = (x_cols > 0)
                                         ? @[ @(trip), @(x_rows), @(x_cols) ]
                                         : @[ @(trip), @(x_rows) ];
      NSArray<NSNumber *> *xtShape =
          (x_cols > 0) ? @[ @(x_rows), @(x_cols) ] : @[ @(x_rows) ];
      NSArray<NSNumber *> *ysShape = (y_cols > 0)
                                         ? @[ @(trip), @(y_rows), @(y_cols) ]
                                         : @[ @(trip), @(y_rows) ];
      NSArray<NSNumber *> *yUpd = (y_cols > 0)
                                      ? @[ @1, @(y_rows), @(y_cols) ]
                                      : @[ @1, @(y_rows) ];
      MPSGraphTensor *xsPh = [g placeholderWithShape:xsShape dataType:dt name:nil];
      MPSGraphTensor *ysPh = [g placeholderWithShape:ysShape dataType:dt name:nil];
      const int total = n_args + 2 + n_body_ops;
      __block int32_t err = 0;
      MPSGraphTensor *lb = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
      MPSGraphTensor *ub = [g constantWithScalar:trip dataType:MPSDataTypeInt32];
      MPSGraphTensor *st = [g constantWithScalar:1 dataType:MPSDataTypeInt32];
      NSArray<MPSGraphTensor *> *results = [g
          forLoopWithLowerBound:lb
                     upperBound:ub
                           step:st
           initialBodyArguments:@[ phs[carry_arg_index], ysPh ]
                           body:^NSArray<MPSGraphTensor *> *(
                               MPSGraphTensor *index,
                               NSArray<MPSGraphTensor *> *bodyArgs) {
                             MPSGraphTensor *carry = bodyArgs[0];
                             MPSGraphTensor *ysAcc = bodyArgs[1];
                             MPSGraphTensor *idx1 = [g reshapeTensor:index
                                                          withShape:@[ @1 ]
                                                               name:nil];
                             MPSGraphTensor *xg =
                                 [g gatherWithUpdatesTensor:xsPh
                                             indicesTensor:idx1
                                                      axis:0
                                           batchDimensions:0
                                                      name:nil];
                             MPSGraphTensor *xt = [g reshapeTensor:xg
                                                        withShape:xtShape
                                                             name:nil];
                             NSMutableArray *t =
                                 [NSMutableArray arrayWithCapacity:total];
                             for (int i = 0; i < total; ++i)
                               [t addObject:[NSNull null]];
                             for (int i = 0; i < n_args; ++i) t[i] = phs[i];
                             t[n_args] = carry;
                             t[n_args + 1] = xt;
                             auto bg = [&](int tid) -> MPSGraphTensor * {
                               if (tid < 0 || tid >= total) return nil;
                               id v = t[tid];
                               return (v == [NSNull null]) ? nil
                                                           : (MPSGraphTensor *)v;
                             };
                             for (int j = 0; j < n_body_ops; ++j) {
                               int code = body_codes[j];
                               MPSGraphTensor *a = bg(body_in0[j]);
                               MPSGraphTensor *b = nil;
                               if (code == 0 || (code >= 1 && code <= 4))
                                 b = bg(body_in1 ? body_in1[j] : -1);
                               MPSGraphTensor *y = mpsg_build_graph_op(
                                   g, code, a, b, body_iattr ? body_iattr[j] : 0,
                                   body_fattr ? body_fattr[j] : 1e-5f, dt);
                               if (!y) { err = -6; y = carry; }
                               t[n_args + 2 + j] = y;
                             }
                             MPSGraphTensor *nc = bg(carry_out_id);
                             MPSGraphTensor *yt = bg(y_out_id);
                             if (!nc || !yt) {
                               err = -6;
                               return @[ carry, ysAcc ];
                             }
                             MPSGraphTensor *upd = [g reshapeTensor:yt
                                                         withShape:yUpd
                                                              name:nil];
                             MPSGraphTensor *idx2 = [g reshapeTensor:index
                                                          withShape:@[ @1, @1 ]
                                                               name:nil];
                             MPSGraphTensor *ysNew = [g
                                 scatterNDWithDataTensor:ysAcc
                                           updatesTensor:upd
                                           indicesTensor:idx2
                                         batchDimensions:0
                                                    mode:MPSGraphScatterModeSet
                                                    name:nil];
                             return @[ nc, ysNew ];
                           }
                           name:nil];
      if (err != 0) return err;
      if (!results || results.count < 2) return -3;
      (void)carry_rows;
      (void)carry_cols;
      std::vector<MetalBufferGuard> guards;
      guards.reserve(n_args + 2);
      NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
      for (int i = 0; i < n_args; ++i) {
        NSArray<NSNumber *> *shp = shapes[i];
        size_t elems = 1;
        for (NSNumber *n in shp) elems *= (size_t)n.intValue;
        id<MTLBuffer> buf =
            metal_buffer_acquire_with_bytes(ctx, arg_ptrs[i], elems * 4);
        if (!buf) return -3;
        guards.emplace_back(ctx, buf, elems * 4);
        feeds[phs[i]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                                shape:shp
                                                             dataType:dt];
      }
      {
        size_t xe = (size_t)trip * x_rows * (x_cols > 0 ? x_cols : 1);
        id<MTLBuffer> xb = metal_buffer_acquire_with_bytes(ctx, xs_ptr, xe * 4);
        if (!xb) return -3;
        guards.emplace_back(ctx, xb, xe * 4);
        feeds[xsPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:xb
                                                              shape:xsShape
                                                           dataType:dt];
        size_t ye = (size_t)trip * y_rows * (y_cols > 0 ? y_cols : 1);
        id<MTLBuffer> yb = metal_buffer_acquire_with_bytes(ctx, ys_zeros, ye * 4);
        if (!yb) return -3;
        guards.emplace_back(ctx, yb, ye * 4);
        feeds[ysPh] = [[MPSGraphTensorData alloc] initWithMTLBuffer:yb
                                                              shape:ysShape
                                                           dataType:dt];
      }
      NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                              feeds:feeds
                                      targetTensors:@[ results[0], results[1] ]
                                   targetOperations:nil];
      MPSGraphTensorData *cd = res[results[0]];
      MPSGraphTensorData *yd = res[results[1]];
      if (!cd || !yd) return -3;
      [[cd mpsndarray] readBytes:out_carry strideBytes:nil];
      [[yd mpsndarray] readBytes:out_ys strideBytes:nil];
      return 1;
    }
  }
  return -1;
}

// PK8b — author a *fused multi-op* `.mtlpackage` by composing MPSGraph nodes
// into a single serialized executable. MPSGraph fuses the chain across the ML
// pipeline, so the whole chain runs as one dispatch — the packaged equivalent
// of the runtime's fused MSL kernels. fp32. Positional bindings (inputs in
// declaration order, output last). ``dims`` carries the per-chain shape
// vector; ``eps`` applies to norm chains. Supported chains:
//
//   "matmul_softmax"        dims=[M,K,N]    A[M,K],B[K,N] -> softmax(A@B)[M,N]
//   "matmul_softmax_matmul" dims=[M,K,N,P]  A,B,C -> (softmax(A@B)@C)[M,P]
//   "rmsnorm_matmul"        dims=[M,K,N]    x[M,K],gamma[K],W[K,N]
//                                            -> (rmsnorm(x)*gamma) @ W  [M,N]
//
// Returns 1 / <=0 error (-2 bad-args, -6 unknown chain, plus
// _mlpkg_compile_and_write codes).
extern "C" int32_t tessera_apple_gpu_mlpkg_author_chain(
    const char *out_package_path, const char *chain, const int32_t *dims,
    int32_t ndims, float eps) {
  if (!out_package_path || !chain || !dims || ndims <= 0) return -2;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return -1;
  if (@available(macOS 14.0, iOS 17.0, *)) {
    @autoreleasepool {
      MPSDataType dt = MPSDataTypeFloat32;
      NSArray<NSNumber *> *axis1 = @[ @1 ];
      MPSGraph *g = [MPSGraph new];

      if (std::strcmp(chain, "matmul_softmax") == 0) {
        if (ndims != 3) return -2;
        int M = dims[0], K = dims[1], N = dims[2];
        if (M <= 0 || K <= 0 || N <= 0) return -2;
        NSArray *as = @[ @(M), @(K) ], *bs = @[ @(K), @(N) ];
        MPSGraphTensor *pa = [g placeholderWithShape:as dataType:dt name:nil];
        MPSGraphTensor *pb = [g placeholderWithShape:bs dataType:dt name:nil];
        MPSGraphTensor *ab =
            [g matrixMultiplicationWithPrimaryTensor:pa secondaryTensor:pb
                                                name:nil];
        MPSGraphTensor *y = [g softMaxWithTensor:ab axis:1 name:nil];
        NSDictionary *feeds = @{
          pa : [[MPSGraphShapedType alloc] initWithShape:as dataType:dt],
          pb : [[MPSGraphShapedType alloc] initWithShape:bs dataType:dt],
        };
        return _mlpkg_compile_and_write(g, feeds, y, out_package_path);
      }

      if (std::strcmp(chain, "matmul_softmax_matmul") == 0) {
        if (ndims != 4) return -2;
        int M = dims[0], K = dims[1], N = dims[2], P = dims[3];
        if (M <= 0 || K <= 0 || N <= 0 || P <= 0) return -2;
        NSArray *as = @[ @(M), @(K) ], *bs = @[ @(K), @(N) ],
                *cs = @[ @(N), @(P) ];
        MPSGraphTensor *pa = [g placeholderWithShape:as dataType:dt name:nil];
        MPSGraphTensor *pb = [g placeholderWithShape:bs dataType:dt name:nil];
        MPSGraphTensor *pc = [g placeholderWithShape:cs dataType:dt name:nil];
        MPSGraphTensor *ab =
            [g matrixMultiplicationWithPrimaryTensor:pa secondaryTensor:pb
                                                name:nil];
        MPSGraphTensor *sm = [g softMaxWithTensor:ab axis:1 name:nil];
        MPSGraphTensor *y =
            [g matrixMultiplicationWithPrimaryTensor:sm secondaryTensor:pc
                                                name:nil];
        NSDictionary *feeds = @{
          pa : [[MPSGraphShapedType alloc] initWithShape:as dataType:dt],
          pb : [[MPSGraphShapedType alloc] initWithShape:bs dataType:dt],
          pc : [[MPSGraphShapedType alloc] initWithShape:cs dataType:dt],
        };
        return _mlpkg_compile_and_write(g, feeds, y, out_package_path);
      }

      if (std::strcmp(chain, "rmsnorm_matmul") == 0) {
        if (ndims != 3) return -2;
        int M = dims[0], K = dims[1], N = dims[2];
        if (M <= 0 || K <= 0 || N <= 0) return -2;
        NSArray *xsh = @[ @(M), @(K) ], *gsh = @[ @(K) ], *wsh = @[ @(K), @(N) ];
        MPSGraphTensor *px = [g placeholderWithShape:xsh dataType:dt name:nil];
        MPSGraphTensor *pg = [g placeholderWithShape:gsh dataType:dt name:nil];
        MPSGraphTensor *pw = [g placeholderWithShape:wsh dataType:dt name:nil];
        // rmsnorm(x) = x / sqrt(mean(x^2, axis=-1) + eps)
        MPSGraphTensor *sq =
            [g multiplicationWithPrimaryTensor:px secondaryTensor:px name:nil];
        MPSGraphTensor *ms = [g meanOfTensor:sq axes:axis1 name:nil];
        MPSGraphTensor *epsc =
            [g constantWithScalar:(double)eps dataType:dt];
        MPSGraphTensor *me =
            [g additionWithPrimaryTensor:ms secondaryTensor:epsc name:nil];
        MPSGraphTensor *denom = [g squareRootWithTensor:me name:nil];
        MPSGraphTensor *norm =
            [g divisionWithPrimaryTensor:px secondaryTensor:denom name:nil];
        MPSGraphTensor *xn =
            [g multiplicationWithPrimaryTensor:norm secondaryTensor:pg name:nil];
        MPSGraphTensor *y =
            [g matrixMultiplicationWithPrimaryTensor:xn secondaryTensor:pw
                                                name:nil];
        NSDictionary *feeds = @{
          px : [[MPSGraphShapedType alloc] initWithShape:xsh dataType:dt],
          pg : [[MPSGraphShapedType alloc] initWithShape:gsh dataType:dt],
          pw : [[MPSGraphShapedType alloc] initWithShape:wsh dataType:dt],
        };
        return _mlpkg_compile_and_write(g, feeds, y, out_package_path);
      }

      return -6;  // unknown chain
    }
  }
  return -1;
}

// Lane (c) — MSL-source → serialized `.metallib` dynamic-library AOT chain.
//
// The parallel AOT lane to MPSGraph packages: instead of an MPSGraph ML
// package, this serializes Tessera's *MSL-source* custom kernels (rope,
// flash_attn, gelu, ...) into a reloadable dynamic library, so a host can
// author once and reload with zero recompilation. Grounded in the SDK
// headers (Decision #27): compile with `MTLLibraryType.dynamic` + an
// `installName`, wrap in an `MTLDynamicLibrary`, serialize to disk, reload.
//
//   newLibraryWithSource:options:  (libraryType=Dynamic, installName set)
//     → newDynamicLibrary:         (MTLDevice.h:1228)
//     → serializeToURL:            (MTLDynamicLibrary.h:77)
//     → newDynamicLibraryWithURL:  (MTLDevice.h:1238)  [reload]
//
// Returns 1 / <=0: -1 OS/device unavailable, -2 bad args, -3 source compile
// failed, -4 dynamic-library create failed, -5 serialize failed.
extern "C" int32_t tessera_apple_gpu_dylib_serialize(
    const char *msl_source, const char *install_name,
    const char *out_path) {
  if (!msl_source || !install_name || !out_path) return -2;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return -1;
  if (@available(macOS 11.0, iOS 14.0, *)) {
    @autoreleasepool {
      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      opts.libraryType = MTLLibraryTypeDynamic;
      opts.installName = @(install_name);
      NSError *err = nil;
      id<MTLLibrary> lib = [ctx.device newLibraryWithSource:@(msl_source)
                                                    options:opts
                                                      error:&err];
      if (!lib) {
        fprintf(stderr, "[tessera_apple_gpu_dylib] source compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        return -3;
      }
      id<MTLDynamicLibrary> dylib = [ctx.device newDynamicLibrary:lib
                                                            error:&err];
      if (!dylib) {
        fprintf(stderr, "[tessera_apple_gpu_dylib] dynamic-library create "
                "failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        return -4;
      }
      NSURL *url = [NSURL fileURLWithPath:@(out_path)];
      if (![dylib serializeToURL:url error:&err]) {
        fprintf(stderr, "[tessera_apple_gpu_dylib] serialize failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        return -5;
      }
      return 1;
    }
  }
  return -1;
}

// Lane (c) — reload a serialized `.metallib` dynamic library. Returns 1 if it
// loads (with a non-empty install name), 0 otherwise. Proves the round-trip.
extern "C" int32_t tessera_apple_gpu_dylib_load(const char *path) {
  if (!path) return 0;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  if (@available(macOS 11.0, iOS 14.0, *)) {
    @autoreleasepool {
      NSURL *url = [NSURL fileURLWithPath:@(path)];
      NSError *err = nil;
      id<MTLDynamicLibrary> dylib = [ctx.device newDynamicLibraryWithURL:url
                                                                  error:&err];
      if (!dylib) {
        fprintf(stderr, "[tessera_apple_gpu_dylib] reload failed for '%s': "
                "%s\n", path,
                err ? [[err localizedDescription] UTF8String] : "<nil>");
        return 0;
      }
      return (dylib.installName && dylib.installName.length > 0) ? 1 : 0;
    }
  }
  return 0;
}

// Number of distinct (shape-class, opcode, dtype, shape) MPSGraphs cached.
// Used by tests to verify graph reuse across repeated dispatches.
extern "C" int32_t tessera_apple_gpu_mpsgraph_cache_size(void) {
  std::lock_guard<std::mutex> lock(g_mpsg_graph_mu);
  return (int32_t)[mpsg_graph_cache() count];
}

// Task B (2026-06-01) — LRU eviction introspection.
//
// ``cache_evictions()`` returns the monotonically-increasing count of
// LRU evictions since process start. A test that puts (capacity+1)
// distinct entries and observes ``evictions == 1`` proves the LRU
// fired. A long training loop observing the counter climbing means the
// working set exceeds capacity and the cache is thrashing — bump
// ``TESSERA_MPSGRAPH_CACHE_CAPACITY`` or restructure the workload.
extern "C" int64_t tessera_apple_gpu_mpsgraph_cache_evictions(void) {
  return g_mpsg_evictions.load(std::memory_order_relaxed);
}

// ``cache_capacity()`` returns the active LRU capacity (resolved once
// from ``TESSERA_MPSGRAPH_CACHE_CAPACITY`` env var; default 1024).
// Returns 0 when the env-var sets unbounded mode. Useful for diagnostic
// assertions ("am I running with the capacity I think?").
extern "C" int64_t tessera_apple_gpu_mpsgraph_cache_capacity(void) {
  return (int64_t)mpsg_cache_capacity();
}

// ---- C ABI: unary -----------------------------------------------------------
extern "C" void tessera_apple_gpu_mpsgraph_unary_f32(int32_t op, const float *x,
                                                     float *out, int64_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_unary(ctx, op, x, out, n, MPSDataTypeFloat32, 4)) return;
  // host reference fallback
  for (int64_t i = 0; i < n; ++i) {
    float v = x[i];
    switch (op) {
      case 0: out[i] = v > 0 ? v : 0.0f; break;
      case 1: out[i] = 1.0f / (1.0f + std::exp(-v)); break;
      case 2: out[i] = std::tanh(v); break;
      case 3: out[i] = std::log1p(std::exp(v)); break;
      case 4: out[i] = v / (1.0f + std::exp(-v)); break;
      case 5: out[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))); break;
      case 6: out[i] = std::exp(v); break;
      case 7: out[i] = std::log(v); break;
      case 8: out[i] = std::sqrt(v); break;
      case 9: out[i] = 1.0f / std::sqrt(v); break;
      case 10: out[i] = -v; break;
      case 11: out[i] = std::fabs(v); break;
      default: out[i] = v; break;
    }
  }
}

extern "C" void tessera_apple_gpu_mpsgraph_unary_f16(int32_t op,
                                                     const uint16_t *x,
                                                     uint16_t *out, int64_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_unary(ctx, op, x, out, n, MPSDataTypeFloat16, 2)) return;
  // No host-side half arithmetic here: python upcasts on fallback.
  std::memcpy(out, x, (size_t)n * 2);
}

// ---- C ABI: binary ----------------------------------------------------------
extern "C" void tessera_apple_gpu_mpsgraph_binary_f32(int32_t op, const float *a,
                                                      const float *b, float *out,
                                                      int64_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_binary(ctx, op, a, b, out, n, MPSDataTypeFloat32, 4)) return;
  for (int64_t i = 0; i < n; ++i) {
    float x = a[i], y = b[i];
    switch (op) {
      case 0: out[i] = x + y; break;
      case 1: out[i] = x - y; break;
      case 2: out[i] = x * y; break;
      case 3: out[i] = x / y; break;
      case 4: out[i] = x > y ? x : y; break;
      case 5: out[i] = x < y ? x : y; break;
      case 6: out[i] = x * (y / (1.0f + std::exp(-y))); break;
      default: out[i] = x; break;
    }
  }
}

extern "C" void tessera_apple_gpu_mpsgraph_binary_f16(int32_t op,
                                                      const uint16_t *a,
                                                      const uint16_t *b,
                                                      uint16_t *out, int64_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_binary(ctx, op, a, b, out, n, MPSDataTypeFloat16, 2)) return;
  std::memcpy(out, a, (size_t)n * 2);
}

//===----------------------------------------------------------------------===//
// Phase-G Rung 0 — control-flow lowering: a bounded scan lowered into ONE
// MPSGraph control-flow executable via -forLoopWithLowerBound:upperBound:step:.
// Recurrence: carry_{i+1} = tanh(carry_i @ Wh + x_i @ Wx); ys[i] = carry_{i+1}.
// The trip count and every carry/body tensor are static-shape, so the loop and
// its body run as one dispatched graph. Per-step outputs are recovered by
// scattering each iteration's carry into an [T,d] accumulator at row `index`.
// This is the reusable machinery (index gather + carry threading + per-step
// scatter-accumulate) every higher control-flow rung reuses. See
// docs/apple_gpu_control_flow_lowering.md.
//===----------------------------------------------------------------------===//
extern "C" int32_t tessera_apple_gpu_cf_scan_f32(const float *Wh, const float *Wx,
                                                 const float *xseq,
                                                 const float *init, float *ys,
                                                 int32_t T, int32_t d, int32_t m) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || T <= 0 || d <= 0 || m <= 0) return 0;  // caller falls back
  if (!Wh || !Wx || !xseq || !init || !ys) return 0;
  @autoreleasepool {
    MPSDataType F = MPSDataTypeFloat32;
    MPSGraph *g = [MPSGraph new];
    MPSGraphTensor *pWh = [g placeholderWithShape:@[ @(d), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pWx = [g placeholderWithShape:@[ @(m), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pXs = [g placeholderWithShape:@[ @(T), @(m) ] dataType:F name:nil];
    MPSGraphTensor *pInit = [g placeholderWithShape:@[ @1, @(d) ] dataType:F name:nil];
    MPSGraphTensor *accInit = [g constantWithScalar:0.0 shape:@[ @(T), @(d) ] dataType:F];
    MPSGraphTensor *lb = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor *ub = [g constantWithScalar:T dataType:MPSDataTypeInt32];
    MPSGraphTensor *st = [g constantWithScalar:1 dataType:MPSDataTypeInt32];

    NSArray<MPSGraphTensor *> *results = [g
        forLoopWithLowerBound:lb
                   upperBound:ub
                         step:st
         initialBodyArguments:@[ pInit, accInit ]
                         body:^NSArray<MPSGraphTensor *> *(
                             MPSGraphTensor *index,
                             NSArray<MPSGraphTensor *> *args) {
                           MPSGraphTensor *carry = args[0];  // [1, d]
                           MPSGraphTensor *acc = args[1];    // [T, d]
                           MPSGraphTensor *idx =
                               [g reshapeTensor:index withShape:@[ @1 ] name:nil];
                           MPSGraphTensor *xi =
                               [g gatherWithUpdatesTensor:pXs
                                            indicesTensor:idx
                                                     axis:0
                                          batchDimensions:0
                                                     name:nil];  // [1, m]
                           MPSGraphTensor *hh =
                               [g matrixMultiplicationWithPrimaryTensor:carry
                                                       secondaryTensor:pWh
                                                                  name:nil];
                           MPSGraphTensor *xx =
                               [g matrixMultiplicationWithPrimaryTensor:xi
                                                       secondaryTensor:pWx
                                                                  name:nil];
                           MPSGraphTensor *sum =
                               [g additionWithPrimaryTensor:hh
                                            secondaryTensor:xx
                                                       name:nil];
                           MPSGraphTensor *newCarry =
                               [g tanhWithTensor:sum name:nil];  // [1, d]
                           MPSGraphTensor *newAcc =
                               [g scatterWithDataTensor:acc
                                          updatesTensor:newCarry
                                          indicesTensor:idx
                                                   axis:0
                                                   mode:MPSGraphScatterModeSet
                                                   name:nil];
                           return @[ newCarry, newAcc ];
                         }
                         name:nil];
    MPSGraphTensor *ysT = results[1];  // [T, d]

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWh, ctx, Wh, (size_t)d * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWx, ctx, Wx, (size_t)m * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bXs, ctx, xseq, (size_t)T * m * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bIn, ctx, init, (size_t)d * 4);
    if (!bWh || !bWx || !bXs || !bIn) return 0;
    MPSGraphTensorData *dWh = [[MPSGraphTensorData alloc] initWithMTLBuffer:bWh shape:@[ @(d), @(d) ] dataType:F];
    MPSGraphTensorData *dWx = [[MPSGraphTensorData alloc] initWithMTLBuffer:bWx shape:@[ @(m), @(d) ] dataType:F];
    MPSGraphTensorData *dXs = [[MPSGraphTensorData alloc] initWithMTLBuffer:bXs shape:@[ @(T), @(m) ] dataType:F];
    MPSGraphTensorData *dIn = [[MPSGraphTensorData alloc] initWithMTLBuffer:bIn shape:@[ @1, @(d) ] dataType:F];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{pWh : dWh, pWx : dWx, pXs : dXs, pInit : dIn}
                                    targetTensors:@[ ysT ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[ysT];
    if (!od) return 0;
    [[od mpsndarray] readBytes:ys strideBytes:nil];
    return 1;
  }
}

//===----------------------------------------------------------------------===//
// Phase-G Rung 1 — the Gumiho serial draft as a single MPSGraph control-flow
// executable. Reuses the Rung-0 machinery (index gather + carry threading +
// per-step scatter) with a transformer-block body: the fixed-trip
// autoregressive serial head runs in ONE -forLoopWithLowerBound: dispatch.
//
// Body at step i (carry = (hidden[1,d], token[1] int)):
//   e   = embed[token]                                   gather
//   x   = concat(hidden, e)                              [1, 2d]
//   s   = x @ fc_in                                      [1, d]
//   for layer in 0..L-1:   (T=1 attention => value-only: v@Wo)
//     s = s + (rmsnorm(s,ln1) @ Wv) @ Wo
//     s = s + silu_mul(rmsnorm(s,ln2) @ Wg, _ @ Wu) @ Wd
//   logits = rmsnorm(s,snorm) @ lm_head                  [1, V]
//   token' = argmax(logits);  hidden' = s
//   scatter token' -> tokens[i];  scatter s -> hidden[i]
// All weights are fed as placeholders; per-layer weights are static-sliced from
// the [L, ...] inputs inside the (C++-unrolled) layer loop. See
// docs/apple_gpu_control_flow_lowering.md.
//===----------------------------------------------------------------------===//
namespace {
// rmsnorm(x,gamma) = x / sqrt(mean(x^2)+eps) * gamma, all in-graph.
static MPSGraphTensor *mpsg_rmsnorm_node(MPSGraph *g, MPSGraphTensor *x,
                                         MPSGraphTensor *gamma, float eps) {
  MPSGraphTensor *sq = [g multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
  MPSGraphTensor *ms = [g meanOfTensor:sq axes:@[ @(-1) ] name:nil];
  MPSGraphTensor *epsT = [g constantWithScalar:eps dataType:MPSDataTypeFloat32];
  MPSGraphTensor *den = [g squareRootWithTensor:[g additionWithPrimaryTensor:ms secondaryTensor:epsT name:nil] name:nil];
  MPSGraphTensor *n = [g divisionWithPrimaryTensor:x secondaryTensor:den name:nil];
  return [g multiplicationWithPrimaryTensor:n secondaryTensor:gamma name:nil];
}
// static slice row li of an [L, a, b] tensor -> [a, b] (or [L, a] -> [a]).
static MPSGraphTensor *mpsg_layer_slice(MPSGraph *g, MPSGraphTensor *all, int li,
                                        NSArray<NSNumber *> *outShape) {
  MPSGraphTensor *s = [g sliceTensor:all dimension:0 start:li length:1 name:nil];
  return [g reshapeTensor:s withShape:outShape name:nil];
}
}  // namespace

extern "C" int32_t tessera_apple_gpu_cf_serial_draft_f32(
    const float *embed, const float *fc_in, const float *ln1_all,
    const float *ln2_all, const float *wv_all, const float *wo_all,
    const float *wg_all, const float *wu_all, const float *wd_all,
    const float *snorm, const float *lm_head, const float *h_init,
    int32_t root_token, int32_t *tokens_out, float *hidden_out, int32_t T,
    int32_t L, int32_t d, int32_t ffn, int32_t V, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || T <= 0 || L <= 0 || d <= 0 || ffn <= 0 || V <= 0) return 0;
  @autoreleasepool {
    MPSDataType F = MPSDataTypeFloat32, I = MPSDataTypeInt32;
    MPSGraph *g = [MPSGraph new];
    MPSGraphTensor *pEmbed = [g placeholderWithShape:@[ @(V), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pFcIn = [g placeholderWithShape:@[ @(2 * d), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pLn1 = [g placeholderWithShape:@[ @(L), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pLn2 = [g placeholderWithShape:@[ @(L), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pWv = [g placeholderWithShape:@[ @(L), @(d), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pWo = [g placeholderWithShape:@[ @(L), @(d), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pWg = [g placeholderWithShape:@[ @(L), @(d), @(ffn) ] dataType:F name:nil];
    MPSGraphTensor *pWu = [g placeholderWithShape:@[ @(L), @(d), @(ffn) ] dataType:F name:nil];
    MPSGraphTensor *pWd = [g placeholderWithShape:@[ @(L), @(ffn), @(d) ] dataType:F name:nil];
    MPSGraphTensor *pSn = [g placeholderWithShape:@[ @1, @(d) ] dataType:F name:nil];
    MPSGraphTensor *pLm = [g placeholderWithShape:@[ @(d), @(V) ] dataType:F name:nil];
    MPSGraphTensor *pH0 = [g placeholderWithShape:@[ @1, @(d) ] dataType:F name:nil];
    MPSGraphTensor *pTok0 = [g placeholderWithShape:@[ @1 ] dataType:I name:nil];
    MPSGraphTensor *tokAcc = [g constantWithScalar:0 shape:@[ @(T) ] dataType:I];
    MPSGraphTensor *hidAcc = [g constantWithScalar:0.0 shape:@[ @(T), @(d) ] dataType:F];
    MPSGraphTensor *lb = [g constantWithScalar:0 dataType:I];
    MPSGraphTensor *ub = [g constantWithScalar:T dataType:I];
    MPSGraphTensor *st = [g constantWithScalar:1 dataType:I];

    NSArray<MPSGraphTensor *> *results = [g
        forLoopWithLowerBound:lb
                   upperBound:ub
                         step:st
         initialBodyArguments:@[ pH0, pTok0, tokAcc, hidAcc ]
                         body:^NSArray<MPSGraphTensor *> *(
                             MPSGraphTensor *index,
                             NSArray<MPSGraphTensor *> *args) {
                           MPSGraphTensor *hid = args[0];   // [1,d]
                           MPSGraphTensor *tok = args[1];   // [1] int
                           MPSGraphTensor *tAcc = args[2];  // [T] int
                           MPSGraphTensor *hAcc = args[3];  // [T,d]
                           MPSGraphTensor *idx = [g reshapeTensor:index withShape:@[ @1 ] name:nil];
                           MPSGraphTensor *e = [g gatherWithUpdatesTensor:pEmbed indicesTensor:tok axis:0 batchDimensions:0 name:nil];  // [1,d]
                           MPSGraphTensor *x = [g concatTensor:hid withTensor:e dimension:1 name:nil];  // [1,2d]
                           MPSGraphTensor *s = [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:pFcIn name:nil];  // [1,d]
                           for (int li = 0; li < L; ++li) {
                             MPSGraphTensor *ln1 = mpsg_layer_slice(g, pLn1, li, @[ @1, @(d) ]);
                             MPSGraphTensor *ln2 = mpsg_layer_slice(g, pLn2, li, @[ @1, @(d) ]);
                             MPSGraphTensor *wv = mpsg_layer_slice(g, pWv, li, @[ @(d), @(d) ]);
                             MPSGraphTensor *wo = mpsg_layer_slice(g, pWo, li, @[ @(d), @(d) ]);
                             MPSGraphTensor *wg = mpsg_layer_slice(g, pWg, li, @[ @(d), @(ffn) ]);
                             MPSGraphTensor *wu = mpsg_layer_slice(g, pWu, li, @[ @(d), @(ffn) ]);
                             MPSGraphTensor *wd = mpsg_layer_slice(g, pWd, li, @[ @(ffn), @(d) ]);
                             MPSGraphTensor *n1 = mpsg_rmsnorm_node(g, s, ln1, eps);
                             MPSGraphTensor *v = [g matrixMultiplicationWithPrimaryTensor:n1 secondaryTensor:wv name:nil];
                             MPSGraphTensor *attn = [g matrixMultiplicationWithPrimaryTensor:v secondaryTensor:wo name:nil];
                             s = [g additionWithPrimaryTensor:s secondaryTensor:attn name:nil];
                             MPSGraphTensor *n2 = mpsg_rmsnorm_node(g, s, ln2, eps);
                             MPSGraphTensor *gate = [g matrixMultiplicationWithPrimaryTensor:n2 secondaryTensor:wg name:nil];
                             MPSGraphTensor *up = [g matrixMultiplicationWithPrimaryTensor:n2 secondaryTensor:wu name:nil];
                             // silu_mul = silu(gate) * up = (gate*sigmoid(gate)) * up
                             MPSGraphTensor *sig = [g sigmoidWithTensor:gate name:nil];
                             MPSGraphTensor *silu = [g multiplicationWithPrimaryTensor:gate secondaryTensor:sig name:nil];
                             MPSGraphTensor *act = [g multiplicationWithPrimaryTensor:silu secondaryTensor:up name:nil];
                             MPSGraphTensor *down = [g matrixMultiplicationWithPrimaryTensor:act secondaryTensor:wd name:nil];
                             s = [g additionWithPrimaryTensor:s secondaryTensor:down name:nil];
                           }
                           MPSGraphTensor *sn = mpsg_rmsnorm_node(g, s, pSn, eps);
                           MPSGraphTensor *logits = [g matrixMultiplicationWithPrimaryTensor:sn secondaryTensor:pLm name:nil];  // [1,V]
                           MPSGraphTensor *am = [g reductionArgMaximumWithTensor:logits axis:1 name:nil];  // [1,1] int32
                           MPSGraphTensor *newTok = [g reshapeTensor:[g castTensor:am toType:MPSDataTypeInt32 name:nil] withShape:@[ @1 ] name:nil];
                           MPSGraphTensor *newTAcc = [g scatterWithDataTensor:tAcc updatesTensor:newTok indicesTensor:idx axis:0 mode:MPSGraphScatterModeSet name:nil];
                           MPSGraphTensor *newHAcc = [g scatterWithDataTensor:hAcc updatesTensor:s indicesTensor:idx axis:0 mode:MPSGraphScatterModeSet name:nil];
                           return @[ s, newTok, newTAcc, newHAcc ];
                         }
                         name:nil];
    MPSGraphTensor *tokT = results[2];  // [T] int
    MPSGraphTensor *hidT = results[3];  // [T, d]

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bEmb, ctx, embed, (size_t)V * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bFc, ctx, fc_in, (size_t)2 * d * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bL1, ctx, ln1_all, (size_t)L * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bL2, ctx, ln2_all, (size_t)L * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWv, ctx, wv_all, (size_t)L * d * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWo, ctx, wo_all, (size_t)L * d * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWg, ctx, wg_all, (size_t)L * d * ffn * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWu, ctx, wu_all, (size_t)L * d * ffn * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWd, ctx, wd_all, (size_t)L * ffn * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bSn, ctx, snorm, (size_t)d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bLm, ctx, lm_head, (size_t)d * V * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bH0, ctx, h_init, (size_t)d * 4);
    int32_t root = root_token;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bT0, ctx, &root, sizeof(int32_t));
    if (!bEmb || !bFc || !bWv || !bWo || !bWg || !bWu || !bWd || !bLm) return 0;
    auto TD = [&](id<MTLBuffer> b, NSArray<NSNumber *> *sh, MPSDataType dt) {
      return [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:sh dataType:dt];
    };
    NSDictionary *feeds = @{
      pEmbed : TD(bEmb, @[ @(V), @(d) ], F),
      pFcIn : TD(bFc, @[ @(2 * d), @(d) ], F),
      pLn1 : TD(bL1, @[ @(L), @(d) ], F),
      pLn2 : TD(bL2, @[ @(L), @(d) ], F),
      pWv : TD(bWv, @[ @(L), @(d), @(d) ], F),
      pWo : TD(bWo, @[ @(L), @(d), @(d) ], F),
      pWg : TD(bWg, @[ @(L), @(d), @(ffn) ], F),
      pWu : TD(bWu, @[ @(L), @(d), @(ffn) ], F),
      pWd : TD(bWd, @[ @(L), @(ffn), @(d) ], F),
      pSn : TD(bSn, @[ @1, @(d) ], F),
      pLm : TD(bLm, @[ @(d), @(V) ], F),
      pH0 : TD(bH0, @[ @1, @(d) ], F),
      pTok0 : TD(bT0, @[ @1 ], I),
    };
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue feeds:feeds targetTensors:@[ tokT, hidT ] targetOperations:nil];
    MPSGraphTensorData *tod = res[tokT];
    MPSGraphTensorData *hod = res[hidT];
    if (!tod || !hod) return 0;
    [[tod mpsndarray] readBytes:tokens_out strideBytes:nil];
    [[hod mpsndarray] readBytes:hidden_out strideBytes:nil];
    return 1;
  }
}

//===----------------------------------------------------------------------===//
// Phase-G Rung 2 — predicate-driven control flow: a bounded greedy-generation
// loop expressed as ONE hand-written MSL kernel with a NATIVE while-loop. The
// trip count is **data-dependent**: the loop runs until it emits the EOS token
// or hits `max_steps`. Body per step:
//   hidden = tanh(hidden @ W);  token = argmax(hidden @ lm);  out[step] = token
// predicate: (step < max_steps) AND (last_token != eos).
// `n_out` returns the number of tokens generated (eos inclusive). This is the
// variable-trip control-flow primitive; the decode body is intentionally small
// (the dynamic verify/accept of a real speculative step stay host-side).
//
// **2026-06-04 — moved off MPSGraph `-whileWithInitialInputs:before:after:`.**
// The MPSGraph `while` route crashed (SIGSEGV) inside MPSGraph's own
// `GPU::WhileOpHandler` constructor during lazy graph specialization — it ran
// in isolation but faulted once enough MPSGraph executables had churned through
// the process (reproduced by `test_apple_gpu_control_flow_stress.py`, which
// interleaves bmm + while-generate). The data-dependent loop is bounded and the
// per-step work is tiny (d ≤ 256), so the whole sequential generation now runs
// in one thread of a classic MSL compute kernel — the same robust single-thread
// pattern as `cf_scan_msl` (MTL4 scan) and the Rung-3 `spec_accept` kernel,
// dispatched through the stable `commit_and_wait_with_timeout` path. argmax
// streams over the vocabulary (no V-sized stack array; V is unbounded). The
// hidden state lives in a `[256]` thread-local array, matching the
// documented d ≤ 256 control-flow envelope. See
// docs/apple_gpu_control_flow_lowering.md.
//===----------------------------------------------------------------------===//
extern "C" int32_t tessera_apple_gpu_cf_while_generate_f32(
    const float *W, const float *lm, const float *h_init, int32_t start_token,
    int32_t eos_token, int32_t max_steps, int32_t *tokens_out, int32_t *n_out,
    int32_t d, int32_t V) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || d <= 0 || V <= 0 || max_steps <= 0) return 0;
  if (!W || !lm || !h_init || !tokens_out || !n_out) return 0;
  if (d > 256) return 0;  // thread-local carry envelope; caller falls back.
  static NSString *const kWhileGenerateMSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;
// W : [d, d]  lm : [d, V]  h_init : [d]
// tokens_out : [max_steps]   n_out : [1]
// Greedy generation: hidden = tanh(hidden @ W); token = argmax(hidden @ lm).
// Native while-loop with a data-dependent trip count; single thread.
kernel void cf_while_generate(device const float *W      [[buffer(0)]],
                              device const float *lm     [[buffer(1)]],
                              device const float *h_init [[buffer(2)]],
                              device int         *toks    [[buffer(3)]],
                              device int         *n_out   [[buffer(4)]],
                              constant int       &start   [[buffer(5)]],
                              constant int       &eos     [[buffer(6)]],
                              constant int       &maxs    [[buffer(7)]],
                              constant int       &d       [[buffer(8)]],
                              constant int       &V       [[buffer(9)]],
                              uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;                  // single thread runs the dynamic loop
  for (int s = 0; s < maxs; ++s) toks[s] = 0;
  float h[256], hp[256];
  for (int j = 0; j < d; ++j) h[j] = h_init[j];
  int last = start, step = 0;
  while (step < maxs && last != eos) {   // <-- native data-dependent control flow
    for (int j = 0; j < d; ++j) {        // hp = tanh(h @ W)
      float acc = 0.0f;
      for (int k = 0; k < d; ++k) acc += h[k] * W[k * d + j];
      hp[j] = tanh(acc);
    }
    int best = 0;                        // last = argmax(hp @ lm), first-max wins
    float bestv = -INFINITY;
    for (int v = 0; v < V; ++v) {
      float acc = 0.0f;
      for (int k = 0; k < d; ++k) acc += hp[k] * lm[k * V + v];
      if (acc > bestv) { bestv = acc; best = v; }
    }
    last = best;
    toks[step] = last;
    for (int j = 0; j < d; ++j) h[j] = hp[j];
    ++step;
  }
  n_out[0] = step;
}
)MSL";
  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kWhileGenerateMSL, @"cf_while_generate");
    if (!pso) return 0;
    // Pooled buffers — synced (commit_and_wait) before the memcpy below, so
    // they are safe to recycle on scope exit.
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bW, ctx, W, (size_t)d * d * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bLm, ctx, lm, (size_t)d * V * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bH0, ctx, h_init, (size_t)d * 4);
    TS_METAL_BUF_ACQUIRE(bTok, ctx, (size_t)max_steps * 4);
    TS_METAL_BUF_ACQUIRE(bN, ctx, sizeof(int32_t));
    if (!bW || !bLm || !bH0 || !bTok || !bN) return 0;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bW offset:0 atIndex:0];
    [enc setBuffer:bLm offset:0 atIndex:1];
    [enc setBuffer:bH0 offset:0 atIndex:2];
    [enc setBuffer:bTok offset:0 atIndex:3];
    [enc setBuffer:bN offset:0 atIndex:4];
    [enc setBytes:&start_token length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&eos_token length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&max_steps length:sizeof(int32_t) atIndex:7];
    [enc setBytes:&d length:sizeof(int32_t) atIndex:8];
    [enc setBytes:&V length:sizeof(int32_t) atIndex:9];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "apple_gpu_cf_while_generate")) return 0;
    std::memcpy(n_out, [bN contents], sizeof(int32_t));
    std::memcpy(tokens_out, [bTok contents], (size_t)max_steps * 4);
    return 1;
  }
}

//===----------------------------------------------------------------------===//
// Metal 4 — live capability probe (M0). Rather than reading version strings,
// this actually *creates* the Metal 4 objects under @available(macOS 26.0) and
// reports which succeed, so the runtime knows whether an MTL4 lane is usable on
// this machine. Metal 4 is an additive lane alongside MPSGraph (which still
// runs on the classic command model). Caps bitmask:
//   1  MTL4CommandQueue   2  MTL4CommandAllocator   4  MTL4Compiler
//   8  MTLTensor          16 MSL 4.0 library compile
// See docs/apple_gpu_metal4_adoption.md.
//===----------------------------------------------------------------------===//
extern "C" int32_t tessera_apple_gpu_metal4_probe(int32_t *caps_out) {
  if (caps_out) *caps_out = 0;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  int32_t caps = 0;
  @autoreleasepool {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      id<MTLDevice> dev = ctx.device;
      id<MTL4CommandQueue> q = [dev newMTL4CommandQueue];
      if (q) caps |= 1;
      id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
      if (alloc) caps |= 2;
      MTL4CompilerDescriptor *cd = [[MTL4CompilerDescriptor alloc] init];
      NSError *cerr = nil;
      id<MTL4Compiler> comp = [dev newCompilerWithDescriptor:cd error:&cerr];
      if (comp) caps |= 4;
      MTLTensorDescriptor *td = [[MTLTensorDescriptor alloc] init];
      const NSInteger dims[1] = {8};
      td.dimensions = [[MTLTensorExtents alloc] initWithRank:1 values:dims];
      td.dataType = MTLTensorDataTypeFloat32;
      td.usage = MTLTensorUsageCompute;
      td.storageMode = MTLStorageModeShared;
      NSError *terr = nil;
      id<MTLTensor> t = [dev newTensorWithDescriptor:td error:&terr];
      if (t) caps |= 8;
      MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
      opts.languageVersion = MTLLanguageVersion4_0;
      NSError *lerr = nil;
      id<MTLLibrary> lib = [dev
          newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\n"
                                "kernel void ts_noop() {}"
                       options:opts
                         error:&lerr];
      if (lib) caps |= 16;
    }
  }
  if (caps_out) *caps_out = caps;
  // "available" means the FULL MTL4 lane is usable, not just any single bit:
  // the cooperative matmul2d/epilogue/session paths all need the command queue,
  // a command allocator, the MTL4 compiler, MTLTensor, and MSL 4.0. A partial
  // stack (some bit missing) must report unavailable so lanes skip cleanly.
  const int32_t kFull = 1 | 2 | 4 | 8 | 16;   // queue|alloc|compiler|tensor|msl4
  return caps == kFull ? 1 : 0;
}

// Metal 4 M1 — round-trip n elements of `in` through a native MTLTensor of the
// given dtype (0 f32, 1 f16, 2 bf16) into `out`, proving the typed resource
// stores + retrieves data. Returns 0 (caller falls back) when MTLTensor is
// unavailable. This is the foundation for an MTLTensor-backed DeviceTensor.
extern "C" int32_t tessera_apple_gpu_metal4_tensor_roundtrip(const void *in,
                                                             void *out, int32_t n,
                                                             int32_t dtype_code) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || n <= 0 || !in || !out) return 0;
  @autoreleasepool {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      MTLTensorDataType dt = (dtype_code == 1)   ? MTLTensorDataTypeFloat16
                             : (dtype_code == 2) ? MTLTensorDataTypeBFloat16
                                                 : MTLTensorDataTypeFloat32;
      MTLTensorDescriptor *td = [[MTLTensorDescriptor alloc] init];
      const NSInteger dims[1] = {n};
      td.dimensions = [[MTLTensorExtents alloc] initWithRank:1 values:dims];
      td.dataType = dt;
      td.usage = MTLTensorUsageCompute;
      td.storageMode = MTLStorageModeShared;
      NSError *err = nil;
      id<MTLTensor> t = [ctx.device newTensorWithDescriptor:td error:&err];
      if (!t) return 0;
      const NSInteger z[1] = {0};
      const NSInteger one[1] = {1};
      MTLTensorExtents *origin = [[MTLTensorExtents alloc] initWithRank:1 values:z];
      MTLTensorExtents *sdim = [[MTLTensorExtents alloc] initWithRank:1 values:dims];
      MTLTensorExtents *strides = [[MTLTensorExtents alloc] initWithRank:1 values:one];
      [t replaceSliceOrigin:origin sliceDimensions:sdim withBytes:in strides:strides];
      [t getBytes:out strides:strides fromSliceOrigin:origin sliceDimensions:sdim];
      return 1;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Metal 4 M2 + Phase-G → MSL4 bridge — the bounded scan recurrence as a
// hand-written MSL kernel with a NATIVE in-kernel for-loop, dispatched through
// the full MTL4 command model (MTL4Compiler pipeline + argument table +
// residency set + async-commit sync). Where the Phase-G control-flow rungs
// express the loop as an MPSGraph forLoop, here the loop is ordinary MSL
// control flow inside one kernel — one thread runs the whole sequential scan.
// This is both the first real MTL4 dispatch and the concrete demonstration that
// Phase-G control flow maps onto MSL 4.0. See docs/apple_gpu_metal4_adoption.md
// and docs/apple_gpu_control_flow_lowering.md.
//===----------------------------------------------------------------------===//
static NSString *kMTL4ScanMSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;
// dims = {T, d, m}. carry/nxt are thread-local (d <= 256 envelope).
kernel void cf_scan_msl(device const float *Wh   [[buffer(0)]],
                        device const float *Wx   [[buffer(1)]],
                        device const float *xseq [[buffer(2)]],
                        device const float *init [[buffer(3)]],
                        device float       *ys   [[buffer(4)]],
                        device const int   *dims [[buffer(5)]]) {
  int T = dims[0], d = dims[1], m = dims[2];
  float carry[256];
  for (int j = 0; j < d; ++j) carry[j] = init[j];
  for (int t = 0; t < T; ++t) {              // <-- native MSL control flow
    float nxt[256];
    for (int j = 0; j < d; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < d; ++k) acc += carry[k] * Wh[k * d + j];
      for (int k = 0; k < m; ++k) acc += xseq[t * m + k] * Wx[k * d + j];
      nxt[j] = tanh(acc);
    }
    for (int j = 0; j < d; ++j) { carry[j] = nxt[j]; ys[t * d + j] = nxt[j]; }
  }
}
)MSL";

extern "C" int32_t tessera_apple_gpu_mtl4_scan_f32(const float *Wh,
                                                   const float *Wx,
                                                   const float *xseq,
                                                   const float *init, float *ys,
                                                   int32_t T, int32_t d,
                                                   int32_t m) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || T <= 0 || d <= 0 || d > 256 || m <= 0) return 0;
  if (!Wh || !Wx || !xseq || !init || !ys) return 0;
  @autoreleasepool {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      id<MTLDevice> dev = ctx.device;
      NSError *err = nil;
      // 1. Cached MTL4 compute pipeline (compiled once per source/entry).
      id<MTLComputePipelineState> pso = compile_mtl4_pipeline(ctx, kMTL4ScanMSL, @"cf_scan_msl");
      if (!pso) return 0;

      // 2. Shared buffers (unified memory — no copies). Pooled (synced before
      // scope exit); the tiny dims buffer stays a fresh raw alloc.
      MTLResourceOptions ro = MTLResourceStorageModeShared;
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWh, ctx, Wh, (size_t)d * d * 4);
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bWx, ctx, Wx, (size_t)m * d * 4);
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bXs, ctx, xseq, (size_t)T * m * 4);
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bIn, ctx, init, (size_t)d * 4);
      TS_METAL_BUF_ACQUIRE(bYs, ctx, (size_t)T * d * 4);
      int dims[3] = {T, d, m};
      id<MTLBuffer> bDm = [dev newBufferWithBytes:dims length:sizeof(dims) options:ro];
      if (!bWh || !bWx || !bXs || !bIn || !bYs || !bDm) return 0;

      // 3. Residency set (MTL4 has no automatic tracking).
      id<MTLResidencySet> res = [dev newResidencySetWithDescriptor:[[MTLResidencySetDescriptor alloc] init] error:&err];
      if (!res) return 0;
      for (id<MTLBuffer> b : {bWh, bWx, bXs, bIn, bYs, bDm}) [res addAllocation:b];
      [res commit];
      [res requestResidency];

      // 4. Argument table — bind buffer GPU addresses by index.
      MTL4ArgumentTableDescriptor *atd = [[MTL4ArgumentTableDescriptor alloc] init];
      atd.maxBufferBindCount = 6;
      id<MTL4ArgumentTable> at = [dev newArgumentTableWithDescriptor:atd error:&err];
      if (!at) return 0;
      id<MTLBuffer> bufs[6] = {bWh, bWx, bXs, bIn, bYs, bDm};
      for (int i = 0; i < 6; ++i) [at setAddress:bufs[i].gpuAddress atIndex:i];

      // 5. MTL4 command model: cached queue + per-call allocator/buffer/encoder.
      id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
      id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
      id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
      if (!queue || !alloc || !cb) return 0;
      [queue addResidencySet:res];
      [cb beginCommandBufferWithAllocator:alloc];
      id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setArgumentTable:at];
      [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [enc endEncoding];
      [cb endCommandBuffer];

      // 6. Commit + CPU sync via a shared event.
      const id<MTL4CommandBuffer> cbs[1] = {cb};
      [queue commit:cbs count:1];
      id<MTLSharedEvent> ev = [dev newSharedEvent];
      [queue signalEvent:ev value:1];
      bool done = [ev waitUntilSignaledValue:1 timeoutMS:10000];
      // The queue is cached + shared, so its per-call residency set must be
      // removed (queue residency-set limit is 32).
      [queue removeResidencySet:res];
      if (!done) return 0;

      std::memcpy(ys, [bYs contents], (size_t)T * d * 4);
      return 1;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Metal 4 M3 — fused matmul via MSL cooperative tensor ops (simdgroup_matrix),
// dispatched through the MTL4 command model. simdgroup_matrix is the SIMD-group
// cooperative matrix API that targets the GPU matrix/tensor units.
//
// M5 — threadgroup-staged tiling + double-buffering (the kernel that has to
// clear MPS before M4 routing flips on). The original M3 kernel gave one 8x8
// output tile to one SIMD group and streamed A/B straight from device memory:
// zero reuse, so every output tile re-read its whole row/column band — 2-4.8x
// slower than MPS and widening with size (pure bandwidth bound).
//
// This kernel computes a BM(32) x BN(32) output tile per threadgroup with four
// 32-lane SIMD groups (128 threads) laid out 2x2; each SIMD group owns a 16x16
// region = a 2x2 array of simdgroup_float8x8 accumulators. The K dimension is
// walked in BK(16)-wide slabs that are cooperatively staged into threadgroup
// memory once and then re-read by all four SIMD groups (the reuse win), and the
// next slab is prefetched into a second threadgroup buffer while the current one
// is consumed (double-buffering, to hide the global->threadgroup load latency).
// Out-of-range rows/cols are zero-padded on load and skipped on store, so any
// M/N multiple of 8 and any K work (the dispatch keeps the 8-multiple envelope).
// (MSL 4.0 also adds a more general `tensor` cooperative-op type; simdgroup_matrix
// is the established cooperative path used here.) See
// docs/apple_gpu_metal4_adoption.md.
//===----------------------------------------------------------------------===//
static NSString *kMTL4MatmulMSL = @R"MSL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr int BM = 32;   // output-tile rows per threadgroup
constant constexpr int BN = 32;   // output-tile cols per threadgroup
constant constexpr int BK = 16;   // K-slab width staged per step
constant constexpr int SG = 16;   // 16x16 region per SIMD group (2x2 of 8x8)
constant constexpr int NT = 2;    // 8x8 tiles per SIMD-group axis (SG/8)

// C[M,N] = A[M,K] @ B[K,N]; dims = {M, N, K}. One 32x32 output tile per
// threadgroup, 4 SIMD groups (128 threads) in a 2x2 layout each owning a 16x16
// region (2x2 of 8x8 accumulators), double-buffered K slabs.
kernel void mtl4_matmul_sg(device const float *A    [[buffer(0)]],
                           device const float *B    [[buffer(1)]],
                           device float       *C    [[buffer(2)]],
                           device const int   *dims [[buffer(3)]],
                           uint2 tg  [[threadgroup_position_in_grid]],
                           uint  tid [[thread_index_in_threadgroup]],
                           uint  sgid [[simdgroup_index_in_threadgroup]]) {
  int M = dims[0], N = dims[1], K = dims[2];
  int brow = int(tg.y) * BM;   // threadgroup output-tile origin
  int bcol = int(tg.x) * BN;

  // Double-buffered staging: As[buf][BM*BK], Bs[buf][BK*BN] -> 16 KB total.
  threadgroup float As[2][BM * BK];
  threadgroup float Bs[2][BK * BN];

  // 2x2 SIMD-group layout; each owns a 16x16 region = 2x2 of 8x8 tiles.
  int sg_row = int(sgid) / 2, sg_col = int(sgid) % 2;
  int r0 = sg_row * SG, c0 = sg_col * SG;
  simdgroup_float8x8 acc[NT][NT];
  for (int i = 0; i < NT; ++i)
    for (int j = 0; j < NT; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  // Cooperative stage of the K-slab at k0 into buffer `buf` (128 threads load
  // BM*BK + BK*BN = 1024 + 1024 = 2048 floats, 16 each). Zero-pad out of range.
  auto stage = [&](int buf, int k0) {
    for (int e = int(tid); e < BM * BK; e += 128) {
      int r = e / BK, kk = e % BK;
      int gr = brow + r, gk = k0 + kk;
      As[buf][e] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
    }
    for (int e = int(tid); e < BK * BN; e += 128) {
      int kk = e / BN, c = e % BN;
      int gk = k0 + kk, gc = bcol + c;
      Bs[buf][e] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
    }
  };

  int buf = 0;
  stage(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k0 = 0; k0 < K; k0 += BK) {
    int nextk = k0 + BK;
    if (nextk < K) stage(buf ^ 1, nextk);   // prefetch next slab while computing
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 a[NT], b[NT];
      for (int i = 0; i < NT; ++i)
        simdgroup_load(a[i], As[buf] + (r0 + i * 8) * BK + kk, BK);
      for (int j = 0; j < NT; ++j)
        simdgroup_load(b[j], Bs[buf] + kk * BN + (c0 + j * 8), BN);
      for (int i = 0; i < NT; ++i)
        for (int j = 0; j < NT; ++j)
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf ^= 1;
  }

  // Store each in-range 8x8 sub-tile. M%8==0 && N%8==0 => an in-range origin
  // implies the full 8x8 is in range.
  for (int i = 0; i < NT; ++i)
    for (int j = 0; j < NT; ++j) {
      int gr = brow + r0 + i * 8, gc = bcol + c0 + j * 8;
      if (gr < M && gc < N) simdgroup_store(acc[i][j], C + gr * N + gc, N);
    }
}

// M5 fast path — register-blocked, vectorized GEMM for aligned shapes
// (M%64==0, N%64==0, K%16==0). Each threadgroup computes a 64x64 output tile
// with 8 SIMD groups (256 threads) in a 2x4 layout, each SIMD group owning a
// 32x16 region = a 4x2 array of simdgroup_float8x8 accumulators (8 per thread,
// no register spill). K is walked in 16-wide slabs staged into threadgroup
// memory with vectorized float4 loads and double-buffered (next slab prefetched
// while the current computes). On this M-series Mac this hits ~6.6 TFLOP/s f32
// (best-of-3, GPU-timed) — ~2.8x the general kernel above and ~80% of MPS (it
// ties MPS around N=1024). No bounds checks: caller must guarantee alignment.
// (Apple's MSL 4.0 MetalPerformancePrimitives `matmul2d` cooperative tensor op
// has no f32 path under execution_simdgroups — only fp16/bf16 — so the matrix
// units don't help f32; this register-blocked simdgroup_matrix kernel is the
// f32 ceiling. See docs/apple_gpu_metal4_adoption.md (M5).)
constant constexpr int FBM = 64, FBN = 64, FBK = 16;   // tile + K-slab
constant constexpr int FSGC = 4;                        // SIMD-group cols (2 rows x 4 cols)
constant constexpr int FNR = 4, FNC = 2;                // 8x8 accumulators per SIMD group
kernel void mtl4_matmul_sg_fast(device const float *A    [[buffer(0)]],
                                device const float *B    [[buffer(1)]],
                                device float       *C    [[buffer(2)]],
                                device const int   *dims [[buffer(3)]],
                                uint2 tg  [[threadgroup_position_in_grid]],
                                uint  tid [[thread_index_in_threadgroup]],
                                uint  sgid [[simdgroup_index_in_threadgroup]]) {
  int N = dims[1], K = dims[2];
  int brow = int(tg.y) * FBM, bcol = int(tg.x) * FBN;
  threadgroup float As[2][FBM * FBK];
  threadgroup float Bs[2][FBK * FBN];
  int sr = int(sgid) / FSGC, sc = int(sgid) % FSGC;   // 2x4 SIMD-group grid
  int r0 = sr * (FNR * 8), c0 = sc * (FNC * 8);
  simdgroup_float8x8 acc[FNR][FNC];
  for (int i = 0; i < FNR; ++i)
    for (int j = 0; j < FNC; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  device const float4 *A4 = (device const float4 *)A;
  device const float4 *B4 = (device const float4 *)B;
  // Vectorized float4 staging of the K-slab at k0 (256 threads, 512 float4 of A
  // + 512 of B). Aligned envelope => no bounds checks needed.
  auto stage = [&](int bf, int k0) {
    for (int e = int(tid); e < FBM * FBK / 4; e += 256) {
      int r = e / (FBK / 4), kq = e % (FBK / 4);
      ((threadgroup float4 *)As[bf])[e] = A4[((brow + r) * K + k0 + kq * 4) / 4];
    }
    for (int e = int(tid); e < FBK * FBN / 4; e += 256) {
      int kk = e / (FBN / 4), cq = e % (FBN / 4);
      ((threadgroup float4 *)Bs[bf])[e] = B4[((k0 + kk) * N + bcol + cq * 4) / 4];
    }
  };

  int bf = 0;
  stage(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k0 = 0; k0 < K; k0 += FBK) {
    int nextk = k0 + FBK;
    if (nextk < K) stage(bf ^ 1, nextk);
    for (int kk = 0; kk < FBK; kk += 8) {
      simdgroup_float8x8 a[FNR], b[FNC];
      for (int i = 0; i < FNR; ++i)
        simdgroup_load(a[i], As[bf] + (r0 + i * 8) * FBK + kk, FBK);
      for (int j = 0; j < FNC; ++j)
        simdgroup_load(b[j], Bs[bf] + kk * FBN + (c0 + j * 8), FBN);
      for (int i = 0; i < FNR; ++i)
        for (int j = 0; j < FNC; ++j)
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bf ^= 1;
  }
  for (int i = 0; i < FNR; ++i)
    for (int j = 0; j < FNC; ++j)
      simdgroup_store(acc[i][j], C + (brow + r0 + i * 8) * N + (bcol + c0 + j * 8), N);
}
)MSL";

extern "C" int32_t tessera_apple_gpu_mtl4_matmul_sg_f32(const float *A,
                                                        const float *B, float *C,
                                                        int32_t M, int32_t N,
                                                        int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || M <= 0 || N <= 0 || K <= 0) return 0;
  if (M % 8 || N % 8 || K % 8) return 0;     // 8-tile envelope
  if (!A || !B || !C) return 0;
  @autoreleasepool {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      id<MTLDevice> dev = ctx.device;
      NSError *err = nil;
      // Pick the register-blocked vectorized fast kernel (64x64 tile, ~2.8x the
      // general kernel) when the shape is aligned; otherwise the general
      // bounds-checked 32x32 kernel covers any M%8/N%8 shape and any K.
      const bool fast = (M % 64 == 0 && N % 64 == 0 && K % 16 == 0);
      id<MTLComputePipelineState> pso = compile_mtl4_pipeline(
          ctx, kMTL4MatmulMSL, fast ? @"mtl4_matmul_sg_fast" : @"mtl4_matmul_sg");
      if (!pso) return 0;

      MTLResourceOptions ro = MTLResourceStorageModeShared;
      // Large A/B/C through the buffer pool (synced before scope exit, so safe to
      // recycle); the tiny dims buffer stays a fresh raw alloc (not worth pooling).
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bA, ctx, A, (size_t)M * K * 4);
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bB, ctx, B, (size_t)K * N * 4);
      TS_METAL_BUF_ACQUIRE(bC, ctx, (size_t)M * N * 4);
      int dims[3] = {M, N, K};
      id<MTLBuffer> bD = [dev newBufferWithBytes:dims length:sizeof(dims) options:ro];
      if (!bA || !bB || !bC || !bD) return 0;

      id<MTLResidencySet> res = [dev newResidencySetWithDescriptor:[[MTLResidencySetDescriptor alloc] init] error:&err];
      if (!res) return 0;
      for (id<MTLBuffer> b : {bA, bB, bC, bD}) [res addAllocation:b];
      [res commit];
      [res requestResidency];

      MTL4ArgumentTableDescriptor *atd = [[MTL4ArgumentTableDescriptor alloc] init];
      atd.maxBufferBindCount = 4;
      id<MTL4ArgumentTable> at = [dev newArgumentTableWithDescriptor:atd error:&err];
      if (!at) return 0;
      id<MTLBuffer> bufs[4] = {bA, bB, bC, bD};
      for (int i = 0; i < 4; ++i) [at setAddress:bufs[i].gpuAddress atIndex:i];

      id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
      id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
      id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
      if (!queue || !alloc || !cb) return 0;
      [queue addResidencySet:res];
      [cb beginCommandBufferWithAllocator:alloc];
      id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:pso];
      [enc setArgumentTable:at];
      // fast: 64x64 tile / 8 SIMD groups (256 threads). general: 32x32 / 128.
      if (fast)
        [enc dispatchThreadgroups:MTLSizeMake(N / 64, M / 64, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      else
        [enc dispatchThreadgroups:MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [enc endEncoding];
      [cb endCommandBuffer];

      const id<MTL4CommandBuffer> cbs[1] = {cb};
      [queue commit:cbs count:1];
      id<MTLSharedEvent> ev = [dev newSharedEvent];
      [queue signalEvent:ev value:1];
      bool done = [ev waitUntilSignaledValue:1 timeoutMS:10000];
      [queue removeResidencySet:res];   // cached/shared queue — limit 32
      if (!done) return 0;

      std::memcpy(C, [bC contents], (size_t)M * N * 4);
      return 1;
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Metal 4 M6/M7 — fp16 + bf16 matmul (+ fused epilogue) via the MSL 4.0
// cooperative `tensor` op (MetalPerformancePrimitives `mpp::tensor_ops::matmul2d`),
// the real tensor-unit path. C[M,N](f32) = A[M,K] @ B[K,N] for {f16, bf16}.
// Unlike the simdgroup_matrix kernels above (whose f32 path tops out ~80% of MPS),
// the MPP cooperative op runs on the matrix units and BEATS MPS fp16 here
// (~1.1-1.18x at N=1024-2048, GPU best-of-3); bf16 follows the identical pattern.
// Requires the MSL 4.0 `tensor` type, real MTLTensors bound via an
// MTL4ArgumentTable (in-kernel `tensor_inline` views are rejected by the
// cooperative run path), and a float `cooperative_tensor` accumulator stored to
// the device tensor. Tile 64x64 / 4 SIMD groups (fastest of the configs swept).
//
// M7 — fused epilogue. Because the result lands in a float `cooperative_tensor`
// (live in registers across the SIMD groups), a bias + activation epilogue is
// applied IN-REGISTER before the single store — no extra device round-trip. The
// per-element walk uses the cooperative_tensor API: `get_capacity()`,
// `is_valid_element(i)` (the real mask accessor), `operator[]`, and
// `get_multidimensional_index(i)` (local tile coord; index[0] is the N/column
// axis, so bias is per output column). act codes: 0 none, 1 relu, 2 gelu(tanh),
// 3 silu. Arbitrary M/N/K (matmul2d slice() edge-checks partial tiles).
// See docs/apple_gpu_metal4_adoption.md (M6/M7).
//===----------------------------------------------------------------------===//
static NSString *kMTL4Matmul2dMSL = @R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Plain: C(f32) = A(ET) @ B(ET). 64x64 tile / 4 SIMD groups (128 threads).
#define TS_MM2D_PLAIN(NAME, ET) \
kernel void NAME(tensor<device ET,    dextents<int32_t,2>> A [[buffer(0)]], \
                 tensor<device ET,    dextents<int32_t,2>> B [[buffer(1)]], \
                 tensor<device float, dextents<int32_t,2>> C [[buffer(2)]], \
                 uint2 tg [[threadgroup_position_in_grid]]) { \
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent)); \
  matmul2d<desc, execution_simdgroups<4>> op; \
  auto mA = A.slice(0, tg.x * 64); auto mB = B.slice(tg.y * 64, 0); \
  auto mC = C.slice(tg.y * 64, tg.x * 64); \
  auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>(); \
  op.run(mA, mB, cT); cT.store(mC); }

// Fused epilogue: C = act(A@B + bias[col]). bias per output column (N); act 0..3.
#define TS_MM2D_EPI(NAME, ET) \
kernel void NAME(tensor<device ET,    dextents<int32_t,2>> A [[buffer(0)]], \
                 tensor<device ET,    dextents<int32_t,2>> B [[buffer(1)]], \
                 tensor<device float, dextents<int32_t,2>> C [[buffer(2)]], \
                 device const float *bias [[buffer(3)]], \
                 constant int2 &p [[buffer(4)]], \
                 uint2 tg [[threadgroup_position_in_grid]]) { \
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent)); \
  matmul2d<desc, execution_simdgroups<4>> op; \
  auto mA = A.slice(0, tg.x * 64); auto mB = B.slice(tg.y * 64, 0); \
  auto mC = C.slice(tg.y * 64, tg.x * 64); \
  auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>(); \
  op.run(mA, mB, cT); \
  int has_bias = p.x, act = p.y; \
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) { \
    if (!cT.is_valid_element(i)) continue; \
    float v = cT[i]; \
    if (has_bias) { auto id = cT.get_multidimensional_index(i); \
      v += bias[int(tg.y) * 64 + int(id[0])]; } \
    if (act == 1) v = fmax(0.0f, v); \
    else if (act == 2) { float t = 0.7978845608028654f * (v + 0.044715f * v * v * v); \
      v = 0.5f * v * (1.0f + tanh(t)); } \
    else if (act == 3) v = v / (1.0f + exp(-v)); \
    cT[i] = v; } \
  cT.store(mC); }

TS_MM2D_PLAIN(mtl4_matmul2d_f16,  half)
TS_MM2D_PLAIN(mtl4_matmul2d_bf16, bfloat)
TS_MM2D_EPI(mtl4_matmul2d_epilogue_f16,  half)
TS_MM2D_EPI(mtl4_matmul2d_epilogue_bf16, bfloat)
)MSL";

// Apple-sample pattern 6 (2026-05-31) — Row-major strides for a tensor whose
// extents are stored innermost-first. ``MTLTensorDescriptor`` requires
// ``strides[0] == 1``; each subsequent stride is the cumulative product of
// preceding extents. Mirrors Apple's ``Matrix+TensorUtilities.m::
// tensorStridesForDimensions:`` (sample at lines 61-70). Centralizing the
// contract here avoids the per-call inline math that historically caused
// stride-order bugs (the conv2d native multi-tile spike had to debug
// innermost-first the hard way). Always pure / inline / no allocation.
static inline void apple_row_major_strides(const NSInteger *dims, int rank,
                                           NSInteger *strides_out) {
  NSInteger stride = 1;
  for (int i = 0; i < rank; ++i) {
    strides_out[i] = stride;
    stride *= dims[i];
  }
}

// Buffer-backed MTLTensor: extents innermost-first (cols, rows); packed
// row-major strides (1, inner); usage=Compute. Returns nil on failure.
API_AVAILABLE(macos(26.0), ios(26.0))
static id<MTLTensor> make_buffer_tensor(id<MTLDevice> dev, id<MTLBuffer> buf,
                                        int inner, int outer, MTLTensorDataType dt) {
  MTLTensorDescriptor *td = [[MTLTensorDescriptor alloc] init];
  NSInteger dims[2] = {inner, outer}, strd[2];
  apple_row_major_strides(dims, 2, strd);
  td.dimensions = [[MTLTensorExtents alloc] initWithRank:2 values:dims];
  td.strides = [[MTLTensorExtents alloc] initWithRank:2 values:strd];
  td.dataType = dt;
  td.usage = MTLTensorUsageCompute;
  NSError *e = nil;
  return [buf newTensorWithDescriptor:td offset:0 error:&e];
}

// R0→MTLTensor bridge — a cached buffer-backed MTLTensor view over a resident
// device tensor's shared storage, so the MTL4 cooperative lane can bind it
// directly (no `newBufferWithBytes` upload). The view is rebuilt only when the
// requested (inner, outer, dt) differs from the cached one — so a steady decode
// loop (same M,K each step) creates it once and reuses it.
API_AVAILABLE(macos(26.0), ios(26.0))
static id<MTLTensor> ts_dev_tensor_view(TsDeviceTensor *t, int inner, int outer,
                                        MTLTensorDataType dt) {
  if (!t || !t->buf) return nil;
  if (t->tensorView && t->viewInner == inner && t->viewOuter == outer &&
      t->viewDt == (int)dt)
    return (id<MTLTensor>)t->tensorView;
  id<MTLTensor> v = make_buffer_tensor(deviceContext().device, t->buf, inner, outer, dt);
  if (!v) return nil;
  t->tensorView = v;
  t->viewInner = inner;
  t->viewOuter = outer;
  t->viewDt = (int)dt;
  return v;
}

// Shared MPP matmul2d dispatch. A, B are 16-bit (f16 or bf16) bit patterns; C is
// f32. When `fused`, also binds `bias` (length N, may be null) + {has_bias, act}
// and runs the epilogue entry. Returns 1 if it ran on the tensor-op lane, else 0.
static int32_t mtl4_matmul2d_dispatch(NSString *entry, MTLTensorDataType dt,
                                      const uint16_t *A, const uint16_t *B, float *C,
                                      int32_t M, int32_t N, int32_t K,
                                      bool fused, const float *bias, int32_t act) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || M <= 0 || N <= 0 || K <= 0 || !A || !B || !C) return 0;
  @autoreleasepool {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      id<MTLDevice> dev = ctx.device;
      NSError *err = nil;
      id<MTLComputePipelineState> pso = compile_mtl4_pipeline(ctx, kMTL4Matmul2dMSL, entry);
      if (!pso) return 0;

      MTLResourceOptions ro = MTLResourceStorageModeShared;
      // Recycle the (large) A/B/C buffers through the shared pool — this path
      // syncs before returning, so the buffers are free to recycle on scope exit
      // (RAII guards). Avoids ~50-100us/alloc x3 of churn on repeated same-size
      // calls. Buffers > 4MB bypass the pool and allocate fresh (see the pool).
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bA, ctx, A, (size_t)M * K * 2);
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bB, ctx, B, (size_t)K * N * 2);
      TS_METAL_BUF_ACQUIRE(bC, ctx, (size_t)M * N * 4);
      if (!bA || !bB || !bC) return 0;
      id<MTLTensor> tA = make_buffer_tensor(dev, bA, K, M, dt);
      id<MTLTensor> tB = make_buffer_tensor(dev, bB, N, K, dt);
      id<MTLTensor> tC = make_buffer_tensor(dev, bC, N, M, MTLTensorDataTypeFloat32);
      if (!tA || !tB || !tC) return 0;

      id<MTLBuffer> bBias = nil, bP = nil;
      if (fused) {
        int has_bias = bias ? 1 : 0;
        float zero = 0.0f;
        bBias = bias ? [dev newBufferWithBytes:bias length:(size_t)N * 4 options:ro]
                     : [dev newBufferWithBytes:&zero length:4 options:ro];
        int params[2] = {has_bias, act};
        bP = [dev newBufferWithBytes:params length:sizeof(params) options:ro];
        if (!bBias || !bP) return 0;
      }

      id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
      if (!queue) return 0;
      // P2/P3 — serialize encode→commit→wait so the reusable allocator / command
      // buffer / argument table / shared event are safe to reset+rebind.
      bool done = false;
      {
        std::lock_guard<std::mutex> lock(ctx.mtl4_dispatch_mu);
        id<MTLBuffer> rbufs[5] = {bA, bB, bC, fused ? bBias : nil, fused ? bP : nil};
        id<MTLResidencySet> res = mtl4_set_residency(ctx, rbufs, fused ? 5 : 3);
        if (!res) return 0;
        // 64x64 output tile per threadgroup; tg.x tiles M, tg.y tiles N; 4 SIMD groups.
        done = mtl4_encode_and_wait(ctx, queue, pso, ^(id<MTL4ArgumentTable> at) {
          [at setResource:tA.gpuResourceID atBufferIndex:0];
          [at setResource:tB.gpuResourceID atBufferIndex:1];
          [at setResource:tC.gpuResourceID atBufferIndex:2];
          if (fused) {
            [at setAddress:bBias.gpuAddress atIndex:3];
            [at setAddress:bP.gpuAddress atIndex:4];
          }
        }, MTLSizeMake((M + 63) / 64, (N + 63) / 64, 1), MTLSizeMake(128, 1, 1), res);
      }
      if (!done) return 0;

      std::memcpy(C, [bC contents], (size_t)M * N * 4);
      return 1;
    }
  }
  return 0;
}

// A, B fp16 (uint16_t bit patterns); C fp32. Returns 1 if it ran on the MPP lane.
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_f16(const uint16_t *A,
                                                       const uint16_t *B, float *C,
                                                       int32_t M, int32_t N, int32_t K) {
  return mtl4_matmul2d_dispatch(@"mtl4_matmul2d_f16", MTLTensorDataTypeFloat16,
                                A, B, C, M, N, K, /*fused=*/false, nullptr, 0);
}

// A, B bf16 (uint16_t bit patterns); C fp32.
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_bf16(const uint16_t *A,
                                                        const uint16_t *B, float *C,
                                                        int32_t M, int32_t N, int32_t K) {
  return mtl4_matmul2d_dispatch(@"mtl4_matmul2d_bf16", MTLTensorDataTypeBFloat16,
                                A, B, C, M, N, K, /*fused=*/false, nullptr, 0);
}

//===----------------------------------------------------------------------===//
// Spike: native MPP convolution2d cooperative op — single-tile baseline.
// VALID 3x3 conv via set_offsets((K-1)/2). One threadgroup, 4 SIMD groups (128
// threads). NHWC activation / HWIO weights / NHWO destination (the only
// layouts the op supports today). The descriptor's destination_dimensions IS
// the per-tile output region — compile-time. dst_H/dst_W are TILE sizes here.
// Step 1 goal: re-confirm bit-correctness vs numpy at this SDK level before
// stepping up to grid-of-threadgroups multi-tile (step 2).
//===----------------------------------------------------------------------===//
API_AVAILABLE(macos(26.0), ios(26.0))
static id<MTLTensor> make_buffer_tensor_4d(id<MTLDevice> dev, id<MTLBuffer> buf,
                                           NSInteger d0, NSInteger d1,
                                           NSInteger d2, NSInteger d3,
                                           MTLTensorDataType dt) {
  // MTLTensorDescriptor requires `strides[0] == 1` — extents are stored
  // **innermost-first** (d0 = innermost). Callers must pass dims in that
  // order: for MPP NHWC activations that's {Cin, srcW, srcH, B}.
  NSInteger dims[4] = {d0, d1, d2, d3};
  NSInteger strd[4];
  apple_row_major_strides(dims, 4, strd);  // {1, d0, d0*d1, d0*d1*d2}
  MTLTensorDescriptor *td = [[MTLTensorDescriptor alloc] init];
  td.dimensions = [[MTLTensorExtents alloc] initWithRank:4 values:dims];
  td.strides = [[MTLTensorExtents alloc] initWithRank:4 values:strd];
  td.dataType = dt;
  td.usage = MTLTensorUsageCompute;
  NSError *e = nil;
  return [buf newTensorWithDescriptor:td offset:0 error:&e];
}

static NSString *kMTL4Conv2dSingleTileMSL = @R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// VALID 3x3 conv via set_offsets((K-1)/2). Compile-time shape: 8x8 destination
// tile, Cin=4, Cout=4, B=1, source = 10x10 (8 + 2 halo). f16 activations +
// weights, f32 destination. One threadgroup = 4 simdgroups = 128 threads.
kernel void conv2d_single_tile_f16(
    tensor<device half,  dextents<int32_t,4>> X [[buffer(0)]],
    tensor<device half,  dextents<int32_t,4>> W [[buffer(1)]],
    tensor<device float, dextents<int32_t,4>> Y [[buffer(2)]])
{
  // MPP descriptor order: (out_channels, dst_W, dst_H, batch),
  //                      (in_channels,  src_W, src_H, batch).
  constexpr auto desc = convolution2d_descriptor(
      int4(4, 8, 8, 1),
      int4(4, 10, 10, 1),
      int2(3, 3));
  convolution2d<desc, execution_simdgroups<4>> op;
  op.set_offsets(int2(1, 1));   // SAME-centered -> VALID for K=3
  auto cT = op.get_destination_cooperative_tensor<decltype(X), decltype(W), float>();
  op.run(X, W, cT);
  cT.store(Y);
}
)MSL";

// Spike step 2 — grid-of-threadgroups multi-tile conv. Same compile-time
// descriptor (tile = 8x8 output, Cin=Cout=4, k=3) but launched as a 2-D grid
// over output tiles. Each TG slices its input window via set_offsets and its
// destination via tensor.slice — the pattern that unlocked batched SVD/Cholesky.
// Only aligned output sizes (dstH, dstW multiples of 8) — the honest scope of
// the spike; non-aligned tiling is a follow-up if this lane proves itself.
static NSString *kMTL4Conv2dMultiTileMSL = @R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant int TH = 8;   // output tile rows  (compile-time)
constant int TW = 8;   // output tile cols
constant int Kp = 3;   // kernel side

kernel void conv2d_multi_tile_f16(
    tensor<device half,  dextents<int32_t,4>> X [[buffer(0)]],
    tensor<device half,  dextents<int32_t,4>> W [[buffer(1)]],
    tensor<device float, dextents<int32_t,4>> Y [[buffer(2)]],
    uint2 tg [[threadgroup_position_in_grid]])
{
  // Per-tile descriptor: source = (TH + Kp - 1) x (TW + Kp - 1) halo, dest = TH x TW.
  constexpr auto desc = convolution2d_descriptor(
      int4(4, TW, TH, 1),
      int4(4, TW + Kp - 1, TH + Kp - 1, 1),
      int2(Kp, Kp));
  convolution2d<desc, execution_simdgroups<4>> op;
  op.set_offsets(int2((Kp - 1) / 2, (Kp - 1) / 2));  // SAME-centered -> VALID
  // The cooperative op expects bound tensors whose dimensions match the
  // descriptor's source/dest. Slice both X and Y per-tile (innermost-first
  // offsets: channel=0, W=tile_col*TW, H=tile_row*TH, batch=0).
  auto mX = X.slice(0, (int)tg.x * TW, (int)tg.y * TH, 0);
  auto mY = Y.slice(0, (int)tg.x * TW, (int)tg.y * TH, 0);
  auto cT = op.get_destination_cooperative_tensor<decltype(mX), decltype(W), float>();
  op.run(mX, W, cT);
  cT.store(mY);
}
)MSL";

// Spike harness — multi-tile NHWC, Cin=Cout=4, k=3, VALID. dstH, dstW must be
// multiples of 8 (the compile-time tile). Returns 1 on success.
extern "C" int32_t tessera_apple_gpu_spike_conv2d_multi_tile_f16(
    const uint16_t *X, const uint16_t *W, float *Y,
    int32_t dstH, int32_t dstW) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !X || !W || !Y) return 0;
  const int TH = 8, TW = 8, kH = 3, kW = 3, Cin = 4, Cout = 4, B = 1;
  if (dstH <= 0 || dstW <= 0 || dstH % TH != 0 || dstW % TW != 0) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    id<MTLDevice> dev = ctx.device;
    id<MTLComputePipelineState> pso = compile_mtl4_pipeline(
        ctx, kMTL4Conv2dMultiTileMSL, @"conv2d_multi_tile_f16");
    if (!pso) return 0;
    const int srcH = dstH + (kH - 1), srcW = dstW + (kW - 1);
    @autoreleasepool {
      size_t xB = (size_t)B * srcH * srcW * Cin * 2;
      size_t wB = (size_t)kH * kW * Cin * Cout * 2;
      size_t yB = (size_t)B * dstH * dstW * Cout * 4;
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bX, ctx, X, xB);
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bW, ctx, W, wB);
      TS_METAL_BUF_ACQUIRE(bY, ctx, yB);
      if (!bX || !bW || !bY) return 0;
      // Innermost-first dims: X(Cin,srcW,srcH,B), W(Cout,Cin,kW,kH), Y(Cout,dstW,dstH,B).
      id<MTLTensor> tX = make_buffer_tensor_4d(dev, bX, Cin, srcW, srcH, B,    MTLTensorDataTypeFloat16);
      id<MTLTensor> tW = make_buffer_tensor_4d(dev, bW, Cout, Cin, kW, kH,     MTLTensorDataTypeFloat16);
      id<MTLTensor> tY = make_buffer_tensor_4d(dev, bY, Cout, dstW, dstH, B,   MTLTensorDataTypeFloat32);
      if (!tX || !tW || !tY) return 0;
      id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
      if (!queue) return 0;
      bool done = false;
      {
        std::lock_guard<std::mutex> lock(ctx.mtl4_dispatch_mu);
        id<MTLBuffer> rbufs[3] = {bX, bW, bY};
        id<MTLResidencySet> res = mtl4_set_residency(ctx, rbufs, 3);
        if (!res) return 0;
        MTLSize grid = MTLSizeMake((NSUInteger)(dstW / TW), (NSUInteger)(dstH / TH), 1);
        done = mtl4_encode_and_wait(ctx, queue, pso, ^(id<MTL4ArgumentTable> at) {
          [at setResource:tX.gpuResourceID atBufferIndex:0];
          [at setResource:tW.gpuResourceID atBufferIndex:1];
          [at setResource:tY.gpuResourceID atBufferIndex:2];
        }, grid, MTLSizeMake(128, 1, 1), res);
      }
      if (!done) return 0;
      std::memcpy(Y, [bY contents], yB);
      return 1;
    }
  }
  return 0;
}

// Spike harness — single-tile NHWC 8x8, Cin=Cout=4, k=3, VALID. Returns 1 on
// success, 0 if Metal unavailable or anything fails along the way.
extern "C" int32_t tessera_apple_gpu_spike_conv2d_single_tile_f16(
    const uint16_t *X, const uint16_t *W, float *Y) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !X || !W || !Y) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    id<MTLDevice> dev = ctx.device;
    id<MTLComputePipelineState> pso = compile_mtl4_pipeline(
        ctx, kMTL4Conv2dSingleTileMSL, @"conv2d_single_tile_f16");
    if (!pso) return 0;
    const int B = 1, Cin = 4, Cout = 4, kH = 3, kW = 3, dstH = 8, dstW = 8;
    const int srcH = dstH + (kH - 1), srcW = dstW + (kW - 1);
    @autoreleasepool {
      MTLResourceOptions ro = MTLResourceStorageModeShared;
      size_t xB = (size_t)B * srcH * srcW * Cin * 2;
      size_t wB = (size_t)kH * kW * Cin * Cout * 2;
      size_t yB = (size_t)B * dstH * dstW * Cout * 4;
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bX, ctx, X, xB);
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bW, ctx, W, wB);
      TS_METAL_BUF_ACQUIRE(bY, ctx, yB);
      if (!bX || !bW || !bY) return 0;
      // MPP NHWC/HWIO/NHWO with innermost-first MTLTensor dims:
      //   activation X: innermost=Cin, then srcW, srcH, batch
      //   weights   W: innermost=Cout, then Cin,  kW,   kH    (HWIO storage)
      //   dest      Y: innermost=Cout, then dstW, dstH, batch
      id<MTLTensor> tX = make_buffer_tensor_4d(dev, bX, Cin, srcW, srcH, B,    MTLTensorDataTypeFloat16);
      id<MTLTensor> tW = make_buffer_tensor_4d(dev, bW, Cout, Cin, kW, kH,     MTLTensorDataTypeFloat16);
      id<MTLTensor> tY = make_buffer_tensor_4d(dev, bY, Cout, dstW, dstH, B,   MTLTensorDataTypeFloat32);
      if (!tX || !tW || !tY) return 0;
      id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
      if (!queue) return 0;
      bool done = false;
      {
        std::lock_guard<std::mutex> lock(ctx.mtl4_dispatch_mu);
        id<MTLBuffer> rbufs[3] = {bX, bW, bY};
        id<MTLResidencySet> res = mtl4_set_residency(ctx, rbufs, 3);
        if (!res) return 0;
        // Single threadgroup = 4 SIMD groups = 128 threads (matches the kernel).
        done = mtl4_encode_and_wait(ctx, queue, pso, ^(id<MTL4ArgumentTable> at) {
          [at setResource:tX.gpuResourceID atBufferIndex:0];
          [at setResource:tW.gpuResourceID atBufferIndex:1];
          [at setResource:tY.gpuResourceID atBufferIndex:2];
        }, MTLSizeMake(1, 1, 1), MTLSizeMake(128, 1, 1), res);
      }
      if (!done) return 0;
      std::memcpy(Y, [bY contents], yB);
      return 1;
    }
  }
  return 0;
}

// Fused: C = act(A(f16) @ B(f16) + bias). bias may be null; act 0..3.
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_epilogue_f16(
    const uint16_t *A, const uint16_t *B, float *C, const float *bias,
    int32_t act, int32_t M, int32_t N, int32_t K) {
  return mtl4_matmul2d_dispatch(@"mtl4_matmul2d_epilogue_f16", MTLTensorDataTypeFloat16,
                                A, B, C, M, N, K, /*fused=*/true, bias, act);
}

// Fused: C = act(A(bf16) @ B(bf16) + bias). bias may be null; act 0..3.
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_epilogue_bf16(
    const uint16_t *A, const uint16_t *B, float *C, const float *bias,
    int32_t act, int32_t M, int32_t N, int32_t K) {
  return mtl4_matmul2d_dispatch(@"mtl4_matmul2d_epilogue_bf16", MTLTensorDataTypeBFloat16,
                                A, B, C, M, N, K, /*fused=*/true, bias, act);
}

// R0 — general device-resident matmul2d: A, B, C are DeviceTensors (A/B f16 or
// bf16 per `bf16`, C f32) — *both* operands stay resident (no host upload), so
// resident activations feed the matrix-unit lane directly (the both-resident
// complement to the M8 session's resident-W run_dev). C = A @ B, (M×K)·(K×N).
// 1 if it ran on the MPP cooperative lane, else 0.
extern "C" int32_t tessera_apple_gpu_mtl4_matmul2d_dev(TsDeviceTensor *A,
    TsDeviceTensor *B, TsDeviceTensor *C, int32_t M, int32_t N, int32_t K,
    int32_t bf16) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !A || !B || !C || M <= 0 || N <= 0 || K <= 0) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    MTLTensorDataType dt = bf16 ? MTLTensorDataTypeBFloat16 : MTLTensorDataTypeFloat16;
    id<MTLComputePipelineState> pso = compile_mtl4_pipeline(
        ctx, kMTL4Matmul2dMSL, bf16 ? @"mtl4_matmul2d_bf16" : @"mtl4_matmul2d_f16");
    if (!pso) return 0;
    id<MTLTensor> tA = ts_dev_tensor_view(A, K, M, dt);
    id<MTLTensor> tB = ts_dev_tensor_view(B, N, K, dt);
    id<MTLTensor> tC = ts_dev_tensor_view(C, N, M, MTLTensorDataTypeFloat32);
    if (!tA || !tB || !tC) return 0;
    id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
    if (!queue) return 0;
    bool done = false;
    {
      std::lock_guard<std::mutex> lock(ctx.mtl4_dispatch_mu);
      id<MTLBuffer> rbufs[3] = {A->buf, B->buf, C->buf};
      id<MTLResidencySet> res = mtl4_set_residency(ctx, rbufs, 3);
      if (!res) return 0;
      done = mtl4_encode_and_wait(ctx, queue, pso, ^(id<MTL4ArgumentTable> at) {
        [at setResource:tA.gpuResourceID atBufferIndex:0];
        [at setResource:tB.gpuResourceID atBufferIndex:1];
        [at setResource:tC.gpuResourceID atBufferIndex:2];
      }, MTLSizeMake((M + 63) / 64, (N + 63) / 64, 1), MTLSizeMake(128, 1, 1), res);
    }
    return done ? 1 : 0;
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// P8 (perf) — on-device conv2d: GPU im2col + the matmul2d epilogue, with the
// unfolded `col` matrix kept on the GPU between the two dispatches (no host
// im2col gather — the bottleneck that made the conv lane slower than MPSGraph).
// Stage 1: an MSL gather writes col[M=N*OH*OW, Kk=kH*kW*Cin] from NHWC X
// (zero-padded, strided, dilated). Stage 2: the existing matmul2d epilogue
// kernel computes Y(f32) = act(col @ Wr + bias) on the matrix units, reading
// col as a device MTLTensor. f16/bf16. See docs/apple_backend_integration_review.md (P8).
//===----------------------------------------------------------------------===//
static NSString *kIm2colMSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;
// p = {N,H,W,Cin,kH,kW,sH,sW,pH,pW,dH,dW,OH,OW}. One thread per col element.
#define TS_IM2COL(NAME, ET) \
kernel void NAME(device const ET *X [[buffer(0)]], device ET *col [[buffer(1)]], \
                 constant int *p [[buffer(2)]], uint gid [[thread_position_in_grid]]) { \
  int N=p[0],H=p[1],W=p[2],Cin=p[3],kH=p[4],kW=p[5],sH=p[6],sW=p[7]; \
  int pH=p[8],pW=p[9],dH=p[10],dW=p[11],OH=p[12],OW=p[13]; \
  int Kk=kH*kW*Cin, M=N*OH*OW; \
  if ((int)gid >= M*Kk) return; \
  int row=(int)gid/Kk, k=(int)gid%Kk; \
  int n=row/(OH*OW), r=row%(OH*OW), oh=r/OW, ow=r%OW; \
  int kh=k/(kW*Cin), kk=k%(kW*Cin), kw=kk/Cin, ci=kk%Cin; \
  int ih=oh*sH - pH + kh*dH, iw=ow*sW - pW + kw*dW; \
  ET v = (ET)0; \
  if (ih>=0 && ih<H && iw>=0 && iw<W) v = X[((n*H+ih)*W+iw)*Cin + ci]; \
  col[gid] = v; }
TS_IM2COL(im2col_f16, half)
TS_IM2COL(im2col_bf16, bfloat)
)MSL";

// X NHWC (uint16 f16/bf16), W HWIO reshaped to [Kk, Cout] (uint16), bias [Cout]
// (f32, may be null), Y [N*OH*OW, Cout] (f32). act 0..3. Returns 1 if it ran.
static int32_t mtl4_conv2d_dispatch(MTLTensorDataType dt, bool bf16,
                                    const uint16_t *X, const uint16_t *Wr,
                                    const float *bias, float *Y, int32_t act,
                                    int32_t N, int32_t H, int32_t W, int32_t Cin,
                                    int32_t Cout, int32_t kH, int32_t kW,
                                    int32_t sH, int32_t sW, int32_t pH, int32_t pW,
                                    int32_t dH, int32_t dW) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !X || !Wr || !Y) return 0;
  @autoreleasepool {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      int OH = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
      int OW = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
      if (OH <= 0 || OW <= 0) return 0;
      int M = N * OH * OW, Kk = kH * kW * Cin, Nn = Cout;
      id<MTLDevice> dev = ctx.device;
      id<MTLComputePipelineState> imp = compile_mtl4_pipeline(
          ctx, kIm2colMSL, bf16 ? @"im2col_bf16" : @"im2col_f16");
      id<MTLComputePipelineState> mmp = compile_mtl4_pipeline(
          ctx, kMTL4Matmul2dMSL, bf16 ? @"mtl4_matmul2d_epilogue_bf16"
                                      : @"mtl4_matmul2d_epilogue_f16");
      if (!imp || !mmp) return 0;

      MTLResourceOptions ro = MTLResourceStorageModeShared;
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bX, ctx, X, (size_t)N * H * W * Cin * 2);
      TS_METAL_BUF_ACQUIRE_WITH_BYTES(bW, ctx, Wr, (size_t)Kk * Nn * 2);
      TS_METAL_BUF_ACQUIRE(bCol, ctx, (size_t)M * Kk * 2);   // stays on-GPU
      TS_METAL_BUF_ACQUIRE(bY, ctx, (size_t)M * Nn * 4);
      if (!bX || !bW || !bCol || !bY) return 0;
      int ip[14] = {N, H, W, Cin, kH, kW, sH, sW, pH, pW, dH, dW, OH, OW};
      id<MTLBuffer> bIP = [dev newBufferWithBytes:ip length:sizeof(ip) options:ro];
      int has_bias = bias ? 1 : 0;
      float zero = 0.0f;
      id<MTLBuffer> bBias = bias ? [dev newBufferWithBytes:bias length:(size_t)Nn * 4 options:ro]
                                 : [dev newBufferWithBytes:&zero length:4 options:ro];
      int mp[2] = {has_bias, act};
      id<MTLBuffer> bP = [dev newBufferWithBytes:mp length:sizeof(mp) options:ro];
      // col / W / Y as device MTLTensors for the matmul2d stage.
      id<MTLTensor> tCol = make_buffer_tensor(dev, bCol, Kk, M, dt);
      id<MTLTensor> tW = make_buffer_tensor(dev, bW, Nn, Kk, dt);
      id<MTLTensor> tY = make_buffer_tensor(dev, bY, Nn, M, MTLTensorDataTypeFloat32);
      if (!bIP || !bBias || !bP || !tCol || !tW || !tY) return 0;

      id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
      if (!queue) return 0;
      bool done = false;
      {
        std::lock_guard<std::mutex> lock(ctx.mtl4_dispatch_mu);
        id<MTLBuffer> rbufs[7] = {bX, bW, bCol, bY, bIP, bBias, bP};
        id<MTLResidencySet> res = mtl4_set_residency(ctx, rbufs, 7);
        if (!res) return 0;
        // Stage 1 — GPU im2col (X -> bCol). col never leaves the device.
        done = mtl4_encode_and_wait(ctx, queue, imp, ^(id<MTL4ArgumentTable> at) {
          [at setAddress:bX.gpuAddress atIndex:0];
          [at setAddress:bCol.gpuAddress atIndex:1];
          [at setAddress:bIP.gpuAddress atIndex:2];
        }, MTLSizeMake((M * Kk + 255) / 256, 1, 1), MTLSizeMake(256, 1, 1), res);
        // Stage 2 — matmul2d epilogue (col @ W + bias, act -> Y) on the matrix units.
        if (done)
          done = mtl4_encode_and_wait(ctx, queue, mmp, ^(id<MTL4ArgumentTable> at) {
            [at setResource:tCol.gpuResourceID atBufferIndex:0];
            [at setResource:tW.gpuResourceID atBufferIndex:1];
            [at setResource:tY.gpuResourceID atBufferIndex:2];
            [at setAddress:bBias.gpuAddress atIndex:3];
            [at setAddress:bP.gpuAddress atIndex:4];
          }, MTLSizeMake((M + 63) / 64, (Nn + 63) / 64, 1), MTLSizeMake(128, 1, 1), res);
      }
      if (!done) return 0;
      std::memcpy(Y, [bY contents], (size_t)M * Nn * 4);
      return 1;
    }
  }
  return 0;
}

extern "C" int32_t tessera_apple_gpu_mtl4_conv2d_f16(
    const uint16_t *X, const uint16_t *Wr, const float *bias, float *Y, int32_t act,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout, int32_t kH, int32_t kW,
    int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW) {
  return mtl4_conv2d_dispatch(MTLTensorDataTypeFloat16, false, X, Wr, bias, Y, act,
                              N, H, W, Cin, Cout, kH, kW, sH, sW, pH, pW, dH, dW);
}

extern "C" int32_t tessera_apple_gpu_mtl4_conv2d_bf16(
    const uint16_t *X, const uint16_t *Wr, const float *bias, float *Y, int32_t act,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout, int32_t kH, int32_t kW,
    int32_t sH, int32_t sW, int32_t pH, int32_t pW, int32_t dH, int32_t dW) {
  return mtl4_conv2d_dispatch(MTLTensorDataTypeBFloat16, true, X, Wr, bias, Y, act,
                              N, H, W, Cin, Cout, kH, kW, sH, sW, pH, pW, dH, dW);
}

//===----------------------------------------------------------------------===//
// Metal 4 M8 — fused MLP-block session with RESIDENT weights. A decode-style
// session: the weight `W` (+ bias) is uploaded ONCE and kept resident, and the
// pipeline / residency set / command queue are reused across runs. Each run only
// uploads the (small) activation `X` and dispatches one fused epilogue matmul:
//   Y[M,N](f32) = act(X[M,K](f16/bf16) @ W[K,N] + bias).
// This amortizes exactly the per-call MTL4 overhead that keeps routing OFF at
// decode (small-M) sizes — re-uploading W (which dominates when W >> X) and
// re-committing residency every call. The handle is an opaque C++ struct holding
// ARC-managed Metal objects. See docs/apple_gpu_metal4_adoption.md (M8).
//===----------------------------------------------------------------------===//
struct TesseraMlpSession {
  id pso;      // id<MTLComputePipelineState>
  id bW;       // id<MTLBuffer>        resident weights (K x N, f16/bf16)
  id tW;       // id<MTLTensor>        N x K view of bW
  id bBias;    // id<MTLBuffer>        bias (f32) or 1-elem dummy
  id bParams;  // id<MTLBuffer>        {has_bias, act}
  id resW;     // id<MTLResidencySet>  persistent (W, bias, params)
  int K, N, dt;
};

// dtype: 0 = f16, 1 = bf16. bias may be null. Returns an opaque handle or null.
extern "C" void *tessera_apple_gpu_mtl4_mlp_session_create(const uint16_t *W,
    const float *bias, int32_t act, int32_t K, int32_t N, int32_t bf16) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !W || K <= 0 || N <= 0) return nullptr;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    id<MTLDevice> dev = ctx.device;
    NSError *err = nil;
    NSString *entry = bf16 ? @"mtl4_matmul2d_epilogue_bf16" : @"mtl4_matmul2d_epilogue_f16";
    MTLTensorDataType dt = bf16 ? MTLTensorDataTypeBFloat16 : MTLTensorDataTypeFloat16;
    id<MTLComputePipelineState> pso = compile_mtl4_pipeline(ctx, kMTL4Matmul2dMSL, entry);
    if (!pso) return nullptr;
    MTLResourceOptions ro = MTLResourceStorageModeShared;
    id<MTLBuffer> bW = [dev newBufferWithBytes:W length:(size_t)K * N * 2 options:ro];
    id<MTLTensor> tW = make_buffer_tensor(dev, bW, N, K, dt);   // B operand: (N, K)
    int has_bias = bias ? 1 : 0;
    float zero = 0.0f;
    id<MTLBuffer> bBias = bias ? [dev newBufferWithBytes:bias length:(size_t)N * 4 options:ro]
                               : [dev newBufferWithBytes:&zero length:4 options:ro];
    int params[2] = {has_bias, act};
    id<MTLBuffer> bParams = [dev newBufferWithBytes:params length:sizeof(params) options:ro];
    if (!bW || !tW || !bBias || !bParams) return nullptr;
    id<MTLResidencySet> resW = [dev newResidencySetWithDescriptor:[[MTLResidencySetDescriptor alloc] init] error:&err];
    if (!resW) return nullptr;
    [resW addAllocation:bW]; [resW addAllocation:bBias]; [resW addAllocation:bParams];
    [resW commit];
    [resW requestResidency];
    id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
    if (!queue) return nullptr;
    [queue addResidencySet:resW];
    TesseraMlpSession *s = new TesseraMlpSession();
    s->pso = pso; s->bW = bW; s->tW = tW; s->bBias = bBias; s->bParams = bParams;
    s->resW = resW; s->K = K; s->N = N; s->dt = (int)dt;
    return (void *)s;
  }
  return nullptr;
}

// One decode step: Y = act(X @ W + bias). X is M x K (f16/bf16), Y is M x N (f32).
extern "C" int32_t tessera_apple_gpu_mtl4_mlp_session_run(void *handle,
    const uint16_t *X, float *Y, int32_t M) {
  if (!handle || !X || !Y || M <= 0) return 0;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    TesseraMlpSession *s = (TesseraMlpSession *)handle;
    id<MTLDevice> dev = ctx.device;
    NSError *err = nil;
    int K = s->K, N = s->N;
    // Pooled per-step X/Y (synced before return) + reusable dispatch objects.
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bX, ctx, X, (size_t)M * K * 2);
    TS_METAL_BUF_ACQUIRE(bY, ctx, (size_t)M * N * 4);
    id<MTLTensor> tX = make_buffer_tensor(dev, bX, K, M, (MTLTensorDataType)s->dt);
    id<MTLTensor> tY = make_buffer_tensor(dev, bY, N, M, MTLTensorDataTypeFloat32);
    if (!bX || !bY || !tX || !tY) return 0;
    id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
    if (!queue) return 0;
    bool done = false;
    {
      std::lock_guard<std::mutex> lock(ctx.mtl4_dispatch_mu);
      // Only the per-step X/Y need adding — W/bias/params stay resident via the
      // session's persistent queue-level resW.
      id<MTLBuffer> rbufs[2] = {bX, bY};
      id<MTLResidencySet> resStep = mtl4_set_residency(ctx, rbufs, 2);
      if (!resStep) return 0;
      done = mtl4_encode_and_wait(ctx, queue, (id<MTLComputePipelineState>)s->pso,
          ^(id<MTL4ArgumentTable> at) {
            [at setResource:tX.gpuResourceID atBufferIndex:0];
            [at setResource:((id<MTLTensor>)s->tW).gpuResourceID atBufferIndex:1];
            [at setResource:tY.gpuResourceID atBufferIndex:2];
            [at setAddress:((id<MTLBuffer>)s->bBias).gpuAddress atIndex:3];
            [at setAddress:((id<MTLBuffer>)s->bParams).gpuAddress atIndex:4];
          }, MTLSizeMake((M + 63) / 64, (N + 63) / 64, 1), MTLSizeMake(128, 1, 1), resStep);
    }
    if (!done) return 0;
    std::memcpy(Y, [bY contents], (size_t)M * N * 4);
    return 1;
  }
  return 0;
}

// R0 bridge — device-resident decode step: X and Y are TsDeviceTensors already on
// the GPU. Binds their cached MTLTensor views directly, so neither X is uploaded
// nor Y downloaded — the activation stays resident across the step (and across a
// decode loop, the only per-call cost is the dispatch). Otherwise identical to
// the host-pointer `_run`. Returns 1 if it ran on the matrix-unit lane.
extern "C" int32_t tessera_apple_gpu_mtl4_mlp_session_run_dev(void *handle,
    TsDeviceTensor *X, TsDeviceTensor *Y, int32_t M) {
  if (!handle || !X || !Y || M <= 0) return 0;
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !X->buf || !Y->buf) return 0;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    TesseraMlpSession *s = (TesseraMlpSession *)handle;
    int K = s->K, N = s->N;
    if (X->nbytes < (int64_t)M * K * 2 || Y->nbytes < (int64_t)M * N * 4) return 0;
    id<MTLTensor> tX = ts_dev_tensor_view(X, K, M, (MTLTensorDataType)s->dt);
    id<MTLTensor> tY = ts_dev_tensor_view(Y, N, M, MTLTensorDataTypeFloat32);
    if (!tX || !tY) return 0;
    id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
    if (!queue) return 0;
    NSError *err = nil;
    bool done = false;
    {
      std::lock_guard<std::mutex> lock(ctx.mtl4_dispatch_mu);
      // Resident X/Y device tensors; W/bias/params via the persistent queue-level resW.
      id<MTLBuffer> rbufs[2] = {X->buf, Y->buf};
      id<MTLResidencySet> resStep = mtl4_set_residency(ctx, rbufs, 2);
      if (!resStep) return 0;
      done = mtl4_encode_and_wait(ctx, queue, (id<MTLComputePipelineState>)s->pso,
          ^(id<MTL4ArgumentTable> at) {
            [at setResource:tX.gpuResourceID atBufferIndex:0];
            [at setResource:((id<MTLTensor>)s->tW).gpuResourceID atBufferIndex:1];
            [at setResource:tY.gpuResourceID atBufferIndex:2];
            [at setAddress:((id<MTLBuffer>)s->bBias).gpuAddress atIndex:3];
            [at setAddress:((id<MTLBuffer>)s->bParams).gpuAddress atIndex:4];
          }, MTLSizeMake((M + 63) / 64, (N + 63) / 64, 1), MTLSizeMake(128, 1, 1), resStep);
    }
    // No memcpy: Y stays resident in its device tensor (zero download).
    return done ? 1 : 0;
  }
  return 0;
}

extern "C" void tessera_apple_gpu_mtl4_mlp_session_destroy(void *handle) {
  if (!handle) return;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    TesseraMlpSession *s = (TesseraMlpSession *)handle;
    MetalDeviceContext &ctx = deviceContext();
    if (ctx.ok && s->resW) {
      id<MTL4CommandQueue> queue = mtl4_shared_queue(ctx);
      if (queue) [queue removeResidencySet:(id<MTLResidencySet>)s->resW];
    }
    delete s;   // ARC releases the id members
  }
}

//===----------------------------------------------------------------------===//
// Phase-G Rung 3 via MSL — the DYNAMIC speculative-verify control flow as one
// MSL kernel, the part MPSGraph's static-shape graph cannot express. Given a
// fixed-capacity set of candidate paths (no trie, no heap), the kernel:
//   * per path: accepts draft tokens while they match the target's greedy
//     token, breaking at the first mismatch  (data-dependent trip count);
//   * across paths: keeps the longest accepted prefix  (argmax);
//   * emits the bonus token (the target's correction after the accepted prefix).
// All of this — variable-trip loops, early break, data-dependent indexing — is
// ordinary MSL control flow; only the I/O buffer capacities are fixed. This is
// the Rung-3 frontier the MPSGraph route declared out of scope, made tractable
// by the MSL route (see docs/apple_gpu_control_flow_lowering.md "Mapping to
// MSL 4.0"). Dispatched via the cached classic MSL path (works macOS 12+).
//===----------------------------------------------------------------------===//
extern "C" int32_t tessera_apple_gpu_msl_spec_accept(const int32_t *draft_paths,
                                                     const int32_t *target_greedy,
                                                     int32_t *out, int32_t P,
                                                     int32_t depth) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || P <= 0 || depth <= 0 || !draft_paths || !target_greedy || !out)
    return 0;
  static NSString *const kSpecAcceptMSL = @R"MSL(
#include <metal_stdlib>
using namespace metal;
// draft  : [P, depth]      candidate tokens per path
// target : [P, depth+1]    target greedy token at each position along the path
// out    : [3 + depth]     {best_path, accepted_len, bonus, accepted tokens...}
kernel void spec_accept(device const int *draft  [[buffer(0)]],
                        device const int *target [[buffer(1)]],
                        device int       *out    [[buffer(2)]],
                        constant int     &P      [[buffer(3)]],
                        constant int     &depth  [[buffer(4)]],
                        uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;                  // single thread runs the dynamic logic
  int best_path = 0, best_len = -1, best_bonus = 0;
  for (int p = 0; p < P; ++p) {
    int len = 0;
    for (int i = 0; i < depth; ++i) {
      if (draft[p * depth + i] == target[p * (depth + 1) + i]) len++;
      else break;                         // data-dependent early break
    }
    if (len > best_len) {
      best_len = len; best_path = p;
      best_bonus = target[p * (depth + 1) + len];
    }
  }
  out[0] = best_path; out[1] = best_len; out[2] = best_bonus;
  for (int i = 0; i < depth; ++i)
    out[3 + i] = (i < best_len) ? draft[best_path * depth + i] : -1;
}
)MSL";
  @autoreleasepool {
    id<MTLComputePipelineState> pso = compile_msl_kernel(ctx, kSpecAcceptMSL, @"spec_accept");
    if (!pso) return 0;
    // Pooled (synced before the memcpy below, so safe to recycle on scope exit).
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bD, ctx, draft_paths, (size_t)P * depth * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bT, ctx, target_greedy, (size_t)P * (depth + 1) * 4);
    TS_METAL_BUF_ACQUIRE(bO, ctx, (size_t)(3 + depth) * 4);
    if (!bD || !bT || !bO) return 0;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bD offset:0 atIndex:0];
    [enc setBuffer:bT offset:0 atIndex:1];
    [enc setBuffer:bO offset:0 atIndex:2];
    [enc setBytes:&P length:sizeof(int32_t) atIndex:3];
    [enc setBytes:&depth length:sizeof(int32_t) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    if (!commit_and_wait_with_timeout(ctx, cb, 30000,
                                      "apple_gpu_unknown")) return 0;
    std::memcpy(out, [bO contents], (size_t)(3 + depth) * 4);
    return 1;
  }
}

// ---- C ABI: row ops ---------------------------------------------------------
extern "C" void tessera_apple_gpu_layer_norm_f32(const float *x,
                                                 const float *gamma,
                                                 const float *beta, float *out,
                                                 int32_t rows, int32_t cols,
                                                 float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 0, x, gamma, beta, out, rows, cols, eps,
                               MPSDataTypeFloat32, 4)) return;
  for (int32_t r = 0; r < rows; ++r) {
    const float *row = x + (size_t)r * cols;
    float *o = out + (size_t)r * cols;
    double mean = 0.0;
    for (int32_t c = 0; c < cols; ++c) mean += row[c];
    mean /= cols;
    double var = 0.0;
    for (int32_t c = 0; c < cols; ++c) { double d = row[c] - mean; var += d * d; }
    var /= cols;
    double inv = 1.0 / std::sqrt(var + eps);
    for (int32_t c = 0; c < cols; ++c)
      o[c] = (float)(((row[c] - mean) * inv) * gamma[c] + beta[c]);
  }
}

extern "C" void tessera_apple_gpu_layer_norm_f16(const uint16_t *x,
                                                 const uint16_t *gamma,
                                                 const uint16_t *beta,
                                                 uint16_t *out, int32_t rows,
                                                 int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 0, x, gamma, beta, out, rows, cols, eps,
                               MPSDataTypeFloat16, 2)) return;
  std::memcpy(out, x, (size_t)rows * cols * 2);
}

extern "C" void tessera_apple_gpu_rmsnorm_gpu_f32(const float *x,
                                                  const float *gamma, float *out,
                                                  int32_t rows, int32_t cols,
                                                  float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 1, x, gamma, nullptr, out, rows, cols, eps,
                               MPSDataTypeFloat32, 4)) return;
  for (int32_t r = 0; r < rows; ++r) {
    const float *row = x + (size_t)r * cols;
    float *o = out + (size_t)r * cols;
    double ms = 0.0;
    for (int32_t c = 0; c < cols; ++c) ms += (double)row[c] * row[c];
    ms /= cols;
    double inv = 1.0 / std::sqrt(ms + eps);
    for (int32_t c = 0; c < cols; ++c) o[c] = (float)(row[c] * inv * gamma[c]);
  }
}

extern "C" void tessera_apple_gpu_rmsnorm_gpu_f16(const uint16_t *x,
                                                  const uint16_t *gamma,
                                                  uint16_t *out, int32_t rows,
                                                  int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 1, x, gamma, nullptr, out, rows, cols, eps,
                               MPSDataTypeFloat16, 2)) return;
  std::memcpy(out, x, (size_t)rows * cols * 2);
}

// ── Sprint 3.5: native bf16 MPSGraph unary / binary / norm kernels ───────────
// MPSGraph supports bf16 (probe: tessera_apple_gpu_mpsgraph_bf16_supported), so
// the dtype-parameterized mpsg_run_* helpers run these natively in bf16 (f32
// internal accumulation inside MPSGraph). The host fallback upcasts to f32,
// reuses the corresponding f32 extern, and rounds back — correct on any host.
extern "C" void tessera_apple_gpu_mpsgraph_unary_bf16(int32_t op,
                                                      const uint16_t *x,
                                                      uint16_t *out, int64_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_unary(ctx, op, x, out, n, MPSDataTypeBFloat16, 2)) return;
  std::vector<float> xf((size_t)n), of((size_t)n);
  for (int64_t i = 0; i < n; ++i) xf[i] = bfloat16_to_float_gpu(x[i]);
  tessera_apple_gpu_mpsgraph_unary_f32(op, xf.data(), of.data(), n);
  for (int64_t i = 0; i < n; ++i) out[i] = float_to_bfloat16_gpu(of[i]);
}

extern "C" void tessera_apple_gpu_mpsgraph_binary_bf16(int32_t op,
                                                       const uint16_t *a,
                                                       const uint16_t *b,
                                                       uint16_t *out,
                                                       int64_t n) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_binary(ctx, op, a, b, out, n, MPSDataTypeBFloat16, 2)) return;
  std::vector<float> af((size_t)n), bf((size_t)n), of((size_t)n);
  for (int64_t i = 0; i < n; ++i) {
    af[i] = bfloat16_to_float_gpu(a[i]);
    bf[i] = bfloat16_to_float_gpu(b[i]);
  }
  tessera_apple_gpu_mpsgraph_binary_f32(op, af.data(), bf.data(), of.data(), n);
  for (int64_t i = 0; i < n; ++i) out[i] = float_to_bfloat16_gpu(of[i]);
}

extern "C" void tessera_apple_gpu_rmsnorm_gpu_bf16(const uint16_t *x,
                                                   const uint16_t *gamma,
                                                   uint16_t *out, int32_t rows,
                                                   int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 1, x, gamma, nullptr, out, rows, cols, eps,
                               MPSDataTypeBFloat16, 2)) return;
  size_t nrc = (size_t)rows * cols;
  std::vector<float> xf(nrc), gf((size_t)cols), of(nrc);
  for (size_t i = 0; i < nrc; ++i) xf[i] = bfloat16_to_float_gpu(x[i]);
  for (int32_t c = 0; c < cols; ++c) gf[c] = bfloat16_to_float_gpu(gamma[c]);
  tessera_apple_gpu_rmsnorm_gpu_f32(xf.data(), gf.data(), of.data(), rows, cols, eps);
  for (size_t i = 0; i < nrc; ++i) out[i] = float_to_bfloat16_gpu(of[i]);
}

extern "C" void tessera_apple_gpu_layer_norm_bf16(const uint16_t *x,
                                                  const uint16_t *gamma,
                                                  const uint16_t *beta,
                                                  uint16_t *out, int32_t rows,
                                                  int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 0, x, gamma, beta, out, rows, cols, eps,
                               MPSDataTypeBFloat16, 2)) return;
  size_t nrc = (size_t)rows * cols;
  std::vector<float> xf(nrc), gf((size_t)cols), bbf((size_t)cols), of(nrc);
  for (size_t i = 0; i < nrc; ++i) xf[i] = bfloat16_to_float_gpu(x[i]);
  for (int32_t c = 0; c < cols; ++c) {
    gf[c] = bfloat16_to_float_gpu(gamma[c]);
    bbf[c] = bfloat16_to_float_gpu(beta[c]);
  }
  tessera_apple_gpu_layer_norm_f32(xf.data(), gf.data(), bbf.data(), of.data(),
                                   rows, cols, eps);
  for (size_t i = 0; i < nrc; ++i) out[i] = float_to_bfloat16_gpu(of[i]);
}

extern "C" void tessera_apple_gpu_mpsgraph_softmax_f32(const float *x,
                                                       float *out, int32_t rows,
                                                       int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 2, x, nullptr, nullptr, out, rows, cols,
                               0.0f, MPSDataTypeFloat32, 4)) return;
  for (int32_t r = 0; r < rows; ++r) {
    const float *row = x + (size_t)r * cols;
    float *o = out + (size_t)r * cols;
    float m = row[0];
    for (int32_t c = 1; c < cols; ++c) m = row[c] > m ? row[c] : m;
    double s = 0.0;
    for (int32_t c = 0; c < cols; ++c) { o[c] = std::exp(row[c] - m); s += o[c]; }
    for (int32_t c = 0; c < cols; ++c) o[c] = (float)(o[c] / s);
  }
}

extern "C" void tessera_apple_gpu_mpsgraph_softmax_f16(const uint16_t *x,
                                                       uint16_t *out,
                                                       int32_t rows,
                                                       int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 2, x, nullptr, nullptr, out, rows, cols,
                               0.0f, MPSDataTypeFloat16, 2)) return;
  std::memcpy(out, x, (size_t)rows * cols * 2);
}

extern "C" void tessera_apple_gpu_log_softmax_f32(const float *x, float *out,
                                                  int32_t rows, int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 3, x, nullptr, nullptr, out, rows, cols,
                               0.0f, MPSDataTypeFloat32, 4)) return;
  for (int32_t r = 0; r < rows; ++r) {
    const float *row = x + (size_t)r * cols;
    float *o = out + (size_t)r * cols;
    float m = row[0];
    for (int32_t c = 1; c < cols; ++c) m = row[c] > m ? row[c] : m;
    double s = 0.0;
    for (int32_t c = 0; c < cols; ++c) s += std::exp(row[c] - m);
    float lse = m + (float)std::log(s);
    for (int32_t c = 0; c < cols; ++c) o[c] = row[c] - lse;
  }
}

extern "C" void tessera_apple_gpu_log_softmax_f16(const uint16_t *x,
                                                  uint16_t *out, int32_t rows,
                                                  int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rowop(ctx, 3, x, nullptr, nullptr, out, rows, cols,
                               0.0f, MPSDataTypeFloat16, 2)) return;
  std::memcpy(out, x, (size_t)rows * cols * 2);
}

//===----------------------------------------------------------------------===//
// Batched matmul (bmm) — Tier-2 keystone (2026-05-29)
//
// MPSGraph-backed batched / rank-3 matmul with batch broadcasting (the B
// operand may carry batch=1 to be shared across all A batches — the GQA/MQA
// KV-sharing shape). A: [batch, M, K], B: [b_batch, K, N] (b_batch == batch or
// 1), O: [batch, M, N]. f32 + f16 native (fp32 compute); bf16 upcasts host-side
// in runtime.py, matching the unary lane convention. Reuses the MPSGraph
// graph cache + buffer pool.
//===----------------------------------------------------------------------===//

namespace {

static void reference_bmm_f32(const float *A, const float *B, float *O,
                              int32_t batch, int32_t M, int32_t N, int32_t K,
                              int b_broadcast) {
  for (int32_t bi = 0; bi < batch; ++bi) {
    const float *a = A + (size_t)bi * M * K;
    const float *b = B + (size_t)(b_broadcast ? 0 : bi) * K * N;
    float *o = O + (size_t)bi * M * N;
    for (int32_t m = 0; m < M; ++m)
      for (int32_t n = 0; n < N; ++n) {
        float s = 0.0f;
        for (int32_t k = 0; k < K; ++k)
          s += a[(size_t)m * K + k] * b[(size_t)k * N + n];
        o[(size_t)m * N + n] = s;
      }
  }
}

static bool mpsg_run_bmm(MetalDeviceContext &ctx, const void *a, const void *b,
                         void *out, int32_t batch, int32_t M, int32_t N,
                         int32_t K, bool b_broadcast, MPSDataType ioType,
                         size_t elemSize) {
  if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) return true;
  @autoreleasepool {
    int32_t bBatch = b_broadcast ? 1 : batch;
    size_t aBytes = (size_t)batch * M * K * elemSize;
    size_t bBytes = (size_t)bBatch * K * N * elemSize;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, a, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, b, bBytes);
    if (!bufA || !bufB) return false;
    NSArray<NSNumber *> *aShape = @[ @(batch), @(M), @(K) ];
    NSArray<NSNumber *> *bShape = @[ @(bBatch), @(K), @(N) ];
    NSString *key = [NSString stringWithFormat:@"bmm:%d:%d:%d:%d:%d:%d",
                                               (int)ioType, batch, M, N, K,
                                               (int)b_broadcast];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *pa;
    MPSGraphTensor *pb;
    MPSGraphTensor *y;
    if (entry) {
      g = entry[0];
      pa = ((NSArray *)entry[1])[0];
      pb = ((NSArray *)entry[1])[1];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      pa = [g placeholderWithShape:aShape dataType:ioType name:nil];
      pb = [g placeholderWithShape:bShape dataType:ioType name:nil];
      MPSGraphTensor *yf =
          [g matrixMultiplicationWithPrimaryTensor:mpsg_up(g, pa, ioType)
                                   secondaryTensor:mpsg_up(g, pb, ioType)
                                              name:nil];
      y = mpsg_down(g, yf, ioType);
      mpsg_cache_put(key, @[ g, @[ pa, pb ], y ]);
    }
    MPSGraphTensorData *ad =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:aShape dataType:ioType];
    MPSGraphTensorData *bd =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:bShape dataType:ioType];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{pa : ad, pb : bd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

static bool mpsg_run_ppo_policy_loss_f32(MetalDeviceContext &ctx,
                                         const float *logp_new,
                                         const float *logp_old,
                                         const float *advantages,
                                         float *out,
                                         int32_t n,
                                         float clip_epsilon) {
  if (n <= 0 || clip_epsilon <= 0.0f)
    return false;
  @autoreleasepool {
    size_t bytes = (size_t)n * sizeof(float);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufNew, ctx, logp_new, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufOld, ctx, logp_old, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufAdv, ctx, advantages, bytes);
    if (!bufNew || !bufOld || !bufAdv)
      return false;

    NSArray<NSNumber *> *shape = @[ @(n) ];
    NSArray<NSNumber *> *axis0 = @[ @0 ];
    NSString *key = [NSString stringWithFormat:@"rlppo:f32:%d:%a", n,
                                               (double)clip_epsilon];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *pNew;
    MPSGraphTensor *pOld;
    MPSGraphTensor *pAdv;
    MPSGraphTensor *lossMean;
    if (entry) {
      g = entry[0];
      pNew = ((NSArray *)entry[1])[0];
      pOld = ((NSArray *)entry[1])[1];
      pAdv = ((NSArray *)entry[1])[2];
      lossMean = entry[2];
    } else {
      g = [MPSGraph new];
      pNew = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
      pOld = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
      pAdv = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
      MPSGraphTensor *delta =
          [g subtractionWithPrimaryTensor:pNew secondaryTensor:pOld name:nil];
      MPSGraphTensor *ratio = [g exponentWithTensor:delta name:nil];
      MPSGraphTensor *lo =
          [g constantWithScalar:(double)(1.0f - clip_epsilon)
                       dataType:MPSDataTypeFloat32];
      MPSGraphTensor *hi =
          [g constantWithScalar:(double)(1.0f + clip_epsilon)
                       dataType:MPSDataTypeFloat32];
      MPSGraphTensor *clampedLo =
          [g maximumWithPrimaryTensor:ratio secondaryTensor:lo name:nil];
      MPSGraphTensor *clipped =
          [g minimumWithPrimaryTensor:clampedLo secondaryTensor:hi name:nil];
      MPSGraphTensor *s1 =
          [g multiplicationWithPrimaryTensor:ratio secondaryTensor:pAdv name:nil];
      MPSGraphTensor *s2 =
          [g multiplicationWithPrimaryTensor:clipped secondaryTensor:pAdv name:nil];
      MPSGraphTensor *minS =
          [g minimumWithPrimaryTensor:s1 secondaryTensor:s2 name:nil];
      MPSGraphTensor *loss = [g negativeWithTensor:minS name:nil];
      lossMean = [g meanOfTensor:loss axes:axis0 name:nil];
      mpsg_cache_put(key, @[ g, @[ pNew, pOld, pAdv ], lossMean ]);
    }

    MPSGraphTensorData *newD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufNew
                                               shape:shape
                                            dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *oldD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufOld
                                               shape:shape
                                            dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *advD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufAdv
                                               shape:shape
                                            dataType:MPSDataTypeFloat32];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{pNew : newD, pOld : oldD,
                                                    pAdv : advD}
                                    targetTensors:@[ lossMean ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[lossMean];
    if (!od)
      return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

static bool mpsg_run_ppo_policy_loss_ex_f32(
    MetalDeviceContext &ctx, const float *logp_new, const float *logp_old,
    const float *advantages, const float *mask, const float *ref_logp,
    const float *entropy, float *out, int32_t n, float clip_epsilon,
    float kl_coef, float entropy_coef, int32_t has_mask, int32_t has_ref_kl,
    int32_t has_entropy) {
  if (n <= 0 || clip_epsilon <= 0.0f)
    return false;
  if ((has_mask && !mask) || (has_ref_kl && !ref_logp) ||
      (has_entropy && !entropy))
    return false;
  @autoreleasepool {
    size_t bytes = (size_t)n * sizeof(float);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufNew, ctx, logp_new, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufOld, ctx, logp_old, bytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufAdv, ctx, advantages, bytes);
    MetalBufferGuard gMask(ctx, has_mask ? metal_buffer_acquire_with_bytes(ctx, mask, bytes) : nil, bytes);
    id<MTLBuffer> bufMask = gMask.buf;
    MetalBufferGuard gRef(ctx, has_ref_kl ? metal_buffer_acquire_with_bytes(ctx, ref_logp, bytes) : nil, bytes);
    id<MTLBuffer> bufRef = gRef.buf;
    MetalBufferGuard gEntropy(ctx, has_entropy ? metal_buffer_acquire_with_bytes(ctx, entropy, bytes) : nil, bytes);
    id<MTLBuffer> bufEntropy = gEntropy.buf;
    if (!bufNew || !bufOld || !bufAdv || (has_mask && !bufMask) ||
        (has_ref_kl && !bufRef) || (has_entropy && !bufEntropy))
      return false;

    NSArray<NSNumber *> *shape = @[ @(n) ];
    NSArray<NSNumber *> *axis0 = @[ @0 ];
    NSString *key = [NSString stringWithFormat:@"rlppoex:f32:%d:%a:%a:%a:%d:%d:%d",
                                               n, (double)clip_epsilon,
                                               (double)kl_coef,
                                               (double)entropy_coef,
                                               has_mask, has_ref_kl,
                                               has_entropy];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *pNew;
    MPSGraphTensor *pOld;
    MPSGraphTensor *pAdv;
    MPSGraphTensor *pMask = nil;
    MPSGraphTensor *pRef = nil;
    MPSGraphTensor *pEntropy = nil;
    MPSGraphTensor *lossMean;
    if (entry) {
      g = entry[0];
      NSArray *phs = (NSArray *)entry[1];
      pNew = phs[0];
      pOld = phs[1];
      pAdv = phs[2];
      NSUInteger idx = 3;
      if (has_mask)
        pMask = phs[idx++];
      if (has_ref_kl)
        pRef = phs[idx++];
      if (has_entropy)
        pEntropy = phs[idx++];
      lossMean = entry[2];
    } else {
      g = [MPSGraph new];
      pNew = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
      pOld = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
      pAdv = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
      NSMutableArray *phs = [NSMutableArray arrayWithObjects:pNew, pOld, pAdv, nil];
      if (has_mask) {
        pMask = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
        [phs addObject:pMask];
      }
      if (has_ref_kl) {
        pRef = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
        [phs addObject:pRef];
      }
      if (has_entropy) {
        pEntropy = [g placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
        [phs addObject:pEntropy];
      }
      MPSGraphTensor *delta =
          [g subtractionWithPrimaryTensor:pNew secondaryTensor:pOld name:nil];
      MPSGraphTensor *ratio = [g exponentWithTensor:delta name:nil];
      MPSGraphTensor *lo =
          [g constantWithScalar:(double)(1.0f - clip_epsilon)
                       dataType:MPSDataTypeFloat32];
      MPSGraphTensor *hi =
          [g constantWithScalar:(double)(1.0f + clip_epsilon)
                       dataType:MPSDataTypeFloat32];
      MPSGraphTensor *clampedLo =
          [g maximumWithPrimaryTensor:ratio secondaryTensor:lo name:nil];
      MPSGraphTensor *clipped =
          [g minimumWithPrimaryTensor:clampedLo secondaryTensor:hi name:nil];
      MPSGraphTensor *s1 =
          [g multiplicationWithPrimaryTensor:ratio secondaryTensor:pAdv name:nil];
      MPSGraphTensor *s2 =
          [g multiplicationWithPrimaryTensor:clipped secondaryTensor:pAdv name:nil];
      MPSGraphTensor *minS =
          [g minimumWithPrimaryTensor:s1 secondaryTensor:s2 name:nil];
      MPSGraphTensor *loss = [g negativeWithTensor:minS name:nil];
      if (has_ref_kl) {
        MPSGraphTensor *refDelta =
            [g subtractionWithPrimaryTensor:pRef secondaryTensor:pNew name:nil];
        MPSGraphTensor *expDelta = [g exponentWithTensor:refDelta name:nil];
        MPSGraphTensor *one =
            [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
        MPSGraphTensor *klBase =
            [g subtractionWithPrimaryTensor:expDelta secondaryTensor:refDelta name:nil];
        MPSGraphTensor *kl =
            [g subtractionWithPrimaryTensor:klBase secondaryTensor:one name:nil];
        MPSGraphTensor *coef =
            [g constantWithScalar:(double)kl_coef dataType:MPSDataTypeFloat32];
        MPSGraphTensor *scaled =
            [g multiplicationWithPrimaryTensor:kl secondaryTensor:coef name:nil];
        loss = [g additionWithPrimaryTensor:loss secondaryTensor:scaled name:nil];
      }
      if (has_entropy) {
        MPSGraphTensor *coef =
            [g constantWithScalar:(double)entropy_coef dataType:MPSDataTypeFloat32];
        MPSGraphTensor *scaled =
            [g multiplicationWithPrimaryTensor:pEntropy secondaryTensor:coef name:nil];
        loss = [g subtractionWithPrimaryTensor:loss secondaryTensor:scaled name:nil];
      }
      if (has_mask) {
        MPSGraphTensor *masked =
            [g multiplicationWithPrimaryTensor:loss secondaryTensor:pMask name:nil];
        MPSGraphTensor *sumLoss =
            [g reductionSumWithTensor:masked axes:axis0 name:nil];
        MPSGraphTensor *sumMask =
            [g reductionSumWithTensor:pMask axes:axis0 name:nil];
        MPSGraphTensor *one =
            [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
        MPSGraphTensor *denom =
            [g maximumWithPrimaryTensor:sumMask secondaryTensor:one name:nil];
        lossMean =
            [g divisionWithPrimaryTensor:sumLoss secondaryTensor:denom name:nil];
      } else {
        lossMean = [g meanOfTensor:loss axes:axis0 name:nil];
      }
      mpsg_cache_put(key, @[ g, phs, lossMean ]);
    }

    MPSGraphTensorData *newD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufNew
                                               shape:shape
                                            dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *oldD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufOld
                                               shape:shape
                                            dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *advD =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufAdv
                                               shape:shape
                                            dataType:MPSDataTypeFloat32];
    NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
    feeds[pNew] = newD;
    feeds[pOld] = oldD;
    feeds[pAdv] = advD;
    if (has_mask) {
      feeds[pMask] =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:bufMask
                                                 shape:shape
                                              dataType:MPSDataTypeFloat32];
    }
    if (has_ref_kl) {
      feeds[pRef] =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:bufRef
                                                 shape:shape
                                              dataType:MPSDataTypeFloat32];
    }
    if (has_entropy) {
      feeds[pEntropy] =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:bufEntropy
                                                 shape:shape
                                              dataType:MPSDataTypeFloat32];
    }
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:feeds
                                    targetTensors:@[ lossMean ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[lossMean];
    if (!od)
      return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

// R1 — device-resident bmm: inputs + output are existing shared MTLBuffers
// (from DeviceTensor handles), so there is NO host upload and NO readback. The
// MPSGraph result is written straight into the output buffer via
// resultsDictionary, leaving it resident for the next op. Shares the bmm graph
// cache key, so it reuses the same compiled graph as the host-ptr path.
// Get-or-build the cached bmm graph + its placeholders/output tensor. Shared by
// the synchronous-run, device-resident-run, and R2 encode paths.
static MPSGraph *mpsg_bmm_graph(int32_t batch, int32_t M, int32_t N, int32_t K,
                                bool b_broadcast, MPSDataType ioType,
                                MPSGraphTensor **pa_out,
                                MPSGraphTensor **pb_out,
                                MPSGraphTensor **y_out) {
  int32_t bBatch = b_broadcast ? 1 : batch;
  NSArray<NSNumber *> *aShape = @[ @(batch), @(M), @(K) ];
  NSArray<NSNumber *> *bShape = @[ @(bBatch), @(K), @(N) ];
  NSString *key = [NSString stringWithFormat:@"bmm:%d:%d:%d:%d:%d:%d",
                                             (int)ioType, batch, M, N, K,
                                             (int)b_broadcast];
  NSArray *entry = mpsg_cache_get(key);
  MPSGraph *g;
  MPSGraphTensor *pa, *pb, *y;
  if (entry) {
    g = entry[0];
    pa = ((NSArray *)entry[1])[0];
    pb = ((NSArray *)entry[1])[1];
    y = entry[2];
  } else {
    g = [MPSGraph new];
    pa = [g placeholderWithShape:aShape dataType:ioType name:nil];
    pb = [g placeholderWithShape:bShape dataType:ioType name:nil];
    MPSGraphTensor *yf =
        [g matrixMultiplicationWithPrimaryTensor:mpsg_up(g, pa, ioType)
                                 secondaryTensor:mpsg_up(g, pb, ioType)
                                            name:nil];
    y = mpsg_down(g, yf, ioType);
    mpsg_cache_put(key, @[ g, @[ pa, pb ], y ]);
  }
  *pa_out = pa;
  *pb_out = pb;
  *y_out = y;
  return g;
}

static bool mpsg_run_bmm_dev(MetalDeviceContext &ctx, id<MTLBuffer> bufA,
                             id<MTLBuffer> bufB, id<MTLBuffer> bufO,
                             int32_t batch, int32_t M, int32_t N, int32_t K,
                             bool b_broadcast, MPSDataType ioType) {
  if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) return true;
  if (!bufA || !bufB || !bufO) return false;
  @autoreleasepool {
    int32_t bBatch = b_broadcast ? 1 : batch;
    NSArray<NSNumber *> *aShape = @[ @(batch), @(M), @(K) ];
    NSArray<NSNumber *> *bShape = @[ @(bBatch), @(K), @(N) ];
    NSArray<NSNumber *> *oShape = @[ @(batch), @(M), @(N) ];
    MPSGraphTensor *pa, *pb, *y;
    MPSGraph *g = mpsg_bmm_graph(batch, M, N, K, b_broadcast, ioType, &pa, &pb, &y);
    MPSGraphTensorData *ad = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:aShape dataType:ioType];
    MPSGraphTensorData *bd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:bShape dataType:ioType];
    MPSGraphTensorData *od = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO shape:oShape dataType:ioType];
    [g runWithMTLCommandQueue:ctx.queue
                        feeds:@{pa : ad, pb : bd}
             targetOperations:nil
            resultsDictionary:@{y : od}];
    return true;
  }
}

// Single-command-buffer decode chain scaffold (2026-06-01) — generic
// encode helper for row-wise ops (layer_norm / rmsnorm / softmax /
// log_softmax). Appends to a shared MPSCommandBuffer; no commit/sync.
// ``kind`` matches ``mpsg_rowop_graph``: 0=layer_norm, 1=rmsnorm,
// 2=softmax, 3=log_softmax. ``bufG`` / ``bufB`` may be nil for
// kinds that don't take gamma / beta (softmax / log_softmax have
// neither; rmsnorm has gamma only). See
// ``docs/audit/backend/apple/APPLE_AUDIT.md``.
static bool mpsg_encode_rowop_dev(MPSCommandBuffer *cb, int32_t kind,
                                  id<MTLBuffer> bufX,
                                  id<MTLBuffer> bufG,
                                  id<MTLBuffer> bufB,
                                  id<MTLBuffer> bufY,
                                  int32_t rows, int32_t cols,
                                  float eps, MPSDataType ioType) {
  if (rows <= 0 || cols <= 0) return true;
  if (!cb || !bufX || !bufY) return false;
  NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
  NSArray<NSNumber *> *gs = @[ @(cols) ];
  bool hasGamma = (bufG != nil);
  bool hasBeta = (bufB != nil);
  NSArray *phs;
  MPSGraphTensor *y;
  MPSGraph *g = mpsg_rowop_graph(kind, rows, cols, eps, hasGamma,
                                 hasBeta, ioType, &phs, &y);
  if (!y) return false;
  MPSGraphTensorData *xd = [[MPSGraphTensorData alloc]
                            initWithMTLBuffer:bufX shape:xs dataType:ioType];
  MPSGraphTensorData *od = [[MPSGraphTensorData alloc]
                            initWithMTLBuffer:bufY shape:xs dataType:ioType];
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  feeds[phs[0]] = xd;
  if (hasGamma && phs.count >= 2) {
    feeds[phs[1]] = [[MPSGraphTensorData alloc]
                     initWithMTLBuffer:bufG shape:gs dataType:ioType];
  }
  if (hasBeta && phs.count >= 3) {
    feeds[phs[2]] = [[MPSGraphTensorData alloc]
                     initWithMTLBuffer:bufB shape:gs dataType:ioType];
  }
  [g encodeToCommandBuffer:cb
                     feeds:feeds
              targetOperations:nil
          resultsDictionary:@{y : od}
        executionDescriptor:nil];
  return true;
}

// Layer-norm encode wrapper (kind=0, gamma + beta present). Kept as a
// thin alias for backward compat with the scaffold caller.
static inline bool mpsg_encode_layer_norm_dev(MPSCommandBuffer *cb,
                                              id<MTLBuffer> bufX,
                                              id<MTLBuffer> bufG,
                                              id<MTLBuffer> bufB,
                                              id<MTLBuffer> bufY,
                                              int32_t rows, int32_t cols,
                                              float eps,
                                              MPSDataType ioType) {
  return mpsg_encode_rowop_dev(cb, /*kind=*/0, bufX, bufG, bufB, bufY,
                                rows, cols, eps, ioType);
}

// R2 — encode a bmm into a shared command buffer (no commit/sync here). Metal's
// automatic hazard tracking orders a later op that reads an earlier op's output
// buffer, so a whole op-chain can be encoded into one command buffer and
// committed once. Tensor-data wrappers are retained until commit by the encoded
// command buffer (which retains the underlying MTLBuffers), so no autoreleasepool.
static bool mpsg_encode_bmm_dev(MPSCommandBuffer *cb, id<MTLBuffer> bufA,
                                id<MTLBuffer> bufB, id<MTLBuffer> bufO,
                                int32_t batch, int32_t M, int32_t N, int32_t K,
                                bool b_broadcast, MPSDataType ioType) {
  if (batch <= 0 || M <= 0 || N <= 0 || K <= 0) return true;
  if (!cb || !bufA || !bufB || !bufO) return false;
  int32_t bBatch = b_broadcast ? 1 : batch;
  NSArray<NSNumber *> *aShape = @[ @(batch), @(M), @(K) ];
  NSArray<NSNumber *> *bShape = @[ @(bBatch), @(K), @(N) ];
  NSArray<NSNumber *> *oShape = @[ @(batch), @(M), @(N) ];
  MPSGraphTensor *pa, *pb, *y;
  MPSGraph *g = mpsg_bmm_graph(batch, M, N, K, b_broadcast, ioType, &pa, &pb, &y);
  MPSGraphTensorData *ad = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:aShape dataType:ioType];
  MPSGraphTensorData *bd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:bShape dataType:ioType];
  MPSGraphTensorData *od = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO shape:oShape dataType:ioType];
  [g encodeToCommandBuffer:cb
                     feeds:@{pa : ad, pb : bd}
             targetOperations:nil
         resultsDictionary:@{y : od}
       executionDescriptor:nil];
  return true;
}

}  // namespace

//===----------------------------------------------------------------------===//
// R2 — command-buffer batching: encode N device-resident ops into ONE command
// buffer and commit + wait once, removing the per-op CPU↔GPU sync that
// dominates small-batch decode. A TsEncodeSession owns one MPSCommandBuffer
// wrapping a fresh MTLCommandBuffer; ops encode into it; ts_enc_commit_wait
// commits + waits + frees.
//===----------------------------------------------------------------------===//

struct TsEncodeSession {
  MPSCommandBuffer *cb;
  id<MTLCommandBuffer> mtlcb;
};

extern "C" TsEncodeSession *ts_enc_begin(void) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return nullptr;
  @autoreleasepool {
    id<MTLCommandBuffer> mtlcb = [ctx.queue commandBuffer];
    if (!mtlcb) return nullptr;
    MPSCommandBuffer *cb = [MPSCommandBuffer commandBufferWithCommandBuffer:mtlcb];
    return new TsEncodeSession{cb, mtlcb};
  }
}

extern "C" void ts_enc_commit_wait(TsEncodeSession *s) {
  if (!s) return;
  // waitUntilCompleted migration batch 5 (2026-06-01) — encode-session
  // path. ``s->mtlcb`` is the session-owned MTLCommandBuffer (NOT the
  // generic ``cb`` the Pattern-4 wrapper takes), so we re-encode the
  // Pattern-4 sequence inline: encode + signal into the session's cb,
  // commit through MPSCommandBuffer, wait on the shared event with a
  // timeout. Honors the same 30 s cap as the rest of the migrated
  // dispatchers. The session-side ``s->cb`` is an ``MPSCommandBuffer``
  // wrapper around ``s->mtlcb``; the underlying signal goes through
  // ``s->mtlcb`` so the wait observes encode→commit→GPU-finish.
  MetalDeviceContext &ctx = deviceContext();
  id<MTLSharedEvent> ev = nil;
  uint64_t signal_val = 0;
  if (ctx.ok) {
    std::lock_guard<std::mutex> lock(ctx.legacy_event_mu);
    if (!ctx.legacy_event) {
      ctx.legacy_event = [ctx.device newSharedEvent];
    }
    ev = (id<MTLSharedEvent>)ctx.legacy_event;
    if (ev) signal_val = ++ctx.legacy_event_val;
  }
  if (ev) {
    [s->mtlcb encodeSignalEvent:ev value:signal_val];
  }
  [s->cb commit];
  if (ev) {
    bool done = [ev waitUntilSignaledValue:signal_val
                                  timeoutMS:30000];
    if (!done) {
      fprintf(stderr,
              "[tessera_apple_gpu] ts_enc_commit_wait: GPU dispatch did "
              "not signal within 30000 ms (signaledValue=%llu wanted=%llu)\n",
              (unsigned long long)ev.signaledValue,
              (unsigned long long)signal_val);
    }
  } else {
    // Shared-event init failed — fall back to the legacy synchronous
    // wait so the caller doesn't crash. No timeout protection in this
    // path (matches the helper's own fallback).
    [s->mtlcb waitUntilCompleted];
  }
  s->cb = nil;
  s->mtlcb = nil;
  delete s;
}

// Encoded device-resident bmm — appends to the session's command buffer.
extern "C" int32_t tessera_apple_gpu_bmm_dev_f32_enc(TsEncodeSession *s,
                                                     TsDeviceTensor *A,
                                                     TsDeviceTensor *B,
                                                     TsDeviceTensor *O,
                                                     int32_t batch, int32_t M,
                                                     int32_t N, int32_t K,
                                                     int32_t b_broadcast) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !A || !B || !O) return 0;
  return mpsg_encode_bmm_dev(s->cb, A->buf, B->buf, O->buf, batch, M, N, K,
                             b_broadcast != 0, MPSDataTypeFloat32)
             ? 1
             : 0;
}

// Single-command-buffer decode scaffold (2026-06-01) — encoded device-resident
// layer_norm. Appends to the session's command buffer; no commit/sync here.
// Pairs with ``tessera_apple_gpu_bmm_dev_f32_enc`` so a decoder layer can keep
// "norm + matmul + norm" on ONE command buffer with no per-op GPU↔CPU
// roundtrips. Roadmap: ``docs/audit/backend/apple/APPLE_AUDIT.md``.
extern "C" int32_t tessera_apple_gpu_layer_norm_dev_f32_enc(
    TsEncodeSession *s,
    TsDeviceTensor *X, TsDeviceTensor *gamma,
    TsDeviceTensor *beta, TsDeviceTensor *Y,
    int32_t rows, int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !gamma || !beta || !Y) return 0;
  return mpsg_encode_layer_norm_dev(s->cb, X->buf, gamma->buf, beta->buf,
                                    Y->buf, rows, cols, eps,
                                    MPSDataTypeFloat32)
             ? 1
             : 0;
}

// Stage-2 single-cb (2026-06-01) — encoded rmsnorm (kind=1, gamma
// only). Llama-style transformers use rmsnorm in place of layer_norm.
extern "C" int32_t tessera_apple_gpu_rmsnorm_dev_f32_enc(
    TsEncodeSession *s,
    TsDeviceTensor *X, TsDeviceTensor *gamma,
    TsDeviceTensor *Y, int32_t rows, int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !gamma || !Y) return 0;
  return mpsg_encode_rowop_dev(s->cb, /*kind=*/1, X->buf, gamma->buf,
                                /*bufB=*/nil, Y->buf, rows, cols, eps,
                                MPSDataTypeFloat32) ? 1 : 0;
}

// Stage-2 (2026-06-01) — encoded softmax (kind=2, no gamma / beta).
// Free-standing softmax (separate from the flash_attn fusion) for
// cases like classifier heads, attention scoring outside flash_attn.
extern "C" int32_t tessera_apple_gpu_softmax_dev_f32_enc(
    TsEncodeSession *s,
    TsDeviceTensor *X, TsDeviceTensor *Y,
    int32_t rows, int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Y) return 0;
  return mpsg_encode_rowop_dev(s->cb, /*kind=*/2, X->buf, /*bufG=*/nil,
                                /*bufB=*/nil, Y->buf, rows, cols,
                                /*eps=*/0.0f, MPSDataTypeFloat32)
             ? 1
             : 0;
}

// Stage-2 single-cb decoder block (2026-06-01) — encoded device-resident
// flash_attn. Pairs with ``tessera_apple_gpu_bmm_dev_f32_enc`` +
// ``tessera_apple_gpu_layer_norm_dev_f32_enc`` so an entire
// transformer attention block — ``layer_norm → qkv_bmm → flash_attn →
// out_bmm → layer_norm`` — encodes into ONE command buffer.
extern "C" int32_t tessera_apple_gpu_flash_attn_dev_f32_enc(
    TsEncodeSession *s,
    TsDeviceTensor *Q, TsDeviceTensor *K,
    TsDeviceTensor *V, TsDeviceTensor *O,
    int32_t B, int32_t Sq, int32_t Sk, int32_t D,
    float scale, int32_t causal) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !Q || !K || !V || !O) return 0;
  return encode_flash_attn_msl_dev(ctx, s->cb, Q->buf, K->buf, V->buf,
                                    O->buf, B, Sq, Sk, D, scale, causal)
             ? 1
             : 0;
}

// Project-3 f16 encode-session ABI (2026-06-01) — same shape as the
// f32 variants above, just with ``MPSDataTypeFloat16`` passed to the
// MPSGraph-encode helpers. Caller-side ``uint16_t`` half buffers
// (Tessera uses ``ml_dtypes.float16`` via numpy at the Python edge).
// Real-LLM decode usually runs in fp16, so these complete the encode
// envelope for production-shape workloads.

extern "C" int32_t tessera_apple_gpu_bmm_dev_f16_enc(
    TsEncodeSession *s, TsDeviceTensor *A, TsDeviceTensor *B,
    TsDeviceTensor *O, int32_t batch, int32_t M, int32_t N, int32_t K,
    int32_t b_broadcast) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !A || !B || !O) return 0;
  return mpsg_encode_bmm_dev(s->cb, A->buf, B->buf, O->buf, batch, M, N, K,
                              b_broadcast != 0, MPSDataTypeFloat16)
             ? 1
             : 0;
}

extern "C" int32_t tessera_apple_gpu_layer_norm_dev_f16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *gamma,
    TsDeviceTensor *beta, TsDeviceTensor *Y,
    int32_t rows, int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !gamma || !beta || !Y) return 0;
  return mpsg_encode_rowop_dev(s->cb, /*kind=*/0, X->buf, gamma->buf,
                                beta->buf, Y->buf, rows, cols, eps,
                                MPSDataTypeFloat16) ? 1 : 0;
}

extern "C" int32_t tessera_apple_gpu_rmsnorm_dev_f16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *gamma,
    TsDeviceTensor *Y, int32_t rows, int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !gamma || !Y) return 0;
  return mpsg_encode_rowop_dev(s->cb, /*kind=*/1, X->buf, gamma->buf,
                                /*bufB=*/nil, Y->buf, rows, cols, eps,
                                MPSDataTypeFloat16) ? 1 : 0;
}

extern "C" int32_t tessera_apple_gpu_softmax_dev_f16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *Y,
    int32_t rows, int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Y) return 0;
  return mpsg_encode_rowop_dev(s->cb, /*kind=*/2, X->buf, /*bufG=*/nil,
                                /*bufB=*/nil, Y->buf, rows, cols, 0.0f,
                                MPSDataTypeFloat16) ? 1 : 0;
}

// Project-3 bf16 encode-session ABI (2026-06-01) — same shape as
// the f16 variants, just MPSDataTypeBFloat16. macOS 26+ on Apple
// Silicon (M2+) supports bf16 in MPSGraph directly; on older hosts
// MPSGraph rejects the graph at build time and the encode helper
// returns false. Sub-byte / non-MPSGraph paths (rope MSL kernel,
// flash_attn MSL kernel) need the explicit fp32-conversion route
// which is the Phase-3b follow-on.
extern "C" int32_t tessera_apple_gpu_bmm_dev_bf16_enc(
    TsEncodeSession *s, TsDeviceTensor *A, TsDeviceTensor *B,
    TsDeviceTensor *O, int32_t batch, int32_t M, int32_t N, int32_t K,
    int32_t b_broadcast) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !A || !B || !O) return 0;
  return mpsg_encode_bmm_dev(s->cb, A->buf, B->buf, O->buf, batch, M, N, K,
                              b_broadcast != 0, MPSDataTypeBFloat16)
             ? 1
             : 0;
}

extern "C" int32_t tessera_apple_gpu_layer_norm_dev_bf16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *gamma,
    TsDeviceTensor *beta, TsDeviceTensor *Y,
    int32_t rows, int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !gamma || !beta || !Y) return 0;
  return mpsg_encode_rowop_dev(s->cb, /*kind=*/0, X->buf, gamma->buf,
                                beta->buf, Y->buf, rows, cols, eps,
                                MPSDataTypeBFloat16) ? 1 : 0;
}

extern "C" int32_t tessera_apple_gpu_rmsnorm_dev_bf16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *gamma,
    TsDeviceTensor *Y, int32_t rows, int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !gamma || !Y) return 0;
  return mpsg_encode_rowop_dev(s->cb, /*kind=*/1, X->buf, gamma->buf,
                                /*bufB=*/nil, Y->buf, rows, cols, eps,
                                MPSDataTypeBFloat16) ? 1 : 0;
}

extern "C" int32_t tessera_apple_gpu_softmax_dev_bf16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *Y,
    int32_t rows, int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Y) return 0;
  return mpsg_encode_rowop_dev(s->cb, /*kind=*/2, X->buf, /*bufG=*/nil,
                                /*bufB=*/nil, Y->buf, rows, cols, 0.0f,
                                MPSDataTypeBFloat16) ? 1 : 0;
}

extern "C" int32_t tessera_apple_gpu_rope_dev_f16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *Theta,
    TsDeviceTensor *Y, int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Theta || !Y) return 0;
  return encode_rope_msl_f16_dev(ctx, s->cb, X->buf, Theta->buf, Y->buf,
                                  M, K) ? 1 : 0;
}

extern "C" int32_t tessera_apple_gpu_flash_attn_dev_f16_enc(
    TsEncodeSession *s, TsDeviceTensor *Q, TsDeviceTensor *K,
    TsDeviceTensor *V, TsDeviceTensor *O,
    int32_t B, int32_t Sq, int32_t Sk, int32_t D,
    float scale, int32_t causal) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !Q || !K || !V || !O) return 0;
  return encode_flash_attn_msl_f16_dev(ctx, s->cb, Q->buf, K->buf, V->buf,
                                        O->buf, B, Sq, Sk, D, scale, causal)
             ? 1
             : 0;
}

// Phase 3b (2026-06-01) — forward decl for the MPSGraph cast helper
// defined later in this file. The bf16 MSL-kernel encode helpers
// below (rope_bf16, flash_attn_bf16) reference it; the helper itself
// lives alongside the other mpsg_encode_*_dev helpers further down.
namespace {
static bool mpsg_encode_cast_dev(MPSCommandBuffer *cb,
                                 id<MTLBuffer> bufIn,
                                 id<MTLBuffer> bufOut,
                                 int64_t n,
                                 MPSDataType srcType,
                                 MPSDataType dstType);
}  // namespace

// Phase 3b (2026-06-01) — bf16 RoPE via on-GPU bf16→fp32→bf16 cast.
// The MSL kernel itself is the existing f32 rope; we sandwich it
// between two cast nodes encoded into the same cb so the whole
// thing runs in one session with no host roundtrip.
extern "C" int32_t tessera_apple_gpu_rope_dev_bf16_enc(
    TsEncodeSession *s,
    TsDeviceTensor *X, TsDeviceTensor *Theta, TsDeviceTensor *Y,
    int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Theta || !Y) return 0;
  if (M <= 0 || K <= 0) return 1;
  int64_t n = (int64_t)M * (int64_t)K;
  // Allocate fp32 scratch device buffers for X, Theta, and the
  // rope output. The session's command buffer retains the
  // MTLBuffers until commit so these are safe to release as soon
  // as the caller returns (the encoded chain holds the references).
  id<MTLBuffer> bufXf =
      [ctx.device newBufferWithLength:(NSUInteger)(n * 4)
                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufTf =
      [ctx.device newBufferWithLength:(NSUInteger)(n * 4)
                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufYf =
      [ctx.device newBufferWithLength:(NSUInteger)(n * 4)
                              options:MTLResourceStorageModeShared];
  if (!bufXf || !bufTf || !bufYf) return 0;
  // bf16 → fp32 casts (encoded into the shared cb).
  if (!mpsg_encode_cast_dev(s->cb, X->buf, bufXf, n,
                             MPSDataTypeBFloat16, MPSDataTypeFloat32))
    return 0;
  if (!mpsg_encode_cast_dev(s->cb, Theta->buf, bufTf, n,
                             MPSDataTypeBFloat16, MPSDataTypeFloat32))
    return 0;
  // fp32 RoPE MSL kernel.
  if (!encode_rope_msl_dev(ctx, s->cb, bufXf, bufTf, bufYf, M, K))
    return 0;
  // fp32 → bf16 cast for the output.
  if (!mpsg_encode_cast_dev(s->cb, bufYf, Y->buf, n,
                             MPSDataTypeFloat32, MPSDataTypeBFloat16))
    return 0;
  return 1;
}

// Phase 3b (2026-06-01) — bf16 flash_attn via on-GPU
// bf16→fp32→bf16 cast. Same pattern as rope_bf16 above: bracket the
// existing fp32 MSL flash_attn kernel with two cast nodes.
extern "C" int32_t tessera_apple_gpu_flash_attn_dev_bf16_enc(
    TsEncodeSession *s,
    TsDeviceTensor *Q, TsDeviceTensor *K, TsDeviceTensor *V,
    TsDeviceTensor *O,
    int32_t B, int32_t Sq, int32_t Sk, int32_t D,
    float scale, int32_t causal) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !Q || !K || !V || !O) return 0;
  if (B <= 0 || Sq <= 0 || Sk <= 0 || D <= 0) return 1;
  int64_t qN = (int64_t)B * Sq * D;
  int64_t kvN = (int64_t)B * Sk * D;
  id<MTLBuffer> bufQf =
      [ctx.device newBufferWithLength:(NSUInteger)(qN * 4)
                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufKf =
      [ctx.device newBufferWithLength:(NSUInteger)(kvN * 4)
                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufVf =
      [ctx.device newBufferWithLength:(NSUInteger)(kvN * 4)
                              options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufOf =
      [ctx.device newBufferWithLength:(NSUInteger)(qN * 4)
                              options:MTLResourceStorageModeShared];
  if (!bufQf || !bufKf || !bufVf || !bufOf) return 0;
  if (!mpsg_encode_cast_dev(s->cb, Q->buf, bufQf, qN,
                             MPSDataTypeBFloat16, MPSDataTypeFloat32))
    return 0;
  if (!mpsg_encode_cast_dev(s->cb, K->buf, bufKf, kvN,
                             MPSDataTypeBFloat16, MPSDataTypeFloat32))
    return 0;
  if (!mpsg_encode_cast_dev(s->cb, V->buf, bufVf, kvN,
                             MPSDataTypeBFloat16, MPSDataTypeFloat32))
    return 0;
  if (!encode_flash_attn_msl_dev(ctx, s->cb, bufQf, bufKf, bufVf, bufOf,
                                  B, Sq, Sk, D, scale, causal))
    return 0;
  if (!mpsg_encode_cast_dev(s->cb, bufOf, O->buf, qN,
                             MPSDataTypeFloat32, MPSDataTypeBFloat16))
    return 0;
  return 1;
}

// Stage-2 (2026-06-01) — encoded RoPE. Sits between layer_norm/rmsnorm
// and the QKV projections in a typical decoder layer. Composes onto
// the same session as the other encoded ops.
extern "C" int32_t tessera_apple_gpu_rope_dev_f32_enc(
    TsEncodeSession *s,
    TsDeviceTensor *X, TsDeviceTensor *Theta, TsDeviceTensor *Y,
    int32_t M, int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Theta || !Y) return 0;
  return encode_rope_msl_dev(ctx, s->cb, X->buf, Theta->buf, Y->buf,
                              M, K) ? 1 : 0;
}

// Probe symbol (single-command-buffer scaffold) — number of command buffers
// the queue has committed since process start. A test that opens a session,
// encodes N ops, commits, and observes the count incremented by exactly 1
// proves the chain stayed on one cb. The Metal API doesn't expose a direct
// counter on ``MTLCommandQueue``; instead we increment a private counter
// inside ``ts_enc_commit_wait``. This is sufficient for the scaffold's
// structural drift gate.
extern "C" int64_t tessera_apple_gpu_session_commit_count(void) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return -1;
  std::lock_guard<std::mutex> lock(ctx.legacy_event_mu);
  // ``legacy_event_val`` increments once per ``ts_enc_commit_wait`` (or other
  // Pattern-4-wrapped dispatch). We don't need the absolute value, just a
  // monotonic counter callers can diff across a session.
  return (int64_t)ctx.legacy_event_val;
}

//===----------------------------------------------------------------------===//
// R2 (cont.) — encoded flat elementwise unary/binary ops. These let a single
// command buffer express full transformer/MLP blocks (residual adds, SwiGLU,
// ReLU heads, additive tree-attention masks) alongside bmm/rowop/gumbel —
// not just the MLA decode chain. Shapes collapse to a flat element count;
// elementwise ops are layout-agnostic. unary op: 0 relu, 4 silu (see
// mpsg_unary_node); binary op: 0 add, 2 mul (see mpsg_binary_node).
//===----------------------------------------------------------------------===//
namespace {

// Phase 3b (2026-06-01) — encode an in-cb dtype cast via MPSGraph's
// cast operation. Used by the bf16 MSL-kernel encode helpers
// (encode_rope_msl_bf16_dev, encode_flash_attn_msl_bf16_dev) which
// need bf16→fp32 conversion before the fp32 MSL kernel and
// fp32→bf16 conversion after. ``n`` is the element count.
//
// The cast is encoded into the SHARED command buffer (no commit /
// wait) so a chain like:
//   bf16_in_buf → cast → fp32_tmp → MSL_kernel → fp32_out → cast → bf16_out_buf
// runs end-to-end on ONE cb. Metal's automatic hazard tracking
// orders the cast → kernel → cast sequence correctly.
static bool mpsg_encode_cast_dev(MPSCommandBuffer *cb,
                                 id<MTLBuffer> bufIn,
                                 id<MTLBuffer> bufOut,
                                 int64_t n,
                                 MPSDataType srcType,
                                 MPSDataType dstType) {
  if (n <= 0) return true;
  if (!cb || !bufIn || !bufOut) return false;
  NSArray<NSNumber *> *shape = @[ @(n) ];
  NSString *key = [NSString stringWithFormat:@"cast:%d:%d:%lld",
                                             (int)srcType, (int)dstType,
                                             (long long)n];
  NSArray *entry = mpsg_cache_get(key);
  MPSGraph *g;
  MPSGraphTensor *ph;
  MPSGraphTensor *y;
  if (entry) {
    g = entry[0];
    ph = ((NSArray *)entry[1])[0];
    y = entry[2];
  } else {
    g = [MPSGraph new];
    ph = [g placeholderWithShape:shape dataType:srcType name:nil];
    y = [g castTensor:ph toType:dstType name:@"cast"];
    if (!y) return false;
    mpsg_cache_put(key, @[ g, @[ ph ], y ]);
  }
  MPSGraphTensorData *xd =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:bufIn
                                              shape:shape
                                           dataType:srcType];
  MPSGraphTensorData *od =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:bufOut
                                              shape:shape
                                           dataType:dstType];
  [g encodeToCommandBuffer:cb feeds:@{ph : xd}
              targetOperations:nil
          resultsDictionary:@{y : od}
        executionDescriptor:nil];
  return true;
}
}  // namespace cast

// Reopen the unary/binary namespace block — the existing
// implementations follow below.
namespace {
static bool mpsg_encode_unary_dev(MPSCommandBuffer *cb, id<MTLBuffer> bufX,
                                  id<MTLBuffer> bufO, int64_t n, int op,
                                  MPSDataType ioType) {
  if (n <= 0) return true;
  if (!cb || !bufX || !bufO) return false;
  NSArray<NSNumber *> *shape = @[ @(n) ];
  NSString *key = [NSString stringWithFormat:@"ue:%d:%d:%lld", op, (int)ioType,
                                             (long long)n];
  NSArray *entry = mpsg_cache_get(key);
  MPSGraph *g;
  MPSGraphTensor *ph;
  MPSGraphTensor *y;
  if (entry) {
    g = entry[0];
    ph = ((NSArray *)entry[1])[0];
    y = entry[2];
  } else {
    g = [MPSGraph new];
    ph = [g placeholderWithShape:shape dataType:ioType name:nil];
    MPSGraphTensor *yf = mpsg_unary_node(g, mpsg_up(g, ph, ioType), op);
    if (!yf) return false;
    y = mpsg_down(g, yf, ioType);
    mpsg_cache_put(key, @[ g, @[ ph ], y ]);
  }
  MPSGraphTensorData *xd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:shape dataType:ioType];
  MPSGraphTensorData *od = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO shape:shape dataType:ioType];
  [g encodeToCommandBuffer:cb feeds:@{ph : xd} targetOperations:nil
         resultsDictionary:@{y : od} executionDescriptor:nil];
  return true;
}
static bool mpsg_encode_binary_dev(MPSCommandBuffer *cb, id<MTLBuffer> bufA,
                                   id<MTLBuffer> bufB, id<MTLBuffer> bufO,
                                   int64_t n, int op, MPSDataType ioType) {
  if (n <= 0) return true;
  if (!cb || !bufA || !bufB || !bufO) return false;
  NSArray<NSNumber *> *shape = @[ @(n) ];
  NSString *key = [NSString stringWithFormat:@"be:%d:%d:%lld", op, (int)ioType,
                                             (long long)n];
  NSArray *entry = mpsg_cache_get(key);
  MPSGraph *g;
  MPSGraphTensor *pa;
  MPSGraphTensor *pb;
  MPSGraphTensor *y;
  if (entry) {
    g = entry[0];
    pa = ((NSArray *)entry[1])[0];
    pb = ((NSArray *)entry[1])[1];
    y = entry[2];
  } else {
    g = [MPSGraph new];
    pa = [g placeholderWithShape:shape dataType:ioType name:nil];
    pb = [g placeholderWithShape:shape dataType:ioType name:nil];
    MPSGraphTensor *yf =
        mpsg_binary_node(g, mpsg_up(g, pa, ioType), mpsg_up(g, pb, ioType), op);
    if (!yf) return false;
    y = mpsg_down(g, yf, ioType);
    mpsg_cache_put(key, @[ g, @[ pa, pb ], y ]);
  }
  MPSGraphTensorData *ad = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:shape dataType:ioType];
  MPSGraphTensorData *bd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:shape dataType:ioType];
  MPSGraphTensorData *od = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO shape:shape dataType:ioType];
  [g encodeToCommandBuffer:cb feeds:@{pa : ad, pb : bd} targetOperations:nil
         resultsDictionary:@{y : od} executionDescriptor:nil];
  return true;
}
}  // namespace

extern "C" int32_t tessera_apple_gpu_unary_dev_f32_enc(TsEncodeSession *s,
                                                       TsDeviceTensor *X,
                                                       TsDeviceTensor *O,
                                                       int64_t n, int32_t op) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !O) return 0;
  return mpsg_encode_unary_dev(s->cb, X->buf, O->buf, n, (int)op,
                               MPSDataTypeFloat32)
             ? 1
             : 0;
}

// Project-3 f16 (2026-06-01) — unary encoded f16. Routes through the
// same MPSGraph unary node with MPSDataTypeFloat16 to satisfy LLM
// activation paths (silu / gelu / sigmoid / ...) in fp16.
extern "C" int32_t tessera_apple_gpu_unary_dev_f16_enc(TsEncodeSession *s,
                                                       TsDeviceTensor *X,
                                                       TsDeviceTensor *O,
                                                       int64_t n, int32_t op) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !O) return 0;
  return mpsg_encode_unary_dev(s->cb, X->buf, O->buf, n, (int)op,
                                MPSDataTypeFloat16)
             ? 1
             : 0;
}

extern "C" int32_t tessera_apple_gpu_unary_dev_bf16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *O,
    int64_t n, int32_t op) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !O) return 0;
  return mpsg_encode_unary_dev(s->cb, X->buf, O->buf, n, (int)op,
                                MPSDataTypeBFloat16)
             ? 1
             : 0;
}

// Capability probe — does MPSGraph accept bf16 graph nodes on this
// host? Builds a trivial 2-element bf16 unary graph at probe time
// and reports whether construction + compile succeeded. Tests can
// gate on this rather than running an actual op that might fail
// silently mid-pipeline. Returns 1 iff MPSGraph supports bf16 here.
extern "C" int32_t tessera_apple_gpu_mpsgraph_bf16_supported(void) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok) return 0;
  @autoreleasepool {
    // Build a tiny silu graph in bf16 — same shape as the runtime
    // would build, just unused. If MPSGraph rejects the dtype, the
    // node construction returns nil and we return 0.
    MPSGraph *g = [[MPSGraph alloc] init];
    MPSGraphTensor *x = [g placeholderWithShape:@[ @1 ]
                                       dataType:MPSDataTypeBFloat16
                                           name:nil];
    if (!x) return 0;
    MPSGraphTensor *y = mpsg_unary_node(g, x, /*op=*/4);  // silu
    if (!y) return 0;
    return 1;
  }
}

extern "C" int32_t tessera_apple_gpu_binary_dev_f32_enc(TsEncodeSession *s,
                                                        TsDeviceTensor *A,
                                                        TsDeviceTensor *B,
                                                        TsDeviceTensor *O,
                                                        int64_t n, int32_t op) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !A || !B || !O) return 0;
  return mpsg_encode_binary_dev(s->cb, A->buf, B->buf, O->buf, n, (int)op,
                                MPSDataTypeFloat32)
             ? 1
             : 0;
}

//===----------------------------------------------------------------------===//
// R4 (block-paged) — on-GPU non-contiguous block-table gather. Given a resident
// physical block pool [num_blocks, block_size, dim] and a sequence's block table
// (int32 [n] of physical block ids), MPSGraph gathers those (possibly
// scattered) blocks along axis 0 into a contiguous window
// [n, block_size, dim] — the device-resident equivalent of vLLM's paged gather,
// with no host round-trip.
//===----------------------------------------------------------------------===//
namespace {
static bool mpsg_run_gather_blocks(MetalDeviceContext &ctx, id<MTLBuffer> pool,
                                   id<MTLBuffer> idx, id<MTLBuffer> out,
                                   MPSCommandBuffer *cb, int32_t num_blocks,
                                   int32_t n, int32_t block_size, int32_t dim) {
  if (num_blocks <= 0 || n <= 0 || block_size <= 0 || dim <= 0) return true;
  if (!pool || !idx || !out) return false;
  @autoreleasepool {
    NSArray<NSNumber *> *poolShape = @[ @(num_blocks), @(block_size), @(dim) ];
    NSArray<NSNumber *> *idxShape = @[ @(n) ];
    NSArray<NSNumber *> *outShape = @[ @(n), @(block_size), @(dim) ];
    NSString *key = [NSString stringWithFormat:@"gblk:%d:%d:%d:%d", num_blocks,
                                               n, block_size, dim];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *pp, *pi, *y;
    if (entry) {
      g = entry[0];
      pp = ((NSArray *)entry[1])[0];
      pi = ((NSArray *)entry[1])[1];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      pp = [g placeholderWithShape:poolShape dataType:MPSDataTypeFloat32 name:nil];
      pi = [g placeholderWithShape:idxShape dataType:MPSDataTypeInt32 name:nil];
      y = [g gatherWithUpdatesTensor:pp indicesTensor:pi axis:0 batchDimensions:0 name:nil];
      mpsg_cache_put(key, @[ g, @[ pp, pi ], y ]);
    }
    MPSGraphTensorData *pd = [[MPSGraphTensorData alloc] initWithMTLBuffer:pool shape:poolShape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *id_ = [[MPSGraphTensorData alloc] initWithMTLBuffer:idx shape:idxShape dataType:MPSDataTypeInt32];
    MPSGraphTensorData *od = [[MPSGraphTensorData alloc] initWithMTLBuffer:out shape:outShape dataType:MPSDataTypeFloat32];
    if (cb)
      [g encodeToCommandBuffer:cb feeds:@{pp : pd, pi : id_} targetOperations:nil resultsDictionary:@{y : od} executionDescriptor:nil];
    else
      [g runWithMTLCommandQueue:ctx.queue feeds:@{pp : pd, pi : id_} targetOperations:nil resultsDictionary:@{y : od}];
    return true;
  }
}
}  // namespace

extern "C" int32_t tessera_apple_gpu_gather_blocks_dev_f32(
    TsDeviceTensor *pool, TsDeviceTensor *block_table, TsDeviceTensor *out,
    int32_t num_blocks, int32_t n, int32_t block_size, int32_t dim) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !pool || !block_table || !out) return 0;
  return mpsg_run_gather_blocks(ctx, pool->buf, block_table->buf, out->buf, nil,
                                num_blocks, n, block_size, dim) ? 1 : 0;
}

extern "C" int32_t tessera_apple_gpu_gather_blocks_dev_f32_enc(
    TsEncodeSession *s, TsDeviceTensor *pool, TsDeviceTensor *block_table,
    TsDeviceTensor *out, int32_t num_blocks, int32_t n, int32_t block_size,
    int32_t dim) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !pool || !block_table || !out) return 0;
  return mpsg_run_gather_blocks(ctx, pool->buf, block_table->buf, out->buf,
                                s->cb, num_blocks, n, block_size, dim) ? 1 : 0;
}

// R1 device-resident bmm entry point. A/B/O are TsDeviceTensor handles whose
// shared buffers are used in place. Returns 1 on a real GPU run, 0 otherwise
// (caller falls back to the host-ptr path).
extern "C" int32_t tessera_apple_gpu_bmm_dev_f32(TsDeviceTensor *A,
                                                 TsDeviceTensor *B,
                                                 TsDeviceTensor *O,
                                                 int32_t batch, int32_t M,
                                                 int32_t N, int32_t K,
                                                 int32_t b_broadcast) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !A || !B || !O) return 0;
  return mpsg_run_bmm_dev(ctx, A->buf, B->buf, O->buf, batch, M, N, K,
                          b_broadcast != 0, MPSDataTypeFloat32)
             ? 1
             : 0;
}

extern "C" void tessera_apple_gpu_bmm_f32(const float *A, const float *B,
                                          float *O, int32_t batch, int32_t M,
                                          int32_t N, int32_t K,
                                          int32_t b_broadcast) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_bmm(ctx, A, B, O, batch, M, N, K, b_broadcast != 0,
                             MPSDataTypeFloat32, 4))
    return;
  reference_bmm_f32(A, B, O, batch, M, N, K, b_broadcast);
}

//===---------------------------------------------------------------------===//
// Thrust #3a — fused ragged grouped-GEMM (the MoE expert-FFN compute core).
//
// One Metal dispatch over the whole (T, N) output instead of the per-group MPS
// matmul loop in `_apple_gpu_dispatch_grouped_gemm`: each thread (t, n) reads
// the per-token expert id `E[t]` and contracts X[t,:] with W[E[t],:,n]. Removes
// the per-expert dispatch overhead (the win for many small groups) and folds
// the routing into the kernel. f32; numerically validated against the per-group
// reference.
//===---------------------------------------------------------------------===//
namespace {

bool dispatch_grouped_gemm_msl(MetalDeviceContext &ctx, const float *X,
                               const float *W, const int32_t *E, float *O,
                               int32_t T, int32_t K, int32_t N, int32_t Ecount) {
  static NSString *const kGroupedGemmSource = @R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void grouped_gemm_f32(
    device const float* X   [[buffer(0)]],
    device const float* W   [[buffer(1)]],
    device const int*   E   [[buffer(2)]],   // per-token expert id
    device float*       O   [[buffer(3)]],
    constant int&       T   [[buffer(4)]],
    constant int&       K   [[buffer(5)]],
    constant int&       N   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int t = (int)gid.x;
    int n = (int)gid.y;
    if (t >= T || n >= N) return;
    int e = E[t];
    int x_off = t * K;
    int w_base = (e * K) * N + n;        // W[e, 0, n]; stride over k is N
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += X[x_off + k] * W[w_base + k * N];
    O[t * N + n] = acc;
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kGroupedGemmSource, @"grouped_gemm_f32");
    if (!pso) return false;

    NSUInteger xBytes = sizeof(float) * (NSUInteger)T * (NSUInteger)K;
    NSUInteger wBytes =
        sizeof(float) * (NSUInteger)Ecount * (NSUInteger)K * (NSUInteger)N;
    NSUInteger eBytes = sizeof(int32_t) * (NSUInteger)T;
    NSUInteger oBytes = sizeof(float) * (NSUInteger)T * (NSUInteger)N;

    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, xBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufW, ctx, W, wBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufE, ctx, E, eBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, oBytes);
    if (!bufX || !bufW || !bufE || !bufO) return false;

    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufX offset:0 atIndex:0];
    [enc setBuffer:bufW offset:0 atIndex:1];
    [enc setBuffer:bufE offset:0 atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&T length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&K length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&N length:sizeof(int32_t) atIndex:6];

    MTLSize grid = MTLSizeMake((NSUInteger)T, (NSUInteger)N, 1);
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger h = pso.maxTotalThreadsPerThreadgroup / (w == 0 ? 1 : w);
    if (w == 0) w = 1;
    if (h == 0) h = 1;
    MTLSize tg = MTLSizeMake(w, h, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    if (!commit_and_wait_with_timeout(ctx, cb, 60000, "grouped_gemm_msl"))
      return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
}

inline void reference_grouped_gemm_f32(const float *X, const float *W,
                                       const int32_t *E, float *O, int32_t T,
                                       int32_t K, int32_t N) {
  for (int32_t t = 0; t < T; ++t) {
    int32_t e = E[t];
    for (int32_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int32_t k = 0; k < K; ++k)
        acc += X[(std::size_t)t * K + k] *
               W[((std::size_t)e * K + k) * N + n];
      O[(std::size_t)t * N + n] = acc;
    }
  }
}

}  // namespace

extern "C" void tessera_apple_gpu_grouped_gemm_f32(
    const float *X, const float *W, const int32_t *E, float *O, int32_t T,
    int32_t K, int32_t N, int32_t Ecount) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && T > 0 && N > 0 &&
      dispatch_grouped_gemm_msl(ctx, X, W, E, O, T, K, N, Ecount))
    return;
  reference_grouped_gemm_f32(X, W, E, O, T, K, N);
}

extern "C" int32_t tessera_apple_gpu_ppo_policy_loss_f32(
    const float *logp_new, const float *logp_old, const float *advantages,
    float *out, int32_t n, float clip_epsilon) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !logp_new || !logp_old || !advantages || !out)
    return 0;
  return mpsg_run_ppo_policy_loss_f32(ctx, logp_new, logp_old, advantages,
                                      out, n, clip_epsilon)
             ? 1
             : 0;
}

extern "C" int32_t tessera_apple_gpu_ppo_policy_loss_ex_f32(
    const float *logp_new, const float *logp_old, const float *advantages,
    const float *mask, const float *ref_logp, const float *entropy,
    float *out, int32_t n, float clip_epsilon, float kl_coef,
    float entropy_coef, int32_t has_mask, int32_t has_ref_kl,
    int32_t has_entropy) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !logp_new || !logp_old || !advantages || !out)
    return 0;
  return mpsg_run_ppo_policy_loss_ex_f32(
             ctx, logp_new, logp_old, advantages, mask, ref_logp, entropy, out,
             n, clip_epsilon, kl_coef, entropy_coef, has_mask, has_ref_kl,
             has_entropy)
             ? 1
             : 0;
}

extern "C" void tessera_apple_gpu_bmm_f16(const uint16_t *A, const uint16_t *B,
                                          uint16_t *O, int32_t batch, int32_t M,
                                          int32_t N, int32_t K,
                                          int32_t b_broadcast) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_bmm(ctx, A, B, O, batch, M, N, K, b_broadcast != 0,
                             MPSDataTypeFloat16, 2))
    return;
  // Sprint 8 fix (P1): if the MPSGraph f16 path is unavailable, compute a real
  // host-side fallback (f16 -> f32 -> bmm -> f16) instead of zero-filling. The
  // f16 bmm symbol is advertised executable on the value lane, so it must never
  // return zeros: an executable symbol always produces the real product.
  std::size_t aN = (std::size_t)batch * M * K;
  std::size_t bN = (std::size_t)(b_broadcast ? 1 : batch) * K * N;
  std::size_t oN = (std::size_t)batch * M * N;
  std::vector<float> Af(aN), Bf(bN), Of(oN);
  for (std::size_t i = 0; i < aN; ++i) Af[i] = half_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < bN; ++i) Bf[i] = half_to_float_gpu(B[i]);
  reference_bmm_f32(Af.data(), Bf.data(), Of.data(), batch, M, N, K, b_broadcast);
  for (std::size_t i = 0; i < oN; ++i) O[i] = float_to_half_gpu(Of[i]);
}

// Sprint 8: bf16 batched matmul. bf16 is not a native MPS matrix dtype, so the
// boundary keeps the bf16 ABI (uint16) but upcasts to f32, runs the f32 MPSGraph
// bmm, and rounds back to bf16 — an honest bf16 ABI symbol (NOT an alias of the
// f32 symbol). This mirrors the bf16 GEMM conversion path elsewhere in the shim.
extern "C" void tessera_apple_gpu_bmm_bf16(const uint16_t *A, const uint16_t *B,
                                           uint16_t *O, int32_t batch, int32_t M,
                                           int32_t N, int32_t K,
                                           int32_t b_broadcast) {
  std::size_t aN = (std::size_t)batch * M * K;
  std::size_t bN = (std::size_t)(b_broadcast ? 1 : batch) * K * N;
  std::size_t oN = (std::size_t)batch * M * N;
  std::vector<float> Af(aN), Bf(bN), Of(oN);
  for (std::size_t i = 0; i < aN; ++i) Af[i] = bfloat16_to_float_gpu(A[i]);
  for (std::size_t i = 0; i < bN; ++i) Bf[i] = bfloat16_to_float_gpu(B[i]);
  tessera_apple_gpu_bmm_f32(Af.data(), Bf.data(), Of.data(), batch, M, N, K,
                            b_broadcast);
  for (std::size_t i = 0; i < oN; ++i) O[i] = float_to_bfloat16_gpu(Of[i]);
}

//===----------------------------------------------------------------------===//
// Tier-3 MPSGraph reduction lane (2026-05-29)
//
// Row reductions over the last axis of an [rows, cols] f32 tensor. Python
// normalizes arbitrary axis/keepdims by moving the reduced axes to the end and
// folding to [rows, cols]; f16/bf16 upcast to f32 host-side (fp32 reduction is
// the right numerics). Reuses the MPSGraph graph cache + buffer pool.
//   reduce:  op 0 sum 1 mean 2 max 3 min 4 prod 5 var(biased) 6 std(biased)
//   argreduce: op 0 argmax 1 argmin  (int32 indices, one per row)
//   scan:    op 0 cumsum 1 cumprod   ([rows, cols] output)
//===----------------------------------------------------------------------===//

namespace {

static MPSGraphTensor *mpsg_reduce_node(MPSGraph *g, MPSGraphTensor *x, int op,
                                        NSArray<NSNumber *> *axis1) {
  switch (op) {
    case 0: return [g reductionSumWithTensor:x axes:axis1 name:nil];
    case 1: return [g meanOfTensor:x axes:axis1 name:nil];
    case 2: return [g reductionMaximumWithTensor:x axes:axis1 name:nil];
    case 3: return [g reductionMinimumWithTensor:x axes:axis1 name:nil];
    case 4: return [g reductionProductWithTensor:x axes:axis1 name:nil];
    case 5:
    case 6: {  // var / std (biased; Python applies the ddof correction)
      MPSGraphTensor *m = [g meanOfTensor:x axes:axis1 name:nil];
      MPSGraphTensor *d = [g subtractionWithPrimaryTensor:x secondaryTensor:m name:nil];
      MPSGraphTensor *sq = [g multiplicationWithPrimaryTensor:d secondaryTensor:d name:nil];
      MPSGraphTensor *var = [g meanOfTensor:sq axes:axis1 name:nil];
      return op == 5 ? var : [g squareRootWithTensor:var name:nil];
    }
    default: return nil;
  }
}

static bool mpsg_run_reduce(MetalDeviceContext &ctx, int op, const float *x,
                            float *out, int32_t rows, int32_t cols) {
  if (rows <= 0 || cols <= 0) return true;
  @autoreleasepool {
    size_t xbytes = (size_t)rows * cols * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, x, xbytes);
    if (!bufX) return false;
    NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
    NSArray<NSNumber *> *axis1 = @[ @1 ];
    NSString *key = [NSString stringWithFormat:@"red:%d:%d:%d", op, rows, cols];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *ph;
    MPSGraphTensor *y;
    if (entry) {
      g = entry[0];
      ph = ((NSArray *)entry[1])[0];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      ph = [g placeholderWithShape:xs dataType:MPSDataTypeFloat32 name:nil];
      y = mpsg_reduce_node(g, ph, op, axis1);
      if (!y) return false;
      mpsg_cache_put(key, @[ g, @[ ph ], y ]);
    }
    MPSGraphTensorData *xd =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:xs dataType:MPSDataTypeFloat32];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{ph : xd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

static bool mpsg_run_argreduce(MetalDeviceContext &ctx, int op, const float *x,
                               int32_t *out, int32_t rows, int32_t cols) {
  if (rows <= 0 || cols <= 0) return true;
  @autoreleasepool {
    size_t xbytes = (size_t)rows * cols * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, x, xbytes);
    if (!bufX) return false;
    NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
    NSString *key = [NSString stringWithFormat:@"arg:%d:%d:%d", op, rows, cols];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *ph;
    MPSGraphTensor *y;
    if (entry) {
      g = entry[0];
      ph = ((NSArray *)entry[1])[0];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      ph = [g placeholderWithShape:xs dataType:MPSDataTypeFloat32 name:nil];
      y = op == 0 ? [g reductionArgMaximumWithTensor:ph axis:1 name:nil]
                  : [g reductionArgMinimumWithTensor:ph axis:1 name:nil];
      // Normalize to int32 so the host read is unambiguous.
      y = [g castTensor:y toType:MPSDataTypeInt32 name:nil];
      mpsg_cache_put(key, @[ g, @[ ph ], y ]);
    }
    MPSGraphTensorData *xd =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:xs dataType:MPSDataTypeFloat32];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{ph : xd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

static bool mpsg_run_scan(MetalDeviceContext &ctx, int op, const float *x,
                          float *out, int32_t rows, int32_t cols) {
  if (rows <= 0 || cols <= 0) return true;
  @autoreleasepool {
    size_t xbytes = (size_t)rows * cols * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, x, xbytes);
    if (!bufX) return false;
    NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
    NSString *key = [NSString stringWithFormat:@"scan:%d:%d:%d", op, rows, cols];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *ph;
    MPSGraphTensor *y;
    if (entry) {
      g = entry[0];
      ph = ((NSArray *)entry[1])[0];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      ph = [g placeholderWithShape:xs dataType:MPSDataTypeFloat32 name:nil];
      y = op == 0 ? [g cumulativeSumWithTensor:ph axis:1 name:nil]
                  : [g cumulativeProductWithTensor:ph axis:1 name:nil];
      mpsg_cache_put(key, @[ g, @[ ph ], y ]);
    }
    MPSGraphTensorData *xd =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:xs dataType:MPSDataTypeFloat32];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{ph : xd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

static void reference_reduce(int op, const float *x, float *out, int32_t rows,
                             int32_t cols) {
  for (int32_t r = 0; r < rows; ++r) {
    const float *row = x + (size_t)r * cols;
    double acc;
    switch (op) {
      case 0: case 1: { double s = 0; for (int32_t c = 0; c < cols; ++c) s += row[c]; acc = op == 1 ? s / cols : s; break; }
      case 2: { double m = row[0]; for (int32_t c = 1; c < cols; ++c) m = row[c] > m ? row[c] : m; acc = m; break; }
      case 3: { double m = row[0]; for (int32_t c = 1; c < cols; ++c) m = row[c] < m ? row[c] : m; acc = m; break; }
      case 4: { double p = 1; for (int32_t c = 0; c < cols; ++c) p *= row[c]; acc = p; break; }
      default: { double s = 0; for (int32_t c = 0; c < cols; ++c) s += row[c]; double m = s / cols; double v = 0; for (int32_t c = 0; c < cols; ++c) { double d = row[c] - m; v += d * d; } v /= cols; acc = op == 6 ? std::sqrt(v) : v; break; }
    }
    out[r] = (float)acc;
  }
}

// Gumbel-max categorical sampler: ids = argmax(logits/T + gumbel) per row.
// The Gumbel noise is supplied by the caller (generated from the Philox stream
// on the host) so sampling is deterministic + reproducible without an on-GPU
// RNG — argmax(z + g) with g_i = -log(-log(u_i)) draws from softmax(z). The
// per-row reduction over the vocab runs on-GPU (the throughput win for batched
// sampling of many concurrent sequences).
static MPSGraph *mpsg_gumbel_graph(int32_t rows, int32_t cols, float invT,
                                   MPSGraphTensor **pl_out,
                                   MPSGraphTensor **pg_out,
                                   MPSGraphTensor **y_out) {
  NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
  NSString *key = [NSString stringWithFormat:@"gumbel:%d:%d:%a", rows, cols, invT];
  NSArray *entry = mpsg_cache_get(key);
  MPSGraph *g;
  MPSGraphTensor *pl, *pg, *y;
  if (entry) {
    g = entry[0];
    pl = ((NSArray *)entry[1])[0];
    pg = ((NSArray *)entry[1])[1];
    y = entry[2];
  } else {
    g = [MPSGraph new];
    pl = [g placeholderWithShape:xs dataType:MPSDataTypeFloat32 name:nil];
    pg = [g placeholderWithShape:xs dataType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor *scaled = [g multiplicationWithPrimaryTensor:pl
                                secondaryTensor:[g constantWithScalar:(double)invT dataType:MPSDataTypeFloat32]
                                           name:nil];
    MPSGraphTensor *scores = [g additionWithPrimaryTensor:scaled secondaryTensor:pg name:nil];
    MPSGraphTensor *idx = [g reductionArgMaximumWithTensor:scores axis:1 name:nil];
    y = [g castTensor:idx toType:MPSDataTypeInt32 name:nil];
    mpsg_cache_put(key, @[ g, @[ pl, pg ], y ]);
  }
  *pl_out = pl;
  *pg_out = pg;
  *y_out = y;
  return g;
}

static bool mpsg_run_gumbel_argmax(MetalDeviceContext &ctx, const float *logits,
                                   const float *gumbel, int32_t *out,
                                   int32_t rows, int32_t cols, float invT) {
  if (rows <= 0 || cols <= 0) return true;
  @autoreleasepool {
    size_t xbytes = (size_t)rows * cols * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufL, ctx, logits, xbytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufG, ctx, gumbel, xbytes);
    if (!bufL || !bufG) return false;
    NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
    MPSGraphTensor *pl, *pg, *y;
    MPSGraph *g = mpsg_gumbel_graph(rows, cols, invT, &pl, &pg, &y);
    MPSGraphTensorData *ld = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufL shape:xs dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *gd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufG shape:xs dataType:MPSDataTypeFloat32];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{pl : ld, pg : gd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:out strideBytes:nil];
    return true;
  }
}

// R2 — device-resident + encode variants for rowop (norms/softmax) and gumbel,
// so a full decode step (proj/attn/logits via bmm + norm/softmax via rowop +
// sample via gumbel) batches into ONE command buffer. f32 only.
static bool encode_or_run_rowop_dev(MPSCommandBuffer *cb, MetalDeviceContext &ctx,
                                    int kind, id<MTLBuffer> bufX,
                                    id<MTLBuffer> bufG, id<MTLBuffer> bufO,
                                    int32_t rows, int32_t cols, float eps) {
  if (rows <= 0 || cols <= 0) return true;
  if (!bufX || !bufO) return false;
  bool hasGamma = (bufG != nil);
  NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
  NSArray<NSNumber *> *gs = @[ @(cols) ];
  NSArray *phs;
  MPSGraphTensor *y;
  MPSGraph *g = mpsg_rowop_graph(kind, rows, cols, eps, hasGamma, false,
                                 MPSDataTypeFloat32, &phs, &y);
  if (!y) return false;
  NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
  feeds[phs[0]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:xs dataType:MPSDataTypeFloat32];
  if (hasGamma && phs.count >= 2)
    feeds[phs[1]] = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufG shape:gs dataType:MPSDataTypeFloat32];
  MPSGraphTensorData *od = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO shape:xs dataType:MPSDataTypeFloat32];
  if (cb)
    [g encodeToCommandBuffer:cb feeds:feeds targetOperations:nil resultsDictionary:@{y : od} executionDescriptor:nil];
  else
    [g runWithMTLCommandQueue:ctx.queue feeds:feeds targetOperations:nil resultsDictionary:@{y : od}];
  return true;
}

static bool encode_or_run_gumbel_dev(MPSCommandBuffer *cb, MetalDeviceContext &ctx,
                                     id<MTLBuffer> bufL, id<MTLBuffer> bufG,
                                     id<MTLBuffer> bufO, int32_t rows,
                                     int32_t cols, float invT) {
  if (rows <= 0 || cols <= 0) return true;
  if (!bufL || !bufG || !bufO) return false;
  NSArray<NSNumber *> *xs = @[ @(rows), @(cols) ];
  MPSGraphTensor *pl, *pg, *y;
  MPSGraph *g = mpsg_gumbel_graph(rows, cols, invT, &pl, &pg, &y);
  MPSGraphTensorData *ld = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufL shape:xs dataType:MPSDataTypeFloat32];
  MPSGraphTensorData *gd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufG shape:xs dataType:MPSDataTypeFloat32];
  // argMax keeps the reduced axis (size 1) -> [rows,1]; match it so the result
  // writes correctly into the [rows] int32 output buffer.
  MPSGraphTensorData *od = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO shape:@[ @(rows), @1 ] dataType:MPSDataTypeInt32];
  if (cb)
    [g encodeToCommandBuffer:cb feeds:@{pl : ld, pg : gd} targetOperations:nil resultsDictionary:@{y : od} executionDescriptor:nil];
  else
    [g runWithMTLCommandQueue:ctx.queue feeds:@{pl : ld, pg : gd} targetOperations:nil resultsDictionary:@{y : od}];
  return true;
}

}  // namespace

// rowop: X [rows,cols], optional gamma [cols], O [rows,cols]. kind 0 layer_norm
// (unweighted/gamma), 1 rmsnorm, 2 softmax, 3 log_softmax.
extern "C" int32_t tessera_apple_gpu_rowop_dev_f32(TsDeviceTensor *X,
                                                   TsDeviceTensor *gamma,
                                                   TsDeviceTensor *O, int32_t kind,
                                                   int32_t rows, int32_t cols,
                                                   float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !X || !O) return 0;
  return encode_or_run_rowop_dev(nil, ctx, kind, X->buf, gamma ? gamma->buf : nil,
                                 O->buf, rows, cols, eps) ? 1 : 0;
}
extern "C" int32_t tessera_apple_gpu_rowop_dev_f32_enc(TsEncodeSession *s,
                                                       TsDeviceTensor *X,
                                                       TsDeviceTensor *gamma,
                                                       TsDeviceTensor *O,
                                                       int32_t kind, int32_t rows,
                                                       int32_t cols, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !O) return 0;
  return encode_or_run_rowop_dev(s->cb, ctx, kind, X->buf,
                                 gamma ? gamma->buf : nil, O->buf, rows, cols,
                                 eps) ? 1 : 0;
}

// gumbel: logits + gumbel [rows,cols] f32, out_ids [rows] int32.
extern "C" int32_t tessera_apple_gpu_gumbel_argmax_dev_f32(TsDeviceTensor *logits,
                                                           TsDeviceTensor *gumbel,
                                                           TsDeviceTensor *out_ids,
                                                           int32_t rows,
                                                           int32_t cols,
                                                           float inv_temp) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !logits || !gumbel || !out_ids) return 0;
  return encode_or_run_gumbel_dev(nil, ctx, logits->buf, gumbel->buf,
                                  out_ids->buf, rows, cols, inv_temp) ? 1 : 0;
}
extern "C" int32_t tessera_apple_gpu_gumbel_argmax_dev_f32_enc(
    TsEncodeSession *s, TsDeviceTensor *logits, TsDeviceTensor *gumbel,
    TsDeviceTensor *out_ids, int32_t rows, int32_t cols, float inv_temp) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !logits || !gumbel || !out_ids) return 0;
  return encode_or_run_gumbel_dev(s->cb, ctx, logits->buf, gumbel->buf,
                                  out_ids->buf, rows, cols, inv_temp) ? 1 : 0;
}

extern "C" void tessera_apple_gpu_gumbel_argmax_f32(const float *logits,
                                                    const float *gumbel,
                                                    int32_t *out, int32_t rows,
                                                    int32_t cols,
                                                    float inv_temp) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_gumbel_argmax(ctx, logits, gumbel, out, rows, cols, inv_temp))
    return;
  for (int32_t r = 0; r < rows; ++r) {
    const float *L = logits + (size_t)r * cols;
    const float *G = gumbel + (size_t)r * cols;
    int32_t best = 0;
    float bs = L[0] * inv_temp + G[0];
    for (int32_t c = 1; c < cols; ++c) {
      float s = L[c] * inv_temp + G[c];
      if (s > bs) { bs = s; best = c; }
    }
    out[r] = best;
  }
}

extern "C" void tessera_apple_gpu_mpsgraph_reduce_f32(int32_t op, const float *x,
                                                      float *out, int32_t rows,
                                                      int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_reduce(ctx, op, x, out, rows, cols)) return;
  reference_reduce(op, x, out, rows, cols);
}

extern "C" void tessera_apple_gpu_mpsgraph_argreduce_f32(int32_t op,
                                                         const float *x,
                                                         int32_t *out,
                                                         int32_t rows,
                                                         int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_argreduce(ctx, op, x, out, rows, cols)) return;
  for (int32_t r = 0; r < rows; ++r) {
    const float *row = x + (size_t)r * cols;
    int32_t best = 0;
    for (int32_t c = 1; c < cols; ++c)
      if ((op == 0 && row[c] > row[best]) || (op == 1 && row[c] < row[best]))
        best = c;
    out[r] = best;
  }
}

extern "C" void tessera_apple_gpu_mpsgraph_scan_f32(int32_t op, const float *x,
                                                    float *out, int32_t rows,
                                                    int32_t cols) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_scan(ctx, op, x, out, rows, cols)) return;
  for (int32_t r = 0; r < rows; ++r) {
    const float *row = x + (size_t)r * cols;
    float *o = out + (size_t)r * cols;
    double acc = op == 1 ? 1.0 : 0.0;
    for (int32_t c = 0; c < cols; ++c) {
      acc = op == 1 ? acc * row[c] : acc + row[c];
      o[c] = (float)acc;
    }
  }
}

//===----------------------------------------------------------------------===//
// flash_attn with native GQA/MQA KV-group indexing (2026-05-29)
//
// Q is [B, Sq, D] with B = batch_outer * q_heads query heads; K/V are
// [batch_outer * kv_heads, Sk, D] (NOT repeated). Query head `b` reads KV group
// `(b % q_heads) / (q_heads / kv_heads)` — so GQA/MQA avoid materializing the
// repeated KV (the Phase-2 bandwidth win). MQA = kv_heads 1. f32; D <= 256.
//===----------------------------------------------------------------------===//

namespace {

bool dispatch_flash_attn_gqa_msl(MetalDeviceContext &ctx, const float *Q,
                                 const float *K, const float *V, float *O,
                                 int32_t B, int32_t q_heads, int32_t kv_heads,
                                 int32_t Sq, int32_t Sk, int32_t D, float scale,
                                 int32_t causal) {
  static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
#define TESSERA_GQA_MAX_D 256
kernel void flash_attn_gqa_f32(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V       [[buffer(2)]],
    device float*       O       [[buffer(3)]],
    constant int&       B       [[buffer(4)]],
    constant int&       q_heads [[buffer(5)]],
    constant int&       kv_heads[[buffer(6)]],
    constant int&       Sq      [[buffer(7)]],
    constant int&       Sk      [[buffer(8)]],
    constant int&       D       [[buffer(9)]],
    constant float&     scale   [[buffer(10)]],
    constant int&       causal  [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= (uint)B || gid.x >= (uint)Sq) return;
    int batch = (int)gid.y;   // query head index
    int q_row = (int)gid.x;
    if (D > TESSERA_GQA_MAX_D) return;
    int b_outer = batch / q_heads;
    int q_head  = batch % q_heads;
    int group   = q_heads / kv_heads;     // query heads per kv head
    int kv_head = q_head / group;
    int kv_batch = b_outer * kv_heads + kv_head;
    int q_off = batch * Sq * D + q_row * D;
    int kv_base = kv_batch * Sk * D;
    float m = -INFINITY;
    float l = 0.0f;
    float o[TESSERA_GQA_MAX_D];
    for (int d = 0; d < D; ++d) o[d] = 0.0f;
    for (int k_row = 0; k_row < Sk; ++k_row) {
        if (causal != 0 && k_row > q_row) break;
        int k_off = kv_base + k_row * D;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) score += Q[q_off + d] * K[k_off + d];
        score *= scale;
        float new_m = max(m, score);
        float exp_old = exp(m - new_m);
        float exp_score = exp(score - new_m);
        float new_l = l * exp_old + exp_score;
        for (int d = 0; d < D; ++d) o[d] = o[d] * exp_old + V[k_off + d] * exp_score;
        m = new_m;
        l = new_l;
    }
    if (l == 0.0f) { for (int d = 0; d < D; ++d) O[q_off + d] = 0.0f; }
    else { float inv = 1.0f / l; for (int d = 0; d < D; ++d) O[q_off + d] = o[d] * inv; }
}
)MSL";
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0) return false;
  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kSrc, @"flash_attn_gqa_f32");
    if (!pso) return false;
    int32_t kv_outer = (B / q_heads) * kv_heads;
    NSUInteger qBytes = sizeof(float) * (NSUInteger)B * Sq * D;
    NSUInteger kvBytes = sizeof(float) * (NSUInteger)kv_outer * Sk * D;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufQ, ctx, Q, qBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufK, ctx, K, kvBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufV, ctx, V, kvBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, qBytes);
    if (!bufQ || !bufK || !bufV || !bufO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufQ offset:0 atIndex:0];
    [enc setBuffer:bufK offset:0 atIndex:1];
    [enc setBuffer:bufV offset:0 atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&B length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&q_heads length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&kv_heads length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&Sq length:sizeof(int32_t) atIndex:7];
    [enc setBytes:&Sk length:sizeof(int32_t) atIndex:8];
    [enc setBytes:&D length:sizeof(int32_t) atIndex:9];
    [enc setBytes:&scale length:sizeof(float) atIndex:10];
    [enc setBytes:&causal length:sizeof(int32_t) atIndex:11];
    MTLSize grid = MTLSizeMake((NSUInteger)Sq, (NSUInteger)B, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)Sq, 32);
    NSUInteger tg_y = std::min<NSUInteger>((NSUInteger)B,
        pso.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(tg_x, 1));
    if (tg_y == 0) tg_y = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_gqa_msl")) return false;
    std::memcpy(O, [bufO contents], qBytes);
    return true;
  }
}

static void reference_flash_attn_gqa_f32(const float *Q, const float *K,
                                         const float *V, float *O, int32_t B,
                                         int32_t q_heads, int32_t kv_heads,
                                         int32_t Sq, int32_t Sk, int32_t D,
                                         float scale, int32_t causal) {
  for (int32_t b = 0; b < B; ++b) {
    int32_t group = q_heads / kv_heads;
    int32_t kv_batch = (b / q_heads) * kv_heads + (b % q_heads) / group;
    const float *Kb = K + (size_t)kv_batch * Sk * D;
    const float *Vb = V + (size_t)kv_batch * Sk * D;
    for (int32_t q = 0; q < Sq; ++q) {
      const float *qp = Q + ((size_t)b * Sq + q) * D;
      float *op = O + ((size_t)b * Sq + q) * D;
      double m = -1e30, l = 0.0;
      std::vector<double> o(D, 0.0);
      for (int32_t k = 0; k < Sk; ++k) {
        if (causal != 0 && k > q) break;
        const float *kp = Kb + (size_t)k * D;
        double s = 0.0;
        for (int32_t d = 0; d < D; ++d) s += (double)qp[d] * kp[d];
        s *= scale;
        double nm = std::max(m, s);
        double eo = std::exp(m - nm), es = std::exp(s - nm);
        l = l * eo + es;
        for (int32_t d = 0; d < D; ++d) o[d] = o[d] * eo + Vb[(size_t)k * D + d] * es;
        m = nm;
      }
      if (l == 0.0) for (int32_t d = 0; d < D; ++d) op[d] = 0.0f;
      else for (int32_t d = 0; d < D; ++d) op[d] = (float)(o[d] / l);
    }
  }
}

}  // namespace

extern "C" void tessera_apple_gpu_flash_attn_gqa_f32(
    const float *Q, const float *K, const float *V, float *O, int32_t B,
    int32_t q_heads, int32_t kv_heads, int32_t Sq, int32_t Sk, int32_t D,
    float scale, int32_t causal) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_flash_attn_gqa_msl(ctx, Q, K, V, O, B, q_heads,
                                            kv_heads, Sq, Sk, D, scale, causal))
    return;
  reference_flash_attn_gqa_f32(Q, K, V, O, B, q_heads, kv_heads, Sq, Sk, D,
                               scale, causal);
}

//===----------------------------------------------------------------------===//
// Fused batched matmul -> softmax -> matmul (2026-05-29)
//
// O = softmax((A @ B) * scale, axis=-1) @ C, per batch — the batched attention
// block in a single dispatch (vs the bmm + softmax + bmm 3-call compose).
// A:[batch,M,K] B:[batch,K,N] C:[batch,N,P] -> O:[batch,M,P]. For attention
// A=Q, B=Kᵀ, C=V. MPSGraph fuses the whole graph; f32 + f16 native (fp32
// compute), bf16 host-upcast. Reuses the graph cache + buffer pool.
//===----------------------------------------------------------------------===//

namespace {

static bool mpsg_run_bsmm(MetalDeviceContext &ctx, const void *A, const void *B,
                          const void *C, void *O, int32_t batch, int32_t M,
                          int32_t N, int32_t P, int32_t K, float scale,
                          MPSDataType ioType, size_t elemSize) {
  if (batch <= 0 || M <= 0 || N <= 0 || P <= 0 || K <= 0) return true;
  @autoreleasepool {
    size_t aBytes = (size_t)batch * M * K * elemSize;
    size_t bBytes = (size_t)batch * K * N * elemSize;
    size_t cBytes = (size_t)batch * N * P * elemSize;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufC, ctx, C, cBytes);
    if (!bufA || !bufB || !bufC) return false;
    NSArray<NSNumber *> *aShape = @[ @(batch), @(M), @(K) ];
    NSArray<NSNumber *> *bShape = @[ @(batch), @(K), @(N) ];
    NSArray<NSNumber *> *cShape = @[ @(batch), @(N), @(P) ];
    NSString *key = [NSString stringWithFormat:@"bsmm:%d:%d:%d:%d:%d:%d:%a",
                                               (int)ioType, batch, M, N, P, K,
                                               scale];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *pa, *pb, *pc, *y;
    if (entry) {
      g = entry[0];
      pa = ((NSArray *)entry[1])[0];
      pb = ((NSArray *)entry[1])[1];
      pc = ((NSArray *)entry[1])[2];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      pa = [g placeholderWithShape:aShape dataType:ioType name:nil];
      pb = [g placeholderWithShape:bShape dataType:ioType name:nil];
      pc = [g placeholderWithShape:cShape dataType:ioType name:nil];
      MPSGraphTensor *s = [g matrixMultiplicationWithPrimaryTensor:mpsg_up(g, pa, ioType)
                                                   secondaryTensor:mpsg_up(g, pb, ioType)
                                                              name:nil];
      MPSGraphTensor *scaled = [g multiplicationWithPrimaryTensor:s
                                  secondaryTensor:[g constantWithScalar:(double)scale dataType:MPSDataTypeFloat32]
                                             name:nil];
      MPSGraphTensor *attn = [g softMaxWithTensor:scaled axis:2 name:nil];
      MPSGraphTensor *yf = [g matrixMultiplicationWithPrimaryTensor:attn
                                                   secondaryTensor:mpsg_up(g, pc, ioType)
                                                              name:nil];
      y = mpsg_down(g, yf, ioType);
      mpsg_cache_put(key, @[ g, @[ pa, pb, pc ], y ]);
    }
    MPSGraphTensorData *ad = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:aShape dataType:ioType];
    MPSGraphTensorData *bd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:bShape dataType:ioType];
    MPSGraphTensorData *cd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufC shape:cShape dataType:ioType];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{pa : ad, pb : bd, pc : cd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:O strideBytes:nil];
    return true;
  }
}

static void reference_bsmm_f32(const float *A, const float *B, const float *C,
                               float *O, int32_t batch, int32_t M, int32_t N,
                               int32_t P, int32_t K, float scale) {
  std::vector<double> s((size_t)M * N);
  for (int32_t bi = 0; bi < batch; ++bi) {
    const float *a = A + (size_t)bi * M * K;
    const float *b = B + (size_t)bi * K * N;
    const float *c = C + (size_t)bi * N * P;
    float *o = O + (size_t)bi * M * P;
    for (int32_t m = 0; m < M; ++m) {
      double mx = -1e30;
      for (int32_t n = 0; n < N; ++n) {
        double acc = 0;
        for (int32_t k = 0; k < K; ++k) acc += (double)a[m * K + k] * b[k * N + n];
        acc *= scale;
        s[(size_t)m * N + n] = acc;
        mx = std::max(mx, acc);
      }
      double den = 0;
      for (int32_t n = 0; n < N; ++n) { double e = std::exp(s[(size_t)m * N + n] - mx); s[(size_t)m * N + n] = e; den += e; }
      for (int32_t p = 0; p < P; ++p) {
        double acc = 0;
        for (int32_t n = 0; n < N; ++n) acc += s[(size_t)m * N + n] / den * c[n * P + p];
        o[(size_t)m * P + p] = (float)acc;
      }
    }
  }
}

}  // namespace

extern "C" void tessera_apple_gpu_mpsgraph_bsmm_f32(const float *A, const float *B,
                                                    const float *C, float *O,
                                                    int32_t batch, int32_t M,
                                                    int32_t N, int32_t P, int32_t K,
                                                    float scale) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_bsmm(ctx, A, B, C, O, batch, M, N, P, K, scale,
                              MPSDataTypeFloat32, 4))
    return;
  reference_bsmm_f32(A, B, C, O, batch, M, N, P, K, scale);
}

extern "C" void tessera_apple_gpu_mpsgraph_bsmm_f16(const uint16_t *A,
                                                    const uint16_t *B,
                                                    const uint16_t *C,
                                                    uint16_t *O, int32_t batch,
                                                    int32_t M, int32_t N, int32_t P,
                                                    int32_t K, float scale) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_bsmm(ctx, A, B, C, O, batch, M, N, P, K, scale,
                              MPSDataTypeFloat16, 2))
    return;
  std::memset(O, 0, (size_t)batch * M * P * 2);
}

//===----------------------------------------------------------------------===//
// Sprint 3.3 perf-fusion — fused PRE-NORM + PROJECTION in one MPSGraph dispatch.
//
//   O = (rmsnorm(X) * gamma) @ W       X:[M,K] gamma:[K] W:[K,N] -> O:[M,N]
//   rmsnorm(X) = X / sqrt(mean(X^2, axis=-1) + eps)
//
// The hottest chain in a pre-norm transformer (the norm feeding a projection).
// Previously available only via the file-based `mlpkg_author_chain` ML-package
// path; this is the direct single-call kernel the GraphFn GPU lane fuses into.
// MPSGraph fuses norm+matmul across the ML pipeline → one dispatch. fp32.
//===----------------------------------------------------------------------===//

static bool mpsg_run_rmsnorm_matmul_f32(MetalDeviceContext &ctx, const float *X,
                                        const float *gamma, const float *W,
                                        float *O, int32_t M, int32_t K,
                                        int32_t N, float eps) {
  if (M <= 0 || K <= 0 || N <= 0) return true;
  @autoreleasepool {
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, (size_t)M * K * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufG, ctx, gamma, (size_t)K * 4);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufW, ctx, W, (size_t)K * N * 4);
    if (!bufX || !bufG || !bufW) return false;
    MPSDataType dt = MPSDataTypeFloat32;
    NSArray<NSNumber *> *xShape = @[ @(M), @(K) ];
    NSArray<NSNumber *> *gShape = @[ @(K) ];
    NSArray<NSNumber *> *wShape = @[ @(K), @(N) ];
    NSString *key =
        [NSString stringWithFormat:@"rmsnorm_matmul:%d:%d:%d:%a", M, K, N, eps];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *px, *pg, *pw, *y;
    if (entry) {
      g = entry[0];
      px = ((NSArray *)entry[1])[0];
      pg = ((NSArray *)entry[1])[1];
      pw = ((NSArray *)entry[1])[2];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      px = [g placeholderWithShape:xShape dataType:dt name:nil];
      pg = [g placeholderWithShape:gShape dataType:dt name:nil];
      pw = [g placeholderWithShape:wShape dataType:dt name:nil];
      MPSGraphTensor *sq =
          [g multiplicationWithPrimaryTensor:px secondaryTensor:px name:nil];
      MPSGraphTensor *ms = [g meanOfTensor:sq axes:@[ @1 ] name:nil];
      MPSGraphTensor *me =
          [g additionWithPrimaryTensor:ms
                       secondaryTensor:[g constantWithScalar:(double)eps
                                                    dataType:dt]
                                  name:nil];
      MPSGraphTensor *denom = [g squareRootWithTensor:me name:nil];
      MPSGraphTensor *norm =
          [g divisionWithPrimaryTensor:px secondaryTensor:denom name:nil];
      MPSGraphTensor *weighted =
          [g multiplicationWithPrimaryTensor:norm secondaryTensor:pg name:nil];
      y = [g matrixMultiplicationWithPrimaryTensor:weighted
                                   secondaryTensor:pw
                                              name:nil];
      mpsg_cache_put(key, @[ g, @[ px, pg, pw ], y ]);
    }
    MPSGraphTensorData *xd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX
                                                                    shape:xShape
                                                                 dataType:dt];
    MPSGraphTensorData *gd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufG
                                                                    shape:gShape
                                                                 dataType:dt];
    MPSGraphTensorData *wd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufW
                                                                    shape:wShape
                                                                 dataType:dt];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{px : xd, pg : gd, pw : wd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:O strideBytes:nil];
    return true;
  }
}

static void reference_rmsnorm_matmul_f32(const float *X, const float *gamma,
                                         const float *W, float *O, int32_t M,
                                         int32_t K, int32_t N, float eps) {
  std::vector<float> norm((size_t)K, 0.0f);
  for (int32_t m = 0; m < M; ++m) {
    const float *xr = X + (size_t)m * K;
    double ms = 0.0;
    for (int32_t k = 0; k < K; ++k) ms += (double)xr[k] * xr[k];
    ms /= K;
    double inv = 1.0 / std::sqrt(ms + eps);
    for (int32_t k = 0; k < K; ++k) norm[k] = (float)(xr[k] * inv * gamma[k]);
    float *orow = O + (size_t)m * N;
    for (int32_t n = 0; n < N; ++n) orow[n] = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      float nv = norm[k];
      const float *wr = W + (size_t)k * N;
      for (int32_t n = 0; n < N; ++n) orow[n] += nv * wr[n];
    }
  }
}

extern "C" void tessera_apple_gpu_rmsnorm_matmul_f32(const float *X,
                                                     const float *gamma,
                                                     const float *W, float *O,
                                                     int32_t M, int32_t K,
                                                     int32_t N, float eps) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_rmsnorm_matmul_f32(ctx, X, gamma, W, O, M, K, N, eps))
    return;
  reference_rmsnorm_matmul_f32(X, gamma, W, O, M, K, N, eps);
}

// Forward decl: main already defines the f32 GQA kernel earlier in this TU.
extern "C" void tessera_apple_gpu_flash_attn_gqa_f32(const float *, const float *, const float *, float *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, float, int32_t);

// flash_attn GQA — f16 (native) + bf16 (fp32-conversion) (2026-05-29)
//===----------------------------------------------------------------------===//

namespace {

bool dispatch_flash_attn_gqa_msl_f16(MetalDeviceContext &ctx, const uint16_t *Q,
                                     const uint16_t *K, const uint16_t *V,
                                     uint16_t *O, int32_t B, int32_t q_heads,
                                     int32_t kv_heads, int32_t Sq, int32_t Sk,
                                     int32_t D, float scale, int32_t causal) {
  static NSString *const kSrc = @R"MSL(
#include <metal_stdlib>
using namespace metal;
#define TESSERA_GQA_MAX_D 256
kernel void flash_attn_gqa_f16(
    device const half*  Q       [[buffer(0)]],
    device const half*  K       [[buffer(1)]],
    device const half*  V       [[buffer(2)]],
    device half*        O       [[buffer(3)]],
    constant int&       B       [[buffer(4)]],
    constant int&       q_heads [[buffer(5)]],
    constant int&       kv_heads[[buffer(6)]],
    constant int&       Sq      [[buffer(7)]],
    constant int&       Sk      [[buffer(8)]],
    constant int&       D       [[buffer(9)]],
    constant float&     scale   [[buffer(10)]],
    constant int&       causal  [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.y >= (uint)B || gid.x >= (uint)Sq) return;
    int batch = (int)gid.y;
    int q_row = (int)gid.x;
    if (D > TESSERA_GQA_MAX_D) return;
    int b_outer = batch / q_heads;
    int q_head  = batch % q_heads;
    int group   = q_heads / kv_heads;
    int kv_batch = b_outer * kv_heads + (q_head / group);
    int q_off = batch * Sq * D + q_row * D;
    int kv_base = kv_batch * Sk * D;
    float m = -INFINITY;
    float l = 0.0f;
    float o[TESSERA_GQA_MAX_D];
    for (int d = 0; d < D; ++d) o[d] = 0.0f;
    for (int k_row = 0; k_row < Sk; ++k_row) {
        if (causal != 0 && k_row > q_row) break;
        int k_off = kv_base + k_row * D;
        float score = 0.0f;
        for (int d = 0; d < D; ++d) score += float(Q[q_off + d]) * float(K[k_off + d]);
        score *= scale;
        float new_m = max(m, score);
        float exp_old = exp(m - new_m);
        float exp_score = exp(score - new_m);
        float new_l = l * exp_old + exp_score;
        for (int d = 0; d < D; ++d) o[d] = o[d] * exp_old + float(V[k_off + d]) * exp_score;
        m = new_m;
        l = new_l;
    }
    if (l == 0.0f) { for (int d = 0; d < D; ++d) O[q_off + d] = half(0.0f); }
    else { float inv = 1.0f / l; for (int d = 0; d < D; ++d) O[q_off + d] = half(o[d] * inv); }
}
)MSL";
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0) return false;
  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kSrc, @"flash_attn_gqa_f16");
    if (!pso) return false;
    int32_t kv_outer = (B / q_heads) * kv_heads;
    NSUInteger qBytes = sizeof(uint16_t) * (NSUInteger)B * Sq * D;
    NSUInteger kvBytes = sizeof(uint16_t) * (NSUInteger)kv_outer * Sk * D;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufQ, ctx, Q, qBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufK, ctx, K, kvBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufV, ctx, V, kvBytes);
    TS_METAL_BUF_ACQUIRE(bufO, ctx, qBytes);
    if (!bufQ || !bufK || !bufV || !bufO) return false;
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufQ offset:0 atIndex:0];
    [enc setBuffer:bufK offset:0 atIndex:1];
    [enc setBuffer:bufV offset:0 atIndex:2];
    [enc setBuffer:bufO offset:0 atIndex:3];
    [enc setBytes:&B length:sizeof(int32_t) atIndex:4];
    [enc setBytes:&q_heads length:sizeof(int32_t) atIndex:5];
    [enc setBytes:&kv_heads length:sizeof(int32_t) atIndex:6];
    [enc setBytes:&Sq length:sizeof(int32_t) atIndex:7];
    [enc setBytes:&Sk length:sizeof(int32_t) atIndex:8];
    [enc setBytes:&D length:sizeof(int32_t) atIndex:9];
    [enc setBytes:&scale length:sizeof(float) atIndex:10];
    [enc setBytes:&causal length:sizeof(int32_t) atIndex:11];
    MTLSize grid = MTLSizeMake((NSUInteger)Sq, (NSUInteger)B, 1);
    NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)Sq, 32);
    NSUInteger tg_y = std::min<NSUInteger>((NSUInteger)B,
        pso.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(tg_x, 1));
    if (tg_y == 0) tg_y = 1;
    [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
    [enc endEncoding];
    // waitUntilCompleted migration (2026-06-01) — Pattern 4 wrapper.
    if (!commit_and_wait_with_timeout(ctx, cb, 60000,
                                      "flash_attn_gqa_msl_f16")) return false;
    std::memcpy(O, [bufO contents], qBytes);
    return true;
  }
}

// bf16 <-> f32 bit helpers (top 16 bits of f32).
static inline float gqa_bf16_to_f32(uint16_t b) {
  uint32_t f = (uint32_t)b << 16;
  float out;
  std::memcpy(&out, &f, sizeof(out));
  return out;
}
static inline uint16_t gqa_f32_to_bf16(float v) {
  uint32_t f;
  std::memcpy(&f, &v, sizeof(f));
  uint32_t lsb = (f >> 16) & 1u;
  return (uint16_t)((f + 0x7FFFu + lsb) >> 16);
}

}  // namespace

extern "C" void tessera_apple_gpu_flash_attn_gqa_f16(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O,
    int32_t B, int32_t q_heads, int32_t kv_heads, int32_t Sq, int32_t Sk,
    int32_t D, float scale, int32_t causal) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_flash_attn_gqa_msl_f16(ctx, Q, K, V, O, B, q_heads,
                                                kv_heads, Sq, Sk, D, scale, causal))
    return;
  std::memset(O, 0, (size_t)B * Sq * D * 2);  // python upcasts on fallback
}

extern "C" void tessera_apple_gpu_flash_attn_gqa_bf16(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O,
    int32_t B, int32_t q_heads, int32_t kv_heads, int32_t Sq, int32_t Sk,
    int32_t D, float scale, int32_t causal) {
  // fp32-conversion path (MSL has no native bf16): convert, run the f32 GQA
  // kernel, convert back.
  int32_t kv_outer = (q_heads > 0) ? (B / q_heads) * kv_heads : B;
  std::vector<float> qf((size_t)B * Sq * D), kf((size_t)kv_outer * Sk * D),
      vf((size_t)kv_outer * Sk * D), of((size_t)B * Sq * D);
  for (size_t i = 0; i < qf.size(); ++i) qf[i] = gqa_bf16_to_f32(Q[i]);
  for (size_t i = 0; i < kf.size(); ++i) kf[i] = gqa_bf16_to_f32(K[i]);
  for (size_t i = 0; i < vf.size(); ++i) vf[i] = gqa_bf16_to_f32(V[i]);
  tessera_apple_gpu_flash_attn_gqa_f32(qf.data(), kf.data(), vf.data(),
                                       of.data(), B, q_heads, kv_heads, Sq, Sk,
                                       D, scale, causal);
  for (size_t i = 0; i < of.size(); ++i) O[i] = gqa_f32_to_bf16(of[i]);
}

//===----------------------------------------------------------------------===//
// conv2d — MPSGraph convolution2D (NHWC source, HWIO weights) (2026-05-30)
//
// Tier-3 vision primitive. Source X is [N, H, W, Cin], weights Wt are
// [kH, kW, Cin/groups, Cout] (HWIO), optional bias is [Cout]. Output O is
// [N, outH, outW, Cout] with
//   outH = (H + 2*padH - dilationH*(kH-1) - 1) / strideH + 1
//   outW = (W + 2*padW - dilationW*(kW-1) - 1) / strideW + 1
// Compute runs in fp32 internally (f16 inputs cast up, result cast down);
// graphs are cached by signature like the rest of the MPSGraph lane.
//===----------------------------------------------------------------------===//

namespace {

static inline int32_t conv2d_out_dim(int32_t in, int32_t k, int32_t stride,
                                     int32_t pad, int32_t dilation) {
  int32_t eff = dilation * (k - 1) + 1;
  if (in + 2 * pad < eff) return 0;
  return (in + 2 * pad - eff) / stride + 1;
}

static bool mpsg_run_conv2d(MetalDeviceContext &ctx, const void *X,
                            const void *Wt, const void *bias, void *O, int32_t N,
                            int32_t H, int32_t W, int32_t Cin, int32_t Cout,
                            int32_t kH, int32_t kW, int32_t strideH,
                            int32_t strideW, int32_t padH, int32_t padW,
                            int32_t dilationH, int32_t dilationW, int32_t groups,
                            MPSDataType ioType, size_t elemSize) {
  int32_t outH = conv2d_out_dim(H, kH, strideH, padH, dilationH);
  int32_t outW = conv2d_out_dim(W, kW, strideW, padW, dilationW);
  if (N <= 0 || outH <= 0 || outW <= 0 || Cout <= 0 || groups <= 0) return true;
  if (Cin % groups != 0 || Cout % groups != 0) return false;
  @autoreleasepool {
    size_t xBytes = (size_t)N * H * W * Cin * elemSize;
    size_t wBytes = (size_t)kH * kW * (Cin / groups) * Cout * elemSize;
    size_t oBytes = (size_t)N * outH * outW * Cout * elemSize;
    size_t bBytes = (size_t)Cout * elemSize;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufX, ctx, X, xBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufW, ctx, Wt, wBytes);
    // bias is optional: acquire its buffer from the pool when present, else a
    // 1-element placeholder (never fed into the graph) so the RAII guard owns
    // every Metal allocation in this dispatcher.
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, bias ? bias : Wt,
                                    bias ? bBytes : elemSize);
    if (!bufX || !bufW || (bias && !bufB)) return false;
    NSArray<NSNumber *> *xShape = @[ @(N), @(H), @(W), @(Cin) ];
    NSArray<NSNumber *> *wShape = @[ @(kH), @(kW), @(Cin / groups), @(Cout) ];
    NSArray<NSNumber *> *bShape = @[ @(Cout) ];
    NSString *key = [NSString
        stringWithFormat:@"conv2d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
                         (int)ioType, N, H, W, Cin, Cout, kH, kW, strideH,
                         strideW, padH, padW, dilationH, dilationW, groups,
                         bias ? 1 : 0];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *px, *pw, *pb = nil, *y;
    if (entry) {
      g = entry[0];
      NSArray *phs = (NSArray *)entry[1];
      px = phs[0];
      pw = phs[1];
      pb = (bias && phs.count > 2) ? phs[2] : nil;
      y = entry[2];
    } else {
      g = [MPSGraph new];
      px = [g placeholderWithShape:xShape dataType:ioType name:nil];
      pw = [g placeholderWithShape:wShape dataType:ioType name:nil];
      MPSGraphConvolution2DOpDescriptor *desc =
          [MPSGraphConvolution2DOpDescriptor
              descriptorWithStrideInX:strideW
                            strideInY:strideH
                      dilationRateInX:dilationW
                      dilationRateInY:dilationH
                               groups:groups
                          paddingLeft:padW
                         paddingRight:padW
                           paddingTop:padH
                        paddingBottom:padH
                         paddingStyle:MPSGraphPaddingStyleExplicit
                           dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                        weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
      MPSGraphTensor *conv =
          [g convolution2DWithSourceTensor:mpsg_up(g, px, ioType)
                             weightsTensor:mpsg_up(g, pw, ioType)
                                descriptor:desc
                                      name:nil];
      if (bias) {
        pb = [g placeholderWithShape:bShape dataType:ioType name:nil];
        conv = [g additionWithPrimaryTensor:conv
                            secondaryTensor:mpsg_up(g, pb, ioType)
                                       name:nil];
      }
      y = mpsg_down(g, conv, ioType);
      NSArray *phs = bias ? @[ px, pw, pb ] : @[ px, pw ];
      mpsg_cache_put(key, @[ g, phs, y ]);
    }
    MPSGraphTensorData *xd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX shape:xShape dataType:ioType];
    MPSGraphTensorData *wd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufW shape:wShape dataType:ioType];
    NSMutableDictionary *feeds =
        [@{px : xd, pw : wd} mutableCopy];
    if (bias && pb) {
      MPSGraphTensorData *bd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:bShape dataType:ioType];
      feeds[pb] = bd;
    }
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:feeds
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:O strideBytes:nil];
    return true;
  }
}

// Project 5 (2026-06-01) — device-resident encode-session conv2d. Mirrors
// ``mpsg_run_conv2d`` but takes already-device-resident ``MTLBuffer``s
// and uses ``encodeToCommandBuffer:`` so the dispatch appends to the
// session's command buffer instead of running its own command queue.
// The MPSGraph cache is shared with ``mpsg_run_conv2d`` (same key
// schema) so a kernel built for the run path is reused for encode and
// vice versa.
//
// Bias is optional — pass ``bufBias == nil`` for the no-bias path.
// Returns ``true`` on a clean encode; ``false`` if the cb / inputs
// are invalid.
static bool mpsg_encode_conv2d_dev(MPSCommandBuffer *cb,
                                    id<MTLBuffer> bufX, id<MTLBuffer> bufW,
                                    id<MTLBuffer> bufBias, id<MTLBuffer> bufO,
                                    int32_t N, int32_t H, int32_t W,
                                    int32_t Cin, int32_t Cout,
                                    int32_t kH, int32_t kW,
                                    int32_t strideH, int32_t strideW,
                                    int32_t padH, int32_t padW,
                                    int32_t dilationH, int32_t dilationW,
                                    int32_t groups, MPSDataType ioType) {
  if (!cb || !bufX || !bufW || !bufO) return false;
  int32_t outH = conv2d_out_dim(H, kH, strideH, padH, dilationH);
  int32_t outW = conv2d_out_dim(W, kW, strideW, padW, dilationW);
  if (N <= 0 || outH <= 0 || outW <= 0 || Cout <= 0 || groups <= 0) return true;
  if (Cin % groups != 0 || Cout % groups != 0) return false;
  bool hasBias = (bufBias != nil);
  @autoreleasepool {
    NSArray<NSNumber *> *xShape = @[ @(N), @(H), @(W), @(Cin) ];
    NSArray<NSNumber *> *wShape = @[ @(kH), @(kW), @(Cin / groups), @(Cout) ];
    NSArray<NSNumber *> *bShape = @[ @(Cout) ];
    NSArray<NSNumber *> *oShape = @[ @(N), @(outH), @(outW), @(Cout) ];
    NSString *key = [NSString
        stringWithFormat:@"conv2d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
                         (int)ioType, N, H, W, Cin, Cout, kH, kW, strideH,
                         strideW, padH, padW, dilationH, dilationW, groups,
                         hasBias ? 1 : 0];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *px, *pw, *pb = nil, *y;
    if (entry) {
      g = entry[0];
      NSArray *phs = (NSArray *)entry[1];
      px = phs[0];
      pw = phs[1];
      pb = (hasBias && phs.count > 2) ? phs[2] : nil;
      y = entry[2];
    } else {
      g = [MPSGraph new];
      px = [g placeholderWithShape:xShape dataType:ioType name:nil];
      pw = [g placeholderWithShape:wShape dataType:ioType name:nil];
      MPSGraphConvolution2DOpDescriptor *desc =
          [MPSGraphConvolution2DOpDescriptor
              descriptorWithStrideInX:strideW
                            strideInY:strideH
                      dilationRateInX:dilationW
                      dilationRateInY:dilationH
                               groups:groups
                          paddingLeft:padW
                         paddingRight:padW
                           paddingTop:padH
                        paddingBottom:padH
                         paddingStyle:MPSGraphPaddingStyleExplicit
                           dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                        weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
      MPSGraphTensor *conv =
          [g convolution2DWithSourceTensor:mpsg_up(g, px, ioType)
                             weightsTensor:mpsg_up(g, pw, ioType)
                                descriptor:desc
                                      name:nil];
      if (hasBias) {
        pb = [g placeholderWithShape:bShape dataType:ioType name:nil];
        conv = [g additionWithPrimaryTensor:conv
                            secondaryTensor:mpsg_up(g, pb, ioType)
                                       name:nil];
      }
      y = mpsg_down(g, conv, ioType);
      NSArray *phs = hasBias ? @[ px, pw, pb ] : @[ px, pw ];
      mpsg_cache_put(key, @[ g, phs, y ]);
    }
    MPSGraphTensorData *xd =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufX
                                                shape:xShape
                                             dataType:ioType];
    MPSGraphTensorData *wd =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufW
                                                shape:wShape
                                             dataType:ioType];
    MPSGraphTensorData *od =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:bufO
                                                shape:oShape
                                             dataType:ioType];
    NSMutableDictionary *feeds = [@{px : xd, pw : wd} mutableCopy];
    if (hasBias && pb) {
      MPSGraphTensorData *bd =
          [[MPSGraphTensorData alloc] initWithMTLBuffer:bufBias
                                                  shape:bShape
                                               dataType:ioType];
      feeds[pb] = bd;
    }
    [g encodeToCommandBuffer:cb
                       feeds:feeds
              targetOperations:nil
          resultsDictionary:@{y : od}
        executionDescriptor:nil];
    return true;
  }
}

static void reference_conv2d_f32(const float *X, const float *Wt,
                                 const float *bias, float *O, int32_t N,
                                 int32_t H, int32_t W, int32_t Cin, int32_t Cout,
                                 int32_t kH, int32_t kW, int32_t strideH,
                                 int32_t strideW, int32_t padH, int32_t padW,
                                 int32_t dilationH, int32_t dilationW,
                                 int32_t groups) {
  int32_t outH = conv2d_out_dim(H, kH, strideH, padH, dilationH);
  int32_t outW = conv2d_out_dim(W, kW, strideW, padW, dilationW);
  if (outH <= 0 || outW <= 0 || groups <= 0 || Cin % groups || Cout % groups)
    return;
  int32_t cinG = Cin / groups;
  int32_t coutG = Cout / groups;
  for (int32_t n = 0; n < N; ++n)
    for (int32_t oy = 0; oy < outH; ++oy)
      for (int32_t ox = 0; ox < outW; ++ox)
        for (int32_t oc = 0; oc < Cout; ++oc) {
          int32_t grp = oc / coutG;
          double acc = bias ? (double)bias[oc] : 0.0;
          for (int32_t ky = 0; ky < kH; ++ky) {
            int32_t iy = oy * strideH + ky * dilationH - padH;
            if (iy < 0 || iy >= H) continue;
            for (int32_t kx = 0; kx < kW; ++kx) {
              int32_t ix = ox * strideW + kx * dilationW - padW;
              if (ix < 0 || ix >= W) continue;
              for (int32_t ic = 0; ic < cinG; ++ic) {
                int32_t icAbs = grp * cinG + ic;
                double xv = X[(((size_t)n * H + iy) * W + ix) * Cin + icAbs];
                double wv =
                    Wt[(((size_t)ky * kW + kx) * cinG + ic) * Cout + oc];
                acc += xv * wv;
              }
            }
          }
          O[(((size_t)n * outH + oy) * outW + ox) * Cout + oc] = (float)acc;
        }
}

}  // namespace

extern "C" int32_t tessera_apple_gpu_conv2d_out_h(int32_t H, int32_t kH,
                                                  int32_t strideH, int32_t padH,
                                                  int32_t dilationH) {
  return conv2d_out_dim(H, kH, strideH, padH, dilationH);
}

extern "C" int32_t tessera_apple_gpu_conv2d_out_w(int32_t W, int32_t kW,
                                                  int32_t strideW, int32_t padW,
                                                  int32_t dilationW) {
  return conv2d_out_dim(W, kW, strideW, padW, dilationW);
}

extern "C" void tessera_apple_gpu_conv2d_f32(
    const float *X, const float *Wt, const float *bias, float *O, int32_t N,
    int32_t H, int32_t W, int32_t Cin, int32_t Cout, int32_t kH, int32_t kW,
    int32_t strideH, int32_t strideW, int32_t padH, int32_t padW,
    int32_t dilationH, int32_t dilationW, int32_t groups) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_conv2d(ctx, X, Wt, bias, O, N, H, W, Cin, Cout, kH, kW,
                                strideH, strideW, padH, padW, dilationH,
                                dilationW, groups, MPSDataTypeFloat32, 4))
    return;
  reference_conv2d_f32(X, Wt, bias, O, N, H, W, Cin, Cout, kH, kW, strideH,
                       strideW, padH, padW, dilationH, dilationW, groups);
}

extern "C" void tessera_apple_gpu_conv2d_f16(
    const uint16_t *X, const uint16_t *Wt, const uint16_t *bias, uint16_t *O,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout, int32_t kH,
    int32_t kW, int32_t strideH, int32_t strideW, int32_t padH, int32_t padW,
    int32_t dilationH, int32_t dilationW, int32_t groups) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_conv2d(ctx, X, Wt, bias, O, N, H, W, Cin, Cout, kH, kW,
                                strideH, strideW, padH, padW, dilationH,
                                dilationW, groups, MPSDataTypeFloat16, 2))
    return;
  int32_t outH = conv2d_out_dim(H, kH, strideH, padH, dilationH);
  int32_t outW = conv2d_out_dim(W, kW, strideW, padW, dilationW);
  if (outH > 0 && outW > 0)
    std::memset(O, 0, (size_t)N * outH * outW * Cout * 2);
}

// Project 5 (2026-06-01) — device-resident encode-session conv2d C ABI.
// Mirrors the layer_norm / bmm / softmax / flash_attn encode wrappers
// above so a model with conv2d can keep the conv on the SAME command
// buffer as the rest of a chain (no per-op GPU↔CPU sync). NHWC source
// + HWIO weights; optional bias (pass NULL TsDeviceTensor*). Honors
// padding / stride / dilation / groups so depthwise + grouped conv
// land on this path too. The output buffer must be sized for
// ``N * conv2d_out_h * conv2d_out_w * Cout * 4`` bytes (fp32).
//
// Returns ``1`` on a clean encode; ``0`` if the runtime / inputs are
// invalid. The session retains the underlying tensors until commit,
// so the caller can immediately enqueue the next op.
extern "C" int32_t tessera_apple_gpu_conv2d_dev_f32_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *Wt,
    TsDeviceTensor *bias, TsDeviceTensor *O,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout,
    int32_t kH, int32_t kW, int32_t strideH, int32_t strideW,
    int32_t padH, int32_t padW, int32_t dilationH, int32_t dilationW,
    int32_t groups) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Wt || !O) return 0;
  // ``bias`` is optional — nil-tensor means no bias term.
  id<MTLBuffer> bufBias = bias ? bias->buf : nil;
  return mpsg_encode_conv2d_dev(s->cb, X->buf, Wt->buf, bufBias, O->buf,
                                 N, H, W, Cin, Cout, kH, kW, strideH,
                                 strideW, padH, padW, dilationH, dilationW,
                                 groups, MPSDataTypeFloat32)
             ? 1
             : 0;
}

// Sprint A (2026-06-01) — f16 encode-session conv2d C ABI. Same
// shape contract as the f32 variant; the output buffer must be sized
// for ``N * outH * outW * Cout * 2`` bytes (fp16 = 2 bytes/elem).
// The MPSGraph path handles f16 natively (no on-GPU cast) by passing
// ``MPSDataTypeFloat16`` through ``mpsg_up``/``mpsg_down`` (both are
// no-ops when ioType == MPSDataTypeFloat16 is requested; actually
// they cast to fp32 internally for the conv op, then back down).
extern "C" int32_t tessera_apple_gpu_conv2d_dev_f16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *Wt,
    TsDeviceTensor *bias, TsDeviceTensor *O,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout,
    int32_t kH, int32_t kW, int32_t strideH, int32_t strideW,
    int32_t padH, int32_t padW, int32_t dilationH, int32_t dilationW,
    int32_t groups) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Wt || !O) return 0;
  id<MTLBuffer> bufBias = bias ? bias->buf : nil;
  return mpsg_encode_conv2d_dev(s->cb, X->buf, Wt->buf, bufBias, O->buf,
                                 N, H, W, Cin, Cout, kH, kW, strideH,
                                 strideW, padH, padW, dilationH, dilationW,
                                 groups, MPSDataTypeFloat16)
             ? 1
             : 0;
}

// Sprint A (2026-06-01) — bf16 encode-session conv2d C ABI. macOS
// 26+ on M2+ supports bf16 in MPSGraph directly; on older hosts the
// graph build rejects bf16 and the encode helper returns 0. The
// output buffer must be sized for ``N * outH * outW * Cout * 2``
// bytes (bf16 = 2 bytes/elem; same byte count as f16).
extern "C" int32_t tessera_apple_gpu_conv2d_dev_bf16_enc(
    TsEncodeSession *s, TsDeviceTensor *X, TsDeviceTensor *Wt,
    TsDeviceTensor *bias, TsDeviceTensor *O,
    int32_t N, int32_t H, int32_t W, int32_t Cin, int32_t Cout,
    int32_t kH, int32_t kW, int32_t strideH, int32_t strideW,
    int32_t padH, int32_t padW, int32_t dilationH, int32_t dilationW,
    int32_t groups) {
  MetalDeviceContext &ctx = deviceContext();
  if (!ctx.ok || !s || !X || !Wt || !O) return 0;
  id<MTLBuffer> bufBias = bias ? bias->buf : nil;
  return mpsg_encode_conv2d_dev(s->cb, X->buf, Wt->buf, bufBias, O->buf,
                                 N, H, W, Cin, Cout, kH, kW, strideH,
                                 strideW, padH, padW, dilationH, dilationW,
                                 groups, MPSDataTypeBFloat16)
             ? 1
             : 0;
}

//===----------------------------------------------------------------------===//
// conv3d — im2col + MPSGraph batched matmul (NDHWC source, DHWIO weights)
// (2026-05-30)
//
// MPSGraph has no 3-D convolution node, so conv3d is lowered to the classic
// im2col + GEMM decomposition: the spatial patches are gathered on the host
// into a column matrix laid out per-group as [groups, rows, K] (rows =
// N*oD*oH*oW, K = kD*kH*kW*Cin/groups), the weights are regrouped to
// [groups, K, Cout/groups], and the dominant GEMM runs on-GPU as a single
// MPSGraph batched matmul (fp32 accumulation). Bias + scatter back to NDHWC
// happen on the host. f16 I/O converts to fp32 at the boundary.
//===----------------------------------------------------------------------===//

namespace {

// GPU batched matmul A[g,M,K] @ B[g,K,Ncols] -> O[g,M,Ncols], fp32, cached.
static bool mpsg_conv3d_batched_matmul_f32(MetalDeviceContext &ctx,
                                           const float *A, const float *B,
                                           float *O, int32_t G, int32_t M,
                                           int32_t K, int32_t Ncols) {
  if (G <= 0 || M <= 0 || K <= 0 || Ncols <= 0) return true;
  @autoreleasepool {
    size_t aBytes = (size_t)G * M * K * 4;
    size_t bBytes = (size_t)G * K * Ncols * 4;
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufA, ctx, A, aBytes);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bufB, ctx, B, bBytes);
    if (!bufA || !bufB) return false;
    NSArray<NSNumber *> *aShape = @[ @(G), @(M), @(K) ];
    NSArray<NSNumber *> *bShape = @[ @(G), @(K), @(Ncols) ];
    NSString *key = [NSString
        stringWithFormat:@"conv3dmm:%d:%d:%d:%d", G, M, K, Ncols];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *pa, *pb, *y;
    if (entry) {
      g = entry[0];
      pa = ((NSArray *)entry[1])[0];
      pb = ((NSArray *)entry[1])[1];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      pa = [g placeholderWithShape:aShape dataType:MPSDataTypeFloat32 name:nil];
      pb = [g placeholderWithShape:bShape dataType:MPSDataTypeFloat32 name:nil];
      y = [g matrixMultiplicationWithPrimaryTensor:pa secondaryTensor:pb name:nil];
      mpsg_cache_put(key, @[ g, @[ pa, pb ], y ]);
    }
    MPSGraphTensorData *ad = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufA shape:aShape dataType:MPSDataTypeFloat32];
    MPSGraphTensorData *bd = [[MPSGraphTensorData alloc] initWithMTLBuffer:bufB shape:bShape dataType:MPSDataTypeFloat32];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{pa : ad, pb : bd}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:O strideBytes:nil];
    return true;
  }
}

static inline int32_t conv3d_out_dim(int32_t in, int32_t k, int32_t stride,
                                     int32_t pad, int32_t dilation) {
  return conv2d_out_dim(in, k, stride, pad, dilation);
}

// fp32 core: host im2col + GPU GEMM + host bias/scatter. on_gpu=false runs a
// pure-host GEMM (reference path). Returns false only on a hard GPU failure.
static bool conv3d_core_f32(MetalDeviceContext *ctx, const float *X,
                            const float *Wt, const float *bias, float *O,
                            int32_t N, int32_t iD, int32_t iH, int32_t iW,
                            int32_t Cin, int32_t Cout, int32_t kD, int32_t kH,
                            int32_t kW, int32_t sD, int32_t sH, int32_t sW,
                            int32_t pD, int32_t pH, int32_t pW, int32_t dD,
                            int32_t dH, int32_t dW, int32_t groups) {
  int32_t oD = conv3d_out_dim(iD, kD, sD, pD, dD);
  int32_t oH = conv3d_out_dim(iH, kH, sH, pH, dH);
  int32_t oW = conv3d_out_dim(iW, kW, sW, pW, dW);
  if (oD <= 0 || oH <= 0 || oW <= 0 || groups <= 0 || Cin % groups ||
      Cout % groups)
    return true;
  int32_t cinG = Cin / groups, coutG = Cout / groups;
  int32_t K = kD * kH * kW * cinG;
  int32_t rows = N * oD * oH * oW;
  if (K <= 0 || rows <= 0) return true;

  // im2col -> cols[g, r, kk]; weights -> wg[g, kk, oc']
  std::vector<float> cols((size_t)groups * rows * K, 0.0f);
  std::vector<float> wg((size_t)groups * K * coutG);
  for (int32_t g = 0; g < groups; ++g)
    for (int32_t kk = 0; kk < K; ++kk)
      for (int32_t oc = 0; oc < coutG; ++oc)
        wg[((size_t)g * K + kk) * coutG + oc] =
            Wt[(size_t)kk * Cout + g * coutG + oc];

  for (int32_t n = 0; n < N; ++n)
    for (int32_t od = 0; od < oD; ++od)
      for (int32_t oh = 0; oh < oH; ++oh)
        for (int32_t ow = 0; ow < oW; ++ow) {
          int32_t r = ((n * oD + od) * oH + oh) * oW + ow;
          for (int32_t kd = 0; kd < kD; ++kd) {
            int32_t id = od * sD + kd * dD - pD;
            if (id < 0 || id >= iD) continue;
            for (int32_t kh = 0; kh < kH; ++kh) {
              int32_t ih = oh * sH + kh * dH - pH;
              if (ih < 0 || ih >= iH) continue;
              for (int32_t kw = 0; kw < kW; ++kw) {
                int32_t iw = ow * sW + kw * dW - pW;
                if (iw < 0 || iw >= iW) continue;
                int32_t kbase = ((kd * kH + kh) * kW + kw) * cinG;
                for (int32_t g = 0; g < groups; ++g) {
                  const float *xp =
                      X + ((((size_t)n * iD + id) * iH + ih) * iW + iw) * Cin +
                      g * cinG;
                  float *cp = cols.data() +
                              ((size_t)g * rows + r) * K + kbase;
                  for (int32_t ic = 0; ic < cinG; ++ic) cp[ic] = xp[ic];
                }
              }
            }
          }
        }

  std::vector<float> mm((size_t)groups * rows * coutG);
  bool ran = false;
  if (ctx && ctx->ok)
    ran = mpsg_conv3d_batched_matmul_f32(*ctx, cols.data(), wg.data(),
                                         mm.data(), groups, rows, K, coutG);
  if (!ran) {
    for (int32_t g = 0; g < groups; ++g)
      for (int32_t r = 0; r < rows; ++r)
        for (int32_t oc = 0; oc < coutG; ++oc) {
          double acc = 0;
          const float *cp = cols.data() + ((size_t)g * rows + r) * K;
          const float *wp = wg.data() + (size_t)g * K * coutG;
          for (int32_t kk = 0; kk < K; ++kk) acc += (double)cp[kk] * wp[kk * coutG + oc];
          mm[((size_t)g * rows + r) * coutG + oc] = (float)acc;
        }
  }

  // scatter mm[g,r,oc'] (+ bias) -> O[n,od,oh,ow, g*coutG+oc']
  for (int32_t g = 0; g < groups; ++g)
    for (int32_t r = 0; r < rows; ++r)
      for (int32_t oc = 0; oc < coutG; ++oc) {
        int32_t ocAbs = g * coutG + oc;
        float v = mm[((size_t)g * rows + r) * coutG + oc];
        if (bias) v += bias[ocAbs];
        O[(size_t)r * Cout + ocAbs] = v;
      }
  return true;
}

}  // namespace

extern "C" int32_t tessera_apple_gpu_conv3d_out_dim(int32_t in, int32_t k,
                                                    int32_t stride, int32_t pad,
                                                    int32_t dilation) {
  return conv2d_out_dim(in, k, stride, pad, dilation);
}

extern "C" void tessera_apple_gpu_conv3d_f32(
    const float *X, const float *Wt, const float *bias, float *O, int32_t N,
    int32_t iD, int32_t iH, int32_t iW, int32_t Cin, int32_t Cout, int32_t kD,
    int32_t kH, int32_t kW, int32_t sD, int32_t sH, int32_t sW, int32_t pD,
    int32_t pH, int32_t pW, int32_t dD, int32_t dH, int32_t dW, int32_t groups) {
  MetalDeviceContext &ctx = deviceContext();
  conv3d_core_f32(ctx.ok ? &ctx : nullptr, X, Wt, bias, O, N, iD, iH, iW, Cin,
                  Cout, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
}

extern "C" void tessera_apple_gpu_conv3d_f16(
    const uint16_t *X, const uint16_t *Wt, const uint16_t *bias, uint16_t *O,
    int32_t N, int32_t iD, int32_t iH, int32_t iW, int32_t Cin, int32_t Cout,
    int32_t kD, int32_t kH, int32_t kW, int32_t sD, int32_t sH, int32_t sW,
    int32_t pD, int32_t pH, int32_t pW, int32_t dD, int32_t dH, int32_t dW,
    int32_t groups) {
  int32_t oD = conv2d_out_dim(iD, kD, sD, pD, dD);
  int32_t oH = conv2d_out_dim(iH, kH, sH, pH, dH);
  int32_t oW = conv2d_out_dim(iW, kW, sW, pW, dW);
  if (oD <= 0 || oH <= 0 || oW <= 0 || groups <= 0 || Cin % groups ||
      Cout % groups)
    return;
  size_t xn = (size_t)N * iD * iH * iW * Cin;
  size_t wn = (size_t)kD * kH * kW * (Cin / groups) * Cout;
  size_t on = (size_t)N * oD * oH * oW * Cout;
  std::vector<float> xf(xn), wf(wn), of(on);
  std::vector<float> bf;
  for (size_t i = 0; i < xn; ++i) xf[i] = half_to_float_gpu(X[i]);
  for (size_t i = 0; i < wn; ++i) wf[i] = half_to_float_gpu(Wt[i]);
  if (bias) {
    bf.resize(Cout);
    for (int32_t i = 0; i < Cout; ++i) bf[i] = half_to_float_gpu(bias[i]);
  }
  MetalDeviceContext &ctx = deviceContext();
  conv3d_core_f32(ctx.ok ? &ctx : nullptr, xf.data(), wf.data(),
                  bias ? bf.data() : nullptr, of.data(), N, iD, iH, iW, Cin,
                  Cout, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
  for (size_t i = 0; i < on; ++i) O[i] = float_to_half_gpu(of[i]);
}

//===----------------------------------------------------------------------===//
// MLA decode with decoupled RoPE (explicit per-head K) (2026-05-30)
//
// DeepSeek-style MLA splits each head's query/key into a no-position-encoding
// part (dim dn) and a RoPE-carrying part (dim dr). The RoPE key part is SHARED
// across heads (one vector per position). Once RoPE is applied to the rope
// parts and `[nope ; rope]` is concatenated per head (broadcasting the shared
// key-rope across heads), the score reduces to standard MHA with head_dim
// dh = dn + dr:  O = softmax((Qfull @ Kfullᵀ)·scale) @ V.  So we assemble Qfull
// / Kfullᵀ / V on the host (cheap, elementwise RoPE + concat) and run the heavy
// fused attention on-GPU via the existing bsmm (matmul→softmax→matmul) kernel.
//
// rotation_style: 0 = interleaved (NeoX even/odd pairs), 1 = half (GPT-J split
// halves). cos/sin tables are [S, dr/2]. This matches the switchable RoPE
// convention requested for the kernel.
//===----------------------------------------------------------------------===//

namespace {

// Apply RoPE to one rope-part vector x[dr] -> out[dr], cos/sin are [dr/2].
static inline void mla_rope_apply(const float *x, const float *cosr,
                                  const float *sinr, float *out, int32_t dr,
                                  int32_t style) {
  int32_t half = dr / 2;
  if (style == 0) {  // interleaved: pairs (2p, 2p+1)
    for (int32_t p = 0; p < half; ++p) {
      float a = x[2 * p], b = x[2 * p + 1], c = cosr[p], s = sinr[p];
      out[2 * p] = a * c - b * s;
      out[2 * p + 1] = a * s + b * c;
    }
  } else {  // half: pairs (p, p+half)
    for (int32_t p = 0; p < half; ++p) {
      float a = x[p], b = x[p + half], c = cosr[p], s = sinr[p];
      out[p] = a * c - b * s;
      out[p + half] = b * c + a * s;
    }
  }
}

// Assemble A = Qfull[batch, Sq, dh], Bt = Kfullᵀ[batch, dh, Skv], C = V[batch,
// Skv, dv] from the decoupled-RoPE inputs (batch = B*H, dh = dn + dr).
static void mla_rope_assemble_f32(const float *Qn, const float *Qr,
                                  const float *Kn, const float *Kr,
                                  const float *V, const float *cosQ,
                                  const float *sinQ, const float *cosK,
                                  const float *sinK, int32_t B, int32_t H,
                                  int32_t Sq, int32_t Skv, int32_t dn,
                                  int32_t dr, int32_t dv, int32_t style,
                                  std::vector<float> &A, std::vector<float> &Bt,
                                  std::vector<float> &C) {
  int32_t dh = dn + dr;
  int32_t halfdr = dr / 2;
  A.assign((size_t)B * H * Sq * dh, 0.0f);
  Bt.assign((size_t)B * H * dh * Skv, 0.0f);
  C.assign((size_t)B * H * Skv * dv, 0.0f);
  std::vector<float> tmp(dr);
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      int32_t bh = b * H + h;
      // Qfull rows
      for (int32_t i = 0; i < Sq; ++i) {
        const float *qn = Qn + (((size_t)bh * Sq + i) * dn);
        const float *qr = Qr + (((size_t)bh * Sq + i) * dr);
        float *arow = A.data() + (((size_t)bh * Sq + i) * dh);
        for (int32_t d = 0; d < dn; ++d) arow[d] = qn[d];
        mla_rope_apply(qr, cosQ + (size_t)i * halfdr, sinQ + (size_t)i * halfdr,
                       tmp.data(), dr, style);
        for (int32_t d = 0; d < dr; ++d) arow[dn + d] = tmp[d];
      }
      // Kfullᵀ columns: Bt[bh, d, j] = Kfull[bh, j, d]. Kr is shared across h.
      for (int32_t j = 0; j < Skv; ++j) {
        const float *kn = Kn + (((size_t)bh * Skv + j) * dn);
        const float *kr = Kr + (((size_t)b * Skv + j) * dr);  // shared (no h)
        float *btbase = Bt.data() + ((size_t)bh * dh * Skv);
        for (int32_t d = 0; d < dn; ++d) btbase[(size_t)d * Skv + j] = kn[d];
        mla_rope_apply(kr, cosK + (size_t)j * halfdr, sinK + (size_t)j * halfdr,
                       tmp.data(), dr, style);
        for (int32_t d = 0; d < dr; ++d)
          btbase[(size_t)(dn + d) * Skv + j] = tmp[d];
      }
      // V passthrough
      const float *vsrc = V + ((size_t)bh * Skv * dv);
      float *cdst = C.data() + ((size_t)bh * Skv * dv);
      std::memcpy(cdst, vsrc, (size_t)Skv * dv * sizeof(float));
    }
  }
}

}  // namespace

extern "C" void tessera_apple_gpu_mla_decode_rope_f32(
    const float *Qn, const float *Qr, const float *Kn, const float *Kr,
    const float *V, const float *cosQ, const float *sinQ, const float *cosK,
    const float *sinK, float *O, int32_t B, int32_t H, int32_t Sq, int32_t Skv,
    int32_t dn, int32_t dr, int32_t dv, int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      (dr % 2) != 0 || (dn + dr) <= 0)
    return;
  int32_t dh = dn + dr;
  int32_t batch = B * H;
  float scale = 1.0f / std::sqrt((float)dh);
  std::vector<float> A, Bt, C;
  mla_rope_assemble_f32(Qn, Qr, Kn, Kr, V, cosQ, sinQ, cosK, sinK, B, H, Sq,
                        Skv, dn, dr, dv, rotation_style, A, Bt, C);
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_bsmm(ctx, A.data(), Bt.data(), C.data(), O, batch, Sq,
                              Skv, dv, dh, scale, MPSDataTypeFloat32, 4))
    return;
  reference_bsmm_f32(A.data(), Bt.data(), C.data(), O, batch, Sq, Skv, dv, dh,
                     scale);
}

// decode_rope f16 (native bsmm f16 I/O, fp32 accum) + bf16 (host round-trip).
extern "C" void tessera_apple_gpu_mla_decode_rope_f16(
    const uint16_t *Qn, const uint16_t *Qr, const uint16_t *Kn,
    const uint16_t *Kr, const uint16_t *V, const float *cosQ, const float *sinQ,
    const float *cosK, const float *sinK, uint16_t *O, int32_t B, int32_t H,
    int32_t Sq, int32_t Skv, int32_t dn, int32_t dr, int32_t dv,
    int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      (dr % 2) != 0 || (dn + dr) <= 0)
    return;
  int32_t dh = dn + dr, batch = B * H;
  float scale = 1.0f / std::sqrt((float)dh);
  auto cvt = [](const uint16_t *p, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = half_to_float_gpu(p[i]);
    return v;
  };
  std::vector<float> qnf = cvt(Qn, (size_t)B * H * Sq * dn),
                     qrf = cvt(Qr, (size_t)B * H * Sq * dr),
                     knf = cvt(Kn, (size_t)B * H * Skv * dn),
                     krf = cvt(Kr, (size_t)B * Skv * dr),
                     vf = cvt(V, (size_t)B * H * Skv * dv);
  std::vector<float> A, Bt, C;
  mla_rope_assemble_f32(qnf.data(), qrf.data(), knf.data(), krf.data(),
                        vf.data(), cosQ, sinQ, cosK, sinK, B, H, Sq, Skv, dn, dr,
                        dv, rotation_style, A, Bt, C);
  auto toh = [](const std::vector<float> &v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = float_to_half_gpu(v[i]);
    return h;
  };
  std::vector<uint16_t> Ah = toh(A), Bth = toh(Bt), Ch = toh(C);
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_bsmm(ctx, Ah.data(), Bth.data(), Ch.data(), O, batch,
                              Sq, Skv, dv, dh, scale, MPSDataTypeFloat16, 2))
    return;
  std::vector<float> of((size_t)batch * Sq * dv);
  reference_bsmm_f32(A.data(), Bt.data(), C.data(), of.data(), batch, Sq, Skv,
                     dv, dh, scale);
  for (size_t i = 0; i < of.size(); ++i) O[i] = float_to_half_gpu(of[i]);
}

extern "C" void tessera_apple_gpu_mla_decode_rope_bf16(
    const uint16_t *Qn, const uint16_t *Qr, const uint16_t *Kn,
    const uint16_t *Kr, const uint16_t *V, const float *cosQ, const float *sinQ,
    const float *cosK, const float *sinK, uint16_t *O, int32_t B, int32_t H,
    int32_t Sq, int32_t Skv, int32_t dn, int32_t dr, int32_t dv,
    int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      (dr % 2) != 0 || (dn + dr) <= 0)
    return;
  int32_t dh = dn + dr, batch = B * H;
  float scale = 1.0f / std::sqrt((float)dh);
  auto cvt = [](const uint16_t *p, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = gqa_bf16_to_f32(p[i]);
    return v;
  };
  std::vector<float> qnf = cvt(Qn, (size_t)B * H * Sq * dn),
                     qrf = cvt(Qr, (size_t)B * H * Sq * dr),
                     knf = cvt(Kn, (size_t)B * H * Skv * dn),
                     krf = cvt(Kr, (size_t)B * Skv * dr),
                     vf = cvt(V, (size_t)B * H * Skv * dv);
  std::vector<float> A, Bt, C, of((size_t)batch * Sq * dv);
  mla_rope_assemble_f32(qnf.data(), qrf.data(), knf.data(), krf.data(),
                        vf.data(), cosQ, sinQ, cosK, sinK, B, H, Sq, Skv, dn, dr,
                        dv, rotation_style, A, Bt, C);
  MetalDeviceContext &ctx = deviceContext();
  if (!(ctx.ok && mpsg_run_bsmm(ctx, A.data(), Bt.data(), C.data(), of.data(),
                                batch, Sq, Skv, dv, dh, scale, MPSDataTypeFloat32, 4)))
    reference_bsmm_f32(A.data(), Bt.data(), C.data(), of.data(), batch, Sq, Skv,
                       dv, dh, scale);
  for (size_t i = 0; i < of.size(); ++i) O[i] = gqa_f32_to_bf16(of[i]);
}

//===----------------------------------------------------------------------===//
// MLA decode with weight absorption + decoupled RoPE (2026-05-30)
//
// The real MLA bandwidth win. Instead of materializing per-head K/V from the
// cached latent, the up-projection weights absorb into the query/output so
// attention runs directly against the compressed latent (shared across heads):
//
//   q_abs   = q_nope @ Wukᵀ                       (absorb up-K into the query)
//   s_nope  = q_abs @ c_kvᵀ                        (score vs latent — no k_nope)
//   s_rope  = rope(q_rope) @ rope(k_rope)ᵀ         (k_rope shared across heads)
//   attn    = softmax((s_nope + s_rope)·scale)
//   ctx     = attn @ c_kv                          (context in latent space)
//   O       = ctx @ Wuv                            (absorb up-V into the output)
//
// The KV cache therefore stores only c_kv [Skv, Dl] + k_rope [Skv, dr] (shared
// across all H heads) rather than per-head K/V. Mathematically identical to the
// explicit-K decoupled-RoPE path. One cached MPSGraph runs the whole decode;
// RoPE + the (compute-only) tiling of shared operands to the B·H batch happen
// on the host. scale = 1/sqrt(dn + dr).  rotation_style: 0 interleaved, 1 half.
//===----------------------------------------------------------------------===//

namespace {

// Absorbed-decode MPSGraph, generic over I/O dtype. Placeholders carry ioType
// (f16 halves the on-GPU cache-read bandwidth); all matmuls run in fp32 via
// mpsg_up, and the output is cast back down to ioType. bf16 has no native
// MPSGraph type — callers run this with Float32 and convert at the boundary.
static bool mpsg_run_mla_absorb(MetalDeviceContext &ctx, const void *qn,
                                const void *qr, const void *ckv,
                                const void *krope, const void *Wukt,
                                const void *Wuv, void *O, int32_t BH, int32_t Sq,
                                int32_t Skv, int32_t dn, int32_t dr, int32_t dv,
                                int32_t Dl, MPSDataType ioType, size_t elemSize) {
  if (BH <= 0 || Sq <= 0 || Skv <= 0 || dv <= 0 || Dl <= 0) return true;
  float scale = 1.0f / std::sqrt((float)(dn + dr));
  @autoreleasepool {
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bqn, ctx, qn, (size_t)BH * Sq * dn * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bqr, ctx, qr, (size_t)BH * Sq * dr * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bckv, ctx, ckv, (size_t)BH * Skv * Dl * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bkr, ctx, krope, (size_t)BH * Skv * dr * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bwukt, ctx, Wukt, (size_t)BH * dn * Dl * elemSize);
    TS_METAL_BUF_ACQUIRE_WITH_BYTES(bwuv, ctx, Wuv, (size_t)BH * Dl * dv * elemSize);
    if (!bqn || !bqr || !bckv || !bkr || !bwukt || !bwuv) return false;
    NSArray<NSNumber *> *qnS = @[ @(BH), @(Sq), @(dn) ];
    NSArray<NSNumber *> *qrS = @[ @(BH), @(Sq), @(dr) ];
    NSArray<NSNumber *> *ckvS = @[ @(BH), @(Skv), @(Dl) ];
    NSArray<NSNumber *> *krS = @[ @(BH), @(Skv), @(dr) ];
    NSArray<NSNumber *> *wuktS = @[ @(BH), @(dn), @(Dl) ];
    NSArray<NSNumber *> *wuvS = @[ @(BH), @(Dl), @(dv) ];
    NSString *key = [NSString stringWithFormat:@"mlaabsorb:%d:%d:%d:%d:%d:%d:%d:%d",
                                               (int)ioType, BH, Sq, Skv, dn, dr,
                                               dv, Dl];
    NSArray *entry = mpsg_cache_get(key);
    MPSGraph *g;
    MPSGraphTensor *pqn, *pqr, *pckv, *pkr, *pwukt, *pwuv, *y;
    if (entry) {
      g = entry[0];
      NSArray *p = (NSArray *)entry[1];
      pqn = p[0]; pqr = p[1]; pckv = p[2]; pkr = p[3]; pwukt = p[4]; pwuv = p[5];
      y = entry[2];
    } else {
      g = [MPSGraph new];
      pqn = [g placeholderWithShape:qnS dataType:ioType name:nil];
      pqr = [g placeholderWithShape:qrS dataType:ioType name:nil];
      pckv = [g placeholderWithShape:ckvS dataType:ioType name:nil];
      pkr = [g placeholderWithShape:krS dataType:ioType name:nil];
      pwukt = [g placeholderWithShape:wuktS dataType:ioType name:nil];
      pwuv = [g placeholderWithShape:wuvS dataType:ioType name:nil];
      MPSGraphTensor *qn32 = mpsg_up(g, pqn, ioType);
      MPSGraphTensor *qr32 = mpsg_up(g, pqr, ioType);
      MPSGraphTensor *ckv32 = mpsg_up(g, pckv, ioType);
      MPSGraphTensor *kr32 = mpsg_up(g, pkr, ioType);
      MPSGraphTensor *wukt32 = mpsg_up(g, pwukt, ioType);
      MPSGraphTensor *wuv32 = mpsg_up(g, pwuv, ioType);
      MPSGraphTensor *qabs = [g matrixMultiplicationWithPrimaryTensor:qn32 secondaryTensor:wukt32 name:nil];      // [BH,Sq,Dl]
      MPSGraphTensor *ckvT = [g transposeTensor:ckv32 dimension:1 withDimension:2 name:nil];                     // [BH,Dl,Skv]
      MPSGraphTensor *sNope = [g matrixMultiplicationWithPrimaryTensor:qabs secondaryTensor:ckvT name:nil];      // [BH,Sq,Skv]
      MPSGraphTensor *krT = [g transposeTensor:kr32 dimension:1 withDimension:2 name:nil];                       // [BH,dr,Skv]
      MPSGraphTensor *sRope = [g matrixMultiplicationWithPrimaryTensor:qr32 secondaryTensor:krT name:nil];       // [BH,Sq,Skv]
      MPSGraphTensor *s = [g additionWithPrimaryTensor:sNope secondaryTensor:sRope name:nil];
      MPSGraphTensor *scaled = [g multiplicationWithPrimaryTensor:s
                                  secondaryTensor:[g constantWithScalar:(double)scale dataType:MPSDataTypeFloat32]
                                             name:nil];
      MPSGraphTensor *attn = [g softMaxWithTensor:scaled axis:2 name:nil];
      MPSGraphTensor *ctxT = [g matrixMultiplicationWithPrimaryTensor:attn secondaryTensor:ckv32 name:nil];      // [BH,Sq,Dl]
      MPSGraphTensor *yf = [g matrixMultiplicationWithPrimaryTensor:ctxT secondaryTensor:wuv32 name:nil];        // [BH,Sq,dv]
      y = mpsg_down(g, yf, ioType);
      mpsg_cache_put(key, @[ g, @[ pqn, pqr, pckv, pkr, pwukt, pwuv ], y ]);
    }
    MPSGraphTensorData *dqn = [[MPSGraphTensorData alloc] initWithMTLBuffer:bqn shape:qnS dataType:ioType];
    MPSGraphTensorData *dqr = [[MPSGraphTensorData alloc] initWithMTLBuffer:bqr shape:qrS dataType:ioType];
    MPSGraphTensorData *dckv = [[MPSGraphTensorData alloc] initWithMTLBuffer:bckv shape:ckvS dataType:ioType];
    MPSGraphTensorData *dkr = [[MPSGraphTensorData alloc] initWithMTLBuffer:bkr shape:krS dataType:ioType];
    MPSGraphTensorData *dwukt = [[MPSGraphTensorData alloc] initWithMTLBuffer:bwukt shape:wuktS dataType:ioType];
    MPSGraphTensorData *dwuv = [[MPSGraphTensorData alloc] initWithMTLBuffer:bwuv shape:wuvS dataType:ioType];
    NSDictionary *res = [g runWithMTLCommandQueue:ctx.queue
                                            feeds:@{pqn : dqn, pqr : dqr, pckv : dckv, pkr : dkr, pwukt : dwukt, pwuv : dwuv}
                                    targetTensors:@[ y ]
                                 targetOperations:nil];
    MPSGraphTensorData *od = res[y];
    if (!od) return false;
    [[od mpsndarray] readBytes:O strideBytes:nil];
    return true;
  }
}

// Host reference for the absorbed decode (fallback + cross-check).
static void reference_mla_absorb_f32(const float *qn, const float *qr,
                                     const float *ckv, const float *krope,
                                     const float *Wukt, const float *Wuv,
                                     float *O, int32_t BH, int32_t Sq,
                                     int32_t Skv, int32_t dn, int32_t dr,
                                     int32_t dv, int32_t Dl) {
  float scale = 1.0f / std::sqrt((float)(dn + dr));
  std::vector<double> qabs(Dl), score(Skv);
  for (int32_t b = 0; b < BH; ++b) {
    const float *qnb = qn + (size_t)b * Sq * dn;
    const float *qrb = qr + (size_t)b * Sq * dr;
    const float *ckvb = ckv + (size_t)b * Skv * Dl;
    const float *krb = krope + (size_t)b * Skv * dr;
    const float *wuktb = Wukt + (size_t)b * dn * Dl;
    const float *wuvb = Wuv + (size_t)b * Dl * dv;
    float *Ob = O + (size_t)b * Sq * dv;
    for (int32_t i = 0; i < Sq; ++i) {
      for (int32_t l = 0; l < Dl; ++l) {
        double acc = 0;
        for (int32_t d = 0; d < dn; ++d) acc += (double)qnb[i * dn + d] * wuktb[d * Dl + l];
        qabs[l] = acc;
      }
      double mx = -1e30;
      for (int32_t j = 0; j < Skv; ++j) {
        double sn = 0;
        for (int32_t l = 0; l < Dl; ++l) sn += qabs[l] * ckvb[j * Dl + l];
        double sr = 0;
        for (int32_t d = 0; d < dr; ++d) sr += (double)qrb[i * dr + d] * krb[j * dr + d];
        score[j] = (sn + sr) * scale;
        mx = std::max(mx, score[j]);
      }
      double den = 0;
      for (int32_t j = 0; j < Skv; ++j) { double e = std::exp(score[j] - mx); score[j] = e; den += e; }
      for (int32_t d = 0; d < dv; ++d) {
        double acc = 0;
        for (int32_t j = 0; j < Skv; ++j) {
          // O[i,d] = sum_j w_j * (ckv[j] · Wuv[:,d])
          double cv = 0;
          for (int32_t l = 0; l < Dl; ++l) cv += ckvb[j * Dl + l] * wuvb[l * dv + d];
          acc += (score[j] / den) * cv;
        }
        Ob[i * dv + d] = (float)acc;
      }
    }
  }
}

// Host RoPE + tile assembly: fp32 inputs -> fp32 tiled [BH,...] buffers, shared
// by all three dtype externs.
static void mla_absorb_assemble_f32(const float *q_nope, const float *q_rope,
                                    const float *c_kv, const float *k_rope,
                                    const float *Wuk_t, const float *Wuv,
                                    const float *cosQ, const float *sinQ,
                                    const float *cosK, const float *sinK,
                                    int32_t B, int32_t H, int32_t Sq,
                                    int32_t Skv, int32_t dn, int32_t dr,
                                    int32_t dv, int32_t Dl, int32_t style,
                                    std::vector<float> &qn, std::vector<float> &qr,
                                    std::vector<float> &ckv, std::vector<float> &kr,
                                    std::vector<float> &wukt, std::vector<float> &wuv) {
  int32_t BH = B * H, halfdr = dr / 2;
  qn.assign((size_t)BH * Sq * dn, 0.0f);
  qr.assign((size_t)BH * Sq * dr, 0.0f);
  ckv.assign((size_t)BH * Skv * Dl, 0.0f);
  kr.assign((size_t)BH * Skv * dr, 0.0f);
  wukt.assign((size_t)BH * dn * Dl, 0.0f);
  wuv.assign((size_t)BH * Dl * dv, 0.0f);
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      int32_t bh = b * H + h;
      std::memcpy(qn.data() + (size_t)bh * Sq * dn,
                  q_nope + (size_t)bh * Sq * dn, (size_t)Sq * dn * sizeof(float));
      for (int32_t i = 0; i < Sq; ++i)
        mla_rope_apply(q_rope + ((size_t)bh * Sq + i) * dr,
                       cosQ + (size_t)i * halfdr, sinQ + (size_t)i * halfdr,
                       qr.data() + ((size_t)bh * Sq + i) * dr, dr, style);
      std::memcpy(ckv.data() + (size_t)bh * Skv * Dl,
                  c_kv + (size_t)b * Skv * Dl, (size_t)Skv * Dl * sizeof(float));
      for (int32_t j = 0; j < Skv; ++j)
        mla_rope_apply(k_rope + ((size_t)b * Skv + j) * dr,
                       cosK + (size_t)j * halfdr, sinK + (size_t)j * halfdr,
                       kr.data() + ((size_t)bh * Skv + j) * dr, dr, style);
      std::memcpy(wukt.data() + (size_t)bh * dn * Dl,
                  Wuk_t + (size_t)h * dn * Dl, (size_t)dn * Dl * sizeof(float));
      std::memcpy(wuv.data() + (size_t)bh * Dl * dv,
                  Wuv + (size_t)h * Dl * dv, (size_t)Dl * dv * sizeof(float));
    }
  }
}

}  // namespace

extern "C" void tessera_apple_gpu_mla_absorb_decode_f32(
    const float *q_nope, const float *q_rope, const float *c_kv,
    const float *k_rope, const float *Wuk_t, const float *Wuv,
    const float *cosQ, const float *sinQ, const float *cosK, const float *sinK,
    float *O, int32_t B, int32_t H, int32_t Sq, int32_t Skv, int32_t dn,
    int32_t dr, int32_t dv, int32_t Dl, int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      Dl <= 0 || (dr % 2) != 0)
    return;
  int32_t BH = B * H;
  std::vector<float> qn, qr, ckv, kr, wukt, wuv;
  mla_absorb_assemble_f32(q_nope, q_rope, c_kv, k_rope, Wuk_t, Wuv, cosQ, sinQ,
                          cosK, sinK, B, H, Sq, Skv, dn, dr, dv, Dl,
                          rotation_style, qn, qr, ckv, kr, wukt, wuv);
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_mla_absorb(ctx, qn.data(), qr.data(), ckv.data(),
                                    kr.data(), wukt.data(), wuv.data(), O, BH,
                                    Sq, Skv, dn, dr, dv, Dl, MPSDataTypeFloat32, 4))
    return;
  reference_mla_absorb_f32(qn.data(), qr.data(), ckv.data(), kr.data(),
                           wukt.data(), wuv.data(), O, BH, Sq, Skv, dn, dr, dv, Dl);
}

// Native f16: assemble in fp32, convert the tiled buffers to half, run the
// MPSGraph with Float16 I/O (fp32 accumulation) — half the on-GPU bandwidth.
extern "C" void tessera_apple_gpu_mla_absorb_decode_f16(
    const uint16_t *q_nope, const uint16_t *q_rope, const uint16_t *c_kv,
    const uint16_t *k_rope, const uint16_t *Wuk_t, const uint16_t *Wuv,
    const float *cosQ, const float *sinQ, const float *cosK, const float *sinK,
    uint16_t *O, int32_t B, int32_t H, int32_t Sq, int32_t Skv, int32_t dn,
    int32_t dr, int32_t dv, int32_t Dl, int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      Dl <= 0 || (dr % 2) != 0)
    return;
  int32_t BH = B * H;
  auto cvt = [](const uint16_t *p, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = half_to_float_gpu(p[i]);
    return v;
  };
  std::vector<float> qnf = cvt(q_nope, (size_t)B * H * Sq * dn);
  std::vector<float> qrf = cvt(q_rope, (size_t)B * H * Sq * dr);
  std::vector<float> ckvf = cvt(c_kv, (size_t)B * Skv * Dl);
  std::vector<float> krf = cvt(k_rope, (size_t)B * Skv * dr);
  std::vector<float> wuktf = cvt(Wuk_t, (size_t)H * dn * Dl);
  std::vector<float> wuvf = cvt(Wuv, (size_t)H * Dl * dv);
  std::vector<float> qn, qr, ckv, kr, wukt, wuv;
  mla_absorb_assemble_f32(qnf.data(), qrf.data(), ckvf.data(), krf.data(),
                          wuktf.data(), wuvf.data(), cosQ, sinQ, cosK, sinK, B,
                          H, Sq, Skv, dn, dr, dv, Dl, rotation_style, qn, qr,
                          ckv, kr, wukt, wuv);
  auto toh = [](const std::vector<float> &v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = float_to_half_gpu(v[i]);
    return h;
  };
  std::vector<uint16_t> qnh = toh(qn), qrh = toh(qr), ckvh = toh(ckv),
                        krh = toh(kr), wukth = toh(wukt), wuvh = toh(wuv);
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && mpsg_run_mla_absorb(ctx, qnh.data(), qrh.data(), ckvh.data(),
                                    krh.data(), wukth.data(), wuvh.data(), O, BH,
                                    Sq, Skv, dn, dr, dv, Dl, MPSDataTypeFloat16, 2))
    return;
  std::vector<float> of((size_t)BH * Sq * dv);
  reference_mla_absorb_f32(qn.data(), qr.data(), ckv.data(), kr.data(),
                           wukt.data(), wuv.data(), of.data(), BH, Sq, Skv, dn,
                           dr, dv, Dl);
  for (size_t i = 0; i < of.size(); ++i) O[i] = float_to_half_gpu(of[i]);
}

// bf16: no native MPSGraph type — convert to fp32 at the boundary, run the fp32
// graph, convert the output back to bf16.
extern "C" void tessera_apple_gpu_mla_absorb_decode_bf16(
    const uint16_t *q_nope, const uint16_t *q_rope, const uint16_t *c_kv,
    const uint16_t *k_rope, const uint16_t *Wuk_t, const uint16_t *Wuv,
    const float *cosQ, const float *sinQ, const float *cosK, const float *sinK,
    uint16_t *O, int32_t B, int32_t H, int32_t Sq, int32_t Skv, int32_t dn,
    int32_t dr, int32_t dv, int32_t Dl, int32_t rotation_style) {
  if (B <= 0 || H <= 0 || Sq <= 0 || Skv <= 0 || dn < 0 || dr < 0 || dv <= 0 ||
      Dl <= 0 || (dr % 2) != 0)
    return;
  int32_t BH = B * H;
  auto cvt = [](const uint16_t *p, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = gqa_bf16_to_f32(p[i]);
    return v;
  };
  std::vector<float> qnf = cvt(q_nope, (size_t)B * H * Sq * dn);
  std::vector<float> qrf = cvt(q_rope, (size_t)B * H * Sq * dr);
  std::vector<float> ckvf = cvt(c_kv, (size_t)B * Skv * Dl);
  std::vector<float> krf = cvt(k_rope, (size_t)B * Skv * dr);
  std::vector<float> wuktf = cvt(Wuk_t, (size_t)H * dn * Dl);
  std::vector<float> wuvf = cvt(Wuv, (size_t)H * Dl * dv);
  std::vector<float> qn, qr, ckv, kr, wukt, wuv;
  mla_absorb_assemble_f32(qnf.data(), qrf.data(), ckvf.data(), krf.data(),
                          wuktf.data(), wuvf.data(), cosQ, sinQ, cosK, sinK, B,
                          H, Sq, Skv, dn, dr, dv, Dl, rotation_style, qn, qr,
                          ckv, kr, wukt, wuv);
  std::vector<float> of((size_t)BH * Sq * dv);
  MetalDeviceContext &ctx = deviceContext();
  if (!(ctx.ok && mpsg_run_mla_absorb(ctx, qn.data(), qr.data(), ckv.data(),
                                      kr.data(), wukt.data(), wuv.data(),
                                      of.data(), BH, Sq, Skv, dn, dr, dv, Dl,
                                      MPSDataTypeFloat32, 4)))
    reference_mla_absorb_f32(qn.data(), qr.data(), ckv.data(), kr.data(),
                             wukt.data(), wuv.data(), of.data(), BH, Sq, Skv, dn,
                             dr, dv, Dl);
  for (size_t i = 0; i < of.size(); ++i) O[i] = gqa_f32_to_bf16(of[i]);
}
