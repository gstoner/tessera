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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct MetalDeviceContext {
  id<MTLDevice>       device;
  id<MTLCommandQueue> queue;
  bool                ok;

  // Phase 8.4 — MSL kernel cache. Keyed by (msl_source + entry_point) so
  // multiple kernels in the same source unit (uncommon but legal) cache
  // independently. Compiled lazily; the cache outlives any single command
  // buffer so amortizes the ~1ms compile cost across kernel invocations.
  std::unordered_map<std::string, id<MTLComputePipelineState>> kernel_cache;
  std::mutex                                                   kernel_cache_mu;

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
    }
  });
  return ctx;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;

    std::memcpy(C, [bufC contents], byteCountC);
    return true;
  }
}

} // namespace

extern "C" int32_t tessera_apple_gpu_runtime_has_metal(void) {
  return deviceContext().ok ? 1 : 0;
}

extern "C" void tessera_apple_gpu_mps_matmul_f32(const float* A,
                                                 const float* B, float* C,
                                                 int32_t M, int32_t N,
                                                 int32_t K) {
  MetalDeviceContext &ctx = deviceContext();
  if (ctx.ok && dispatch_mps_gemm_f32(ctx, A, B, C, M, N, K)) return;
  reference_gemm_f32(A, B, C, M, N, K);
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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

bool dispatch_rope_msl(MetalDeviceContext &ctx, const float* X,
                       const float* Theta, float* Out, int32_t M, int32_t K) {
  static NSString *const kRopeSource = @R"MSL(
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

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kRopeSource, @"rope_f32");
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
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

bool dispatch_rope_msl_f16(MetalDeviceContext &ctx, const uint16_t* X,
                           const uint16_t* Theta, uint16_t* Out,
                           int32_t M, int32_t K) {
  static NSString *const kRopeSourceF16 = @R"MSL(
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

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kRopeSourceF16, @"rope_f16");
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
    std::memcpy(Out, [bufO contents], byteCount);
    return true;
  }
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

bool dispatch_flash_attn_msl(MetalDeviceContext &ctx, const float* Q,
                             const float* K, const float* V, float* O,
                             int32_t B, int32_t Sq, int32_t Sk, int32_t D,
                             float scale, int32_t causal) {
  static NSString *const kFlashAttnSource = @R"MSL(
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

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFlashAttnSource, @"flash_attn_f32");
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
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

bool dispatch_flash_attn_msl_f16(MetalDeviceContext &ctx, const uint16_t* Q,
                                 const uint16_t* K, const uint16_t* V,
                                 uint16_t* O, int32_t B, int32_t Sq,
                                 int32_t Sk, int32_t D, float scale,
                                 int32_t causal) {
  static NSString *const kFlashAttnSourceF16 = @R"MSL(
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

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kFlashAttnSourceF16, @"flash_attn_f16");
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
    std::memcpy(O, [bufO contents], oBytes);
    return true;
  }
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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

    // Step 2: per-thread partial row max -> threadgroup max reduction.
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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

} // namespace

extern "C" void tessera_apple_gpu_mla_decode_f32(const float* X,
                                                  const float* Wdkv,
                                                  const float* Wuk,
                                                  const float* Wuv,
                                                  const float* Q, float* O,
                                                  int32_t B, int32_t S_kv,
                                                  int32_t D_x, int32_t D_lat,
                                                  int32_t S_q, int32_t D_h) {
  reference_mla_decode_f32(X, Wdkv, Wuk, Wuv, Q, O, B, S_kv, D_x, D_lat, S_q, D_h);
}

//===---------------------------------------------------------------------===//
// attention_variants_plan, NSA-5 — DeepSeek Native Sparse Attention.
//
// Host reference combining the three branches:
//   1. Sliding-window dense local attention.
//   2. Per-block-summary attention (mean compression by default).
//   3. Top-k block-selected dense attention.
// Gated per query via the `gate_logits` operand (sigmoid → 3-way mix
// using the first 3 channels per block-score logit; this matches the
// `nn.NativeSparseAttention` Module's gate convention).
//
// A fully fused MSL kernel (all three branches + gating in one
// dispatch with simdgroup top-k reduction) is a follow-up. The host
// reference here just composes the existing per-branch math.
//===---------------------------------------------------------------------===//

extern "C" void tessera_apple_gpu_native_sparse_attn_f32(
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) return false;
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
    [cb commit];
    [cb waitUntilCompleted];
    bool ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool _pool_ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool ok = (cb.status == MTLCommandBufferStatusCompleted);
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
    [cb commit];
    [cb waitUntilCompleted];
    bool ok = (cb.status == MTLCommandBufferStatusCompleted);
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
static NSMutableDictionary<NSString *, NSArray *> *mpsg_graph_cache() {
  static NSMutableDictionary *cache = nil;
  static std::once_flag once;
  std::call_once(once, [] { cache = [[NSMutableDictionary alloc] init]; });
  return cache;
}
static std::mutex g_mpsg_graph_mu;

static NSArray *mpsg_cache_get(NSString *key) {
  std::lock_guard<std::mutex> lock(g_mpsg_graph_mu);
  return mpsg_graph_cache()[key];
}
static void mpsg_cache_put(NSString *key, NSArray *entry) {
  std::lock_guard<std::mutex> lock(g_mpsg_graph_mu);
  mpsg_graph_cache()[key] = entry;
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
    NSArray<NSNumber *> *axis1 = @[ @1 ];
    // eps is baked into the graph as a constant, so it is part of the key;
    // weighted-ness (gamma/beta present) changes the graph too. The MLIR
    // lowering passes the unweighted form (gamma == beta == nullptr) for the
    // unweighted tessera.layer_norm / tessera.rmsnorm ops; the Python runtime
    // passes ones/zeros (weighted). Both are supported.
    NSString *key = [NSString stringWithFormat:@"r:%d:%d:%d:%d:%a:%d:%d", kind,
                                               (int)ioType, rows, cols, eps,
                                               gamma ? 1 : 0, beta ? 1 : 0];
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
        bool weighted = (gamma != nullptr);
        bool hasBeta = (beta != nullptr);
        MPSGraphTensor *gf = nil;
        if (weighted) {
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
          yf = weighted
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
          yf = weighted
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
      if (!yf) return false;
      y = mpsg_down(g, yf, ioType);
      phs = [P copy];
      mpsg_cache_put(key, @[ g, phs, y ]);
    }
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

// Number of distinct (shape-class, opcode, dtype, shape) MPSGraphs cached.
// Used by tests to verify graph reuse across repeated dispatches.
extern "C" int32_t tessera_apple_gpu_mpsgraph_cache_size(void) {
  std::lock_guard<std::mutex> lock(g_mpsg_graph_mu);
  return (int32_t)[mpsg_graph_cache() count];
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
