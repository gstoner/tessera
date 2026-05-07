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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

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
};

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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A
                                                  length:byteCountA
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B
                                                  length:byteCountB
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [ctx.device newBufferWithLength:byteCountC
                                                 options:MTLResourceStorageModeShared];
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
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytes:X
                                                  length:byteCount
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufT = [ctx.device newBufferWithBytes:Theta
                                                  length:byteCount
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:byteCount
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufQ = [ctx.device newBufferWithBytes:Q
                                                  length:qBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufK = [ctx.device newBufferWithBytes:K
                                                  length:kvBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufV = [ctx.device newBufferWithBytes:V
                                                  length:kvBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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
