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
