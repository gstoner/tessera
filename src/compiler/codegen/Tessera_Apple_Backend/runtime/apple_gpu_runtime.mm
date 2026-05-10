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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A
                                                  length:byteCountA
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B
                                                  length:byteCountB
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [ctx.device newBufferWithLength:byteCountC
                                                 options:MTLResourceStorageModeShared];
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
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytes:X
                                                  length:byteCount
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:byteCount
                                                 options:MTLResourceStorageModeShared];
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
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytes:X
                                                  length:byteCount
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:byteCount
                                                 options:MTLResourceStorageModeShared];
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
    out[gid] = 0.5f * v * (1.0f + tanh(t));
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kGeluSource, @"gelu_f32");
    if (!pso) return false;

    NSUInteger byteCount = sizeof(float) * static_cast<NSUInteger>(N);
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytes:X
                                                  length:byteCount
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:byteCount
                                                 options:MTLResourceStorageModeShared];
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
    out[gid] = half(0.5f * v * (1.0f + tanh(t)));
}
)MSL";

  @autoreleasepool {
    id<MTLComputePipelineState> pso =
        compile_msl_kernel(ctx, kGeluSourceF16, @"gelu_f16");
    if (!pso) return false;

    NSUInteger byteCount = sizeof(uint16_t) * static_cast<NSUInteger>(N);
    id<MTLBuffer> bufX = [ctx.device newBufferWithBytes:X
                                                  length:byteCount
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:byteCount
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A
                                                  length:aBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B
                                                  length:bBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A length:aBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B length:bBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A
                                                  length:aBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B
                                                  length:bBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A length:aBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B length:bBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [ctx.device newBufferWithBytes:C length:cBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A length:aBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B length:bBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [ctx.device newBufferWithBytes:C length:cBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A length:aBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B length:bBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufA = [ctx.device newBufferWithBytes:A length:aBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [ctx.device newBufferWithBytes:B length:bBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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

    id<MTLBuffer> bufX  = [ctx.device newBufferWithBytes:X  length:xBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufWg = [ctx.device newBufferWithBytes:Wg length:wgBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufWu = [ctx.device newBufferWithBytes:Wu length:wuBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufWd = [ctx.device newBufferWithBytes:Wd length:wdBytes
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO  = [ctx.device newBufferWithLength:oBytes
                                                 options:MTLResourceStorageModeShared];
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
