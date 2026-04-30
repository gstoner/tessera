
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#if __CUDACC_VER_MAJOR__ >= 12
#include <cuda/pipeline>
#endif

#ifndef TESSERA_USE_TMA
#define TESSERA_USE_TMA 0
#endif

namespace tessera {

// Minimal mbarrier wrapper (guarded). On pre-Hopper, we fallback to __syncthreads().
struct mbarr_t {
#if __CUDA_ARCH__ >= 900
  unsigned long long* ptr;
  __device__ inline void init(unsigned long long* p, unsigned count) { ptr = p; asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(ptr), "r"(count)); }
  __device__ inline void arrive(unsigned count) { asm volatile("mbarrier.arrive.shared.b64 [%0], %1;" :: "r"(ptr), "r"(count)); }
  __device__ inline void wait() { asm volatile("mbarrier.try_wait.parity.shared.b64 %0, [%1];" : "=r"(count_dummy) : "r"(ptr)); }
  unsigned count_dummy;
#else
  __device__ inline void init(unsigned long long*, unsigned) {}
  __device__ inline void arrive(unsigned) {}
  __device__ inline void wait() { __syncthreads(); }
#endif
};

// Copy global->shared (contiguous) asynchronously using cuda::pipeline (CUDA 12+),
// then swizzle into a destination SMEM tile. This gives "bulk tensor -> swizzled SMEM".
template <typename T>
__device__ inline void tma_copy_then_swizzle(T* smem_contig, T* smem_swizzled,
                                             const T* gmem,
                                             int rows, int cols,
                                             int ld_src, int ld_dst_swizzled) {
#if TESSERA_USE_TMA && (__CUDACC_VER_MAJOR__ >= 12)
  namespace cp = cuda::experimental;
  auto pipe = cp::make_pipeline();
  for (int r = threadIdx.y; r < rows; r += blockDim.y) {
    const T* src = gmem + r * ld_src;
    T* dst = smem_contig + r * cols;
    cp::memcpy_async(pipe, dst, src, cols * sizeof(T));
  }
  cp::commit(pipe);
  cp::wait_prior<0>(pipe);
  __syncthreads(); // all rows present

  // Swizzle into destination
  for (int r = threadIdx.y; r < rows; r += blockDim.y) {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
      // identity swizzle; replace with your chosen mapping
      int cc = c;
      smem_swizzled[r * ld_dst_swizzled + cc] = smem_contig[r * cols + c];
    }
  }
  __syncthreads();
#else
  // Fallback: direct per-thread copies (acts like cp.async path)
  for (int r=threadIdx.y; r<rows; r+=blockDim.y) {
    for (int c=threadIdx.x; c<cols; c+=blockDim.x) {
      smem_swizzled[r * ld_dst_swizzled + c] = gmem[r*ld_src + c];
    }
  }
  __syncthreads();
#endif
}

} // namespace tessera
