
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

// Feature toggles (set via CMake)
#ifndef TESSERA_USE_WGMMA
#define TESSERA_USE_WGMMA 0
#endif
#ifndef TESSERA_USE_TMA
#define TESSERA_USE_TMA 0
#endif

#if TESSERA_USE_TMA
#include <cuda/pipeline>
#endif

namespace tessera {

template <typename T, int M, int N>
struct tile {
  T data[M*N];
};

template <typename T, int M, int N>
__device__ inline tile<T,M,N> tile_zero() {
  tile<T,M,N> t;
  #pragma unroll
  for (int i=0;i<M*N;++i) t.data[i] = T(0);
  return t;
}

struct pipeline_t {
#if TESSERA_USE_TMA
  using pipe_t = cuda::pipeline<cuda::thread_scope_thread>;
  pipe_t p;
  __device__ pipeline_t() : p(cuda::make_pipeline()) {}
  __device__ inline void commit() const { cuda::pipeline_producer_commit(p); }
  __device__ inline void wait(int) const { cuda::pipeline_consumer_wait_prior<0>(p); __syncthreads(); }
#else
  __device__ inline void commit() const {
  #if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
  #endif
  }
  __device__ inline void wait(int /*slot*/) const {
  #if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
  #endif
    __syncthreads();
  }
#endif
};

// ------------------------------
// Copy paths
// ------------------------------

__device__ inline void cp_async_16B(void* smem, const void* gmem) {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem), "l"(gmem));
#else
  (void)smem; (void)gmem;
#endif
}

template <typename T>
__device__ inline void async_copy_2d_cp_async(T* smem, const T* gmem, int rows, int cols, int ld_src, int ld_dst) {
#if __CUDA_ARCH__ >= 800
  int bytes_per_row = cols * int(sizeof(T));
  int chunks = (bytes_per_row + 15) / 16;
  for (int r = threadIdx.y; r < rows; r += blockDim.y) {
    for (int c16 = threadIdx.x; c16 < chunks; c16 += blockDim.x) {
      char* srow = reinterpret_cast<char*>(smem + r * ld_dst);
      const char* grow = reinterpret_cast<const char*>(gmem + r * ld_src);
      void* sdst = (void*)(srow + c16 * 16);
      const void* gsrc = (const void*)(grow + c16 * 16);
      cp_async_16B(sdst, gsrc);
    }
  }
#else
  for (int r=threadIdx.y; r<rows; r+=blockDim.y) {
    for (int c=threadIdx.x; c<cols; c+=blockDim.x) {
      smem[r*ld_dst + c] = gmem[r*ld_src + c];
    }
  }
#endif
}

#if TESSERA_USE_TMA
template <typename T>
__device__ inline void async_copy_2d_tma(T* smem, const T* gmem, int rows, int cols, int ld_src, int ld_dst, pipeline_t& pipe) {
  // Use cuda::memcpy_async through a local pipeline; Hopper maps to TMA where possible.
  for (int r = threadIdx.y; r < rows; r += blockDim.y) {
    const T* src = gmem + r * ld_src;
    T* dst = smem + r * ld_dst;
    cuda::memcpy_async(pipe.p, dst, src, cols * sizeof(T));
  }
}
#endif

template <typename T>
__device__ inline void async_copy_2d_select(T* smem, const T* gmem, int rows, int cols, int ld_src, int ld_dst, pipeline_t& pipe, int use_tma) {
#if TESSERA_USE_TMA
  if (use_tma) { async_copy_2d_tma(smem, gmem, rows, cols, ld_src, ld_dst, pipe); return; }
#endif
  (void)use_tma;
  async_copy_2d_cp_async(smem, gmem, rows, cols, ld_src, ld_dst);
}

// ------------------------------
// Fallback compute path
// ------------------------------
template <typename TA, typename TB, typename TC, int M, int K, int N>
__device__ inline void mma_sync(tile<TC,M,N>& c, tile<TA,M,K> const& a, tile<TB,K,N> const& b) {
  #pragma unroll
  for (int m=0;m<M;++m){
    #pragma unroll
    for (int n=0;n<N;++n){
      TC acc = c.data[m*N+n];
      #pragma unroll
      for (int k=0;k<K;++k){
        acc += static_cast<TC>(a.data[m*K+k]) * static_cast<TC>(b.data[k*N+n]);
      }
      c.data[m*N+n] = acc;
    }
  }
}

} // namespace tessera
