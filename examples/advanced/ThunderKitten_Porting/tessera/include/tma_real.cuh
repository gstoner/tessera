
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef TESSERA_REAL_TMA
#define TESSERA_REAL_TMA 0
#endif

namespace tessera {

struct tma_desc_2d {
  unsigned long long raw[8];
};

// Create a 2D tensormap descriptor for row-major [rows x cols] with element size elem_bytes and ld_src (bytes).
__device__ inline void tma_create_2d(tma_desc_2d* td, const void* gptr, int rows, int cols, int ld_src_bytes, int elem_bytes) {
#if TESSERA_REAL_TMA && (__CUDA_ARCH__ >= 900)
  asm volatile(
    "tensormap.create.tensor.2d.shared::cluster.global [%0], [%1], %2, %3, %4, %5, %6, %7;"
    :: "r"(td->raw), "l"(gptr), "r"(rows), "r"(cols), "r"(ld_src_bytes), "r"(elem_bytes), "r"(0 /*swizzle*/), "r"(0 /*oob*/)
  );
#else
  (void)td; (void)gptr; (void)rows; (void)cols; (void)ld_src_bytes; (void)elem_bytes;
#endif
}

// Issue a bulk tensor copy into shared memory using the descriptor and an mbarrier.
// Describes a [rows x cols] transfer starting from (row_off, col_off).
__device__ inline void tma_async_bulk_2d(void* smem_dst, const tma_desc_2d* td, int row_off, int col_off, int rows, int cols, unsigned long long* mbar_addr) {
#if TESSERA_REAL_TMA && (__CUDA_ARCH__ >= 900)
  // arrive
  asm volatile("mbarrier.arrive.shared.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
  // bulk tensor copy
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global [%0], [%1], %2, %3, [%4];"
    :: "r"(smem_dst), "r"(td->raw), "r"(row_off), "r"(col_off), "r"(mbar_addr)
  );
  // wait
  asm volatile("mbarrier.try_wait.parity.shared.b64 %0, [%1];" : "=r"(row_off) : "r"(mbar_addr));
#else
  (void)smem_dst; (void)td; (void)row_off; (void)col_off; (void)rows; (void)cols; (void)mbar_addr;
#endif
}

} // namespace tessera
