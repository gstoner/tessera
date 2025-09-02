
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace tessera {

// Lightweight descriptor representing a .shared pointer and element bytes.
// Hopper/Blackwell WGMMA expects "descriptors" that encode SMEM address and layout.
// For simplicity we pass the cvta.to.shared pointer; layout is achieved by how we write SMEM.
struct smem_desc_2d {
  uint64_t smem_ptr;  // 64-bit shared memory address
  int      ld;        // leading dimension in elements
  int      elem_bytes;
};

template <typename T>
__device__ inline uint64_t cvta_to_shared_u64(const T* p) {
  uint64_t a; asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(a) : "l"(p));
  return a;
}

template <typename T>
__device__ inline smem_desc_2d make_smem_desc_2d(const T* base, int ld) {
  smem_desc_2d d;
  d.smem_ptr   = cvta_to_shared_u64(base);
  d.ld         = ld;
  d.elem_bytes = sizeof(T);
  return d;
}

// ---------------- Swizzle helpers ----------------
// Default is identity (row-major). For Hopper/Blackwell bank-optimized layouts,
// replace with an XOR/interleave mapping. We keep the interface fixed.

// Map (r,c) -> linear offset for an MxN tile in shared memory.
__device__ inline int smem_offset_identity(int r, int c, int ld) {
  return r * ld + c;
}

// Example hook: 128B interleaved swizzle (pseudo; adjust for your layout).
__device__ inline int smem_offset_swizzle_128B(int r, int c, int ld) {
  // Interleave columns in 128B groups to reduce bank conflicts.
  // 128B == 64 elements (bf16/fp16). We XOR a small tag from the row into the 64-wide group.
  int group = c / 64;
  int within = c % 64;
  int tag = (r & 0x1) << 5; // simple 32-element flip between even/odd rows
  int cc = group * 64 + (within ^ tag);
  return r * ld + cc;
}

// Choose swizzle: set USE_IDENTITY=1 for safe default.
#ifndef TESSERA_WGMMA_USE_IDENTITY_SWIZZLE
#define TESSERA_WGMMA_USE_IDENTITY_SWIZZLE 1
#endif

__device__ inline int smem_offset(int r, int c, int ld) {
#if TESSERA_WGMMA_USE_IDENTITY_SWIZZLE
  return smem_offset_identity(r,c,ld);
#else
  return smem_offset_swizzle_128B(r,c,ld);
#endif
}

// Convenience writers for bf16/half tiles into shared memory with chosen swizzle.
template <typename T>
__device__ inline void smem_store_tile_swizzled(T* smem, const T* gmem,
                                                int rows, int cols,
                                                int ld_src, int ld_dst) {
  for (int r=threadIdx.y; r<rows; r+=blockDim.y) {
    for (int c=threadIdx.x; c<cols; c+=blockDim.x) {
      smem[ smem_offset(r,c,ld_dst) ] = gmem[r*ld_src + c];
    }
  }
}

} // namespace tessera
