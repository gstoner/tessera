
#pragma once
#include <cuda_runtime.h>

namespace tessera {

// Lane→fragment positions for one 64x64 sub-tile, per 128-thread warpgroup.
// Returns top-left (dr, dc) of a 2x2 micro-fragment for the calling thread, relative to the sub-tile origin.
__device__ inline void lane_fragment_map_64x64(int& dr, int& dc) {
  int lane = threadIdx.x & 31;   // 0..31 within warp
  int warp = (threadIdx.x >> 5) & 3; // 0..3 within warpgroup
  int row_block = (lane / 8) * 8;    // 0,8,16,24
  int col_block = (lane % 8) * 2;    // 0,2,4,...,14
  dr = row_block;
  dc = warp * 16 + col_block;
}

} // namespace tessera
