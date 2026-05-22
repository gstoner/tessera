
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

namespace tessera {

__device__ inline float silu(float x){ return x / (1.0f + __expf(-x)); }
__device__ inline float gelu(float x){
  // approximate GELU
  const float k = 0.7978845608f; // sqrt(2/pi)
  return 0.5f * x * (1.0f + tanhf(k*(x + 0.044715f*x*x*x)));
}

enum EpilogueKind { E_NONE=0, E_BIAS=1, E_BIAS_SILU=2, E_BIAS_GELU=3 };

template <typename AccT=float>
__device__ inline void apply_epilogue(EpilogueKind kind, AccT* tile, int rows, int cols, int ld, const float* bias /*len>=cols*/) {
  if (kind==E_NONE) return;
  for (int r=threadIdx.y; r<rows; r+=blockDim.y){
    for (int c=threadIdx.x; c<cols; c+=blockDim.x){
      float v = tile[r*ld + c];
      float b = (bias? bias[c] : 0.0f);
      if (kind==E_BIAS)        v = v + b;
      else if (kind==E_BIAS_SILU) v = silu(v + b);
      else if (kind==E_BIAS_GELU) v = gelu(v + b);
      tile[r*ld + c] = v;
    }
  }
  __syncthreads();
}

} // namespace tessera
