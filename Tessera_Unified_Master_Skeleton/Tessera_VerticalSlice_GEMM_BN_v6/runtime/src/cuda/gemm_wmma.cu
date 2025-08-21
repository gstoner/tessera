#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

extern "C" __global__
void gemm_wmma_m16n16k16_fp16acc_f32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                     int M, int K, int N){
#if __CUDA_ARCH__ >= 700
  // Each warp computes one 16x16 tile of C
  int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
  int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

  int row = warpM * 16;
  int col = warpN * 16;
  if(row >= M || col >= N) return;

  wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> aFrag;
  wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> bFrag;
  wmma::fragment<wmma::accumulator, 16,16,16, float> cFrag;
  wmma::fill_fragment(cFrag, 0.0f);

  for(int k=0;k<K;k+=16){
    // Load A and B tiles, converting fp32 -> fp16 on the fly
    half aTile[16*16], bTile[16*16];
    #pragma unroll
    for(int i=0;i<16;i++){
      int r = row + i;
      #pragma unroll
      for(int j=0;j<16;j++){
        int c = k + j;
        float v = (r<M && c<K) ? A[r*K + c] : 0.f;
        aTile[i*16+j] = __float2half_rn(v);
      }
    }
    #pragma unroll
    for(int i=0;i<16;i++){
      int r = k + i;
      #pragma unroll
      for(int j=0;j<16;j++){
        int c = col + j;
        float v = (r<K && c<N) ? B[r*N + c] : 0.f;
        bTile[i*16+j] = __float2half_rn(v);
      }
    }

    wmma::load_matrix_sync(aFrag, aTile, 16);
    wmma::load_matrix_sync(bFrag, bTile, 16);
    wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
  }

  // Store back
  float cTile[16*16];
  wmma::store_matrix_sync(cTile, cFrag, 16, wmma::mem_row_major);
  for(int i=0;i<16;i++){
    int r = row + i;
    if(r<M){
      for(int j=0;j<16;j++){
        int c = col + j;
        if(c<N){
          C[r*N + c] = cTile[i*16+j];
        }
      }
    }
  }
#endif
}
