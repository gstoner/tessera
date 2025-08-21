#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__
void gemm_wgmma_sm90_fp16acc_f32(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                 int M, int K, int N){
#if __CUDA_ARCH__ >= 900
  constexpr int BM = 64;
  constexpr int BN = 128;
  constexpr int BK = 32;

  extern __shared__ uint8_t smem[];
  float* As = reinterpret_cast<float*>(smem);
  float* Bs = As + BM * BK;
  float* As1 = Bs + BK * BN;
  float* Bs1 = As1 + BM * BK;

  int blockRow = blockIdx.y * BM;
  int blockCol = blockIdx.x * BN;

  for(int i = threadIdx.y; i < BM; i += blockDim.y){
    for(int j = threadIdx.x; j < BN; j += blockDim.x){
      if(blockRow + i < M && blockCol + j < N){
        C[(blockRow + i)*N + (blockCol + j)] = 0.f;
      }
    }
  }
  __syncthreads();

  bool useBuf0 = true;
  // Preload first tiles
  {
    int t = threadIdx.y * blockDim.x + threadIdx.x;
    int loadElemsA = BM * BK;
    int loadElemsB = BK * BN;
    for(int idx = t; idx < loadElemsA; idx += blockDim.x * blockDim.y){
      int r = idx / BK;
      int c = idx % BK;
      int gRow = blockRow + r;
      int gCol = 0 + c;
      float v = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.f;
      As[idx] = v;
    }
    for(int idx = t; idx < loadElemsB; idx += blockDim.x * blockDim.y){
      int r = idx / BN;
      int c = idx % BN;
      int gRow = 0 + r;
      int gCol = blockCol + c;
      float v = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.f;
      Bs[idx] = v;
    }
  }
  __syncthreads();

  for(int k0 = 0; k0 < K; k0 += BK){
    if(k0 + BK < K){
      int t = threadIdx.y * blockDim.x + threadIdx.x;
      int loadElemsA = BM * BK;
      int loadElemsB = BK * BN;
      for(int idx = t; idx < loadElemsA; idx += blockDim.x * blockDim.y){
        int r = idx / BK;
        int c = idx % BK;
        int gRow = blockRow + r;
        int gCol = k0 + BK + c;
        float v = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.f;
        (useBuf0 ? As1 : As)[idx] = v;
      }
      for(int idx = t; idx < loadElemsB; idx += blockDim.x * blockDim.y){
        int r = idx / BN;
        int c = idx % BN;
        int gRow = k0 + BK + r;
        int gCol = blockCol + c;
        float v = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.f;
        (useBuf0 ? Bs1 : Bs)[idx] = v;
      }
    }

    __syncthreads();
    float* Acur = useBuf0 ? As : As1;
    float* Bcur = useBuf0 ? Bs : Bs1;
    for(int i = threadIdx.y; i < BM; i += blockDim.y){
      for(int j = threadIdx.x; j < BN; j += blockDim.x){
        float acc = 0.f;
        for(int kk=0; kk < BK; ++kk){
          acc += Acur[i*BK + kk] * Bcur[kk*BN + j];
        }
        int r = blockRow + i;
        int c = blockCol + j;
        if(r < M && c < N){
          C[r*N + c] += acc;
        }
      }
    }
    __syncthreads();
    useBuf0 = !useBuf0;
  }
#endif
}
