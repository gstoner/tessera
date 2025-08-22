#include <cuda_runtime.h>
extern "C" {

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                  int M, int K, int N){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(row<M && col<N){
    float acc = 0.f;
    for(int k=0;k<K;k++) acc += A[row*K+k]*B[k*N+col];
    C[row*N+col] = acc;
  }
}

__global__ void bn_infer_lastdim_kernel(const float* X, const float* mean, const float* var,
                                        const float* gamma, const float* beta, float* Y,
                                        int rows, int C, float eps){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows*C;
  if(idx < total){
    int c = idx % C;
    float inv_std = rsqrtf(var[c] + eps);
    Y[idx] = gamma[c] * (X[idx] - mean[c]) * inv_std + beta[c];
  }
}

} // extern "C"
