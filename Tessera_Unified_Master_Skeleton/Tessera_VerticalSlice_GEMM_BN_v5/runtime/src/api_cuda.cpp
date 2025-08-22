#include "tessera/runtime/api.h"
#include <cuda_runtime.h>
#include <new>
#include <cstdio>

extern "C" {

static tsStatus to_status(cudaError_t e){ return e==cudaSuccess ? TS_OK : TS_ERR_BACKEND; }

// Simple helpers: allocate device buffers when tensor.device>=0
static bool is_dev(const tsTensor* t){ return t && t->device>=0; }

tsStatus tesseraMatmul_CUDA(const tsTensor* A, const tsTensor* B, tsTensor* C){
  if(!A||!B||!C) return TS_ERR_INVALID;
  if(!(is_dev(A)&&is_dev(B)&&is_dev(C))) return TS_ERR_INVALID;
  int M=A->shape[0], K=A->shape[1], N=B->shape[1];
  dim3 block(16,16); dim3 grid((N+15)/16, (M+15)/16);
  extern void gemm_naive_kernel(const float*, const float*, float*, int,int,int);
  gemm_naive_kernel<<<grid, block>>>((const float*)A->ptr, (const float*)B->ptr, (float*)C->ptr, M,K,N);
  return to_status(cudaGetLastError());
}

tsStatus tesseraBatchNorm_CUDA(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                               const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y, float eps){
  if(!(is_dev(X)&&is_dev(Y)&&is_dev(Mean)&&is_dev(Var)&&is_dev(Gamma)&&is_dev(Beta))) return TS_ERR_INVALID;
  int C = X->shape[X->rank-1];
  long long elems=1; for(int i=0;i<X->rank;i++) elems*=X->shape[i];
  int rows = (int)(elems / C);
  int threads = 256;
  int blocks = (int)((elems + threads - 1) / threads);
  extern void bn_infer_lastdim_kernel(const float*, const float*, const float*, const float*, const float*, float*, int, int, float);
  bn_infer_lastdim_kernel<<<blocks, threads>>>((const float*)X->ptr, (const float*)Mean->ptr, (const float*)Var->ptr,
                                               (const float*)Gamma->ptr, (const float*)Beta->ptr, (float*)Y->ptr,
                                               rows, C, eps);
  return to_status(cudaGetLastError());
}

} // extern "C"
