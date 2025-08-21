// runtime/src/api.cpp
#include "tessera/runtime/api.h"
#include <new>
#include <cmath>

extern "C" {

tsStatus tesseraMatmul(const tsTensor* A, const tsTensor* B, tsTensor* C){
  if(!A||!B||!C) return TS_ERR_INVALID;
  if(A->rank!=2 || B->rank!=2 || C->rank!=2) return TS_ERR_INVALID;
  const long long M=A->shape[0], K=A->shape[1], K2=B->shape[0], N=B->shape[1];
  if(K!=K2 || C->shape[0]!=M || C->shape[1]!=N) return TS_ERR_INVALID;

  const float* a = reinterpret_cast<const float*>(A->ptr);
  const float* b = reinterpret_cast<const float*>(B->ptr);
  float* c = reinterpret_cast<float*>(C->ptr);

  for(long long i=0;i<M;++i){
    for(long long j=0;j<N;++j){
      double acc=0.0;
      for(long long k=0;k<K;++k) acc += (double)a[i*K+k] * (double)b[k*N+j];
      c[i*N+j] = (float)acc;
    }
  }
  return TS_OK;
}

tsStatus tesseraBatchNorm(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                          const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y, float eps){
  if(!X||!Mean||!Var||!Gamma||!Beta||!Y) return TS_ERR_INVALID;
  if(X->dtype!=Mean->dtype || X->dtype!=Var->dtype || X->dtype!=Gamma->dtype || X->dtype!=Beta->dtype) return TS_ERR_INVALID;
  // Assume last-dim normalization: shape (..., C)
  if(X->rank < 1 || Y->rank != X->rank) return TS_ERR_INVALID;
  for(int i=0;i<X->rank;i++) if (X->shape[i]!=Y->shape[i]) return TS_ERR_INVALID;
  const long long C = X->shape[X->rank-1];

  const float *x = (const float*)X->ptr;
  const float *mean=(const float*)Mean->ptr;
  const float *var=(const float*)Var->ptr;
  const float *gamma=(const float*)Gamma->ptr;
  const float *beta=(const float*)Beta->ptr;
  float *y = (float*)Y->ptr;

  long long elems = 1; for(int i=0;i<X->rank;i++) elems *= X->shape[i];
  long long rows = elems / C;

  for(long long n=0;n<rows;++n){
    const long long base = n*C;
    for(long long c=0;c<C;++c){
      const float xn = x[base + c];
      const float inv_std = 1.0f / std::sqrt(var[c] + eps);
      y[base + c] = gamma[c] * (xn - mean[c]) * inv_std + beta[c];
    }
  }
  return TS_OK;
}

} // extern "C"
