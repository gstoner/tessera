#include "tessera/runtime/api.h"
#include <new>
#include <cmath>
#include <cstring>

extern "C" {

tsStatus tesseraSetDevice(int device){ (void)device; return TS_OK; }

tsStatus tesseraAllocHost(tsTensor* t, int dtype, int rank, const long long* shape){
  if(!t || rank<1 || rank>8) return TS_ERR_INVALID;
  size_t elems=1; for(int i=0;i<rank;i++) elems *= (size_t)shape[i];
  size_t bytes = elems * 4; // f32
  t->ptr = ::operator new(bytes, std::nothrow);
  if(!t->ptr) return TS_ERR_OOM;
  t->device=-1; t->dtype=dtype; t->rank=rank;
  for(int i=0;i<rank;i++) t->shape[i]=shape[i];
  return TS_OK;
}
tsStatus tesseraFreeHost(tsTensor* t){ if(!t||!t->ptr) return TS_ERR_INVALID; ::operator delete(t->ptr); t->ptr=nullptr; return TS_OK; }

tsStatus tesseraAllocDevice(tsTensor*, int, int, int, const long long*){ return TS_ERR_BACKEND; }
tsStatus tesseraFreeDevice(tsTensor*){ return TS_ERR_BACKEND; }
tsStatus tesseraCopyHostToDevice(const tsTensor*, tsTensor*){ return TS_ERR_BACKEND; }
tsStatus tesseraCopyDeviceToHost(const tsTensor*, tsTensor*){ return TS_ERR_BACKEND; }

tsStatus tesseraStreamCreate(int device, tsStream* out){ if(!out) return TS_ERR_INVALID; out->impl=nullptr; out->device=device; return TS_OK; }
tsStatus tesseraStreamDestroy(tsStream* s){ (void)s; return TS_OK; }
tsStatus tesseraStreamSync(tsStream* s){ (void)s; return TS_OK; }

static tsStatus cpu_matmul(const tsTensor* A, const tsTensor* B, tsTensor* C){
  if(A->rank!=2||B->rank!=2||C->rank!=2) return TS_ERR_INVALID;
  long long M=A->shape[0], K=A->shape[1], K2=B->shape[0], N=B->shape[1];
  if(K!=K2 || C->shape[0]!=M || C->shape[1]!=N) return TS_ERR_INVALID;
  const float *a=(const float*)A->ptr, *b=(const float*)B->ptr; float *c=(float*)C->ptr;
  for(long long i=0;i<M;i++){ for(long long j=0;j<N;j++){ double acc=0.0; for(long long k=0;k<K;k++) acc += (double)a[i*K+k]*(double)b[k*N+j]; c[i*N+j]=(float)acc; } }
  return TS_OK;
}

static tsStatus cpu_bn(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                       const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y, float eps){
  if(X->rank<1||Y->rank!=X->rank) return TS_ERR_INVALID;
  for(int i=0;i<X->rank;i++) if(X->shape[i]!=Y->shape[i]) return TS_ERR_INVALID;
  long long C = X->shape[X->rank-1];
  const float *x=(const float*)X->ptr, *mean=(const float*)Mean->ptr, *var=(const float*)Var->ptr;
  const float *g=(const float*)Gamma->ptr, *b=(const float*)Beta->ptr;
  float *y=(float*)Y->ptr;
  long long elems=1; for(int i=0;i<X->rank;i++) elems *= X->shape[i];
  long long rows = elems / C;
  for(long long n=0;n<rows;n++){ long long base=n*C; for(long long c=0;c<C;c++){ float inv=1.0f/std::sqrt(var[c]+eps); y[base+c]=g[c]*(x[base+c]-mean[c])*inv+b[c]; } }
  return TS_OK;
}

tsStatus tesseraMatmulAsync(const tsTensor* A, const tsTensor* B, tsTensor* C, tsStream* stream){
  (void)stream;
  if(A->device!=-1||B->device!=-1||C->device!=-1) return TS_ERR_BACKEND;
  return cpu_matmul(A,B,C);
}

tsStatus tesseraBatchNormAsync(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                               const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y,
                               float eps, tsStream* stream){
  (void)stream;
  if(X->device!=-1||Y->device!=-1) return TS_ERR_BACKEND;
  return cpu_bn(X,Mean,Var,Gamma,Beta,Y,eps);
}

} // extern "C"
