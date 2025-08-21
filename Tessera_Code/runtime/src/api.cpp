#include "tessera/runtime/api.h"
#include <new>

tsStatus tesseraInit(void){ return TS_OK; }
tsStatus tesseraShutdown(void){ return TS_OK; }

tsStatus tesseraAlloc(tsTensor* t, int device, int dtype, int rank, const int64_t* shape){
  if(!t || rank<1 || rank>8) return TS_ERR_INVALID;
  size_t elems = 1;
  for(int i=0;i<rank;++i) elems *= (size_t)shape[i];
  size_t bytes = elems * 4; // assume 4B dtype placeholder
  t->ptr = ::operator new(bytes, std::nothrow);
  if(!t->ptr) return TS_ERR_OOM;
  t->device = device;
  t->dtype = dtype;
  t->rank = rank;
  for(int i=0;i<rank;++i) t->shape[i]=shape[i];
  return TS_OK;
}
tsStatus tesseraFree(tsTensor* t){
  if(!t || !t->ptr) return TS_ERR_INVALID;
  ::operator delete(t->ptr);
  t->ptr=nullptr; return TS_OK;
}

tsStatus tesseraStreamCreate(int device, tsStream* out){ if(!out) return TS_ERR_INVALID; out->device=device; out->id=0; return TS_OK; }
tsStatus tesseraStreamDestroy(tsStream* s){ (void)s; return TS_OK; }
tsStatus tesseraStreamSync(tsStream* s){ (void)s; return TS_OK; }

tsStatus tesseraMatmul(const tsTensor* A, const tsTensor* B, tsTensor* C){
  if(!A||!B||!C) return TS_ERR_INVALID;
  const int64_t M=A->shape[0], K=A->shape[1], K2=B->shape[0], N=B->shape[1];
  if(K!=K2) return TS_ERR_INVALID;
  const float* a = reinterpret_cast<const float*>(A->ptr);
  const float* b = reinterpret_cast<const float*>(B->ptr);
  float* c = reinterpret_cast<float*>(C->ptr);
  for(int64_t i=0;i<M;++i){
    for(int64_t j=0;j<N;++j){
      double acc=0.0;
      for(int64_t k=0;k<K;++k) acc += (double)a[i*K+k] * (double)b[k*N+j];
      c[i*N+j] = (float)acc;
    }
  }
  return TS_OK;
}
