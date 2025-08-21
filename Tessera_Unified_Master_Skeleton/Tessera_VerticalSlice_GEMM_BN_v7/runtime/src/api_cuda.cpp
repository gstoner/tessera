#include "tessera/runtime/api.h"
#include <cuda_runtime.h>
#include <cstdlib>

extern "C" {

static tsStatus to_status(cudaError_t e){ return e==cudaSuccess ? TS_OK : TS_ERR_BACKEND; }

tsStatus tesseraSetDevice(int device){ return to_status(cudaSetDevice(device)); }

static size_t bytes_for(const tsTensor* t){
  size_t elems=1; for(int i=0;i<t->rank;i++) elems *= (size_t)t->shape[i]; return elems*4;
}

tsStatus tesseraAllocDevice(tsTensor* t, int device, int dtype, int rank, const long long* shape){
  if(!t||rank<1||rank>8) return TS_ERR_INVALID;
  t->device=device; t->dtype=dtype; t->rank=rank;
  for(int i=0;i<rank;i++) t->shape[i]=shape[i];
  size_t bytes = bytes_for(t);
  cudaError_t e = cudaMalloc(&t->ptr, bytes);
  return to_status(e);
}
tsStatus tesseraFreeDevice(tsTensor* t){ if(!t||!t->ptr) return TS_ERR_INVALID; return to_status(cudaFree(t->ptr)); }

tsStatus tesseraCopyHostToDevice(const tsTensor* srcHost, tsTensor* dstDevice){
  if(!srcHost||!dstDevice||srcHost->device!=-1||dstDevice->device<0) return TS_ERR_INVALID;
  if(srcHost->rank!=dstDevice->rank) return TS_ERR_INVALID;
  for(int i=0;i<srcHost->rank;i++) if(srcHost->shape[i]!=dstDevice->shape[i]) return TS_ERR_INVALID;
  return to_status(cudaMemcpy(dstDevice->ptr, srcHost->ptr, bytes_for(srcHost), cudaMemcpyHostToDevice));
}
tsStatus tesseraCopyDeviceToHost(const tsTensor* srcDevice, tsTensor* dstHost){
  if(!srcDevice||!dstHost||srcDevice->device<0||dstHost->device!=-1) return TS_ERR_INVALID;
  if(srcDevice->rank!=dstHost->rank) return TS_ERR_INVALID;
  for(int i=0;i<srcDevice->rank;i++) if(srcDevice->shape[i]!=dstHost->shape[i]) return TS_ERR_INVALID;
  return to_status(cudaMemcpy(dstHost->ptr, srcDevice->ptr, bytes_for(srcDevice), cudaMemcpyDeviceToHost));
}

// Streams
tsStatus tesseraStreamCreate(int device, tsStream* out){
  if(!out) return TS_ERR_INVALID;
  cudaStream_t s = nullptr;
  auto e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  out->impl = (void*)s; out->device=device;
  return to_status(e);
}
tsStatus tesseraStreamDestroy(tsStream* s){ if(!s) return TS_ERR_INVALID; return to_status(cudaStreamDestroy((cudaStream_t)s->impl)); }
tsStatus tesseraStreamSync(tsStream* s){ cudaStream_t cs = s?(cudaStream_t)s->impl:nullptr; return to_status(cudaStreamSynchronize(cs)); }

// Kernel decls
void gemm_naive_kernel(const float*, const float*, float*, int,int,int);
void bn_infer_lastdim_kernel(const float*, const float*, const float*, const float*, const float*, float*, int,int,float);
void gemm_wmma_m16n16k16_fp16acc_f32(const float*, const float*, float*, int,int,int);
void gemm_wgmma_sm90_fp16acc_f32(const float*, const float*, float*, int,int,int);

// Env toggles
static bool use_wgmma(){ const char* e = std::getenv("TESSERA_WGMMA"); return e && (e[0]=='1'); }
static bool use_mma(){ const char* e = std::getenv("TESSERA_MMA"); return e && (e[0]=='1'); }
static bool force_naive(){ const char* e = std::getenv("TESSERA_NAIVE"); return e && (e[0]=='1'); }

tsStatus tesseraMatmulAsync(const tsTensor* A, const tsTensor* B, tsTensor* C, tsStream* stream){
  if(!A||!B||!C) return TS_ERR_INVALID;
  if(!(A->device>=0 && B->device>=0 && C->device>=0)) return TS_ERR_INVALID;
  int M=A->shape[0], K=A->shape[1], N=B->shape[1];
  cudaStream_t s = stream ? (cudaStream_t)stream->impl : nullptr;

#if __CUDA_ARCH__ >= 900
  if(!force_naive() && use_wgmma()){
    dim3 block(32, 8); // tune as needed
    dim3 grid((N+127)/128, (M+63)/64);
    size_t smem = (64*32 + 32*128 + 64*32 + 32*128) * sizeof(float); // double-buffer A/B
    gemm_wgmma_sm90_fp16acc_f32<<<grid, block, smem, s>>>((const float*)A->ptr, (const float*)B->ptr, (float*)C->ptr, M,K,N);
    return to_status(cudaGetLastError());
  }
#endif
  if(!force_naive() && use_mma()){
    dim3 block(128, 4);
    dim3 grid((N+15)/16, (M+15)/16);
    gemm_wmma_m16n16k16_fp16acc_f32<<<grid, block, 0, s>>>((const float*)A->ptr, (const float*)B->ptr, (float*)C->ptr, M,K,N);
    return to_status(cudaGetLastError());
  } else {
    dim3 block(16,16); dim3 grid((N+15)/16, (M+15)/16);
    gemm_naive_kernel<<<grid, block, 0, s>>>((const float*)A->ptr, (const float*)B->ptr, (float*)C->ptr, M,K,N);
    return to_status(cudaGetLastError());
  }
}

tsStatus tesseraBatchNormAsync(const tsTensor* X, const tsTensor* Mean, const tsTensor* Var,
                               const tsTensor* Gamma, const tsTensor* Beta, tsTensor* Y,
                               float eps, tsStream* stream){
  if(!(X&&Mean&&Var&&Gamma&&Beta&&Y)) return TS_ERR_INVALID;
  if(!(X->device>=0 && Y->device>=0)) return TS_ERR_INVALID;
  int C = X->shape[X->rank-1];
  long long elems=1; for(int i=0;i<X->rank;i++) elems *= X->shape[i];
  int rows = (int)(elems / C);
  cudaStream_t s = stream ? (cudaStream_t)stream->impl : nullptr;
  int threads=256; int blocks=(int)((elems + threads - 1)/threads);
  bn_infer_lastdim_kernel<<<blocks, threads, 0, s>>>((const float*)X->ptr, (const float*)Mean->ptr, (const float*)Var->ptr,
                                                     (const float*)Gamma->ptr, (const float*)Beta->ptr, (float*)Y->ptr,
                                                     rows, C, eps);
  return to_status(cudaGetLastError());
}

} // extern "C"
