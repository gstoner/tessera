#include "tessera/gpu/target.h"
#include "tessera/gpu/cuda_driver.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <memory>

namespace tessera { namespace gpu {

struct Impl {
    CudaContext ctx;
};

TesseraGpuBackend::TesseraGpuBackend() : impl_(new Impl()) {}
TesseraGpuBackend::~TesseraGpuBackend() { delete static_cast<Impl*>(impl_); }

bool TesseraGpuBackend::cudaAvailable() const noexcept {
    auto* I = static_cast<Impl*>(impl_);
    return I->ctx.ok();
}
int  TesseraGpuBackend::smVersion() const noexcept {
    auto* I = static_cast<Impl*>(impl_);
    return I->ctx.sm;
}

// Kernels (symbols defined in .cu files)
void launch_wmma_fp16_gemm(const __half*, const __half*, float*, int,int,int, float,float, cudaStream_t);
void launch_wmma_bf16_gemm(const __nv_bfloat16*, const __nv_bfloat16*, float*, int,int,int, float,float, cudaStream_t);

void TesseraGpuBackend::wmma_gemm_fp16(const __half* A, const __half* B, float* C,
                                       int M, int N, int K, float alpha, float beta) {
    launch_wmma_fp16_gemm(A,B,C,M,N,K,alpha,beta,0);
}
void TesseraGpuBackend::wmma_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                                       int M, int N, int K, float alpha, float beta) {
    launch_wmma_bf16_gemm(A,B,C,M,N,K,alpha,beta,0);
}

extern "C" {

bool tessera_gpu_available() {
    TesseraGpuBackend b; return b.cudaAvailable();
}
int  tessera_gpu_sm() {
    TesseraGpuBackend b; return b.smVersion();
}
void tessera_gpu_wmma_gemm_fp16(const __half* A, const __half* B, float* C,
                                int M, int N, int K, float alpha, float beta) {
    TesseraGpuBackend b; b.wmma_gemm_fp16(A,B,C,M,N,K,alpha,beta);
}
void tessera_gpu_wmma_gf16(const __half* A, const __half* B, float* C,
                           int M, int N, int K, float alpha, float beta) {
    TesseraGpuBackend b; b.wmma_gemm_fp16(A,B,C,M,N,K,alpha,beta);
}
void tessera_gpu_wmma_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                                int M, int N, int K, float alpha, float beta) {
    TesseraGpuBackend b; b.wmma_gemm_bf16(A,B,C,M,N,K,alpha,beta);
}

} // extern "C"

}} // namespace


// New launchers
void launch_wgmma_bf16_ptx(const __nv_bfloat16*, const __nv_bfloat16*, float*,
                           int,int,int,int,int,int,float,float,const void*,const void*, cudaStream_t);
void launch_imma_int8(const int8_t*, const int8_t*, int32_t*,
                      int,int,int,int,int,int,int, cudaStream_t);

// Convenience C wrappers
extern "C" void tessera_gpu_wgmma_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                                       int M, int N, int K, int lda, int ldb, int ldc,
                                       float alpha, float beta,
                                       const void* tma_desc_A, const void* tma_desc_B) {
    launch_wgmma_bf16_ptx(A,B,C,M,N,K,lda,ldb,ldc,alpha,beta,tma_desc_A,tma_desc_B,0);
}

extern "C" void tessera_gpu_imma_int8(const int8_t* A, const int8_t* B, int32_t* C,
                                      int M, int N, int K, int lda, int ldb, int ldc, int beta) {
    launch_imma_int8(A,B,C,M,N,K,lda,ldb,ldc,beta,0);
}
