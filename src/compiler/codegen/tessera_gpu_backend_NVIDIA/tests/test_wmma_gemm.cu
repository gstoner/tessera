#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cassert>
#include "tessera/gpu/target.h"

static __half h(float x){ return __float2half(x); }
static __nv_bfloat16 b16(float x){ return __float2bfloat16(x); }

int main() {
    using namespace tessera::gpu;
    TesseraGpuBackend be;
    if (!be.cudaAvailable()) { std::printf("No CUDA device.\n"); return 0; }
    std::printf("SM version: %d\n", be.smVersion());

    int M=128,N=128,K=128;
    std::vector<__half> A(M*K), B(K*N);
    std::vector<float>  Cf(M*N, 0.0f);
    for (int i=0;i<M*K;i++) A[i] = h((i%13 - 6) / 7.0f);
    for (int i=0;i<K*N;i++) B[i] = h(((i*3)%17 - 8) / 9.0f);

    __half* dA; __half* dB; float* dC;
    cudaMalloc(&dA, sizeof(__half)*M*K);
    cudaMalloc(&dB, sizeof(__half)*K*N);
    cudaMalloc(&dC, sizeof(float)*M*N);
    cudaMemcpy(dA, A.data(), sizeof(__half)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), sizeof(__half)*K*N, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sizeof(float)*M*N);

    be.wmma_gemm_fp16(dA, dB, dC, M,N,K, 1.0f, 0.0f);
    cudaMemcpy(Cf.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    // Just print a checksum
    double sum=0; for (auto v: Cf) sum += v;
    std::printf("FP16 GEMM checksum: %.6f\n", sum);

    // BF16 (if sm80+)
    if (be.smVersion() >= 80) {
        std::vector<__nv_bfloat16> Ab(M*K), Bb(K*N);
        for (int i=0;i<M*K;i++) Ab[i] = b16((i%11 - 5) / 7.0f);
        for (int i=0;i<K*N;i++) Bb[i] = b16(((i*5)%19 - 9) / 9.0f);
        __nv_bfloat16 *dAb,*dBb; cudaMalloc(&dAb, sizeof(__nv_bfloat16)*M*K);
        cudaMalloc(&dBb, sizeof(__nv_bfloat16)*K*N);
        cudaMemcpy(dAb, Ab.data(), sizeof(__nv_bfloat16)*M*K, cudaMemcpyHostToDevice);
        cudaMemcpy(dBb, Bb.data(), sizeof(__nv_bfloat16)*K*N, cudaMemcpyHostToDevice);
        cudaMemset(dC, 0, sizeof(float)*M*N);
        be.wmma_gemm_bf16(dAb, dBb, dC, M,N,K, 1.0f, 0.0f);
        cudaMemcpy(Cf.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);
        double sum2=0; for (auto v: Cf) sum2 += v;
        std::printf("BF16 GEMM checksum: %.6f\n", sum2);
        cudaFree(dAb); cudaFree(dBb);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
