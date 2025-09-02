#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>

// Forward declare from ptx_wgmma_bf16.cu
extern "C" void launch_wgmma_bf16_gemm(const __nv_bfloat16*, const __nv_bfloat16*, float*,
                                       int,int,int, float,float, cudaStream_t);

static __nv_bfloat16 b16(float x){ return __float2bfloat16(x); }

int main() {
    int M=128,N=128,K=128;
    std::vector<__nv_bfloat16> A(M*K), B(K*N);
    std::vector<float> C(M*N, 1.0f);
    for (int i=0;i<M*K;i++) A[i] = b16((i%7 - 3)/4.0f);
    for (int i=0;i<K*N;i++) B[i] = b16(((i*5)%11 - 5)/6.0f);
    __nv_bfloat16 *dA,*dB; float* dC;
    cudaMalloc(&dA, sizeof(__nv_bfloat16)*M*K);
    cudaMalloc(&dB, sizeof(__nv_bfloat16)*K*N);
    cudaMalloc(&dC, sizeof(float)*M*N);
    cudaMemcpy(dA, A.data(), sizeof(__nv_bfloat16)*M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), sizeof(__nv_bfloat16)*K*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), sizeof(float)*M*N, cudaMemcpyHostToDevice);

    launch_wgmma_bf16_gemm(dA,dB,dC,M,N,K,1.0f,0.0f,0);
    cudaMemcpy(C.data(), dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    double sum=0; for (auto v: C) sum += v;
    std::printf("WGMMA placeholder checksum: %.6f\n", sum);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
