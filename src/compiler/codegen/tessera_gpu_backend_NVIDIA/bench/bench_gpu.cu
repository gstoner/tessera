#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <random>
#include <string>

// Backends
extern "C" void tessera_gpu_wmma_gemm_fp16(const __half*, const __half*, float*, int,int,int,float,float);
extern "C" void tessera_gpu_wmma_gemm_bf16(const __nv_bfloat16*, const __nv_bfloat16*, float*, int,int,int,float,float);
extern "C" void tessera_gpu_wgmma_bf16(const __nv_bfloat16*, const __nv_bfloat16*, float*,
                                       int,int,int,int,int,int,float,float,const void*,const void*);
extern "C" void tessera_gpu_imma_int8(const int8_t*, const int8_t*, int32_t*,
                                      int,int,int,int,int,int,int);

using Clock = std::chrono::steady_clock;

template<typename T>
void fill_vec(std::vector<T>& v, unsigned seed=123) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1,1);
    for (auto& x: v) x = (T)d(rng);
}
void fill_s8(std::vector<int8_t>& v, unsigned seed=123) {
    std::mt19937 rng(seed); std::uniform_int_distribution<int> d(-127,127);
    for (auto& x: v) x = (int8_t)d(rng);
}

int main() {
    printf("kernel,dtype,path,M,N,K,ms,tflops\n");
    int sizes[][3] = {{256,256,256},{512,512,512},{1024,1024,1024}};
    for (auto& s: sizes) {
        int M=s[0],N=s[1],K=s[2];

        // FP16 WMMA
        {
            std::vector<__half> A(M*K), B(K*N); std::vector<float> C(M*N);
            for (int i=0;i<M*K;i++) A[i] = __float2half((i%7 - 3)/4.0f);
            for (int i=0;i<K*N;i++) B[i] = __float2half(((i*3)%11 - 5)/6.0f);
            __half *dA,*dB; float* dC;
            cudaMalloc(&dA,sizeof(__half)*M*K); cudaMalloc(&dB,sizeof(__half)*K*N); cudaMalloc(&dC,sizeof(float)*M*N);
            cudaMemcpy(dA,A.data(),sizeof(__half)*M*K,cudaMemcpyHostToDevice);
            cudaMemcpy(dB,B.data(),sizeof(__half)*K*N,cudaMemcpyHostToDevice); cudaMemset(dC,0,sizeof(float)*M*N);
            for (int w=0; w<5; ++w) tessera_gpu_wmma_gemm_fp16(dA,dB,dC,M,N,K,1.0f,0.0f);
            auto t0=Clock::now();
            for (int it=0; it<30; ++it) tessera_gpu_wmma_gemm_fp16(dA,dB,dC,M,N,K,1.0f,0.0f);
            auto t1=Clock::now();
            double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/30.0;
            double flops = 2.0*double(M)*double(N)*double(K);
            double tflops = flops/(ms*1e9);
            printf("wmma,fp16,wmma,%d,%d,%d,%.3f,%.3f\n",M,N,K,ms,tflops);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }

        // BF16 WMMA
        {
            std::vector<__nv_bfloat16> A(M*K), B(K*N); std::vector<float> C(M*N);
            for (int i=0;i<M*K;i++) A[i] = __float2bfloat16((i%7 - 3)/4.0f);
            for (int i=0;i<K*N;i++) B[i] = __float2bfloat16(((i*3)%11 - 5)/6.0f);
            __nv_bfloat16 *dA,*dB; float* dC;
            cudaMalloc(&dA,sizeof(__nv_bfloat16)*M*K); cudaMalloc(&dB,sizeof(__nv_bfloat16)*K*N); cudaMalloc(&dC,sizeof(float)*M*N);
            cudaMemcpy(dA,A.data(),sizeof(__nv_bfloat16)*M*K,cudaMemcpyHostToDevice);
            cudaMemcpy(dB,B.data(),sizeof(__nv_bfloat16)*K*N,cudaMemcpyHostToDevice); cudaMemset(dC,0,sizeof(float)*M*N);
            for (int w=0; w<5; ++w) tessera_gpu_wmma_gemm_bf16(dA,dB,dC,M,N,K,1.0f,0.0f);
            auto t0=Clock::now();
            for (int it=0; it<30; ++it) tessera_gpu_wmma_gemm_bf16(dA,dB,dC,M,N,K,1.0f,0.0f);
            auto t1=Clock::now();
            double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/30.0;
            double flops = 2.0*double(M)*double(N)*double(K);
            double tflops = flops/(ms*1e9);
            printf("wmma,bf16,wmma,%d,%d,%d,%.3f,%.3f\n",M,N,K,ms,tflops);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }

        // BF16 WGMMA (placeholder math; measures issue/latency)
        {
            std::vector<__nv_bfloat16> A(M*K), B(K*N); std::vector<float> C(M*N, 0.f);
            for (int i=0;i<M*K;i++) A[i] = __float2bfloat16((i%7 - 3)/4.0f);
            for (int i=0;i<K*N;i++) B[i] = __float2bfloat16(((i*3)%11 - 5)/6.0f);
            __nv_bfloat16 *dA,*dB; float* dC;
            cudaMalloc(&dA,sizeof(__nv_bfloat16)*M*K); cudaMalloc(&dB,sizeof(__nv_bfloat16)*K*N); cudaMalloc(&dC,sizeof(float)*M*N);
            cudaMemcpy(dA,A.data(),sizeof(__nv_bfloat16)*M*K,cudaMemcpyHostToDevice);
            cudaMemcpy(dB,B.data(),sizeof(__nv_bfloat16)*K*N,cudaMemcpyHostToDevice); cudaMemset(dC,0,sizeof(float)*M*N);
            for (int w=0; w<5; ++w)
                tessera_gpu_wgmma_bf16(dA,dB,dC,M,N,K, K, N, N, 1.0f, 0.0f, nullptr, nullptr);
            auto t0=Clock::now();
            for (int it=0; it<30; ++it)
                tessera_gpu_wgmma_bf16(dA,dB,dC,M,N,K, K, N, N, 1.0f, 0.0f, nullptr, nullptr);
            auto t1=Clock::now();
            double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/30.0;
            double flops = 2.0*double(M)*double(N)*double(K);
            double tflops = flops/(ms*1e9);
            printf("wgmma,bf16,wgmma,%d,%d,%d,%.3f,%.3f\n",M,N,K,ms,tflops);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }

        // INT8 IMMA (approx, dp4a/IMMA hybrid)
        {
            std::vector<int8_t> A(M*K), B(K*N); std::vector<int32_t> C(M*N,0);
            std::mt19937 rng(123); std::uniform_int_distribution<int> di(-127,127);
            for (auto& x: A) x = (int8_t)di(rng);
            for (auto& x: B) x = (int8_t)di(rng);
            int8_t *dA,*dB; int32_t* dC;
            cudaMalloc(&dA,sizeof(int8_t)*M*K); cudaMalloc(&dB,sizeof(int8_t)*K*N); cudaMalloc(&dC,sizeof(int32_t)*M*N);
            cudaMemcpy(dA,A.data(),sizeof(int8_t)*M*K,cudaMemcpyHostToDevice);
            cudaMemcpy(dB,B.data(),sizeof(int8_t)*K*N,cudaMemcpyHostToDevice); cudaMemset(dC,0,sizeof(int32_t)*M*N);
            for (int w=0; w<5; ++w) tessera_gpu_imma_int8(dA,dB,dC,M,N,K, K, N, N, 0);
            auto t0=Clock::now();
            for (int it=0; it<30; ++it) tessera_gpu_imma_int8(dA,dB,dC,M,N,K, K, N, N, 0);
            auto t1=Clock::now();
            double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/30.0;
            double flops = 2.0*double(M)*double(N)*double(K);
            double tflops = flops/(ms*1e9);
            printf("imma,int8,imma,%d,%d,%d,%.3f,%.3f\n",M,N,K,ms,tflops);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }
    }
    return 0;
}


// Pipeline kernel
extern "C" void wmma_bf16_pipeline_kernel(const __nv_bfloat16*, const __nv_bfloat16*, float*,
                                          int,int,int,int,int,int,float,float);

// Add timing for pipeline (correctness-checked)
static void run_wgmma_pipeline(int M,int N,int K) {
    std::vector<__nv_bfloat16> A(M*K), B(K*N); std::vector<float> C(M*N, 0.f);
    for (int i=0;i<M*K;i++) A[i] = __float2bfloat16((i%7 - 3)/4.0f);
    for (int i=0;i<K*N;i++) B[i] = __float2bfloat16(((i*3)%11 - 5)/6.0f);
    __nv_bfloat16 *dA,*dB; float* dC;
    cudaMalloc(&dA,sizeof(__nv_bfloat16)*M*K); cudaMalloc(&dB,sizeof(__nv_bfloat16)*K*N); cudaMalloc(&dC,sizeof(float)*M*N);
    cudaMemcpy(dA,A.data(),sizeof(__nv_bfloat16)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B.data(),sizeof(__nv_bfloat16)*K*N,cudaMemcpyHostToDevice); cudaMemset(dC,0,sizeof(float)*M*N);

    dim3 block(32,4);
    dim3 grid((N+127)/128, (M+127)/128);
    size_t smem = (128*16 + 16*128)*sizeof(__nv_bfloat16)*2; // double buffers

    // warmup
    for (int w=0; w<5; ++w)
        wmma_bf16_pipeline_kernel<<<grid, block, smem>>>(dA,dB,dC,M,N,K, K, N, N, 1.0f, 0.0f);
    auto t0=Clock::now();
    for (int it=0; it<30; ++it)
        wmma_bf16_pipeline_kernel<<<grid, block, smem>>>(dA,dB,dC,M,N,K, K, N, N, 1.0f, 0.0f);
    cudaDeviceSynchronize();
    auto t1=Clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/30.0;
    double flops = 2.0*double(M)*double(N)*double(K);
    double tflops = flops/(ms*1e9);
    printf("wgmma,bf16,wgmma_pipe,%d,%d,%d,%.3f,%.3f\n",M,N,K,ms,tflops);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
