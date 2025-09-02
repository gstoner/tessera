#pragma once
#include <cstdint>
#include <cstddef>

namespace tessera { namespace gpu {

struct BackendConfig {
    int sm = 0;       // computed at runtime
    bool preferNVVM = true;
    bool preferPTX = true;
};

class TesseraGpuBackend {
public:
    TesseraGpuBackend();
    ~TesseraGpuBackend();

    bool cudaAvailable() const noexcept;
    int  smVersion() const noexcept;

    // Launch utility wrappers
    void wmma_gemm_fp16(const __half* A, const __half* B, float* C,
                        int M, int N, int K, float alpha, float beta);
    void wmma_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                        int M, int N, int K, float alpha, float beta);

private:
    void* impl_;
};

extern "C" {
// C APIs for FFI
bool tessera_gpu_available();
int  tessera_gpu_sm();
void tessera_gpu_wmma_gemm_fp16(const __half* A, const __half* B, float* C,
                                int M, int N, int K, float alpha, float beta);
void tessera_gpu_wmma_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, float* C,
                                int M, int N, int K, float alpha, float beta);
}

}} // namespace
