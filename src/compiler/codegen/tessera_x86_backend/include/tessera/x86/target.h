#pragma once
#include <cstdint>
#include <cstddef>

namespace tessera {
namespace x86 {

// Minimal backend interface for integration.
// Adapt this to your compiler's real Backend interface.
struct X86BackendConfig {
    bool preferAMX = true;
    bool preferAVX512 = true;
};

class X86Backend {
public:
    explicit X86Backend(const X86BackendConfig& cfg = {}) : cfg_(cfg) {}
    bool amxAvailable() const noexcept;
    bool avx512BF16Available() const noexcept;

    // BF16 GEMM (C = A @ B + (beta?C:0)), row-major tiles, simple multiples of (16x64)
    // A: MxK (bf16), B: KxN (bf16), C: MxN (fp32)
    void gemm_bf16(const uint16_t* A, const uint16_t* B, float* C,
                   int M, int N, int K, float beta) const;

private:
    X86BackendConfig cfg_;
};

// C APIs for easy FFI
extern "C" {
bool tessera_x86_amx_available();
bool tessera_x86_avx512bf16_available();
void tessera_x86_amx_gemm_bf16(const uint16_t* A, const uint16_t* B, float* C,
                               int M, int N, int K, float beta);
void tessera_x86_avx512_gemm_bf16(const uint16_t* A, const uint16_t* B, float* C,
                                  int M, int N, int K, float beta);
}

} // namespace x86
} // namespace tessera


// Optional epilogue kinds for FP32 outputs
enum EpilogueKind : int {
    EPILOGUE_NONE = 0,
    EPILOGUE_BIAS = 1,
    EPILOGUE_BIAS_GELU = 2
};

// INT8 GEMMs
extern "C" {
// AMX signed* signed -> s32
void tessera_x86_amx_gemm_s8s8_s32(const int8_t* A, const int8_t* B, int32_t* C,
                                   int M, int N, int K, int beta);

// AVX-512 VNNI unsigned* signed -> s32
void tessera_x86_avx512_vnni_gemm_u8s8_s32(const uint8_t* A, const int8_t* B, int32_t* C,
                                           int M, int N, int K, int beta);

// FP32 epilogues for bias/GELU (row-major MxN)
void tessera_x86_epilogue_bias_fp32(float* C, const float* bias, int M, int N);
void tessera_x86_epilogue_bias_gelu_fp32(float* C, const float* bias, int M, int N);
}
