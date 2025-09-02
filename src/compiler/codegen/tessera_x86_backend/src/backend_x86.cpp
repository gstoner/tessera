#include "tessera/x86/target.h"
#include "tessera/x86/amx_runtime.h"
#include <immintrin.h>

namespace tessera { namespace x86 {

bool X86Backend::amxAvailable() const noexcept {
    return tessera_x86_amx_supported() && tessera_x86_amx_enable_linux();
}

bool X86Backend::avx512BF16Available() const noexcept {
#if defined(__AVX512BF16__)
    return true;
#else
    // Fallback: we can emulate BF16 multiply to FP32 using AVX-512 if needed.
    return false;
#endif
}

void X86Backend::gemm_bf16(const uint16_t* A, const uint16_t* B, float* C,
                           int M, int N, int K, float beta) const {
    if (cfg_.preferAMX && amxAvailable()) {
        tessera_x86_amx_gemm_bf16(A,B,C,M,N,K,beta);
    } else {
        tessera_x86_avx512_gemm_bf16(A,B,C,M,N,K,beta);
    }
}

extern "C" {

bool tessera_x86_amx_available() {
    return tessera_x86_amx_supported() && tessera_x86_amx_enable_linux();
}

bool tessera_x86_avx512bf16_available() {
#if defined(__AVX512BF16__)
    return true;
#else
    return false;
#endif
}

} // extern "C"

}} // namespace
