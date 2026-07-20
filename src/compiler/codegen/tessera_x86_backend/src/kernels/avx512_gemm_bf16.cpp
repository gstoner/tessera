#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

static inline float bf16_to_float(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

extern "C" void tessera_x86_reference_gemm_bf16(
    const uint16_t* A, const uint16_t* B, float* C,
    int M, int N, int K, float beta) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = beta == 0.0f ? 0.0f : beta * C[size_t(m) * N + n];
            for (int k = 0; k < K; ++k)
                acc += bf16_to_float(A[size_t(m) * K + k]) *
                       bf16_to_float(B[size_t(k) * N + n]);
            C[size_t(m) * N + n] = acc;
        }
    }
}

// Row-major BF16 x BF16 -> FP32. VDPBF16PS consumes adjacent BF16 pairs;
// packing B pairs explicitly keeps the ABI correct for arbitrary N/K tails.
extern "C" void tessera_x86_avx512_gemm_bf16(
    const uint16_t* A, const uint16_t* B, float* C,
    int M, int N, int K, float beta) {
#if !defined(__AVX512BF16__)
    tessera_x86_reference_gemm_bf16(A, B, C, M, N, K, beta);
#else
    alignas(64) uint32_t bPairs[16];
    const __mmask16 full = 0xffff;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 16) {
            int width = N - n < 16 ? N - n : 16;
            __mmask16 mask = width == 16 ? full : __mmask16((1u << width) - 1u);
            __m512 acc = beta == 0.0f
                ? _mm512_setzero_ps()
                : _mm512_mul_ps(
                    _mm512_maskz_loadu_ps(mask, C + size_t(m) * N + n),
                    _mm512_set1_ps(beta));
            for (int k = 0; k < K; k += 2) {
                uint16_t a0 = A[size_t(m) * K + k];
                uint16_t a1 = k + 1 < K ? A[size_t(m) * K + k + 1] : 0;
                uint32_t aPair = uint32_t(a0) | (uint32_t(a1) << 16);
                for (int lane = 0; lane < width; ++lane) {
                    uint16_t b0 = B[size_t(k) * N + n + lane];
                    uint16_t b1 = k + 1 < K ? B[size_t(k + 1) * N + n + lane] : 0;
                    bPairs[lane] = uint32_t(b0) | (uint32_t(b1) << 16);
                }
                for (int lane = width; lane < 16; ++lane) bPairs[lane] = 0;
                __m512bh av = (__m512bh)_mm512_set1_epi32(int(aPair));
                __m512bh bv = (__m512bh)_mm512_load_si512((const void*)bPairs);
                acc = _mm512_dpbf16_ps(acc, av, bv);
            }
            _mm512_mask_storeu_ps(C + size_t(m) * N + n, mask, acc);
        }
    }
#endif
}
