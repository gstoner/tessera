#include <immintrin.h>
#include <cstddef>
#include <cstdint>

extern "C" void tessera_x86_reference_gemm_u8s8_s32(
    const uint8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, int beta) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = beta == 0 ? 0 : C[size_t(m) * N + n] * beta;
            for (int k = 0; k < K; ++k)
                acc += int32_t(A[size_t(m) * K + k]) *
                       int32_t(B[size_t(k) * N + n]);
            C[size_t(m) * N + n] = acc;
        }
    }
}

// Row-major U8 x S8 -> S32. VPDPBUSD consumes four byte products per dword;
// explicit lane packing preserves matrix semantics for arbitrary shapes.
extern "C" void tessera_x86_avx512_vnni_gemm_u8s8_s32(
    const uint8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, int beta) {
#if !defined(__AVX512VNNI__)
    tessera_x86_reference_gemm_u8s8_s32(A, B, C, M, N, K, beta);
#else
    alignas(64) uint32_t bQuads[16];
    const __mmask16 full = 0xffff;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 16) {
            int width = N - n < 16 ? N - n : 16;
            __mmask16 mask = width == 16 ? full : __mmask16((1u << width) - 1u);
            __m512i acc = beta == 0
                ? _mm512_setzero_si512()
                : _mm512_mullo_epi32(
                    _mm512_maskz_loadu_epi32(mask, C + size_t(m) * N + n),
                    _mm512_set1_epi32(beta));
            for (int k = 0; k < K; k += 4) {
                uint32_t aQuad = 0;
                for (int t = 0; t < 4 && k + t < K; ++t)
                    aQuad |= uint32_t(A[size_t(m) * K + k + t]) << (8 * t);
                for (int lane = 0; lane < width; ++lane) {
                    uint32_t packed = 0;
                    for (int t = 0; t < 4 && k + t < K; ++t)
                        packed |= uint32_t(uint8_t(B[size_t(k + t) * N + n + lane])) << (8 * t);
                    bQuads[lane] = packed;
                }
                for (int lane = width; lane < 16; ++lane) bQuads[lane] = 0;
                acc = _mm512_dpbusd_epi32(
                    acc, _mm512_set1_epi32(int(aQuad)),
                    _mm512_load_si512((const void*)bQuads));
            }
            _mm512_mask_storeu_epi32(C + size_t(m) * N + n, mask, acc);
        }
    }
#endif
}
