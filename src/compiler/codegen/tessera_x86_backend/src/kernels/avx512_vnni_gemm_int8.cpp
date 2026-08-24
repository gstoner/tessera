#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include "tessera/Rank2Index.h"

using tessera::layout::linearIndex2D;
using tessera::layout::Rank2Order;

extern "C" void tessera_x86_reference_gemm_u8s8_s32(
    const uint8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K, int beta) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const auto outputIndex =
                linearIndex2D<Rank2Order::RowMajor>(m, n, N);
            int32_t acc = beta == 0 ? 0 : C[outputIndex] * beta;
            for (int k = 0; k < K; ++k)
                acc += int32_t(
                           A[linearIndex2D<Rank2Order::RowMajor>(m, k, K)]) *
                       int32_t(
                           B[linearIndex2D<Rank2Order::RowMajor>(k, n, N)]);
            C[outputIndex] = acc;
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
                    _mm512_maskz_loadu_epi32(
                        mask, C + linearIndex2D<Rank2Order::RowMajor>(m, n, N)),
                    _mm512_set1_epi32(beta));
            for (int k = 0; k < K; k += 4) {
                uint32_t aQuad = 0;
                for (int t = 0; t < 4 && k + t < K; ++t)
                    aQuad |= uint32_t(A[linearIndex2D<Rank2Order::RowMajor>(
                                 m, k + t, K)])
                             << (8 * t);
                for (int lane = 0; lane < width; ++lane) {
                    uint32_t packed = 0;
                    for (int t = 0; t < 4 && k + t < K; ++t)
                        packed |= uint32_t(uint8_t(
                                      B[linearIndex2D<Rank2Order::RowMajor>(
                                          k + t, n + lane, N)]))
                                  << (8 * t);
                    bQuads[lane] = packed;
                }
                for (int lane = width; lane < 16; ++lane) bQuads[lane] = 0;
                acc = _mm512_dpbusd_epi32(
                    acc, _mm512_set1_epi32(int(aQuad)),
                    _mm512_load_si512((const void*)bQuads));
            }
            _mm512_mask_storeu_epi32(
                C + linearIndex2D<Rank2Order::RowMajor>(m, n, N), mask, acc);
        }
    }
#endif
}
