#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "tessera/Rank2Index.h"

using tessera::layout::linearIndex2D;
using tessera::layout::Rank2Order;

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
            const auto outputIndex =
                linearIndex2D<Rank2Order::RowMajor>(m, n, N);
            float acc = beta == 0.0f ? 0.0f : beta * C[outputIndex];
            for (int k = 0; k < K; ++k)
                acc += bf16_to_float(
                           A[linearIndex2D<Rank2Order::RowMajor>(m, k, K)]) *
                       bf16_to_float(
                           B[linearIndex2D<Rank2Order::RowMajor>(k, n, N)]);
            C[outputIndex] = acc;
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
                    _mm512_maskz_loadu_ps(
                        mask, C + linearIndex2D<Rank2Order::RowMajor>(m, n, N)),
                    _mm512_set1_ps(beta));
            for (int k = 0; k < K; k += 2) {
                uint16_t a0 =
                    A[linearIndex2D<Rank2Order::RowMajor>(m, k, K)];
                uint16_t a1 = k + 1 < K
                                  ? A[linearIndex2D<Rank2Order::RowMajor>(
                                        m, k + 1, K)]
                                  : 0;
                uint32_t aPair = uint32_t(a0) | (uint32_t(a1) << 16);
                for (int lane = 0; lane < width; ++lane) {
                    uint16_t b0 = B[linearIndex2D<Rank2Order::RowMajor>(
                        k, n + lane, N)];
                    uint16_t b1 = k + 1 < K
                                      ? B[linearIndex2D<Rank2Order::RowMajor>(
                                            k + 1, n + lane, N)]
                                      : 0;
                    bPairs[lane] = uint32_t(b0) | (uint32_t(b1) << 16);
                }
                for (int lane = width; lane < 16; ++lane) bPairs[lane] = 0;
                __m512bh av = (__m512bh)_mm512_set1_epi32(int(aPair));
                __m512bh bv = (__m512bh)_mm512_load_si512((const void*)bPairs);
                acc = _mm512_dpbf16_ps(acc, av, bv);
            }
            _mm512_mask_storeu_ps(
                C + linearIndex2D<Rank2Order::RowMajor>(m, n, N), mask, acc);
        }
    }
#endif
}
