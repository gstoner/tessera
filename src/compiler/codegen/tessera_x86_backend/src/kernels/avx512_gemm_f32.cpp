// AVX-512 f32 GEMM microkernel for the Tessera x86 backend — the ctypes-loadable
// C = A[M,K] @ B[K,N] (row-major, f32) that the runtime matmul-family lane
// builds on (batched_gemm / linear_general / einsum / attention all compose
// around this 2D GEMM in Python, mirroring the ROCm WMMA-family executor).
//
// The bf16/AMX GEMM lives in the static backend lib; this is the pure-f32
// AVX-512 path exposed in libtessera_x86_elementwise.so. Vectorizes over N (16
// f32 lanes/__m512), accumulating over K with a broadcast of A[m,k] and an FMA
// — a real vectorized GEMM (not a scalar triple loop). N % 16 tail via mask.
// f32 accumulate; matches numpy matmul to a K-scaled tolerance.

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include "tessera/Rank2Index.h"

using tessera::layout::linearIndex2D;
using tessera::layout::Rank2Order;

extern "C" void tessera_x86_avx512_gemm_f32(const float* A, const float* B,
                                            int64_t M, int64_t N, int64_t K,
                                            float* C) {
    for (int64_t m = 0; m < M; ++m) {
        const float* a = A + linearIndex2D<Rank2Order::RowMajor>(m, 0, K);
        float* c = C + linearIndex2D<Rank2Order::RowMajor>(m, 0, N);
        int64_t n = 0;
        for (; n + 16 <= N; n += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; ++k)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]),
                                      _mm512_loadu_ps(
                                          B + linearIndex2D<Rank2Order::RowMajor>(k, n, N)),
                                      acc);
            _mm512_storeu_ps(c + n, acc);
        }
        if (n < N) {
            __mmask16 tail = (__mmask16)((1u << (unsigned)(N - n)) - 1u);
            __m512 acc = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; ++k)
                acc = _mm512_fmadd_ps(
                    _mm512_set1_ps(a[k]),
                    _mm512_maskz_loadu_ps(
                        tail, B + linearIndex2D<Rank2Order::RowMajor>(k, n, N)),
                    acc);
            _mm512_mask_storeu_ps(c + n, tail, acc);
        }
    }
}

// T1 evidence ABI: the same AVX-512 microkernel with explicit compiler-owned
// M/N/K blocking.  This is intentionally separate from the promoted ABI until
// measured rank correlation shows that the cache model can select among these
// physically distinct loop nests.  BN is rounded to whole AVX-512 vectors;
// ragged N remains mask-safe.
extern "C" int tessera_x86_avx512_gemm_f32_tiled(
    const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C,
    int64_t BM, int64_t BN, int64_t BK) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 || BM <= 0 || BN <= 0 ||
        BK <= 0)
        return 1;
    BN = ((BN + 15) / 16) * 16;
    for (int64_t m0 = 0; m0 < M; m0 += BM) {
        const int64_t mEnd = std::min(m0 + BM, M);
        for (int64_t n0 = 0; n0 < N; n0 += BN) {
            const int64_t nEnd = std::min(n0 + BN, N);
            for (int64_t m = m0; m < mEnd; ++m) {
                const float* a =
                    A + linearIndex2D<Rank2Order::RowMajor>(m, 0, K);
                float* c =
                    C + linearIndex2D<Rank2Order::RowMajor>(m, 0, N);
                int64_t n = n0;
                for (; n + 16 <= nEnd; n += 16) {
                    __m512 acc = _mm512_setzero_ps();
                    for (int64_t k0 = 0; k0 < K; k0 += BK) {
                        const int64_t kEnd = std::min(k0 + BK, K);
                        for (int64_t k = k0; k < kEnd; ++k)
                            acc = _mm512_fmadd_ps(
                                _mm512_set1_ps(a[k]),
                                _mm512_loadu_ps(
                                    B + linearIndex2D<Rank2Order::RowMajor>(k, n, N)),
                                acc);
                    }
                    _mm512_storeu_ps(c + n, acc);
                }
                if (n < nEnd) {
                    const unsigned width = static_cast<unsigned>(nEnd - n);
                    const __mmask16 tail =
                        width == 16 ? static_cast<__mmask16>(0xffffu)
                                    : static_cast<__mmask16>((1u << width) - 1u);
                    __m512 acc = _mm512_setzero_ps();
                    for (int64_t k0 = 0; k0 < K; k0 += BK) {
                        const int64_t kEnd = std::min(k0 + BK, K);
                        for (int64_t k = k0; k < kEnd; ++k)
                            acc = _mm512_fmadd_ps(
                                _mm512_set1_ps(a[k]),
                                _mm512_maskz_loadu_ps(
                                    tail,
                                    B + linearIndex2D<Rank2Order::RowMajor>(k, n, N)),
                                acc);
                    }
                    _mm512_mask_storeu_ps(c + n, tail, acc);
                }
            }
        }
    }
    return 0;
}
