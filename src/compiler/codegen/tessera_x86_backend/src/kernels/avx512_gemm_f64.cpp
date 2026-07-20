#include <immintrin.h>
#include <cstddef>
#include <cstdint>

extern "C" void tessera_x86_reference_gemm_f64(
    const double* A, const double* B, int64_t M, int64_t N, int64_t K,
    double* C) {
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k)
                acc += A[size_t(m) * K + k] * B[size_t(k) * N + n];
            C[size_t(m) * N + n] = acc;
        }
}

extern "C" void tessera_x86_avx512_gemm_f64(
    const double* A, const double* B, int64_t M, int64_t N, int64_t K,
    double* C) {
    const __mmask8 full = 0xff;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; n += 8) {
            int64_t remaining = N - n;
            __mmask8 mask = remaining >= 8 ? full : __mmask8((1u << remaining) - 1u);
            __m512d acc = _mm512_setzero_pd();
            for (int64_t k = 0; k < K; ++k) {
                __m512d b = _mm512_maskz_loadu_pd(mask, B + size_t(k) * N + n);
                acc = _mm512_fmadd_pd(_mm512_set1_pd(A[size_t(m) * K + k]), b, acc);
            }
            _mm512_mask_storeu_pd(C + size_t(m) * N + n, mask, acc);
        }
    }
}
