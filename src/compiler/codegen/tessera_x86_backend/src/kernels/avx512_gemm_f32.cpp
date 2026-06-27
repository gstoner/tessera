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
#include <cstdint>

extern "C" void tessera_x86_avx512_gemm_f32(const float* A, const float* B,
                                            int64_t M, int64_t N, int64_t K,
                                            float* C) {
    for (int64_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        float* c = C + m * N;
        int64_t n = 0;
        for (; n + 16 <= N; n += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; ++k)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]),
                                      _mm512_loadu_ps(B + k * N + n), acc);
            _mm512_storeu_ps(c + n, acc);
        }
        if (n < N) {
            __mmask16 tail = (__mmask16)((1u << (unsigned)(N - n)) - 1u);
            __m512 acc = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; ++k)
                acc = _mm512_fmadd_ps(
                    _mm512_set1_ps(a[k]),
                    _mm512_maskz_loadu_ps(tail, B + k * N + n), acc);
            _mm512_mask_storeu_ps(c + n, tail, acc);
        }
    }
}
