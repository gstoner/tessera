// AVX-512 sparse-linear-algebra kernels (f32) for the Tessera x86 backend.
//
// The optimized CPU lane for the S-series `sparse` family — genuinely sparse
// kernels (they iterate the nonzero structure, NOT densify-then-GEMM), the
// AVX-512 analog of the GEMM lane so the sparse primitives get a REAL vectorized
// CPU kernel rather than only the numpy reference. A scalar reference is provided
// alongside each for on-device validation.
//
//   spmm_csr : C[M,N] = A_csr[M,K] @ B[K,N].  Row-wise SpMM — for each row i and
//              each nonzero (col, val) in that row, AXPY  C[i,:] += val·B[col,:].
//              The length-N inner AXPY is the SIMD dimension (16 f32/__m512 +
//              scalar tail).  indptr/indices are i32.
//   sddmm    : OUT[M,N] = (A[M,K] @ Bt[N,K]^row) ⊙ mask[M,N].  Sampled dense-
//              dense — only entries with mask≠0 do the length-K dot product
//              (B is passed ROW-MAJOR-TRANSPOSED as Bt[N,K] so both operands of
//              the dot are contiguous → vectorizable AVX-512 FMA reduction).

#include <immintrin.h>
#include <cstdint>

namespace {
inline float hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }
}  // namespace

// ── SpMM (CSR) ──────────────────────────────────────────────────────────────
extern "C" void tessera_x86_reference_spmm_csr_f32(
    const int32_t* indptr, const int32_t* indices, const float* values,
    const float* B, int64_t M, int64_t N, float* C) {
    for (int64_t i = 0; i < M; ++i) {
        float* crow = C + i * N;
        for (int64_t n = 0; n < N; ++n) crow[n] = 0.0f;
        for (int32_t p = indptr[i]; p < indptr[i + 1]; ++p) {
            float v = values[p];
            const float* brow = B + (int64_t)indices[p] * N;
            for (int64_t n = 0; n < N; ++n) crow[n] += v * brow[n];
        }
    }
}

extern "C" void tessera_x86_avx512_spmm_csr_f32(
    const int32_t* indptr, const int32_t* indices, const float* values,
    const float* B, int64_t M, int64_t N, float* C) {
    for (int64_t i = 0; i < M; ++i) {
        float* crow = C + i * N;
        int64_t n = 0;
        for (; n + 16 <= N; n += 16)
            _mm512_storeu_ps(crow + n, _mm512_setzero_ps());
        for (; n < N; ++n) crow[n] = 0.0f;
        for (int32_t p = indptr[i]; p < indptr[i + 1]; ++p) {
            __m512 vv = _mm512_set1_ps(values[p]);
            const float* brow = B + (int64_t)indices[p] * N;
            n = 0;
            for (; n + 16 <= N; n += 16) {
                __m512 acc = _mm512_loadu_ps(crow + n);
                acc = _mm512_fmadd_ps(vv, _mm512_loadu_ps(brow + n), acc);
                _mm512_storeu_ps(crow + n, acc);
            }
            float v = values[p];
            for (; n < N; ++n) crow[n] += v * brow[n];
        }
    }
}

// ── SDDMM ───────────────────────────────────────────────────────────────────
extern "C" void tessera_x86_reference_sddmm_f32(
    const float* A, const float* Bt, const float* mask, int64_t M, int64_t N,
    int64_t K, float* out) {
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j) {
            float m = mask[i * N + j];
            if (m == 0.0f) { out[i * N + j] = 0.0f; continue; }
            const float* a = A + i * K;
            const float* b = Bt + j * K;
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += a[k] * b[k];
            out[i * N + j] = acc * m;
        }
}

extern "C" void tessera_x86_avx512_sddmm_f32(
    const float* A, const float* Bt, const float* mask, int64_t M, int64_t N,
    int64_t K, float* out) {
    for (int64_t i = 0; i < M; ++i) {
        const float* a = A + i * K;
        for (int64_t j = 0; j < N; ++j) {
            float m = mask[i * N + j];
            if (m == 0.0f) { out[i * N + j] = 0.0f; continue; }
            const float* b = Bt + j * K;
            __m512 acc = _mm512_setzero_ps();
            int64_t k = 0;
            for (; k + 16 <= K; k += 16)
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + k),
                                      _mm512_loadu_ps(b + k), acc);
            float s = hsum512(acc);
            for (; k < K; ++k) s += a[k] * b[k];
            out[i * N + j] = s * m;
        }
    }
}
