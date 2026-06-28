// AVX-512 dense linear-algebra kernels (f32) — Cholesky factorization +
// triangular solve, the optimized CPU lane for the S-series `linalg` SPD /
// triangular family.  Genuinely computes the factorization / substitution (does
// NOT wrap LAPACK).  Batched over leading dims; the M right-hand-side columns of
// the solve are the SIMD dimension.  A scalar reference is provided alongside.
//
//   cholesky  : A[..,n,n] SPD → L[..,n,n] lower, A = L·Lᵀ
//               (Cholesky–Banachiewicz; upper triangle zeroed)
//   tri_solve : A[..,n,n] (triangular part used) · X = B[..,n,m]
//               lower → forward substitution, upper → back substitution

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {
// x[:m] = (b[:m] - acc[:m]) / piv, vectorized over the m RHS columns.
inline void scale_row(const float* b, float* acc, float piv, int64_t m,
                      float* x) {
    __m512 vp = _mm512_set1_ps(1.0f / piv);
    int64_t c = 0;
    for (; c + 16 <= m; c += 16) {
        __m512 r = _mm512_sub_ps(_mm512_loadu_ps(b + c), _mm512_loadu_ps(acc + c));
        _mm512_storeu_ps(x + c, _mm512_mul_ps(r, vp));
    }
    for (; c < m; ++c) x[c] = (b[c] - acc[c]) / piv;
}
// acc[:m] += f * row[:m]
inline void axpy_row(float f, const float* row, float* acc, int64_t m) {
    __m512 vf = _mm512_set1_ps(f);
    int64_t c = 0;
    for (; c + 16 <= m; c += 16)
        _mm512_storeu_ps(acc + c, _mm512_fmadd_ps(vf, _mm512_loadu_ps(row + c),
                                                  _mm512_loadu_ps(acc + c)));
    for (; c < m; ++c) acc[c] += f * row[c];
}
}  // namespace

extern "C" void tessera_x86_cholesky_f32(const float* A, int64_t batch,
                                         int64_t n, float* L) {
    for (int64_t b = 0; b < batch; ++b) {
        const float* a = A + b * n * n;
        float* l = L + b * n * n;
        for (int64_t i = 0; i < n * n; ++i) l[i] = 0.0f;
        for (int64_t j = 0; j < n; ++j) {
            float s = a[j * n + j];
            for (int64_t k = 0; k < j; ++k) s -= l[j * n + k] * l[j * n + k];
            float ljj = std::sqrt(s);
            l[j * n + j] = ljj;
            for (int64_t i = j + 1; i < n; ++i) {
                float t = a[i * n + j];
                for (int64_t k = 0; k < j; ++k) t -= l[i * n + k] * l[j * n + k];
                l[i * n + j] = t / ljj;
            }
        }
    }
}

// lower != 0 → forward substitution; lower == 0 → back substitution.
extern "C" void tessera_x86_tri_solve_f32(const float* A, const float* B,
                                          int64_t batch, int64_t n, int64_t m,
                                          int lower, float* X) {
    // Per-row accumulator — allocated ONCE (size is loop-invariant); never alloca
    // inside the batch loop (would grow the stack per iteration).
    float* acc = (float*)__builtin_alloca(sizeof(float) * (m > 0 ? m : 1));
    for (int64_t b = 0; b < batch; ++b) {
        const float* a = A + b * n * n;
        const float* bb = B + b * n * m;
        float* x = X + b * n * m;
        if (lower) {
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t c = 0; c < m; ++c) acc[c] = 0.0f;
                for (int64_t k = 0; k < i; ++k)
                    axpy_row(a[i * n + k], x + k * m, acc, m);
                scale_row(bb + i * m, acc, a[i * n + i], m, x + i * m);
            }
        } else {
            for (int64_t i = n - 1; i >= 0; --i) {
                for (int64_t c = 0; c < m; ++c) acc[c] = 0.0f;
                for (int64_t k = i + 1; k < n; ++k)
                    axpy_row(a[i * n + k], x + k * m, acc, m);
                scale_row(bb + i * m, acc, a[i * n + i], m, x + i * m);
            }
        }
    }
}
