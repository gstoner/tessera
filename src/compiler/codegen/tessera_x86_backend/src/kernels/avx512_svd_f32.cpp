// AVX-512 SVD kernel (f32) — one-sided Jacobi, the optimized CPU lane for the
// S-series `linalg` SVD.  Genuinely computes the decomposition (does NOT wrap
// LAPACK).  Requires m >= n (the runtime transposes the wide case).  Batched.
//
//   A[m,n] = U[m,n] · diag(S[n]) · Vᵀ[n,n]
//
// One-sided Jacobi orthogonalizes the columns of a working copy by sweeps of
// 2×2 column rotations; afterwards S = column norms, U = normalized columns, V =
// accumulated rotations, sorted by descending singular value.  The working copy
// is stored COLUMN-MAJOR so the per-column dot products and rotations are
// contiguous (AVX-512); that layout also makes Vᵀ a direct copy of the V buffer.
// A scalar reference is provided alongside.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kSweeps = 30;

inline float dotc(const float* a, const float* b, int64_t len) {
    __m512 acc = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= len; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float s = _mm512_reduce_add_ps(acc);
    for (; i < len; ++i) s += a[i] * b[i];
    return s;
}
// (col_p, col_q) <- (c*p - s*q, s*p + c*q), contiguous over len.
inline void rot(float* p, float* q, float c, float s, int64_t len) {
    __m512 vc = _mm512_set1_ps(c), vs = _mm512_set1_ps(s);
    int64_t i = 0;
    for (; i + 16 <= len; i += 16) {
        __m512 x = _mm512_loadu_ps(p + i), y = _mm512_loadu_ps(q + i);
        _mm512_storeu_ps(p + i, _mm512_sub_ps(_mm512_mul_ps(vc, x),
                                              _mm512_mul_ps(vs, y)));
        _mm512_storeu_ps(q + i, _mm512_add_ps(_mm512_mul_ps(vs, x),
                                              _mm512_mul_ps(vc, y)));
    }
    for (; i < len; ++i) {
        float x = p[i], y = q[i];
        p[i] = c * x - s * y;
        q[i] = s * x + c * y;
    }
}

void svd_one(const float* A, int64_t m, int64_t n, float* U, float* S,
             float* Vt, float* uc, float* vc) {
    // uc: column-major working A [n*m]; vc: column-major V [n*n]
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i) uc[j * m + i] = A[i * n + j];
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i) vc[j * n + i] = (i == j) ? 1.0f : 0.0f;

    for (int sweep = 0; sweep < kSweeps; ++sweep)
        for (int64_t p = 0; p < n; ++p)
            for (int64_t q = p + 1; q < n; ++q) {
                float* cp = uc + p * m;
                float* cq = uc + q * m;
                float app = dotc(cp, cp, m);
                float aqq = dotc(cq, cq, m);
                float apq = dotc(cp, cq, m);
                if (std::fabs(apq) <= 1e-12f * std::sqrt(app * aqq)) continue;
                float tau = (aqq - app) / (2.0f * apq);
                float t = (tau >= 0.0f ? 1.0f : -1.0f) /
                          (std::fabs(tau) + std::sqrt(tau * tau + 1.0f));
                float c = 1.0f / std::sqrt(t * t + 1.0f);
                float s = t * c;
                rot(cp, cq, c, s, m);
                rot(vc + p * n, vc + q * n, c, s, n);
            }

    for (int64_t j = 0; j < n; ++j)
        S[j] = std::sqrt(dotc(uc + j * m, uc + j * m, m));
    // selection sort columns by descending singular value
    for (int64_t i = 0; i < n; ++i) {
        int64_t mx = i;
        for (int64_t j = i + 1; j < n; ++j)
            if (S[j] > S[mx]) mx = j;
        if (mx != i) {
            float ts = S[i]; S[i] = S[mx]; S[mx] = ts;
            for (int64_t r = 0; r < m; ++r) {
                float tmp = uc[i * m + r]; uc[i * m + r] = uc[mx * m + r];
                uc[mx * m + r] = tmp;
            }
            for (int64_t r = 0; r < n; ++r) {
                float tmp = vc[i * n + r]; vc[i * n + r] = vc[mx * n + r];
                vc[mx * n + r] = tmp;
            }
        }
    }
    // U = normalized columns (row-major out); Vt = V buffer (already Vᵀ layout)
    for (int64_t j = 0; j < n; ++j) {
        float inv = S[j] > 1e-20f ? 1.0f / S[j] : 0.0f;
        for (int64_t i = 0; i < m; ++i) U[i * n + j] = uc[j * m + i] * inv;
    }
    for (int64_t i = 0; i < n * n; ++i) Vt[i] = vc[i];
}
}  // namespace

extern "C" void tessera_x86_svd_f32(const float* A, int64_t batch, int64_t m,
                                    int64_t n, float* U, float* S, float* Vt) {
    int64_t uc_sz = n * m, vc_sz = n * n;
    float* uc = (float*)__builtin_alloca(sizeof(float) * (uc_sz > 0 ? uc_sz : 1));
    float* vc = (float*)__builtin_alloca(sizeof(float) * (vc_sz > 0 ? vc_sz : 1));
    for (int64_t b = 0; b < batch; ++b)
        svd_one(A + b * m * n, m, n, U + b * m * n, S + b * n, Vt + b * n * n,
                uc, vc);
}
