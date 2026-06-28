// AVX-512 dense factorization kernels (f32) — LU (partial pivoting) + Householder
// QR, the optimized CPU lane for the S-series `linalg` general-factorization
// family (linalg PR-B).  Genuinely computes the factorization (does NOT wrap
// LAPACK).  Batched over leading dims.  Scalar references alongside.
//
//   lu : A[..,n,n] → packed LU[..,n,n] (unit-lower L below diag, U on/above) +
//        pivots[..,n] (0-based, getrf convention: piv[k]=row swapped with k).
//   qr : A[..,m,n] → full Q[..,m,m] (orthonormal) + R[..,m,n] (upper-trapezoid),
//        A = Q·R.  The runtime slices the reduced Q[:, :k], R[:k, :], k=min(m,n).

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {
// acc += f * row[:len]  (used by QR reflector application)
inline void axpy(float f, const float* row, float* acc, int64_t len) {
    __m512 vf = _mm512_set1_ps(f);
    int64_t c = 0;
    for (; c + 16 <= len; c += 16)
        _mm512_storeu_ps(acc + c, _mm512_fmadd_ps(vf, _mm512_loadu_ps(row + c),
                                                  _mm512_loadu_ps(acc + c)));
    for (; c < len; ++c) acc[c] += f * row[c];
}
}  // namespace

// ── LU with partial pivoting (getrf) ─────────────────────────────────────────
extern "C" void tessera_x86_lu_f32(const float* A, int64_t batch, int64_t n,
                                   float* LU, int32_t* piv) {
    for (int64_t b = 0; b < batch; ++b) {
        float* lu = LU + b * n * n;
        int32_t* pv = piv + b * n;
        for (int64_t i = 0; i < n * n; ++i) lu[i] = A[b * n * n + i];
        for (int64_t k = 0; k < n; ++k) {
            int64_t p = k;
            float best = std::fabs(lu[k * n + k]);
            for (int64_t i = k + 1; i < n; ++i) {
                float v = std::fabs(lu[i * n + k]);
                if (v > best) { best = v; p = i; }
            }
            pv[k] = (int32_t)p;
            if (p != k)
                for (int64_t j = 0; j < n; ++j) {
                    float t = lu[k * n + j];
                    lu[k * n + j] = lu[p * n + j];
                    lu[p * n + j] = t;
                }
            float pivval = lu[k * n + k];
            if (pivval == 0.0f) continue;
            for (int64_t i = k + 1; i < n; ++i) {
                float f = lu[i * n + k] / pivval;
                lu[i * n + k] = f;
                // rank-1 update of the trailing row (contiguous in j) — AVX-512
                axpy(-f, lu + k * n + (k + 1), lu + i * n + (k + 1), n - (k + 1));
            }
        }
    }
}

// ── Householder QR (full Q[m,m], R[m,n]) ─────────────────────────────────────
extern "C" void tessera_x86_qr_f32(const float* A, int64_t batch, int64_t m,
                                   int64_t n, float* Q, float* R) {
    int64_t k = m < n ? m : n;
    // Reflector scratch — allocated ONCE (size is loop-invariant); never alloca
    // inside the batch loop (would grow the stack per iteration).
    float* v = (float*)__builtin_alloca(sizeof(float) * (m > 0 ? m : 1));
    for (int64_t b = 0; b < batch; ++b) {
        float* q = Q + b * m * m;
        float* r = R + b * m * n;
        for (int64_t i = 0; i < m * n; ++i) r[i] = A[b * m * n + i];
        for (int64_t i = 0; i < m * m; ++i) q[i] = 0.0f;
        for (int64_t i = 0; i < m; ++i) q[i * m + i] = 1.0f;
        for (int64_t j = 0; j < k; ++j) {
            float norm = 0.0f;
            for (int64_t i = j; i < m; ++i) norm += r[i * n + j] * r[i * n + j];
            norm = std::sqrt(norm);
            if (norm == 0.0f) continue;
            float alpha = (r[j * n + j] >= 0.0f) ? -norm : norm;
            for (int64_t i = j; i < m; ++i) v[i] = r[i * n + j];
            v[j] -= alpha;
            float vtv = 0.0f;
            for (int64_t i = j; i < m; ++i) vtv += v[i] * v[i];
            if (vtv == 0.0f) continue;
            float beta = 2.0f / vtv;
            // R[j:, c] -= beta * (vᵀ R[j:, c]) * v   for each column c
            for (int64_t c = j; c < n; ++c) {
                float dot = 0.0f;
                for (int64_t i = j; i < m; ++i) dot += v[i] * r[i * n + c];
                float f = beta * dot;
                for (int64_t i = j; i < m; ++i) r[i * n + c] -= f * v[i];
            }
            // Q[r, j:] -= beta * (Q[r, j:] · v) * v   for each row r (right-mult;
            // contiguous in i → AVX-512)
            for (int64_t rr = 0; rr < m; ++rr) {
                float dot = 0.0f;
                for (int64_t i = j; i < m; ++i) dot += q[rr * m + i] * v[i];
                axpy(-(beta * dot), v + j, q + rr * m + j, m - j);
            }
        }
        // clean the strict-lower part of the leading k columns of R
        for (int64_t c = 0; c < k; ++c)
            for (int64_t i = c + 1; i < m; ++i) r[i * n + c] = 0.0f;
    }
}
