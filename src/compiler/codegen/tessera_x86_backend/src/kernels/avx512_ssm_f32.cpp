// AVX-512 selective state-space-model (Mamba2) scan kernel (f32).
//
// The optimized CPU lane for the S-series `state_space` family — a single fused
// sequential scan that maintains the (B,D,N) state in place, vectorized over the
// state dimension N, rather than the numpy reference materializing a (B,D,N)
// temporary per timestep.  Per token t, channel d, state n:
//
//   A_bar = exp(delta[b,t,d]·A[d,n]) ; B_bar = delta[b,t,d]·B[b,t,n]
//   h[b,d,n] = A_bar·h[b,d,n] + B_bar·x[b,t,d]
//   y[b,t,d] = Σ_n C[b,t,n]·h[b,d,n]
//
// The n-loop is the SIMD dimension (16 f32/__m512 + scalar tail) — exp via the
// Cephes AVX-512 core (matches avx512_transcendental_f32).  `h` is in/out
// working state (init from the caller's state or zeros).  A scalar reference is
// provided alongside for on-device validation.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {
// Cephes expf core (identical to avx512_transcendental_f32::exp512).
inline __m512 exp512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));
    __m512 fx = _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f),
                                _mm512_set1_ps(0.5f));
    fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 z = _mm512_mul_ps(r, r);
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, z, _mm512_add_ps(r, one));
    return _mm512_scalef_ps(y, fx);
}
}  // namespace

// h is working state [B,D,N] (in/out). y is output [B,S,D].
extern "C" void tessera_x86_reference_selective_ssm_f32(
    const float* x, const float* A2d, const float* B, const float* C,
    const float* delta, int64_t Bsz, int64_t S, int64_t D, int64_t N,
    float* h, float* y) {
    for (int64_t b = 0; b < Bsz; ++b)
        for (int64_t t = 0; t < S; ++t)
            for (int64_t d = 0; d < D; ++d) {
                float dt = delta[(b * S + t) * D + d];
                float xt = x[(b * S + t) * D + d];
                const float* arow = A2d + d * N;
                const float* brow = B + (b * S + t) * N;
                const float* crow = C + (b * S + t) * N;
                float* hrow = h + (b * D + d) * N;
                float acc = 0.0f;
                for (int64_t n = 0; n < N; ++n) {
                    float ab = std::exp(dt * arow[n]);
                    float bb = dt * brow[n];
                    float hv = ab * hrow[n] + bb * xt;
                    hrow[n] = hv;
                    acc += crow[n] * hv;
                }
                y[(b * S + t) * D + d] = acc;
            }
}

extern "C" void tessera_x86_avx512_selective_ssm_f32(
    const float* x, const float* A2d, const float* B, const float* C,
    const float* delta, int64_t Bsz, int64_t S, int64_t D, int64_t N,
    float* h, float* y) {
    for (int64_t b = 0; b < Bsz; ++b)
        for (int64_t t = 0; t < S; ++t)
            for (int64_t d = 0; d < D; ++d) {
                float dt = delta[(b * S + t) * D + d];
                float xt = x[(b * S + t) * D + d];
                __m512 vdt = _mm512_set1_ps(dt);
                __m512 vxt = _mm512_set1_ps(xt);
                const float* arow = A2d + d * N;
                const float* brow = B + (b * S + t) * N;
                const float* crow = C + (b * S + t) * N;
                float* hrow = h + (b * D + d) * N;
                __m512 vacc = _mm512_setzero_ps();
                int64_t n = 0;
                for (; n + 16 <= N; n += 16) {
                    __m512 ab = exp512(_mm512_mul_ps(vdt, _mm512_loadu_ps(arow + n)));
                    __m512 bb = _mm512_mul_ps(vdt, _mm512_loadu_ps(brow + n));
                    __m512 hv = _mm512_fmadd_ps(ab, _mm512_loadu_ps(hrow + n),
                                                _mm512_mul_ps(bb, vxt));
                    _mm512_storeu_ps(hrow + n, hv);
                    vacc = _mm512_fmadd_ps(_mm512_loadu_ps(crow + n), hv, vacc);
                }
                float acc = _mm512_reduce_add_ps(vacc);
                for (; n < N; ++n) {
                    float ab = std::exp(dt * arow[n]);
                    float hv = ab * hrow[n] + dt * brow[n] * xt;
                    hrow[n] = hv;
                    acc += crow[n] * hv;
                }
                y[(b * S + t) * D + d] = acc;
            }
}
