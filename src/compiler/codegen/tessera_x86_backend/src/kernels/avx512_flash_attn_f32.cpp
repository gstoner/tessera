// Flash-attention forward kernel (f32) for the Tessera x86 backend — the
// attention lane (P10 of S_SERIES_GAP_CLOSURE_PLAN), the AVX-512 *partner* to
// the shipped ROCm WMMA flash_attn. FA-style streaming/online softmax: for each
// query row we sweep the keys once, keeping a running max `m`, running
// denominator `l`, and an accumulator `acc[d]`, rescaling on each new max — so
// the full S×S score matrix is never materialized (O(d) state per query).
//
//   for each (bh, query i):
//     m = -inf, l = 0, acc[0..d) = 0
//     for each valid key j:                  // causal: j <= i + max(sk-sq,0)
//       s = scale * <Q_i, K_j>
//       m' = max(m, s);  c = exp(m - m')
//       l = l*c + exp(s - m');  acc = acc*c + exp(s - m')*V_j
//       m = m'
//     O_i = acc / l
//
// The two hot inner loops over the head dim — the QK dot and the acc += p·V_j
// update — run as AVX-512 FMA over 16 lanes + a scalar tail (head dim need not
// be a multiple of 16, unlike the WMMA lane). f32 throughout (the softmax is
// f32 on every backend). Matches the dense reference (tessera.flash_attn with
// scale + causal). Q/K/V/O are [bh, S, d] row-major.

#include <cstdint>
#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {

// <a, b> over `d` f32 lanes.
inline float dot_f32(const float* a, const float* b, int64_t d) {
    int64_t k = 0;
    float acc = 0.0f;
#if defined(__AVX512F__)
    __m512 v = _mm512_setzero_ps();
    for (; k + 16 <= d; k += 16)
        v = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), v);
    acc = _mm512_reduce_add_ps(v);
#endif
    for (; k < d; ++k)
        acc += a[k] * b[k];
    return acc;
}

// acc[0..d) = acc * c + p * v[0..d)   (rescale-and-accumulate).
inline void axpby_f32(float* acc, float c, float p, const float* v, int64_t d) {
    int64_t k = 0;
#if defined(__AVX512F__)
    const __m512 vc = _mm512_set1_ps(c);
    const __m512 vp = _mm512_set1_ps(p);
    for (; k + 16 <= d; k += 16) {
        __m512 a = _mm512_loadu_ps(acc + k);
        a = _mm512_fmadd_ps(a, vc, _mm512_mul_ps(vp, _mm512_loadu_ps(v + k)));
        _mm512_storeu_ps(acc + k, a);
    }
#endif
    for (; k < d; ++k)
        acc[k] = acc[k] * c + p * v[k];
}

} // namespace

// Q: [bh, sq, d]   K,V: [bh, sk, d]   O: [bh, sq, d]   (row-major).
// causal != 0 applies the lower-triangular mask aligned to the sequence tail
// (key j valid iff j <= i + max(sk - sq, 0)), matching the dense reference.
extern "C" void tessera_x86_flash_attn_f32(const float* Q, const float* K,
                                           const float* V, int64_t bh,
                                           int64_t sq, int64_t sk, int64_t d,
                                           float scale, int causal, float* O) {
    const int64_t off = (sk > sq) ? (sk - sq) : 0;     // causal alignment
    for (int64_t b = 0; b < bh; ++b) {
        const float* Qb = Q + b * sq * d;
        const float* Kb = K + b * sk * d;
        const float* Vb = V + b * sk * d;
        float* Ob = O + b * sq * d;
        for (int64_t i = 0; i < sq; ++i) {
            const float* qi = Qb + i * d;
            float* oi = Ob + i * d;
            for (int64_t t = 0; t < d; ++t)
                oi[t] = 0.0f;
            const int64_t jmax = causal ? (i + off) : (sk - 1);
            float m = -INFINITY, l = 0.0f;
            for (int64_t j = 0; j <= jmax && j < sk; ++j) {
                const float s = scale * dot_f32(qi, Kb + j * d, d);
                const float mn = (s > m) ? s : m;
                const float c = (m == -INFINITY) ? 0.0f : std::exp(m - mn);
                const float p = std::exp(s - mn);
                l = l * c + p;
                axpby_f32(oi, c, p, Vb + j * d, d);
                m = mn;
            }
            const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
            for (int64_t t = 0; t < d; ++t)
                oi[t] *= inv;
        }
    }
}
