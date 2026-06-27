// AVX-512 row-wise normalization / softmax kernels (f32) for the Tessera x86
// backend — the CPU analog of the ROCm warp-shuffle norm/softmax lanes, so
// rmsnorm / layer_norm / softmax get a REAL vectorized CPU kernel instead of
// only the numpy reference. Each operates on a flat [M, D] buffer, reducing
// over the inner dimension D (one row at a time):
//
//   rmsnorm(x)         y = x * rsqrt(mean(x²) + eps)
//   layer_norm(x)      y = (x - mean) * rsqrt(var + eps)
//   softmax(x)         y = exp(x - max) / Σ exp(x - max)
//
// These match the ROCm norm/softmax lanes' op signatures: single operand,
// UNWEIGHTED (no γ/β), reduction over the last axis.
//
// Horizontal reductions use the AVX-512 `_mm512_reduce_*` intrinsics over a
// vector accumulator (16 f32 lanes/__m512; D % 16 tail handled scalar). softmax
// reuses the Cephes exp core (avx_mathfun formulation, ~1 ulp). Validated vs
// numpy at atol/rtol 2e-5 in test_norm.cpp and on-device.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {

// Cephes exp core (same as avx512_transcendental_f32.cpp) — softmax needs it.
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

inline float row_sum(const float* x, int64_t D) {
    __m512 acc = _mm512_setzero_ps();
    int64_t d = 0;
    for (; d + 16 <= D; d += 16) acc = _mm512_add_ps(acc, _mm512_loadu_ps(x + d));
    float s = _mm512_reduce_add_ps(acc);
    for (; d < D; ++d) s += x[d];
    return s;
}

inline float row_sumsq(const float* x, int64_t D) {
    __m512 acc = _mm512_setzero_ps();
    int64_t d = 0;
    for (; d + 16 <= D; d += 16) {
        __m512 v = _mm512_loadu_ps(x + d);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; d < D; ++d) s += x[d] * x[d];
    return s;
}

inline float row_max(const float* x, int64_t D) {
    __m512 acc = _mm512_set1_ps(-INFINITY);
    int64_t d = 0;
    for (; d + 16 <= D; d += 16)
        acc = _mm512_max_ps(acc, _mm512_loadu_ps(x + d));
    float m = _mm512_reduce_max_ps(acc);
    for (; d < D; ++d) m = x[d] > m ? x[d] : m;
    return m;
}

}  // namespace

extern "C" void tessera_x86_avx512_rmsnorm_f32(const float* X, int64_t M,
                                               int64_t D, float eps,
                                               float* out) {
    for (int64_t m = 0; m < M; ++m) {
        const float* x = X + m * D;
        float* o = out + m * D;
        float inv = 1.0f / std::sqrt(row_sumsq(x, D) / (float)D + eps);
        __m512 vinv = _mm512_set1_ps(inv);
        int64_t d = 0;
        for (; d + 16 <= D; d += 16)
            _mm512_storeu_ps(o + d, _mm512_mul_ps(_mm512_loadu_ps(x + d), vinv));
        for (; d < D; ++d) o[d] = x[d] * inv;
    }
}

extern "C" void tessera_x86_avx512_layernorm_f32(const float* X, int64_t M,
                                                 int64_t D, float eps,
                                                 float* out) {
    for (int64_t m = 0; m < M; ++m) {
        const float* x = X + m * D;
        float* o = out + m * D;
        float mean = row_sum(x, D) / (float)D;
        // var = mean(x²) - mean²  (single-pass sufficient at this tolerance)
        float var = row_sumsq(x, D) / (float)D - mean * mean;
        if (var < 0.0f) var = 0.0f;
        float inv = 1.0f / std::sqrt(var + eps);
        __m512 vmean = _mm512_set1_ps(mean);
        __m512 vinv = _mm512_set1_ps(inv);
        int64_t d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 c = _mm512_sub_ps(_mm512_loadu_ps(x + d), vmean);
            _mm512_storeu_ps(o + d, _mm512_mul_ps(c, vinv));
        }
        for (; d < D; ++d) o[d] = (x[d] - mean) * inv;
    }
}

extern "C" void tessera_x86_avx512_softmax_f32(const float* X, int64_t M,
                                               int64_t D, float* out) {
    for (int64_t m = 0; m < M; ++m) {
        const float* x = X + m * D;
        float* o = out + m * D;
        float mx = row_max(x, D);
        __m512 vmx = _mm512_set1_ps(mx);
        // pass 1: o = exp(x - mx), accumulate sum
        __m512 acc = _mm512_setzero_ps();
        int64_t d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 e = exp512(_mm512_sub_ps(_mm512_loadu_ps(x + d), vmx));
            _mm512_storeu_ps(o + d, e);
            acc = _mm512_add_ps(acc, e);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; d < D; ++d) { float e = std::exp(x[d] - mx); o[d] = e; sum += e; }
        // pass 2: normalize
        __m512 vinv = _mm512_set1_ps(1.0f / sum);
        d = 0;
        for (; d + 16 <= D; d += 16)
            _mm512_storeu_ps(o + d, _mm512_mul_ps(_mm512_loadu_ps(o + d), vinv));
        for (; d < D; ++d) o[d] /= sum;
    }
}
