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
// The legacy unary symbols remain ABI-stable. Affine entry points add channel
// gamma/beta vectors and share the same vectorized reduction implementation.
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

inline float row_centered_sumsq(const float* x, int64_t D, float mean) {
    __m512 acc = _mm512_setzero_ps();
    const __m512 vm = _mm512_set1_ps(mean);
    int64_t d = 0;
    for (; d + 16 <= D; d += 16) {
        __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x + d), vm);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; d < D; ++d) {
        float v = x[d] - mean;
        s += v * v;
    }
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

static void rmsnorm_impl(const float* X, const float* gamma, int64_t M,
                         int64_t D, float eps, float* out) {
    for (int64_t m = 0; m < M; ++m) {
        const float* x = X + m * D;
        float* o = out + m * D;
        float inv = 1.0f / std::sqrt(row_sumsq(x, D) / (float)D + eps);
        __m512 vinv = _mm512_set1_ps(inv);
        int64_t d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 y = _mm512_mul_ps(_mm512_loadu_ps(x + d), vinv);
            if (gamma) y = _mm512_mul_ps(y, _mm512_loadu_ps(gamma + d));
            _mm512_storeu_ps(o + d, y);
        }
        for (; d < D; ++d)
            o[d] = x[d] * inv * (gamma ? gamma[d] : 1.0f);
    }
}

static void layernorm_impl(const float* X, const float* gamma,
                           const float* beta, int64_t M, int64_t D, float eps,
                           float* out) {
    for (int64_t m = 0; m < M; ++m) {
        const float* x = X + m * D;
        float* o = out + m * D;
        float mean = row_sum(x, D) / (float)D;
        // Stable two-pass variance, matching ROCm and the Graph statistics
        // carrier for large-offset/small-variance rows.
        float var = row_centered_sumsq(x, D, mean) / (float)D;
        float inv = 1.0f / std::sqrt(var + eps);
        __m512 vmean = _mm512_set1_ps(mean);
        __m512 vinv = _mm512_set1_ps(inv);
        int64_t d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 y = _mm512_mul_ps(
                _mm512_sub_ps(_mm512_loadu_ps(x + d), vmean), vinv);
            if (gamma) y = _mm512_mul_ps(y, _mm512_loadu_ps(gamma + d));
            if (beta) y = _mm512_add_ps(y, _mm512_loadu_ps(beta + d));
            _mm512_storeu_ps(o + d, y);
        }
        for (; d < D; ++d) {
            float y = (x[d] - mean) * inv;
            o[d] = y * (gamma ? gamma[d] : 1.0f) + (beta ? beta[d] : 0.0f);
        }
    }
}

static void norm_backward_impl(const float* X, const float* gamma,
                               const float* dY, int64_t M, int64_t D,
                               float eps, bool layer_norm, bool has_gamma,
                               bool has_beta, float* dX, float* dGamma,
                               float* dBeta) {
    if (has_gamma) {
        int64_t d = 0;
        for (; d + 16 <= D; d += 16)
            _mm512_storeu_ps(dGamma + d, _mm512_setzero_ps());
        for (; d < D; ++d) dGamma[d] = 0.0f;
    }
    if (has_beta) {
        int64_t d = 0;
        for (; d + 16 <= D; d += 16)
            _mm512_storeu_ps(dBeta + d, _mm512_setzero_ps());
        for (; d < D; ++d) dBeta[d] = 0.0f;
    }

    for (int64_t m = 0; m < M; ++m) {
        const float* x = X + m * D;
        const float* dy = dY + m * D;
        float* dx = dX + m * D;
        const float mean = layer_norm ? row_sum(x, D) / (float)D : 0.0f;
        const float stat = layer_norm
            ? row_centered_sumsq(x, D, mean) / (float)D
            : row_sumsq(x, D) / (float)D;
        const float inv = 1.0f / std::sqrt(stat + eps);
        const __m512 vmean = _mm512_set1_ps(mean);
        const __m512 vinv = _mm512_set1_ps(inv);

        __m512 acc_dz = _mm512_setzero_ps();
        __m512 acc_dz_z = _mm512_setzero_ps();
        int64_t d = 0;
        for (; d + 16 <= D; d += 16) {
            const __m512 xv = _mm512_loadu_ps(x + d);
            const __m512 dyv = _mm512_loadu_ps(dy + d);
            const __m512 zv = _mm512_mul_ps(
                layer_norm ? _mm512_sub_ps(xv, vmean) : xv, vinv);
            const __m512 dzv = has_gamma
                ? _mm512_mul_ps(dyv, _mm512_loadu_ps(gamma + d)) : dyv;
            acc_dz = _mm512_add_ps(acc_dz, dzv);
            acc_dz_z = _mm512_fmadd_ps(dzv, zv, acc_dz_z);
        }
        float sum_dz = _mm512_reduce_add_ps(acc_dz);
        float sum_dz_z = _mm512_reduce_add_ps(acc_dz_z);
        for (; d < D; ++d) {
            const float z = (x[d] - (layer_norm ? mean : 0.0f)) * inv;
            const float dz = dy[d] * (has_gamma ? gamma[d] : 1.0f);
            sum_dz += dz;
            sum_dz_z += dz * z;
        }
        const float mean_dz = layer_norm ? sum_dz / (float)D : 0.0f;
        const float mean_dz_z = sum_dz_z / (float)D;
        const __m512 vmean_dz = _mm512_set1_ps(mean_dz);
        const __m512 vmean_dz_z = _mm512_set1_ps(mean_dz_z);

        d = 0;
        for (; d + 16 <= D; d += 16) {
            const __m512 xv = _mm512_loadu_ps(x + d);
            const __m512 dyv = _mm512_loadu_ps(dy + d);
            const __m512 zv = _mm512_mul_ps(
                layer_norm ? _mm512_sub_ps(xv, vmean) : xv, vinv);
            const __m512 dzv = has_gamma
                ? _mm512_mul_ps(dyv, _mm512_loadu_ps(gamma + d)) : dyv;
            __m512 inner = _mm512_sub_ps(dzv, vmean_dz);
            inner = _mm512_fnmadd_ps(zv, vmean_dz_z, inner);
            _mm512_storeu_ps(dx + d, _mm512_mul_ps(inner, vinv));
            if (has_gamma) {
                const __m512 dg = _mm512_fmadd_ps(dyv, zv,
                                                   _mm512_loadu_ps(dGamma + d));
                _mm512_storeu_ps(dGamma + d, dg);
            }
            if (has_beta) {
                _mm512_storeu_ps(
                    dBeta + d,
                    _mm512_add_ps(_mm512_loadu_ps(dBeta + d), dyv));
            }
        }
        for (; d < D; ++d) {
            const float z = (x[d] - (layer_norm ? mean : 0.0f)) * inv;
            const float dz = dy[d] * (has_gamma ? gamma[d] : 1.0f);
            dx[d] = inv * (dz - mean_dz - z * mean_dz_z);
            if (has_gamma) dGamma[d] += dy[d] * z;
            if (has_beta) dBeta[d] += dy[d];
        }
    }
}

extern "C" void tessera_x86_avx512_rmsnorm_f32(const float* X, int64_t M,
                                               int64_t D, float eps,
                                               float* out) {
    rmsnorm_impl(X, nullptr, M, D, eps, out);
}

extern "C" void tessera_x86_avx512_layernorm_f32(const float* X, int64_t M,
                                                 int64_t D, float eps,
                                                 float* out) {
    layernorm_impl(X, nullptr, nullptr, M, D, eps, out);
}

extern "C" void tessera_x86_avx512_rmsnorm_affine_f32(
    const float* X, const float* gamma, int64_t M, int64_t D, float eps,
    float* out) {
    rmsnorm_impl(X, gamma, M, D, eps, out);
}

extern "C" void tessera_x86_avx512_layernorm_affine_f32(
    const float* X, const float* gamma, const float* beta, int64_t M, int64_t D,
    float eps, float* out) {
    layernorm_impl(X, gamma, beta, M, D, eps, out);
}

extern "C" void tessera_x86_avx512_rmsnorm_bwd_f32(
    const float* X, const float* gamma, const float* dY, int64_t M, int64_t D,
    float eps, int has_gamma, float* dX, float* dGamma) {
    norm_backward_impl(X, gamma, dY, M, D, eps, false, has_gamma != 0, false,
                       dX, dGamma, nullptr);
}

extern "C" void tessera_x86_avx512_layernorm_bwd_f32(
    const float* X, const float* gamma, const float* dY, int64_t M, int64_t D,
    float eps, int has_gamma, int has_beta, float* dX, float* dGamma,
    float* dBeta) {
    norm_backward_impl(X, gamma, dY, M, D, eps, true, has_gamma != 0,
                       has_beta != 0, dX, dGamma, dBeta);
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
