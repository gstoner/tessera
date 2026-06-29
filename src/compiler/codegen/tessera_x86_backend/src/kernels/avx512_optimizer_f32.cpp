// AVX-512 fused optimizer-step kernels (f32) — the optimized CPU lane for the
// S-series `functional_optimizer_step` family (P3 of S_SERIES_GAP_CLOSURE_PLAN).
// A single flat per-parameter update over one parameter tensor; the pytree
// orchestration + the bias-correction scalars (1-β^t) are computed on the host
// (the optimizer carries `step`), so the kernel is pure elementwise.
//
//   kind 0 sgd      : p -= lr·g
//   kind 1 momentum : v = μ·v + g ; p -= lr·v
//   kind 2 adam     : m = β1·m+(1-β1)g ; v = β2·v+(1-β2)g² ;
//                     p -= lr·(m/b1c)/(√(v/b2c)+eps)
//   kind 3 adamw    : p *= (1-lr·wd) ; then the adam update (decoupled decay)
//   kind 4 lion     : u = β1·m+(1-β1)g ; m = β2·m+(1-β2)g ;
//                     p *= (1-lr·wd) ; p -= lr·sign(u)
//
// `m`/`v` are in/out state buffers (written to m_out/v_out, may alias in). β1/β2
// = momentum/decay; for momentum, μ is passed as beta1 and the v slot is used.
// √ via the exact hardware `_mm512_sqrt_ps`. A scalar reference is alongside.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kSgd = 0, kMomentum = 1, kAdam = 2, kAdamW = 3, kLion = 4,
              kNesterov = 5;

inline float signf(float x) { return (x > 0.0f) - (x < 0.0f); }
}  // namespace

extern "C" void tessera_x86_reference_optimizer_f32(
    const float* p, const float* g, const float* m, const float* v, int64_t n,
    int kind, float lr, float beta1, float beta2, float eps, float wd,
    float b1c, float b2c, float* p_out, float* m_out, float* v_out) {
    for (int64_t i = 0; i < n; ++i) {
        float pi = p[i], gi = g[i];
        if (kind == kSgd) { p_out[i] = pi - lr * gi; continue; }
        if (kind == kMomentum) {
            float vv = beta1 * v[i] + gi;
            v_out[i] = vv;
            p_out[i] = pi - lr * vv;
            continue;
        }
        if (kind == kNesterov) {
            float vv = beta1 * v[i] + gi;            // new velocity
            v_out[i] = vv;
            p_out[i] = pi - lr * (gi + beta1 * vv);  // look-ahead update
            continue;
        }
        if (kind == kLion) {
            float u = beta1 * m[i] + (1.0f - beta1) * gi;
            m_out[i] = beta2 * m[i] + (1.0f - beta2) * gi;
            if (wd != 0.0f) pi *= (1.0f - lr * wd);
            p_out[i] = pi - lr * signf(u);
            continue;
        }
        // adam / adamw
        float mm = beta1 * m[i] + (1.0f - beta1) * gi;
        float vv = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m_out[i] = mm;
        v_out[i] = vv;
        if (kind == kAdamW && wd != 0.0f) pi *= (1.0f - lr * wd);
        float upd = (mm / b1c) / (std::sqrt(vv / b2c) + eps);
        p_out[i] = pi - lr * upd;
    }
}

extern "C" void tessera_x86_optimizer_f32(
    const float* p, const float* g, const float* m, const float* v, int64_t n,
    int kind, float lr, float beta1, float beta2, float eps, float wd,
    float b1c, float b2c, float* p_out, float* m_out, float* v_out) {
    const __m512 vlr = _mm512_set1_ps(lr);
    const __m512 vb1 = _mm512_set1_ps(beta1);
    const __m512 vb2 = _mm512_set1_ps(beta2);
    const __m512 vone = _mm512_set1_ps(1.0f);
    const __m512 vom1 = _mm512_set1_ps(1.0f - beta1);
    const __m512 vom2 = _mm512_set1_ps(1.0f - beta2);
    const __m512 veps = _mm512_set1_ps(eps);
    const __m512 vb1c = _mm512_set1_ps(b1c);
    const __m512 vb2c = _mm512_set1_ps(b2c);
    const __m512 vdecay = _mm512_set1_ps(1.0f - lr * wd);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 pi = _mm512_loadu_ps(p + i);
        __m512 gi = _mm512_loadu_ps(g + i);
        if (kind == kSgd) {
            _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, gi, pi));
            continue;
        }
        if (kind == kMomentum) {
            __m512 vv = _mm512_fmadd_ps(vb1, _mm512_loadu_ps(v + i), gi);
            _mm512_storeu_ps(v_out + i, vv);
            _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, vv, pi));
            continue;
        }
        if (kind == kNesterov) {
            __m512 vv = _mm512_fmadd_ps(vb1, _mm512_loadu_ps(v + i), gi);
            _mm512_storeu_ps(v_out + i, vv);
            __m512 upd = _mm512_fmadd_ps(vb1, vv, gi);   // g + beta1*vv
            _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, upd, pi));
            continue;
        }
        if (kind == kLion) {
            __m512 mi = _mm512_loadu_ps(m + i);
            __m512 u = _mm512_fmadd_ps(vb1, mi, _mm512_mul_ps(vom1, gi));
            _mm512_storeu_ps(m_out + i,
                             _mm512_fmadd_ps(vb2, mi, _mm512_mul_ps(vom2, gi)));
            // sign(u) = (u>0) - (u<0)
            __mmask16 pos = _mm512_cmp_ps_mask(u, _mm512_setzero_ps(), _CMP_GT_OQ);
            __mmask16 neg = _mm512_cmp_ps_mask(u, _mm512_setzero_ps(), _CMP_LT_OQ);
            __m512 sgn = _mm512_mask_blend_ps(
                pos, _mm512_mask_blend_ps(neg, _mm512_setzero_ps(),
                                          _mm512_set1_ps(-1.0f)),
                vone);
            if (wd != 0.0f) pi = _mm512_mul_ps(pi, vdecay);
            _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, sgn, pi));
            continue;
        }
        // adam / adamw
        __m512 mi = _mm512_loadu_ps(m + i);
        __m512 vi = _mm512_loadu_ps(v + i);
        __m512 mm = _mm512_fmadd_ps(vb1, mi, _mm512_mul_ps(vom1, gi));
        __m512 vv = _mm512_fmadd_ps(vb2, vi,
                                    _mm512_mul_ps(vom2, _mm512_mul_ps(gi, gi)));
        _mm512_storeu_ps(m_out + i, mm);
        _mm512_storeu_ps(v_out + i, vv);
        if (kind == kAdamW && wd != 0.0f) pi = _mm512_mul_ps(pi, vdecay);
        __m512 denom = _mm512_add_ps(
            _mm512_sqrt_ps(_mm512_div_ps(vv, vb2c)), veps);
        __m512 upd = _mm512_div_ps(_mm512_div_ps(mm, vb1c), denom);
        _mm512_storeu_ps(p_out + i, _mm512_fnmadd_ps(vlr, upd, pi));
    }
    if (i < n)
        tessera_x86_reference_optimizer_f32(
            p + i, g + i, m + i, v + i, n - i, kind, lr, beta1, beta2, eps, wd,
            b1c, b2c, p_out + i, m_out + i, v_out + i);
}
