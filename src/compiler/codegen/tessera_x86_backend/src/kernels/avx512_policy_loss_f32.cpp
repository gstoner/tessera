// AVX-512 RL policy-loss kernels (f32) for the Tessera x86 backend — the
// per-element surrogate of the S11 RL losses (ppo / cispo), the CPU lane for
// these previously reference-only ops. Operates on (logp_new, logp_old,
// advantages); the runtime applies the reduction (none/mean/sum) via the reduce
// kernel and the grouped advantage normalization (grpo) via the layer_norm kernel.
//
//   kind 0 = ppo    ratio = exp(ln−lo); c = clip(ratio, 1−ε, 1+ε);
//                   loss = −min(ratio·adv, c·adv)
//   kind 1 = cispo  w = min(exp(ln−lo), ε_high);  loss = −(w·adv·ln)
//
// This is the core surrogate (no KL/entropy/mask add-ons — those are handled /
// rejected in the runtime). exp reuses the Cephes core. 16 f32 lanes/__m512;
// n%16 tail scalar.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kPpo = 0;
constexpr int kCispo = 1;

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

inline float scalar_policy_loss(float ln, float lo, float adv, int kind,
                                float clip) {
    float r = std::exp(ln - lo);
    if (kind == kPpo) {
        float c = std::fmin(std::fmax(r, 1.0f - clip), 1.0f + clip);
        return -std::fmin(r * adv, c * adv);
    }
    float w = std::fmin(r, clip);     // cispo: clip the IS weight to ε_high
    return -(w * adv * ln);
}

}  // namespace

extern "C" void tessera_x86_avx512_policy_loss_f32(const float* LN,
                                                   const float* LO,
                                                   const float* ADV, int64_t n,
                                                   int kind, float clip,
                                                   float* out) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 vlo = _mm512_set1_ps(1.0f - clip);
    const __m512 vhi = _mm512_set1_ps(1.0f + clip);
    const __m512 vclip = _mm512_set1_ps(clip);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 ln = _mm512_loadu_ps(LN + i);
        __m512 lo = _mm512_loadu_ps(LO + i);
        __m512 adv = _mm512_loadu_ps(ADV + i);
        __m512 r = exp512(_mm512_sub_ps(ln, lo));
        __m512 y;
        if (kind == kPpo) {
            __m512 c = _mm512_min_ps(_mm512_max_ps(r, vlo), vhi);
            __m512 s = _mm512_min_ps(_mm512_mul_ps(r, adv),
                                     _mm512_mul_ps(c, adv));
            y = _mm512_sub_ps(zero, s);
        } else {
            __m512 w = _mm512_min_ps(r, vclip);
            y = _mm512_sub_ps(zero, _mm512_mul_ps(_mm512_mul_ps(w, adv), ln));
        }
        _mm512_storeu_ps(out + i, y);
    }
    (void)one;
    for (; i < n; ++i)
        out[i] = scalar_policy_loss(LN[i], LO[i], ADV[i], kind, clip);
}
