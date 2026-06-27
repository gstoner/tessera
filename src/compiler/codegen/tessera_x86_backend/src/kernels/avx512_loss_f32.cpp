// AVX-512 pointwise-loss kernels (f32) for the Tessera x86 backend — the CPU
// lane for the S11 pointwise regression losses. Computes the PER-ELEMENT loss
// loss[i] = f(pred[i], target[i]) over a flat [n] buffer; the runtime applies
// the reduction (none/mean/sum) via the x86 reduce lane. These are the elementwise
// FLOPs that dominate the loss — a real vectorized kernel instead of the numpy
// reference.
//
//   kind 0 = mse        (p−t)²
//   kind 1 = mae        |p−t|
//   kind 2 = huber(δ)   a=|p−t|; a≤δ ? ½a² : δ(a−½δ)
//   kind 3 = smooth_l1(β) a=|p−t|; a<β ? ½a²/β : a−½β
//   kind 4 = log_cosh   e + log1p(exp(−2e)) − log2   (= log cosh(e), stable form)
//
// log_cosh reuses the Cephes exp/log cores. 16 f32 lanes/__m512; n%16 tail scalar.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kMse = 0;
constexpr int kMae = 1;
constexpr int kHuber = 2;
constexpr int kSmoothL1 = 3;
constexpr int kLogCosh = 4;

constexpr float kLn2 = 0.6931471805599453f;

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

inline __m512 log512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __mmask16 invalid = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OQ);
    x = _mm512_max_ps(x, _mm512_castsi512_ps(_mm512_set1_epi32(0x00800000)));
    __m512i xi = _mm512_castps_si512(x);
    __m512 e = _mm512_cvtepi32_ps(
        _mm512_sub_epi32(_mm512_srli_epi32(xi, 23), _mm512_set1_epi32(0x7f)));
    x = _mm512_castsi512_ps(_mm512_and_si512(xi, _mm512_set1_epi32(0x807fffff)));
    x = _mm512_or_ps(x, _mm512_set1_ps(0.5f));
    e = _mm512_add_ps(e, one);
    __mmask16 lt = _mm512_cmp_ps_mask(x, _mm512_set1_ps(0.707106781186547524f),
                                      _CMP_LT_OQ);
    __m512 tmp = _mm512_maskz_mov_ps(lt, x);
    x = _mm512_sub_ps(x, one);
    e = _mm512_mask_sub_ps(e, lt, e, one);
    x = _mm512_add_ps(x, tmp);
    __m512 z = _mm512_mul_ps(x, x);
    __m512 y = _mm512_set1_ps(7.0376836292E-2f);
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.1514610310E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.1676998740E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.2420140846E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(1.4249322787E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-1.6668057665E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(2.0000714765E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(-2.4999993993E-1f));
    y = _mm512_fmadd_ps(y, x, _mm512_set1_ps(3.3333331174E-1f));
    y = _mm512_mul_ps(y, x);
    y = _mm512_mul_ps(y, z);
    y = _mm512_fmadd_ps(e, _mm512_set1_ps(-2.12194440e-4f), y);
    y = _mm512_fnmadd_ps(z, _mm512_set1_ps(0.5f), y);
    x = _mm512_add_ps(x, y);
    x = _mm512_fmadd_ps(e, _mm512_set1_ps(0.693359375f), x);
    return _mm512_mask_blend_ps(invalid, x, _mm512_set1_ps(NAN));
}

inline float scalar_loss(float p, float t, int kind, float param) {
    float e = p - t, a = std::fabs(e);
    switch (kind) {
    case kMse:      return e * e;
    case kMae:      return a;
    case kHuber:    return a <= param ? 0.5f * a * a
                                      : param * (a - 0.5f * param);
    case kSmoothL1: return a < param ? 0.5f * a * a / param
                                     : a - 0.5f * param;
    case kLogCosh:  return e + std::log1p(std::exp(-2.0f * e)) - kLn2;
    default:        return 0.0f;
    }
}

}  // namespace

extern "C" void tessera_x86_avx512_pointwise_loss_f32(const float* P,
                                                      const float* T, int64_t n,
                                                      int kind, float param,
                                                      float* out) {
    const __m512 vparam = _mm512_set1_ps(param);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 two = _mm512_set1_ps(2.0f);
    const __m512 vln2 = _mm512_set1_ps(kLn2);
    const __m512 one = _mm512_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 e = _mm512_sub_ps(_mm512_loadu_ps(P + i), _mm512_loadu_ps(T + i));
        __m512 a = _mm512_abs_ps(e);
        __m512 y;
        switch (kind) {
        case kMse: y = _mm512_mul_ps(e, e); break;
        case kMae: y = a; break;
        case kHuber: {
            __mmask16 le = _mm512_cmp_ps_mask(a, vparam, _CMP_LE_OQ);
            __m512 q = _mm512_mul_ps(half, _mm512_mul_ps(a, a));
            __m512 lin = _mm512_mul_ps(
                vparam, _mm512_fnmadd_ps(half, vparam, a));  // δ(a−½δ)
            y = _mm512_mask_blend_ps(le, lin, q);            // le ? q : lin
            break;
        }
        case kSmoothL1: {
            __mmask16 ltb = _mm512_cmp_ps_mask(a, vparam, _CMP_LT_OQ);
            __m512 q = _mm512_div_ps(_mm512_mul_ps(half, _mm512_mul_ps(a, a)),
                                     vparam);
            __m512 lin = _mm512_fnmadd_ps(half, vparam, a);  // a−½β
            y = _mm512_mask_blend_ps(ltb, lin, q);
            break;
        }
        case kLogCosh: {
            // e + log1p(exp(−2e)) − log2
            __m512 ex = exp512(_mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(),
                                                           two), e));
            __m512 l = log512(_mm512_add_ps(one, ex));
            y = _mm512_sub_ps(_mm512_add_ps(e, l), vln2);
            break;
        }
        default: y = _mm512_setzero_ps(); break;
        }
        _mm512_storeu_ps(out + i, y);
    }
    for (; i < n; ++i) out[i] = scalar_loss(P[i], T[i], kind, param);
}

// ── binary-cross-entropy-with-logits family (per-element, z=logits, t=target) ──
//
//   kind 0 = bce            max(z,0) − z·t + log1p(exp(−|z|))
//   kind 1 = asymmetric_bce pw·t·softplus(−z) + nw·(1−t)·softplus(z)
//             softplus(±z) = max(±z,0) + log1p(exp(−|z|));  pw=nw=1 ⇒ bce.
//
// The stable softplus form (relu(±z) + log1p(exp(−|z|))) never overflows.
namespace {
constexpr int kBce = 0;
constexpr int kAsymBce = 1;

inline float scalar_binary_loss(float z, float t, int kind, float pw, float nw) {
    float L = std::log1p(std::exp(-std::fabs(z)));
    float sp_pos = std::fmax(z, 0.0f) + L;
    if (kind == kBce) return sp_pos - z * t;
    float sp_neg = std::fmax(-z, 0.0f) + L;
    return pw * t * sp_neg + nw * (1.0f - t) * sp_pos;
}
}  // namespace

extern "C" void tessera_x86_avx512_binary_loss_f32(const float* Z,
                                                   const float* T, int64_t n,
                                                   int kind, float pos_w,
                                                   float neg_w, float* out) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 vpw = _mm512_set1_ps(pos_w);
    const __m512 vnw = _mm512_set1_ps(neg_w);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 z = _mm512_loadu_ps(Z + i);
        __m512 t = _mm512_loadu_ps(T + i);
        __m512 az = _mm512_abs_ps(z);
        __m512 L = log512(_mm512_add_ps(one, exp512(_mm512_sub_ps(zero, az))));
        __m512 sp_pos = _mm512_add_ps(_mm512_max_ps(z, zero), L);
        __m512 y;
        if (kind == kBce) {
            y = _mm512_sub_ps(sp_pos, _mm512_mul_ps(z, t));   // sp_pos − z·t
        } else {
            __m512 sp_neg = _mm512_add_ps(_mm512_max_ps(_mm512_sub_ps(zero, z),
                                                        zero), L);
            __m512 term1 = _mm512_mul_ps(_mm512_mul_ps(vpw, t), sp_neg);
            __m512 term2 = _mm512_mul_ps(
                _mm512_mul_ps(vnw, _mm512_sub_ps(one, t)), sp_pos);
            y = _mm512_add_ps(term1, term2);
        }
        _mm512_storeu_ps(out + i, y);
    }
    for (; i < n; ++i) out[i] = scalar_binary_loss(Z[i], T[i], kind, pos_w,
                                                   neg_w);
}
