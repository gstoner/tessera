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

extern "C" void tessera_x86_avx512_pointwise_loss_bwd_f32(
    const float* P, const float* T, const float* DY, int64_t n, int kind,
    float param, float scale, int tensor_cotangent, float* DP, float* DT) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 negative_one = _mm512_set1_ps(-1.0f);
    const __m512 transition = _mm512_set1_ps(param);
    const __m512 reduction_scale = _mm512_set1_ps(scale);
    const __m512 scalar_dy =
        tensor_cotangent ? zero : _mm512_set1_ps(DY[0]);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 error =
            _mm512_sub_ps(_mm512_loadu_ps(P + i), _mm512_loadu_ps(T + i));
        __mmask16 positive =
            _mm512_cmp_ps_mask(error, zero, _CMP_GT_OQ);
        __mmask16 negative =
            _mm512_cmp_ps_mask(error, zero, _CMP_LT_OQ);
        __m512 sign = _mm512_mask_blend_ps(negative, zero, negative_one);
        sign = _mm512_mask_blend_ps(positive, sign, one);
        __m512 local;
        switch (kind) {
        case kMse:
            local = _mm512_add_ps(error, error);
            break;
        case kMae:
            local = sign;
            break;
        case kHuber: {
            __m512 abs_error = _mm512_abs_ps(error);
            __mmask16 inside =
                _mm512_cmp_ps_mask(abs_error, transition, _CMP_LE_OQ);
            __m512 outside = _mm512_mul_ps(transition, sign);
            local = _mm512_mask_blend_ps(inside, outside, error);
            break;
        }
        case kSmoothL1: {
            __m512 abs_error = _mm512_abs_ps(error);
            __mmask16 inside =
                _mm512_cmp_ps_mask(abs_error, transition, _CMP_LT_OQ);
            __m512 inside_gradient = _mm512_div_ps(error, transition);
            local = _mm512_mask_blend_ps(inside, sign, inside_gradient);
            break;
        }
        default:
            local = zero;
            break;
        }
        __m512 dy =
            tensor_cotangent ? _mm512_loadu_ps(DY + i) : scalar_dy;
        __m512 grad =
            _mm512_mul_ps(_mm512_mul_ps(local, dy), reduction_scale);
        _mm512_storeu_ps(DP + i, grad);
        _mm512_storeu_ps(DT + i, _mm512_sub_ps(zero, grad));
    }
    for (; i < n; ++i) {
        float error = P[i] - T[i];
        float sign = error > 0.0f ? 1.0f : (error < 0.0f ? -1.0f : 0.0f);
        float local;
        if (kind == kMse)
            local = 2.0f * error;
        else if (kind == kMae)
            local = sign;
        else if (kind == kHuber)
            local = std::fabs(error) <= param ? error : param * sign;
        else if (kind == kSmoothL1)
            local = std::fabs(error) < param ? error / param : sign;
        else
            local = 0.0f;
        float grad = local * (tensor_cotangent ? DY[i] : DY[0]) * scale;
        DP[i] = grad;
        DT[i] = -grad;
    }
}

// Fused regression-loss VJP → SGD update. DP is deliberately not materialized:
// the local loss gradient is consumed immediately by PARAM - lr*grad, while DT
// preserves the target-gradient result of the unfused backward operation.
extern "C" void tessera_x86_avx512_training_loss_sgd_f32(
    const float* P, const float* T, const float* DY, const float* PARAM,
    int64_t n, int kind, float param, float scale, int tensor_cotangent,
    float lr, float* NEW_PARAM, float* DT) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 negative_one = _mm512_set1_ps(-1.0f);
    const __m512 transition = _mm512_set1_ps(param);
    const __m512 reduction_scale = _mm512_set1_ps(scale);
    const __m512 learning_rate = _mm512_set1_ps(lr);
    const __m512 scalar_dy =
        tensor_cotangent ? zero : _mm512_set1_ps(DY[0]);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 error =
            _mm512_sub_ps(_mm512_loadu_ps(P + i), _mm512_loadu_ps(T + i));
        __mmask16 positive =
            _mm512_cmp_ps_mask(error, zero, _CMP_GT_OQ);
        __mmask16 negative =
            _mm512_cmp_ps_mask(error, zero, _CMP_LT_OQ);
        __m512 sign = _mm512_mask_blend_ps(negative, zero, negative_one);
        sign = _mm512_mask_blend_ps(positive, sign, one);
        __m512 local;
        switch (kind) {
        case kMse:
            local = _mm512_add_ps(error, error);
            break;
        case kMae:
            local = sign;
            break;
        case kHuber: {
            __m512 abs_error = _mm512_abs_ps(error);
            __mmask16 inside =
                _mm512_cmp_ps_mask(abs_error, transition, _CMP_LE_OQ);
            __m512 outside = _mm512_mul_ps(transition, sign);
            local = _mm512_mask_blend_ps(inside, outside, error);
            break;
        }
        case kSmoothL1: {
            __m512 abs_error = _mm512_abs_ps(error);
            __mmask16 inside =
                _mm512_cmp_ps_mask(abs_error, transition, _CMP_LT_OQ);
            __m512 inside_gradient = _mm512_div_ps(error, transition);
            local = _mm512_mask_blend_ps(inside, sign, inside_gradient);
            break;
        }
        case kLogCosh: {
            // Training-only kind 4 is BCE-with-logits. The forward pointwise
            // loss keeps kind 4 as log-cosh; this ABI is selected independently.
            __m512 logits = _mm512_loadu_ps(P + i);
            __mmask16 nonnegative =
                _mm512_cmp_ps_mask(logits, zero, _CMP_GE_OQ);
            __m512 exp_negative = exp512(_mm512_sub_ps(zero, logits));
            __m512 exp_positive = exp512(logits);
            __m512 sigmoid_positive =
                _mm512_div_ps(one, _mm512_add_ps(one, exp_negative));
            __m512 sigmoid_negative =
                _mm512_div_ps(exp_positive, _mm512_add_ps(one, exp_positive));
            __m512 sigmoid = _mm512_mask_blend_ps(
                nonnegative, sigmoid_negative, sigmoid_positive);
            local = _mm512_sub_ps(sigmoid, _mm512_loadu_ps(T + i));
            break;
        }
        default:
            local = zero;
            break;
        }
        __m512 dy =
            tensor_cotangent ? _mm512_loadu_ps(DY + i) : scalar_dy;
        __m512 grad =
            _mm512_mul_ps(_mm512_mul_ps(local, dy), reduction_scale);
        __m512 updated = _mm512_fnmadd_ps(
            learning_rate, grad, _mm512_loadu_ps(PARAM + i));
        _mm512_storeu_ps(NEW_PARAM + i, updated);
        __m512 target_grad =
            kind == kLogCosh
                ? _mm512_mul_ps(
                      _mm512_sub_ps(zero, _mm512_loadu_ps(P + i)),
                      _mm512_mul_ps(dy, reduction_scale))
                : _mm512_sub_ps(zero, grad);
        _mm512_storeu_ps(DT + i, target_grad);
    }
    for (; i < n; ++i) {
        float error = P[i] - T[i];
        float sign = error > 0.0f ? 1.0f : (error < 0.0f ? -1.0f : 0.0f);
        float local;
        if (kind == kMse)
            local = 2.0f * error;
        else if (kind == kMae)
            local = sign;
        else if (kind == kHuber)
            local = std::fabs(error) <= param ? error : param * sign;
        else if (kind == kSmoothL1)
            local = std::fabs(error) < param ? error / param : sign;
        else if (kind == kLogCosh) {
            float logits = P[i];
            float sigmoid = logits >= 0.0f
                ? 1.0f / (1.0f + std::exp(-logits))
                : std::exp(logits) / (1.0f + std::exp(logits));
            local = sigmoid - T[i];
        } else
            local = 0.0f;
        float grad = local * (tensor_cotangent ? DY[i] : DY[0]) * scale;
        NEW_PARAM[i] = PARAM[i] - lr * grad;
        DT[i] = kind == kLogCosh
            ? -P[i] * (tensor_cotangent ? DY[i] : DY[0]) * scale
            : -grad;
    }
}

// Fused loss VJP → AdamW update. The ABI passes bias-correction denominators
// explicitly so the hot loop contains no scalar pow and the runtime cache key
// remains independent of tensor shape.
extern "C" void tessera_x86_avx512_training_loss_adamw_f32(
    const float* P, const float* T, const float* DY, const float* PARAM,
    const float* MOMENT1, const float* MOMENT2, int64_t n, int kind,
    float param, float scale, int tensor_cotangent, float lr, float beta1,
    float beta2, float eps, float weight_decay, float correction1,
    float correction2, float* NEW_PARAM, float* NEW_MOMENT1,
    float* NEW_MOMENT2, float* DT) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 negative_one = _mm512_set1_ps(-1.0f);
    const __m512 transition = _mm512_set1_ps(param);
    const __m512 reduction_scale = _mm512_set1_ps(scale);
    const __m512 learning_rate = _mm512_set1_ps(lr);
    const __m512 vb1 = _mm512_set1_ps(beta1);
    const __m512 vb2 = _mm512_set1_ps(beta2);
    const __m512 one_minus_b1 = _mm512_set1_ps(1.0f - beta1);
    const __m512 one_minus_b2 = _mm512_set1_ps(1.0f - beta2);
    const __m512 veps = _mm512_set1_ps(eps);
    const __m512 vweight_decay = _mm512_set1_ps(weight_decay);
    const __m512 vc1 = _mm512_set1_ps(correction1);
    const __m512 vc2 = _mm512_set1_ps(correction2);
    const __m512 scalar_dy =
        tensor_cotangent ? zero : _mm512_set1_ps(DY[0]);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 prediction = _mm512_loadu_ps(P + i);
        __m512 error =
            _mm512_sub_ps(prediction, _mm512_loadu_ps(T + i));
        __mmask16 positive =
            _mm512_cmp_ps_mask(error, zero, _CMP_GT_OQ);
        __mmask16 negative =
            _mm512_cmp_ps_mask(error, zero, _CMP_LT_OQ);
        __m512 sign = _mm512_mask_blend_ps(negative, zero, negative_one);
        sign = _mm512_mask_blend_ps(positive, sign, one);
        __m512 local;
        switch (kind) {
        case kMse:
            local = _mm512_add_ps(error, error);
            break;
        case kMae:
            local = sign;
            break;
        case kHuber: {
            __m512 abs_error = _mm512_abs_ps(error);
            __mmask16 inside =
                _mm512_cmp_ps_mask(abs_error, transition, _CMP_LE_OQ);
            local = _mm512_mask_blend_ps(
                inside, _mm512_mul_ps(transition, sign), error);
            break;
        }
        case kSmoothL1: {
            __m512 abs_error = _mm512_abs_ps(error);
            __mmask16 inside =
                _mm512_cmp_ps_mask(abs_error, transition, _CMP_LT_OQ);
            local = _mm512_mask_blend_ps(
                inside, sign, _mm512_div_ps(error, transition));
            break;
        }
        case kLogCosh: {
            __mmask16 nonnegative =
                _mm512_cmp_ps_mask(prediction, zero, _CMP_GE_OQ);
            __m512 exp_negative =
                exp512(_mm512_sub_ps(zero, prediction));
            __m512 exp_positive = exp512(prediction);
            __m512 sigmoid_positive =
                _mm512_div_ps(one, _mm512_add_ps(one, exp_negative));
            __m512 sigmoid_negative =
                _mm512_div_ps(exp_positive,
                              _mm512_add_ps(one, exp_positive));
            __m512 sigmoid = _mm512_mask_blend_ps(
                nonnegative, sigmoid_negative, sigmoid_positive);
            local = _mm512_sub_ps(sigmoid, _mm512_loadu_ps(T + i));
            break;
        }
        default:
            local = zero;
            break;
        }
        __m512 dy =
            tensor_cotangent ? _mm512_loadu_ps(DY + i) : scalar_dy;
        __m512 grad =
            _mm512_mul_ps(_mm512_mul_ps(local, dy), reduction_scale);
        __m512 new_m1 = _mm512_fmadd_ps(
            vb1, _mm512_loadu_ps(MOMENT1 + i),
            _mm512_mul_ps(one_minus_b1, grad));
        __m512 new_m2 = _mm512_fmadd_ps(
            vb2, _mm512_loadu_ps(MOMENT2 + i),
            _mm512_mul_ps(one_minus_b2, _mm512_mul_ps(grad, grad)));
        __m512 mhat = _mm512_div_ps(new_m1, vc1);
        __m512 vhat = _mm512_div_ps(new_m2, vc2);
        __m512 parameter_value = _mm512_loadu_ps(PARAM + i);
        __m512 update = _mm512_fmadd_ps(
            vweight_decay, parameter_value,
            _mm512_div_ps(
                mhat, _mm512_add_ps(_mm512_sqrt_ps(vhat), veps)));
        __m512 new_param = _mm512_fnmadd_ps(
            learning_rate, update, parameter_value);
        _mm512_storeu_ps(NEW_PARAM + i, new_param);
        _mm512_storeu_ps(NEW_MOMENT1 + i, new_m1);
        _mm512_storeu_ps(NEW_MOMENT2 + i, new_m2);
        __m512 target_grad =
            kind == kLogCosh
                ? _mm512_mul_ps(
                      _mm512_sub_ps(zero, prediction),
                      _mm512_mul_ps(dy, reduction_scale))
                : _mm512_sub_ps(zero, grad);
        _mm512_storeu_ps(DT + i, target_grad);
    }
    for (; i < n; ++i) {
        float error = P[i] - T[i];
        float sign = error > 0.0f ? 1.0f : (error < 0.0f ? -1.0f : 0.0f);
        float local;
        if (kind == kMse)
            local = 2.0f * error;
        else if (kind == kMae)
            local = sign;
        else if (kind == kHuber)
            local = std::fabs(error) <= param ? error : param * sign;
        else if (kind == kSmoothL1)
            local = std::fabs(error) < param ? error / param : sign;
        else if (kind == kLogCosh) {
            float sigmoid = P[i] >= 0.0f
                ? 1.0f / (1.0f + std::exp(-P[i]))
                : std::exp(P[i]) / (1.0f + std::exp(P[i]));
            local = sigmoid - T[i];
        } else
            local = 0.0f;
        float dy = tensor_cotangent ? DY[i] : DY[0];
        float grad = local * dy * scale;
        float new_m1 = beta1 * MOMENT1[i] + (1.0f - beta1) * grad;
        float new_m2 =
            beta2 * MOMENT2[i] + (1.0f - beta2) * grad * grad;
        float update =
            (new_m1 / correction1) /
                (std::sqrt(new_m2 / correction2) + eps) +
            weight_decay * PARAM[i];
        NEW_PARAM[i] = PARAM[i] - lr * update;
        NEW_MOMENT1[i] = new_m1;
        NEW_MOMENT2[i] = new_m2;
        DT[i] = kind == kLogCosh ? -P[i] * dy * scale : -grad;
    }
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

extern "C" void tessera_x86_avx512_binary_loss_bwd_f32(
    const float* Z, const float* T, const float* dY, int64_t n, float scale,
    int tensor_cotangent, float* dZ, float* dT) {
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 vscale = _mm512_set1_ps(scale);
    const __m512 scalar_dy =
        _mm512_set1_ps(tensor_cotangent ? 0.0f : dY[0]);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 z = _mm512_loadu_ps(Z + i);
        __m512 t = _mm512_loadu_ps(T + i);
        __m512 dy =
            tensor_cotangent ? _mm512_loadu_ps(dY + i) : scalar_dy;
        __mmask16 nonnegative =
            _mm512_cmp_ps_mask(z, zero, _CMP_GE_OQ);
        __m512 exp_neg = exp512(_mm512_sub_ps(zero, z));
        __m512 exp_pos = exp512(z);
        __m512 sig_pos = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg));
        __m512 sig_neg =
            _mm512_div_ps(exp_pos, _mm512_add_ps(one, exp_pos));
        __m512 sigmoid =
            _mm512_mask_blend_ps(nonnegative, sig_neg, sig_pos);
        __m512 weighted = _mm512_mul_ps(dy, vscale);
        _mm512_storeu_ps(
            dZ + i, _mm512_mul_ps(_mm512_sub_ps(sigmoid, t), weighted));
        _mm512_storeu_ps(
            dT + i, _mm512_mul_ps(_mm512_sub_ps(zero, z), weighted));
    }
    for (; i < n; ++i) {
        float sigmoid =
            Z[i] >= 0.0f ? 1.0f / (1.0f + std::exp(-Z[i]))
                         : std::exp(Z[i]) / (1.0f + std::exp(Z[i]));
        float dy = tensor_cotangent ? dY[i] : dY[0];
        dZ[i] = (sigmoid - T[i]) * dy * scale;
        dT[i] = -Z[i] * dy * scale;
    }
}

extern "C" void tessera_x86_avx512_cross_entropy_bwd_f32(
    const float* logits, const int64_t* targets, const float* dY,
    int64_t rows, int64_t classes, float smoothing, int64_t ignore_index,
    float scale, int tensor_cotangent, float* dLogits) {
    for (int64_t row = 0; row < rows; ++row) {
        const float* z = logits + row * classes;
        float* dz = dLogits + row * classes;
        const int64_t target = targets[row];
        if (target == ignore_index) {
            for (int64_t k = 0; k < classes; ++k) dz[k] = 0.0f;
            continue;
        }
        float maximum = -INFINITY;
        for (int64_t k = 0; k < classes; ++k)
            maximum = std::fmax(maximum, z[k]);
        float denominator = 0.0f;
        int64_t k = 0;
        const __m512 vmax = _mm512_set1_ps(maximum);
        __m512 vsum = _mm512_setzero_ps();
        for (; k + 16 <= classes; k += 16)
            vsum = _mm512_add_ps(
                vsum, exp512(_mm512_sub_ps(_mm512_loadu_ps(z + k), vmax)));
        denominator += _mm512_reduce_add_ps(vsum);
        for (; k < classes; ++k) denominator += std::exp(z[k] - maximum);
        const float weighted =
            (tensor_cotangent ? dY[row] : dY[0]) * scale;
        const float off_target =
            smoothing == 0.0f ? 0.0f : smoothing / float(classes - 1);
        const float on_target = 1.0f - smoothing;
        const __m512 vden = _mm512_set1_ps(denominator);
        const __m512 vweighted = _mm512_set1_ps(weighted);
        const __m512 voff = _mm512_set1_ps(off_target);
        k = 0;
        for (; k + 16 <= classes; k += 16) {
            __m512 probability = _mm512_div_ps(
                exp512(_mm512_sub_ps(_mm512_loadu_ps(z + k), vmax)), vden);
            __m512 distribution = voff;
            if (target >= k && target < k + 16)
                distribution = _mm512_mask_mov_ps(
                    distribution, __mmask16(1u << (target - k)),
                    _mm512_set1_ps(on_target));
            _mm512_storeu_ps(
                dz + k,
                _mm512_mul_ps(_mm512_sub_ps(probability, distribution),
                              vweighted));
        }
        for (; k < classes; ++k) {
            float distribution = k == target ? on_target : off_target;
            dz[k] = (std::exp(z[k] - maximum) / denominator - distribution) *
                    weighted;
        }
    }
}
