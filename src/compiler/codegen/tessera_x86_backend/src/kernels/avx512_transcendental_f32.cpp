// AVX-512 elementwise transcendental / activation kernels (f32).
//
// The vectorized CPU lane for the S2 transcendental + activation primitives —
// the AVX-512 analog of the ROCm math->ROCDL unary/activation lanes, so these
// ops get a REAL vectorized CPU kernel instead of only the numpy reference.
// AVX-512 has no native transcendentals, so the cores (exp / log) use the
// battle-tested Cephes minimax polynomials (the avx_mathfun formulation,
// ~1 ulp), erf uses Abramowitz-Stegun 7.1.26 (abs err < 1.5e-7), and the
// activations compose from those. Validated to atol/rtol 2e-5 vs libm in
// test_transcendental.cpp and vs numpy on-device.
//
//   kind  0 = exp        e^x                         (Cephes expf core)
//   kind  1 = log        ln(x), x>0 (else NaN)        (Cephes logf core)
//   kind  2 = tanh       (e^2x - 1)/(e^2x + 1)
//   kind  3 = sigmoid    1/(1 + e^-x)
//   kind  4 = silu       x * sigmoid(x)
//   kind  5 = gelu       0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))  (tanh approx,
//                        matching the ROCm activation reference)
//   kind  6 = erf        Abramowitz-Stegun 7.1.26
//   kind  7 = softplus   log1p(e^-|x|) + max(x, 0)    (overflow-stable)
//   kind  8 = expm1      e^x - 1
//   kind  9 = log1p      ln(1 + x)
//
// 16 f32 lanes per __m512; the tail (n % 16) is handled scalar via libm so the
// reference and the vector path agree on partial blocks.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kExp = 0;
constexpr int kLog = 1;
constexpr int kTanh = 2;
constexpr int kSigmoid = 3;
constexpr int kSilu = 4;
constexpr int kGelu = 5;
constexpr int kErf = 6;
constexpr int kSoftplus = 7;
constexpr int kExpm1 = 8;
constexpr int kLog1p = 9;

constexpr float kSqrt2OverPi = 0.7978845608028654f;  // √(2/π)
constexpr float kGeluC = 0.044715f;

// ── scalar reference (libm) — also used for the n % 16 tail ──────────────────
inline float scalar_transcendental(float v, int kind) {
    switch (kind) {
    case kExp:      return std::exp(v);
    case kLog:      return std::log(v);
    case kTanh:     return std::tanh(v);
    case kSigmoid:  return 1.0f / (1.0f + std::exp(-v));
    case kSilu:     return v / (1.0f + std::exp(-v));
    case kGelu:     return 0.5f * v *
                           (1.0f + std::tanh(kSqrt2OverPi *
                                             (v + kGeluC * v * v * v)));
    case kErf:      return std::erf(v);
    case kSoftplus: return std::log1p(std::exp(-std::fabs(v))) +
                           std::fmax(v, 0.0f);
    case kExpm1:    return std::exp(v) - 1.0f;
    case kLog1p:    return std::log1p(v);
    default:        return v;
    }
}

// ── AVX-512 Cephes exp core (avx_mathfun formulation) ────────────────────────
inline __m512 exp512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));
    // fx = floor(x * log2(e) + 0.5)
    __m512 fx = _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f),
                                _mm512_set1_ps(0.5f));
    fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    // r = x - fx*ln2_hi - fx*ln2_lo
    __m512 r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 z = _mm512_mul_ps(r, r);
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, z, _mm512_add_ps(r, one));  // poly*z + r + 1
    return _mm512_scalef_ps(y, fx);                    // y * 2^fx
}

// ── AVX-512 Cephes log core (avx_mathfun formulation) ────────────────────────
inline __m512 log512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __mmask16 invalid = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OQ);
    // cut off denormals / non-positive to the smallest normal
    x = _mm512_max_ps(x, _mm512_castsi512_ps(_mm512_set1_epi32(0x00800000)));
    __m512i xi = _mm512_castps_si512(x);
    // e = (exponent field) - 127
    __m512 e = _mm512_cvtepi32_ps(
        _mm512_sub_epi32(_mm512_srli_epi32(xi, 23), _mm512_set1_epi32(0x7f)));
    // mantissa in [0.5, 1): (x & ~exp_mask) | 0.5
    x = _mm512_castsi512_ps(
        _mm512_and_si512(xi, _mm512_set1_epi32(0x807fffff)));
    x = _mm512_or_ps(x, _mm512_set1_ps(0.5f));
    e = _mm512_add_ps(e, one);
    __mmask16 lt = _mm512_cmp_ps_mask(x, _mm512_set1_ps(0.707106781186547524f),
                                      _CMP_LT_OQ);
    __m512 tmp = _mm512_maskz_mov_ps(lt, x);     // x where m<SQRTHF else 0
    x = _mm512_sub_ps(x, one);
    e = _mm512_mask_sub_ps(e, lt, e, one);       // e -= 1 where m<SQRTHF
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
    y = _mm512_fnmadd_ps(z, _mm512_set1_ps(0.5f), y);   // y -= 0.5*z
    x = _mm512_add_ps(x, y);
    x = _mm512_fmadd_ps(e, _mm512_set1_ps(0.693359375f), x);
    return _mm512_mask_blend_ps(invalid, x, _mm512_set1_ps(NAN));
}

inline __m512 tanh512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 a = exp512(_mm512_mul_ps(x, _mm512_set1_ps(2.0f)));
    return _mm512_div_ps(_mm512_sub_ps(a, one), _mm512_add_ps(a, one));
}

inline __m512 sigmoid512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 e = exp512(_mm512_sub_ps(_mm512_setzero_ps(), x));
    return _mm512_div_ps(one, _mm512_add_ps(one, e));
}

inline __m512 erf512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __m512 ax = _mm512_abs_ps(x);
    __m512 t = _mm512_div_ps(
        one, _mm512_fmadd_ps(_mm512_set1_ps(0.3275911f), ax, one));
    __m512 p = _mm512_set1_ps(1.061405429f);
    p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(-1.453152027f));
    p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(1.421413741f));
    p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(-0.284496736f));
    p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(0.254829592f));
    p = _mm512_mul_ps(p, t);
    __m512 ex = exp512(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(ax, ax)));
    __m512 y = _mm512_fnmadd_ps(p, ex, one);    // 1 - p*exp(-x^2)
    __mmask16 neg = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    return _mm512_mask_sub_ps(y, neg, _mm512_setzero_ps(), y);  // odd: -y for x<0
}

inline __m512 apply512(__m512 v, int kind) {
    const __m512 one = _mm512_set1_ps(1.0f);
    switch (kind) {
    case kExp:     return exp512(v);
    case kLog:     return log512(v);
    case kTanh:    return tanh512(v);
    case kSigmoid: return sigmoid512(v);
    case kSilu:    return _mm512_mul_ps(v, sigmoid512(v));
    case kGelu: {
        __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
        __m512 inner = _mm512_mul_ps(
            _mm512_set1_ps(kSqrt2OverPi),
            _mm512_fmadd_ps(_mm512_set1_ps(kGeluC), x3, v));
        __m512 t = tanh512(inner);
        return _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(0.5f), v),
                             _mm512_add_ps(one, t));
    }
    case kErf:     return erf512(v);
    case kSoftplus: {
        // log1p(exp(-|x|)) + max(x, 0)  — overflow-stable
        __m512 ax = _mm512_abs_ps(v);
        __m512 l = log512(_mm512_add_ps(
            one, exp512(_mm512_sub_ps(_mm512_setzero_ps(), ax))));
        return _mm512_add_ps(l, _mm512_max_ps(v, _mm512_setzero_ps()));
    }
    case kExpm1:   return _mm512_sub_ps(exp512(v), one);
    case kLog1p:   return log512(_mm512_add_ps(one, v));
    default:       return v;
    }
}

}  // namespace

extern "C" void tessera_x86_reference_transcendental_f32(const float* X,
                                                         int64_t n, float* out,
                                                         int kind) {
    for (int64_t i = 0; i < n; ++i) out[i] = scalar_transcendental(X[i], kind);
}

extern "C" void tessera_x86_avx512_transcendental_f32(const float* X, int64_t n,
                                                      float* out, int kind) {
    const int64_t vstep = 16;
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep)
        _mm512_storeu_ps(out + i, apply512(_mm512_loadu_ps(X + i), kind));
    for (; i < n; ++i) out[i] = scalar_transcendental(X[i], kind);
}
