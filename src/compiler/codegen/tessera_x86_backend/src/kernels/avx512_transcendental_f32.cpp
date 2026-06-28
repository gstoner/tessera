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
#include <limits>

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
// trig / inverse-trig / hyperbolic / erfc siblings (ROCm-parity batch)
constexpr int kCos = 10;
constexpr int kTan = 11;
constexpr int kSinh = 12;
constexpr int kCosh = 13;
constexpr int kAsin = 14;
constexpr int kAcos = 15;
constexpr int kAtan = 16;
constexpr int kErfc = 17;
constexpr int kSin = 18;
constexpr int kLgamma = 19;
constexpr int kDigamma = 20;

// ψ(x) = d/dx ln Γ(x). Scalar reference (also the n%16 tail + the x<=0
// reflection lanes) — matches tessera.ops.digamma's _digamma_scalar exactly:
// recurrence up to x>=8 then the asymptotic series; reflection for x<=0; poles
// at non-positive integers return NaN.
inline double digamma_d(double x) {
    constexpr double kPi = 3.14159265358979323846;
    if (x <= 0.0) {
        if (std::fabs(x - std::round(x)) < 1e-12)
            return std::numeric_limits<double>::quiet_NaN();
        return digamma_d(1.0 - x) - kPi / std::tan(kPi * x);
    }
    double result = 0.0;
    while (x < 8.0) { result -= 1.0 / x; x += 1.0; }
    double inv = 1.0 / x, inv2 = inv * inv;
    return result + std::log(x) - 0.5 * inv - inv2 / 12.0
           + inv2 * inv2 / 120.0 - inv2 * inv2 * inv2 / 252.0
           + inv2 * inv2 * inv2 * inv2 / 240.0;
}

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
    case kSin:      return std::sin(v);
    case kCos:      return std::cos(v);
    case kTan:      return std::tan(v);
    case kSinh:     return std::sinh(v);
    case kCosh:     return std::cosh(v);
    case kAsin:     return std::asin(v);
    case kAcos:     return std::acos(v);
    case kAtan:     return std::atan(v);
    case kErfc:     return std::erfc(v);
    case kLgamma:   return std::lgamma(v);
    case kDigamma:  return static_cast<float>(digamma_d(v));
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

// ── ln Γ(x): Numerical Recipes `gammln` — Lanczos g=5 approximation ───────────
// Valid for x > 0; the SIMD series is accurate to ~1e-6 (f32). Reflection /
// non-positive lanes (x < 0.5) fall back to scalar std::lgamma (the exact libm
// reference for those rarer args) — the same SIMD-core + scalar-edge split the
// sin lane uses for large arguments (trigCorrectLarge). Matches math.lgamma.
inline __m512 lgamma512(__m512 x) {
    static const float cof[6] = {
        76.18009172947146f,    -86.50532032941677f,    24.01409824083091f,
        -1.231739572450155f,    0.1208650973866179e-2f, -0.5395239384953e-5f};
    const __m512 half = _mm512_set1_ps(0.5f);
    __m512 y = x;
    __m512 ser = _mm512_set1_ps(1.000000000190015f);
    for (int j = 0; j < 6; ++j) {
        y = _mm512_add_ps(y, _mm512_set1_ps(1.0f));      // ++y, starting x+1
        ser = _mm512_add_ps(ser, _mm512_div_ps(_mm512_set1_ps(cof[j]), y));
    }
    __m512 tmp = _mm512_add_ps(x, _mm512_set1_ps(5.5f));
    tmp = _mm512_sub_ps(tmp, _mm512_mul_ps(_mm512_add_ps(x, half), log512(tmp)));
    const __m512 sqrt2pi = _mm512_set1_ps(2.5066282746310005f);
    __m512 lg = _mm512_add_ps(
        _mm512_sub_ps(_mm512_setzero_ps(), tmp),
        log512(_mm512_div_ps(_mm512_mul_ps(sqrt2pi, ser), x)));
    __mmask16 refl = _mm512_cmp_ps_mask(x, half, _CMP_LT_OQ);
    if (refl) {
        float vx[16], vr[16];
        _mm512_storeu_ps(vx, x);
        _mm512_storeu_ps(vr, lg);
        for (int i = 0; i < 16; ++i)
            if (refl & (1u << i)) vr[i] = std::lgamma(vx[i]);
        lg = _mm512_loadu_ps(vr);
    }
    return lg;
}

// ── ψ(x) = digamma — recurrence to x>=8 + asymptotic series ───────────────────
// SIMD core for x > 0 (8 masked recurrence steps reach x>=8 for any x>0, then
// the asymptotic expansion). x <= 0 lanes (reflection / poles) fall back to the
// scalar digamma_d. Matches tessera.ops.digamma.
inline __m512 digamma512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 eight = _mm512_set1_ps(8.0f);
    __m512 result = _mm512_setzero_ps();
    __m512 xx = x;
    for (int k = 0; k < 8; ++k) {
        __mmask16 m = _mm512_cmp_ps_mask(xx, eight, _CMP_LT_OQ);
        __m512 recip = _mm512_div_ps(one, xx);
        result = _mm512_mask_sub_ps(result, m, result, recip);  // -= 1/xx
        xx = _mm512_mask_add_ps(xx, m, xx, one);                 // xx += 1
    }
    __m512 inv = _mm512_div_ps(one, xx);
    __m512 inv2 = _mm512_mul_ps(inv, inv);
    __m512 inv4 = _mm512_mul_ps(inv2, inv2);
    __m512 inv6 = _mm512_mul_ps(inv4, inv2);
    __m512 inv8 = _mm512_mul_ps(inv4, inv4);
    // log(xx) - inv/2 - inv2/12 + inv4/120 - inv6/252 + inv8/240
    __m512 r = log512(xx);
    r = _mm512_fnmadd_ps(_mm512_set1_ps(0.5f), inv, r);
    r = _mm512_fnmadd_ps(_mm512_set1_ps(1.0f / 12.0f), inv2, r);
    r = _mm512_fmadd_ps(_mm512_set1_ps(1.0f / 120.0f), inv4, r);
    r = _mm512_fnmadd_ps(_mm512_set1_ps(1.0f / 252.0f), inv6, r);
    r = _mm512_fmadd_ps(_mm512_set1_ps(1.0f / 240.0f), inv8, r);
    result = _mm512_add_ps(result, r);
    __mmask16 refl = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OQ);
    if (refl) {
        float vx[16], vr[16];
        _mm512_storeu_ps(vx, x);
        _mm512_storeu_ps(vr, result);
        for (int i = 0; i < 16; ++i)
            if (refl & (1u << i)) vr[i] = static_cast<float>(digamma_d(vx[i]));
        result = _mm512_loadu_ps(vr);
    }
    return result;
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

// Cephes sincosf (avx_mathfun formulation): computes sin and cos together via a
// 4/π argument reduction + the cos/sin minimax polynomials on [-π/4, π/4].
inline void sincos512(__m512 x, __m512* sptr, __m512* cptr) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    __mmask16 sin_sign = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    __m512 ax = _mm512_abs_ps(x);
    // j = (int)(ax * 4/π); round up to even
    __m512i j = _mm512_cvttps_epi32(
        _mm512_mul_ps(ax, _mm512_set1_ps(1.27323954473516f)));
    j = _mm512_and_si512(_mm512_add_epi32(j, _mm512_set1_epi32(1)),
                         _mm512_set1_epi32(~1));
    __m512 y = _mm512_cvtepi32_ps(j);
    __mmask16 poly2 = _mm512_test_epi32_mask(j, _mm512_set1_epi32(2));
    __mmask16 sin_swap = _mm512_test_epi32_mask(j, _mm512_set1_epi32(4));
    __mmask16 cos_keep = _mm512_test_epi32_mask(
        _mm512_sub_epi32(j, _mm512_set1_epi32(2)), _mm512_set1_epi32(4));
    // extended-precision argument reduction: ax -= y*(DP1+DP2+DP3)
    __m512 xr = _mm512_fmadd_ps(y, _mm512_set1_ps(-0.78515625f), ax);
    xr = _mm512_fmadd_ps(y, _mm512_set1_ps(-2.4187564849853515625e-4f), xr);
    xr = _mm512_fmadd_ps(y, _mm512_set1_ps(-3.77489497744594108e-8f), xr);
    __m512 z = _mm512_mul_ps(xr, xr);
    // cos poly on [-π/4, π/4]
    __m512 cp = _mm512_set1_ps(2.443315711809948E-005f);
    cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(-1.388731625493765E-003f));
    cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(4.166664568298827E-002f));
    cp = _mm512_mul_ps(cp, _mm512_mul_ps(z, z));
    cp = _mm512_fnmadd_ps(half, z, cp);   // - 0.5*z
    cp = _mm512_add_ps(cp, one);
    // sin poly
    __m512 sp = _mm512_set1_ps(-1.9515295891E-4f);
    sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(8.3321608736E-3f));
    sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(-1.6666654611E-1f));
    sp = _mm512_mul_ps(sp, _mm512_mul_ps(z, xr));
    sp = _mm512_add_ps(sp, xr);
    // select poly per octant
    __m512 s = _mm512_mask_blend_ps(poly2, sp, cp);
    __m512 c = _mm512_mask_blend_ps(poly2, cp, sp);
    // signs
    s = _mm512_mask_sub_ps(s, sin_swap, _mm512_setzero_ps(), s);
    s = _mm512_mask_sub_ps(s, sin_sign, _mm512_setzero_ps(), s);
    // cos is negated where (j-2)&4 == 0  (i.e. cos_keep false)
    c = _mm512_mask_sub_ps(c, _knot_mask16(cos_keep), _mm512_setzero_ps(), c);
    *sptr = s;
    *cptr = c;
}

// sincos512's f32 4/π reduction loses precision for large |x| (catastrophic
// cancellation in `ax - y*DP` once ax exceeds f32's ~24-bit window), so the SIMD
// trig result diverges from libm there while the scalar n%16 tail stays accurate
// — making the result both wrong and block-size-dependent. Past this magnitude
// we recompute the offending lanes with libm (the SIMD fast path still covers
// the common small-argument case). Threshold is conservative (Cephes sincosf is
// good well past π·2^11); the slow path is rare. kind: 0=sin, 1=cos, 2=tan.
constexpr float kTrigReduceLimit = 8192.0f;  // 2^13
inline __m512 trigCorrectLarge(__m512 v, __m512 simd, int kind) {
    __mmask16 big = _mm512_cmp_ps_mask(_mm512_abs_ps(v),
                                       _mm512_set1_ps(kTrigReduceLimit),
                                       _CMP_GT_OQ);
    if (!big) return simd;  // common case: every lane in range
    float vv[16], rr[16];
    _mm512_storeu_ps(vv, v);
    _mm512_storeu_ps(rr, simd);
    for (int i = 0; i < 16; ++i)
        if (big & (1u << i))
            rr[i] = (kind == 0) ? std::sin(vv[i])
                    : (kind == 1) ? std::cos(vv[i])
                                  : std::tan(vv[i]);
    return _mm512_loadu_ps(rr);
}

// Cephes asinf: poly on [0, 0.5]; |x|>0.5 maps via x = √(0.5(1-|x|)).
inline __m512 asin512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 PIO2 = _mm512_set1_ps(1.5707963267948966f);
    __mmask16 sign = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    __m512 a = _mm512_abs_ps(x);
    __mmask16 big = _mm512_cmp_ps_mask(a, _mm512_set1_ps(0.5f), _CMP_GT_OQ);
    __m512 z_big = _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_sub_ps(one, a));
    __m512 x_big = _mm512_sqrt_ps(z_big);
    __m512 z = _mm512_mask_blend_ps(big, _mm512_mul_ps(a, a), z_big);
    __m512 xx = _mm512_mask_blend_ps(big, a, x_big);
    __m512 p = _mm512_set1_ps(4.2163199048E-2f);
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(2.4181311049E-2f));
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(4.5470025998E-2f));
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(7.4953002686E-2f));
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(1.6666752422E-1f));
    p = _mm512_mul_ps(p, _mm512_mul_ps(z, xx));
    p = _mm512_add_ps(p, xx);
    // big: p = π/2 - 2p
    __m512 p_big = _mm512_fnmadd_ps(_mm512_set1_ps(2.0f), p, PIO2);
    p = _mm512_mask_blend_ps(big, p, p_big);
    return _mm512_mask_sub_ps(p, sign, _mm512_setzero_ps(), p);
}

// Cephes atanf: 3-region reduction (|x|>tan(3π/8), >tan(π/8), else) + poly.
inline __m512 atan512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    __mmask16 sign = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    __m512 a = _mm512_abs_ps(x);
    __mmask16 big = _mm512_cmp_ps_mask(a, _mm512_set1_ps(2.414213562373095f),
                                       _CMP_GT_OQ);
    __mmask16 mid = _mm512_cmp_ps_mask(a, _mm512_set1_ps(0.4142135623730950f),
                                       _CMP_GT_OQ);
    __m512 y = _mm512_setzero_ps();
    __m512 xr = a;
    // mid (and not big): y=π/4, xr=(a-1)/(a+1)
    __m512 xr_mid = _mm512_div_ps(_mm512_sub_ps(a, one), _mm512_add_ps(a, one));
    y = _mm512_mask_blend_ps(mid, y, _mm512_set1_ps(0.7853981633974483f));
    xr = _mm512_mask_blend_ps(mid, xr, xr_mid);
    // big: y=π/2, xr=-1/a
    __m512 xr_big = _mm512_div_ps(_mm512_set1_ps(-1.0f), a);
    y = _mm512_mask_blend_ps(big, y, _mm512_set1_ps(1.5707963267948966f));
    xr = _mm512_mask_blend_ps(big, xr, xr_big);
    __m512 z = _mm512_mul_ps(xr, xr);
    __m512 p = _mm512_set1_ps(8.05374449538e-2f);
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(-1.38776856032E-1f));
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(1.99777106478E-1f));
    p = _mm512_fmadd_ps(p, z, _mm512_set1_ps(-3.33329491539E-1f));
    p = _mm512_mul_ps(p, _mm512_mul_ps(z, xr));
    p = _mm512_add_ps(p, xr);
    p = _mm512_add_ps(y, p);
    return _mm512_mask_sub_ps(p, sign, _mm512_setzero_ps(), p);
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
    case kLgamma:  return lgamma512(v);
    case kDigamma: return digamma512(v);
    case kSoftplus: {
        // log1p(exp(-|x|)) + max(x, 0)  — overflow-stable
        __m512 ax = _mm512_abs_ps(v);
        __m512 l = log512(_mm512_add_ps(
            one, exp512(_mm512_sub_ps(_mm512_setzero_ps(), ax))));
        return _mm512_add_ps(l, _mm512_max_ps(v, _mm512_setzero_ps()));
    }
    case kExpm1:   return _mm512_sub_ps(exp512(v), one);
    case kLog1p:   return log512(_mm512_add_ps(one, v));
    case kSin: {
        __m512 s, c;
        sincos512(v, &s, &c);
        return trigCorrectLarge(v, s, 0);
    }
    case kCos: {
        __m512 s, c;
        sincos512(v, &s, &c);
        return trigCorrectLarge(v, c, 1);
    }
    case kTan: {
        __m512 s, c;
        sincos512(v, &s, &c);
        return trigCorrectLarge(v, _mm512_div_ps(s, c), 2);
    }
    case kSinh: {
        // 0.5 (e^x - e^-x)
        __m512 ex = exp512(v);
        __m512 enx = exp512(_mm512_sub_ps(_mm512_setzero_ps(), v));
        return _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_sub_ps(ex, enx));
    }
    case kCosh: {
        __m512 ex = exp512(v);
        __m512 enx = exp512(_mm512_sub_ps(_mm512_setzero_ps(), v));
        return _mm512_mul_ps(_mm512_set1_ps(0.5f), _mm512_add_ps(ex, enx));
    }
    case kAsin:    return asin512(v);
    case kAcos:    return _mm512_sub_ps(_mm512_set1_ps(1.5707963267948966f),
                                        asin512(v));
    case kAtan:    return atan512(v);
    case kErfc:    return _mm512_sub_ps(one, erf512(v));
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

// ── transcendental-backed BINARY ops (share the exp/log/sigmoid cores) ───────
//
// pow(a, b) = a^b via exp(b·log(a)) — POSITIVE BASE (a>0); a≤0 → NaN/0 like
// exp(b·log(a)). The scalar tail uses std::pow so partial blocks match libm.
extern "C" void tessera_x86_avx512_pow_f32(const float* A, const float* B,
                                           int64_t n, float* out) {
    const int64_t vstep = 16;
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep) {
        __m512 a = _mm512_loadu_ps(A + i);
        __m512 b = _mm512_loadu_ps(B + i);
        _mm512_storeu_ps(out + i, exp512(_mm512_mul_ps(b, log512(a))));
    }
    for (; i < n; ++i) out[i] = std::pow(A[i], B[i]);
}

// silu_mul(a, b) = silu(a)·b = a·sigmoid(a)·b  (SwiGLU gate-multiply).
extern "C" void tessera_x86_avx512_silu_mul_f32(const float* A, const float* B,
                                                int64_t n, float* out) {
    const int64_t vstep = 16;
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep) {
        __m512 a = _mm512_loadu_ps(A + i);
        __m512 b = _mm512_loadu_ps(B + i);
        __m512 s = _mm512_mul_ps(a, sigmoid512(a));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(s, b));
    }
    for (; i < n; ++i) {
        float s = A[i] / (1.0f + std::exp(-A[i]));
        out[i] = s * B[i];
    }
}
