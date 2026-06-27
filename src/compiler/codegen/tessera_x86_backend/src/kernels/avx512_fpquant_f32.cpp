// AVX-512 low-precision float-grid quantization kernel (f32) for the Tessera x86
// backend — the CPU lane for the S9 fp4 / fp6 / nvfp4 quantize ops (the fp8 / fp6
// / fp4 family). Snaps each value to the nearest representable low-precision
// float grid, the format-agnostic mantissa-snap the reference defines:
//
//   ax      = min(|x|, max_normal)
//   ulp     = 2^(floor(log2(ax)) - mantissa_bits)
//   rounded = min(round_to_nearest_even(ax / ulp) * ulp, max_normal)
//   out     = sign(x) * rounded
//
// Parameterized by (max_normal, mantissa_bits, min_exp) so one kernel covers
// e2m1 (fp4), e2m3 / e3m2 (fp6) and e4m3 / e5m2 (fp8 grid). ``min_exp`` clamps
// the per-value exponent so the subnormal grid is FLAT below the format's
// smallest normal — fp8 (e4m3 = −6, e5m2 = −14) follows IEEE gradual-underflow
// (matching ml_dtypes), while fp4/fp6 pass a very negative min_exp to keep the
// reference's pure mantissa-snap (no subnormal flattening). NaNs propagate (a
// NaN sentinel must NOT become a saturated finite value). Uses the AVX-512
// getexp (floor log2), roundscale (RNE) and scalef (2^n) intrinsics — no
// polynomial. The runtime applies the per-tensor scale around this. 16 f32
// lanes/__m512; n%16 tail scalar.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {

inline float scalar_fpquant(float v, float max_normal, int mantissa_bits,
                            int min_exp) {
    if (std::isnan(v)) return v;                       // propagate NaN
    float s = v < 0.0f ? -1.0f : 1.0f;
    float ax = std::fmin(std::fabs(v), max_normal);
    if (ax <= 0.0f) return 0.0f;
    int e = (int)std::floor(std::log2(ax));
    if (e < min_exp) e = min_exp;                      // flat subnormal grid
    float ulp = std::ldexp(1.0f, e - mantissa_bits);
    float rounded = std::nearbyint(ax / ulp) * ulp;    // RNE (FE_TONEAREST)
    rounded = std::fmin(rounded, max_normal);
    return s * rounded;
}

}  // namespace

extern "C" void tessera_x86_avx512_fpquant_f32(const float* X, int64_t n,
                                               float max_normal,
                                               int mantissa_bits, int min_exp,
                                               float* out) {
    const __m512 vmax = _mm512_set1_ps(max_normal);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 vmant = _mm512_set1_ps((float)mantissa_bits);
    const __m512 vminexp = _mm512_set1_ps((float)min_exp);
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(X + i);
        __mmask16 nan = _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q);  // v != v
        __m512 ax = _mm512_min_ps(_mm512_abs_ps(v), vmax);
        __mmask16 pos = _mm512_cmp_ps_mask(ax, zero, _CMP_GT_OQ);  // ax > 0
        // e = max(floor(log2(ax)), min_exp) — flat subnormal grid below min_exp
        __m512 e = _mm512_max_ps(_mm512_getexp_ps(ax), vminexp);
        __m512 ulp = _mm512_scalef_ps(_mm512_set1_ps(1.0f),
                                      _mm512_sub_ps(e, vmant));
        __m512 q = _mm512_roundscale_ps(_mm512_div_ps(ax, ulp),
                                        _MM_FROUND_TO_NEAREST_INT |
                                        _MM_FROUND_NO_EXC);
        __m512 rounded = _mm512_min_ps(_mm512_mul_ps(q, ulp), vmax);
        rounded = _mm512_maskz_mov_ps(pos, rounded);   // ax==0 -> 0
        // restore sign of the original input
        __m512 y = _mm512_mask_blend_ps(
            _mm512_cmp_ps_mask(v, zero, _CMP_LT_OQ), rounded,
            _mm512_sub_ps(zero, rounded));
        y = _mm512_mask_blend_ps(nan, y, _mm512_set1_ps(NAN));  // propagate NaN
        _mm512_storeu_ps(out + i, y);
    }
    for (; i < n; ++i)
        out[i] = scalar_fpquant(X[i], max_normal, mantissa_bits, min_exp);
}
