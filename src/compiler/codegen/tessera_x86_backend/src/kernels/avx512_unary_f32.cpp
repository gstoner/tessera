// AVX-512 elementwise unary math kernels (f32) for the Tessera x86 backend.
//
// Applies a pointwise unary math fn over a flat [n] f32 buffer, producing
// out[n].  This is the optimized CPU lane for the algebraic subset of the
// S2 scalar-math / numeric-helper family — the AVX-512 analog of the ROCm
// `generate-rocm-unary-kernel` lane, so these primitives get a REAL vectorized
// CPU kernel rather than only the numpy reference.  A scalar reference is
// provided alongside for on-device validation (the test compares the two + a
// hand-computed expectation).
//
// Covered here are the ops with a DIRECT AVX-512 intrinsic (no polynomial
// approximation required):
//
//   kind 0 = sqrt        _mm512_sqrt_ps
//   kind 1 = rsqrt       1 / sqrt(x)   (full-precision div, not rsqrt14 approx)
//   kind 2 = reciprocal  1 / x
//   kind 3 = abs         _mm512_abs_ps
//   kind 4 = neg         0 - x
//   kind 5 = sign        (x>0) - (x<0), sign(0)=0  (NaN -> 0, finite-domain use)
//
// The transcendentals (exp/log/erf/tanh/…) lower through the ROCm math->ROCDL
// path on GPU; on CPU they remain the numpy reference for now (no fused x86
// claim).  16 f32 lanes per __m512; the tail (n % 16) is handled scalar.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kSqrt = 0;
constexpr int kRsqrt = 1;
constexpr int kRecip = 2;
constexpr int kAbs = 3;
constexpr int kNeg = 4;
constexpr int kSign = 5;
// rounding tail (2026-06-26) — direct AVX-512 roundscale intrinsics
constexpr int kFloor = 6;
constexpr int kCeil = 7;
constexpr int kTrunc = 8;
constexpr int kRound = 9;   // round-half-to-even (numpy.round)

inline float scalar_unary(float v, int kind) {
    switch (kind) {
    case kSqrt:  return std::sqrt(v);
    case kRsqrt: return 1.0f / std::sqrt(v);
    case kRecip: return 1.0f / v;
    case kAbs:   return std::fabs(v);
    case kNeg:   return -v;
    case kSign:  return (v > 0.0f) ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
    case kFloor: return std::floor(v);
    case kCeil:  return std::ceil(v);
    case kTrunc: return std::trunc(v);
    case kRound: return std::nearbyint(v);  // FE_TONEAREST -> ties-to-even
    default:     return v;
    }
}
}  // namespace

extern "C" void tessera_x86_reference_unary_f32(const float* X, int64_t n,
                                                float* out, int kind) {
    for (int64_t i = 0; i < n; ++i) out[i] = scalar_unary(X[i], kind);
}

extern "C" void tessera_x86_avx512_unary_f32(const float* X, int64_t n,
                                             float* out, int kind) {
    const int64_t vstep = 16;  // f32 lanes per __m512
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 negOne = _mm512_set1_ps(-1.0f);
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep) {
        __m512 v = _mm512_loadu_ps(X + i);
        __m512 y;
        switch (kind) {
        case kSqrt:  y = _mm512_sqrt_ps(v); break;
        case kRsqrt: y = _mm512_div_ps(one, _mm512_sqrt_ps(v)); break;
        case kRecip: y = _mm512_div_ps(one, v); break;
        case kAbs:   y = _mm512_abs_ps(v); break;
        case kNeg:   y = _mm512_sub_ps(zero, v); break;
        case kSign: {
            // (x>0) ? 1 : ((x<0) ? -1 : 0)   — ordered compares, NaN -> 0
            __mmask16 pos = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
            __mmask16 neg = _mm512_cmp_ps_mask(v, zero, _CMP_LT_OQ);
            y = _mm512_mask_blend_ps(pos, _mm512_mask_blend_ps(neg, zero, negOne),
                                     one);
            break;
        }
        case kFloor:
            y = _mm512_roundscale_ps(v, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
            break;
        case kCeil:
            y = _mm512_roundscale_ps(v, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
            break;
        case kTrunc:
            y = _mm512_roundscale_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            break;
        case kRound:  // round-half-to-even
            y = _mm512_roundscale_ps(v,
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            break;
        default: y = v; break;
        }
        _mm512_storeu_ps(out + i, y);
    }
    for (; i < n; ++i) out[i] = scalar_unary(X[i], kind);
}
