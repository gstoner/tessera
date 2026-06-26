// AVX-512 elementwise binary arithmetic kernels (f32) for the Tessera x86
// backend.
//
// Applies a pointwise binary arithmetic fn over two flat [n] f32 buffers,
// producing out[n].  This is the optimized CPU lane for the algebraic subset of
// the S2 binary-arithmetic family — the AVX-512 analog of the ROCm
// `generate-rocm-binary-kernel` lane, so these primitives get a REAL vectorized
// CPU kernel rather than only the numpy reference.  A scalar reference is
// provided alongside for on-device validation (the test compares the two + a
// hand-computed expectation).
//
// Covered here are the ops with a DIRECT AVX-512 intrinsic (no polynomial
// approximation required):
//
//   kind 0 = sub        _mm512_sub_ps        a - b
//   kind 1 = div        _mm512_div_ps        a / b
//   kind 2 = maximum     NaN-propagating max(a, b)   (matches numpy.maximum)
//   kind 3 = minimum     NaN-propagating min(a, b)   (matches numpy.minimum)
//
// `pow` is transcendental (exp(b·log a)) and lowers through the ROCm math->ROCDL
// path on GPU; on CPU it remains the numpy reference for now (no fused x86
// claim), exactly as the unary lane leaves exp/log/erf/…  to the reference.
//
// NaN handling: _mm512_max_ps / _mm512_min_ps follow the SSE convention (return
// the *second* operand when either input is NaN), which does NOT match numpy's
// NaN-propagating maximum/minimum.  We therefore detect NaN inputs (unordered
// compare) and blend a NaN into those lanes, matching the ROCm `arith.maximumf`
// / `minimumf` semantics.  16 f32 lanes per __m512; the tail (n % 16) is scalar.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {
constexpr int kSub = 0;
constexpr int kDiv = 1;
constexpr int kMax = 2;
constexpr int kMin = 3;

inline float scalar_binary(float a, float b, int kind) {
    switch (kind) {
    case kSub: return a - b;
    case kDiv: return a / b;
    case kMax:  // NaN-propagating (numpy.maximum), not fmax
        if (std::isnan(a) || std::isnan(b)) return std::nanf("");
        return a > b ? a : b;
    case kMin:
        if (std::isnan(a) || std::isnan(b)) return std::nanf("");
        return a < b ? a : b;
    default: return a;
    }
}
}  // namespace

extern "C" void tessera_x86_reference_binary_f32(const float* A, const float* B,
                                                 int64_t n, float* out,
                                                 int kind) {
    for (int64_t i = 0; i < n; ++i) out[i] = scalar_binary(A[i], B[i], kind);
}

extern "C" void tessera_x86_avx512_binary_f32(const float* A, const float* B,
                                              int64_t n, float* out, int kind) {
    const int64_t vstep = 16;  // f32 lanes per __m512
    const __m512 qnan = _mm512_set1_ps(std::nanf(""));
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep) {
        __m512 a = _mm512_loadu_ps(A + i);
        __m512 b = _mm512_loadu_ps(B + i);
        __m512 y;
        switch (kind) {
        case kSub: y = _mm512_sub_ps(a, b); break;
        case kDiv: y = _mm512_div_ps(a, b); break;
        case kMax: {
            // NaN-propagating: where (a or b) is unordered (NaN), force NaN.
            __mmask16 un = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
            y = _mm512_mask_blend_ps(un, _mm512_max_ps(a, b), qnan);
            break;
        }
        case kMin: {
            __mmask16 un = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
            y = _mm512_mask_blend_ps(un, _mm512_min_ps(a, b), qnan);
            break;
        }
        default: y = a; break;
        }
        _mm512_storeu_ps(out + i, y);
    }
    for (; i < n; ++i) out[i] = scalar_binary(A[i], B[i], kind);
}
