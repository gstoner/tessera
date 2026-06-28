// AVX-512 unary predicate kernels (f32 -> i8 bool) for the Tessera x86 backend
// (P2b of S_SERIES_GAP_CLOSURE_PLAN) — the numeric_helper predicate family:
//
//   kind 0 isnan    : x != x
//   kind 1 isinf    : |x| == +inf
//   kind 2 isfinite : not (nan or inf)
//
// Output is one byte per element (0 / 1), matching the compare lane's float-in /
// bool-out ABI. 16 f32 lanes per __m512; the mask -> 0/1 bytes via
// `_mm_maskz_set1_epi8`. A scalar reference is provided alongside.

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
constexpr int kIsNan = 0, kIsInf = 1, kIsFinite = 2;
}  // namespace

extern "C" void tessera_x86_reference_predicate_f32(const float* X, int64_t n,
                                                    int8_t* out, int kind) {
    for (int64_t i = 0; i < n; ++i) {
        float x = X[i];
        bool r;
        if (kind == kIsNan) r = std::isnan(x);
        else if (kind == kIsInf) r = std::isinf(x);
        else r = std::isfinite(x);
        out[i] = r ? 1 : 0;
    }
}

extern "C" void tessera_x86_avx512_predicate_f32(const float* X, int64_t n,
                                                 int8_t* out, int kind) {
    const __m512 inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_loadu_ps(X + i);
        __m512 ax = _mm512_abs_ps(x);
        __mmask16 m;
        if (kind == kIsNan) {
            m = _mm512_cmp_ps_mask(x, x, _CMP_UNORD_Q);           // x != x
        } else if (kind == kIsInf) {
            m = _mm512_cmp_ps_mask(ax, inf, _CMP_EQ_OQ);          // |x| == inf
        } else {  // isfinite = ordered AND |x| < inf
            __mmask16 ord = _mm512_cmp_ps_mask(x, x, _CMP_ORD_Q);
            __mmask16 lt = _mm512_cmp_ps_mask(ax, inf, _CMP_LT_OQ);
            m = ord & lt;
        }
        // mask -> 0/1 bytes
        __m128i bytes = _mm_maskz_set1_epi8(m, 1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), bytes);
    }
    for (; i < n; ++i) {
        float x = X[i];
        bool r = (kind == kIsNan)   ? std::isnan(x)
                 : (kind == kIsInf) ? std::isinf(x)
                                    : std::isfinite(x);
        out[i] = r ? 1 : 0;
    }
}
