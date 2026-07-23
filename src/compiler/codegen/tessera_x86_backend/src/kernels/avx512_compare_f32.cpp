// AVX-512 elementwise comparison kernels (f32 -> bool) for the Tessera x86
// backend.
//
// Applies a pointwise comparison over two flat [n] f32 buffers, producing an
// [n] uint8 mask (0/1) — the optimized CPU lane for the S2 comparison family,
// the AVX-512 analog of the ROCm `generate-rocm-compare-kernel` lane. A scalar
// reference is provided alongside for on-device validation.
//
//   kind 0 = eq   a == b      kind 3 = le   a <= b
//   kind 1 = ne   a != b      kind 4 = gt   a >  b
//   kind 2 = lt   a <  b      kind 5 = ge   a >= b
//
// NaN semantics match numpy: every predicate is ORDERED (NaN -> false) EXCEPT
// `ne`, which is unordered-not-equal (NaN -> true).  C's native float operators
// already follow exactly this rule, so the scalar reference uses them directly;
// the AVX-512 lane uses the matching `_CMP_*_OQ` / `_CMP_NEQ_UQ` predicates.
//
// 16 f32 lanes per __m512; the comparison yields a __mmask16 which
// `_mm_maskz_set1_epi8(mask, 1)` expands to 16 bytes of 0/1 (needs AVX512BW +
// AVX512VL). The tail (n % 16) is handled scalar.

#include <immintrin.h>
#include <cstdint>

namespace {
constexpr int kEq = 0;
constexpr int kNe = 1;
constexpr int kLt = 2;
constexpr int kLe = 3;
constexpr int kGt = 4;
constexpr int kGe = 5;

inline uint8_t scalar_compare(float a, float b, int kind) {
    switch (kind) {
    case kEq: return a == b ? 1 : 0;
    case kNe: return a != b ? 1 : 0;
    case kLt: return a < b ? 1 : 0;
    case kLe: return a <= b ? 1 : 0;
    case kGt: return a > b ? 1 : 0;
    case kGe: return a >= b ? 1 : 0;
    default: return 0;
    }
}

inline __mmask16 vec_compare(__m512 a, __m512 b, int kind) {
    switch (kind) {
    case kEq: return _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ);
    case kNe: return _mm512_cmp_ps_mask(a, b, _CMP_NEQ_UQ);
    case kLt: return _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
    case kLe: return _mm512_cmp_ps_mask(a, b, _CMP_LE_OQ);
    case kGt: return _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
    case kGe: return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
    default: return 0;
    }
}

template <typename T>
inline uint8_t scalar_compare_integer(T a, T b, int kind) {
    switch (kind) {
    case kEq: return a == b;
    case kNe: return a != b;
    case kLt: return a < b;
    case kLe: return a <= b;
    case kGt: return a > b;
    case kGe: return a >= b;
    default: return 0;
    }
}

inline __mmask16 vec_compare_i32(__m512i a, __m512i b, int kind) {
    switch (kind) {
    case kEq: return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_EQ);
    case kNe: return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NE);
    case kLt: return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT);
    case kLe: return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LE);
    case kGt: return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLE);
    case kGe: return _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_NLT);
    default: return 0;
    }
}

inline __mmask16 vec_compare_u32(__m512i a, __m512i b, int kind) {
    switch (kind) {
    case kEq: return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_EQ);
    case kNe: return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NE);
    case kLt: return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_LT);
    case kLe: return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_LE);
    case kGt: return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLE);
    case kGe: return _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_NLT);
    default: return 0;
    }
}
}  // namespace

extern "C" void tessera_x86_reference_compare_f32(const float* A, const float* B,
                                                  int64_t n, uint8_t* out,
                                                  int kind) {
    for (int64_t i = 0; i < n; ++i) out[i] = scalar_compare(A[i], B[i], kind);
}

extern "C" void tessera_x86_avx512_compare_f32(const float* A, const float* B,
                                               int64_t n, uint8_t* out,
                                               int kind) {
    const int64_t vstep = 16;  // f32 lanes per __m512
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep) {
        __m512 a = _mm512_loadu_ps(A + i);
        __m512 b = _mm512_loadu_ps(B + i);
        __mmask16 m = vec_compare(a, b, kind);
        __m128i bytes = _mm_maskz_set1_epi8(m, 1);  // 0/1 per lane
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), bytes);
    }
    for (; i < n; ++i) out[i] = scalar_compare(A[i], B[i], kind);
}

extern "C" void tessera_x86_avx512_compare_i32(const int32_t* A,
                                               const int32_t* B, int64_t n,
                                               uint8_t* out, int kind) {
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512i a = _mm512_loadu_si512(A + i);
        __m512i b = _mm512_loadu_si512(B + i);
        __mmask16 m = vec_compare_i32(a, b, kind);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i),
                         _mm_maskz_set1_epi8(m, 1));
    }
    for (; i < n; ++i) out[i] = scalar_compare_integer(A[i], B[i], kind);
}

extern "C" void tessera_x86_avx512_compare_u32(const uint32_t* A,
                                               const uint32_t* B, int64_t n,
                                               uint8_t* out, int kind) {
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512i a = _mm512_loadu_si512(A + i);
        __m512i b = _mm512_loadu_si512(B + i);
        __mmask16 m = vec_compare_u32(a, b, kind);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i),
                         _mm_maskz_set1_epi8(m, 1));
    }
    for (; i < n; ++i) out[i] = scalar_compare_integer(A[i], B[i], kind);
}
