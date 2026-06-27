// AVX-512 elementwise ternary select (where) for the Tessera x86 backend.
//
// out[i] = cond[i] ? a[i] : b[i]  — numpy `where`/select. cond is an i8 boolean
// (normalized via != 0; any nonzero is true). a/b/out are f32. The mask from the
// i8 cond drives `_mm512_mask_blend_ps`. A scalar reference is provided alongside
// for on-device validation. 16 f32 lanes per __m512; the tail (n % 16) is scalar.

#include <immintrin.h>
#include <cstdint>

extern "C" void tessera_x86_reference_where_f32(const uint8_t* C, const float* A,
                                                const float* B, int64_t n,
                                                float* out) {
    for (int64_t i = 0; i < n; ++i) out[i] = C[i] ? A[i] : B[i];
}

extern "C" void tessera_x86_avx512_where_f32(const uint8_t* C, const float* A,
                                             const float* B, int64_t n,
                                             float* out) {
    const int64_t vstep = 16;
    int64_t i = 0;
    for (; i + vstep <= n; i += vstep) {
        // load 16 i8 cond bytes → mask of (byte != 0)
        __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(C + i));
        __mmask16 m = _mm_cmpneq_epi8_mask(c, _mm_setzero_si128());
        __m512 a = _mm512_loadu_ps(A + i);
        __m512 b = _mm512_loadu_ps(B + i);
        // blend: lanes where m → a, else b  (_mm512_mask_blend_ps(m, b, a))
        __m512 y = _mm512_mask_blend_ps(m, b, a);
        _mm512_storeu_ps(out + i, y);
    }
    for (; i < n; ++i) out[i] = C[i] ? A[i] : B[i];
}
