// AVX-512 position-encoding kernels (f32): rope (interleaved-pair rotary
// embedding) and alibi (positional-bias generator) — the CPU analog of the ROCm
// rope/alibi lanes, matching their op signatures.
//
//   rope(x, theta)   O[..,2p]   = e·cos(a) − o·sin(a)
//                    O[..,2p+1] = e·sin(a) + o·cos(a)
//        where e=x[..,2p], o=x[..,2p+1], a=theta[..,2p]   (D even; one row at a
//        time). x and theta are both [.., D].
//   alibi(slopes,H,S) bias[h,i,j] = slope[h]·(j − i), default slope ramp
//        2^(−8k/H), k=1..H. Output [H, S, S].
//
// rope vectorizes 16 pairs (32 floats) at a time: deinterleave even/odd via
// permutex2var, a vectorized Cephes sincos on the angle vector, then
// re-interleave. D%32 tail handled scalar. alibi is a flat FMA over j.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {

// Cephes sincosf (avx_mathfun) — same core as avx512_transcendental_f32.cpp.
inline void sincos512(__m512 x, __m512* sptr, __m512* cptr) {
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    __mmask16 sin_sign = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OQ);
    __m512 ax = _mm512_abs_ps(x);
    __m512i j = _mm512_cvttps_epi32(
        _mm512_mul_ps(ax, _mm512_set1_ps(1.27323954473516f)));
    j = _mm512_and_si512(_mm512_add_epi32(j, _mm512_set1_epi32(1)),
                         _mm512_set1_epi32(~1));
    __m512 y = _mm512_cvtepi32_ps(j);
    __mmask16 poly2 = _mm512_test_epi32_mask(j, _mm512_set1_epi32(2));
    __mmask16 sin_swap = _mm512_test_epi32_mask(j, _mm512_set1_epi32(4));
    __mmask16 cos_keep = _mm512_test_epi32_mask(
        _mm512_sub_epi32(j, _mm512_set1_epi32(2)), _mm512_set1_epi32(4));
    __m512 xr = _mm512_fmadd_ps(y, _mm512_set1_ps(-0.78515625f), ax);
    xr = _mm512_fmadd_ps(y, _mm512_set1_ps(-2.4187564849853515625e-4f), xr);
    xr = _mm512_fmadd_ps(y, _mm512_set1_ps(-3.77489497744594108e-8f), xr);
    __m512 z = _mm512_mul_ps(xr, xr);
    __m512 cp = _mm512_set1_ps(2.443315711809948E-005f);
    cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(-1.388731625493765E-003f));
    cp = _mm512_fmadd_ps(cp, z, _mm512_set1_ps(4.166664568298827E-002f));
    cp = _mm512_mul_ps(cp, _mm512_mul_ps(z, z));
    cp = _mm512_fnmadd_ps(half, z, cp);
    cp = _mm512_add_ps(cp, one);
    __m512 sp = _mm512_set1_ps(-1.9515295891E-4f);
    sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(8.3321608736E-3f));
    sp = _mm512_fmadd_ps(sp, z, _mm512_set1_ps(-1.6666654611E-1f));
    sp = _mm512_mul_ps(sp, _mm512_mul_ps(z, xr));
    sp = _mm512_add_ps(sp, xr);
    __m512 s = _mm512_mask_blend_ps(poly2, sp, cp);
    __m512 c = _mm512_mask_blend_ps(poly2, cp, sp);
    s = _mm512_mask_sub_ps(s, sin_swap, _mm512_setzero_ps(), s);
    s = _mm512_mask_sub_ps(s, sin_sign, _mm512_setzero_ps(), s);
    c = _mm512_mask_sub_ps(c, _knot_mask16(cos_keep), _mm512_setzero_ps(), c);
    *sptr = s;
    *cptr = c;
}

inline float scalar_sin(float a) { return std::sin(a); }
inline float scalar_cos(float a) { return std::cos(a); }

}  // namespace

extern "C" void tessera_x86_avx512_rope_f32(const float* X, const float* Theta,
                                            int64_t M, int64_t D, float* out) {
    // permute index vectors (constant): deinterleave 32 lanes -> 16 even/16 odd
    const __m512i idx_even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                               16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i idx_odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15,
                                              17, 19, 21, 23, 25, 27, 29, 31);
    // re-interleave: lanes pick from {oe(0..15), oo(16..31)}
    const __m512i idx_lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19,
                                             4, 20, 5, 21, 6, 22, 7, 23);
    const __m512i idx_hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27,
                                             12, 28, 13, 29, 14, 30, 15, 31);
    for (int64_t m = 0; m < M; ++m) {
        const float* x = X + m * D;
        const float* th = Theta + m * D;
        float* o = out + m * D;
        int64_t d = 0;
        for (; d + 32 <= D; d += 32) {
            __m512 lo = _mm512_loadu_ps(x + d);
            __m512 hi = _mm512_loadu_ps(x + d + 16);
            __m512 tlo = _mm512_loadu_ps(th + d);
            __m512 thi = _mm512_loadu_ps(th + d + 16);
            __m512 even = _mm512_permutex2var_ps(lo, idx_even, hi);
            __m512 odd = _mm512_permutex2var_ps(lo, idx_odd, hi);
            __m512 a = _mm512_permutex2var_ps(tlo, idx_even, thi);
            __m512 s, c;
            sincos512(a, &s, &c);
            __m512 oe = _mm512_fmsub_ps(even, c, _mm512_mul_ps(odd, s));
            __m512 oo = _mm512_fmadd_ps(even, s, _mm512_mul_ps(odd, c));
            _mm512_storeu_ps(o + d, _mm512_permutex2var_ps(oe, idx_lo, oo));
            _mm512_storeu_ps(o + d + 16,
                             _mm512_permutex2var_ps(oe, idx_hi, oo));
        }
        for (; d + 2 <= D; d += 2) {
            float e = x[d], od = x[d + 1], a = th[d];
            float cs = scalar_cos(a), sn = scalar_sin(a);
            o[d] = e * cs - od * sn;
            o[d + 1] = e * sn + od * cs;
        }
    }
}

extern "C" void tessera_x86_avx512_alibi_f32(const float* Slopes, int64_t H,
                                             int64_t S, float* out) {
    // bias[h, i, j] = slope[h] * (j - i)
    for (int64_t h = 0; h < H; ++h) {
        float slope = Slopes[h];
        __m512 vslope = _mm512_set1_ps(slope);
        for (int64_t i = 0; i < S; ++i) {
            float* row = out + (h * S + i) * S;
            __m512 vi = _mm512_set1_ps((float)i);
            int64_t j = 0;
            for (; j + 16 <= S; j += 16) {
                __m512 vj = _mm512_add_ps(
                    _mm512_set1_ps((float)j),
                    _mm512_setr_ps(0, 1, 2, 3, 4, 5, 6, 7,
                                   8, 9, 10, 11, 12, 13, 14, 15));
                _mm512_storeu_ps(row + j,
                                 _mm512_mul_ps(vslope, _mm512_sub_ps(vj, vi)));
            }
            for (; j < S; ++j) row[j] = slope * (float)(j - i);
        }
    }
}
