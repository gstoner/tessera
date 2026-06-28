// AVX-512 radix-2 complex FFT (f32) for the Tessera x86 backend — Spectral PR2.
//
// In-place iterative Cooley-Tukey (decimation-in-time) C2C transform of a
// batch of power-of-two-length rows. Data is interleaved complex (re, im) f32,
// row-major [batch, 2*n]. Forward uses e^{-2πi kn/N}; inverse uses e^{+...} and
// is UNNORMALIZED (the runtime applies the plan's 1/N / 1/√N scale, per
// SpectralPlan). The strategy/normalization decisions live in the planner
// (compiler/spectral_plan.py); this kernel only executes a radix-2 plan.
//
// Each stage's twiddles are gathered into a contiguous table so the butterfly
// inner loop over k is a flat complex run — vectorized 8 complex (one __m512) at
// a time via deinterleave/interleave permutes + an FMA complex multiply; the
// half<8 tail (early stages) is scalar. Validated vs np.fft in test_fft.cpp.

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

inline uint32_t bit_reverse(uint32_t x, int bits) {
    uint32_t r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

// deinterleave 8 complex (16 floats) -> (re[8] in lo lanes, im[8] in lo lanes)
const __m512i kEven = []{ return _mm512_setr_epi32(0,2,4,6,8,10,12,14,
                                                   16,18,20,22,24,26,28,30); }();
const __m512i kOdd  = []{ return _mm512_setr_epi32(1,3,5,7,9,11,13,15,
                                                   17,19,21,23,25,27,29,31); }();
const __m512i kIntLo = []{ return _mm512_setr_epi32(0,16,1,17,2,18,3,19,
                                                    4,20,5,21,6,22,7,23); }();
const __m512i kIntHi = []{ return _mm512_setr_epi32(8,24,9,25,10,26,11,27,
                                                    12,28,13,29,14,30,15,31); }();

}  // namespace

extern "C" void tessera_x86_fft_c2c_f32(float* data, int64_t batch, int64_t n,
                                        int inverse) {
    if (n <= 1) return;
    int bits = 0;
    while ((int64_t(1) << bits) < n) ++bits;

    // twiddle table tw[k] for k in [0, n/2): exp(sign * -2π i k / n)
    const double sign = inverse ? 1.0 : -1.0;
    const int64_t nh = n / 2;
    std::vector<float> twr(nh), twi(nh);
    for (int64_t k = 0; k < nh; ++k) {
        double ang = sign * 2.0 * M_PI * (double)k / (double)n;
        twr[k] = (float)std::cos(ang);
        twi[k] = (float)std::sin(ang);
    }
    std::vector<float> str, sti;  // per-stage contiguous twiddles (reused)

    for (int64_t b = 0; b < batch; ++b) {
        float* x = data + b * 2 * n;
        // bit-reversal permutation (swap complex pairs)
        for (int64_t i = 1; i < n; ++i) {
            int64_t j = bit_reverse((uint32_t)i, bits);
            if (i < j) {
                std::swap(x[2 * i], x[2 * j]);
                std::swap(x[2 * i + 1], x[2 * j + 1]);
            }
        }
        // stages: len = 2, 4, ..., n
        for (int64_t len = 2; len <= n; len <<= 1) {
            int64_t half = len / 2;
            int64_t step = n / len;  // twiddle stride
            if ((int64_t)str.size() < half) { str.resize(half); sti.resize(half); }
            for (int64_t k = 0; k < half; ++k) {
                str[k] = twr[k * step]; sti[k] = twi[k * step];
            }
            for (int64_t base = 0; base < n; base += len) {
                float* lo = x + 2 * base;             // a[base + k]
                float* hi = x + 2 * (base + half);    // a[base + half + k]
                int64_t k = 0;
                // 16 complex per iteration: two __m512 loads (32 floats) per
                // half, deinterleaved to 16 re + 16 im lanes.
                for (; k + 16 <= half; k += 16) {
                    __m512 alo = _mm512_loadu_ps(lo + 2 * k);
                    __m512 ahi = _mm512_loadu_ps(lo + 2 * k + 16);
                    __m512 blo = _mm512_loadu_ps(hi + 2 * k);
                    __m512 bhi = _mm512_loadu_ps(hi + 2 * k + 16);
                    __m512 ar = _mm512_permutex2var_ps(alo, kEven, ahi);
                    __m512 ai = _mm512_permutex2var_ps(alo, kOdd, ahi);
                    __m512 br = _mm512_permutex2var_ps(blo, kEven, bhi);
                    __m512 bi = _mm512_permutex2var_ps(blo, kOdd, bhi);
                    __m512 tr = _mm512_loadu_ps(&str[k]);   // already deinterleaved
                    __m512 ti = _mm512_loadu_ps(&sti[k]);
                    // v = tw * b : vr = br*tr - bi*ti ; vi = br*ti + bi*tr
                    __m512 vr = _mm512_fmsub_ps(br, tr, _mm512_mul_ps(bi, ti));
                    __m512 vi = _mm512_fmadd_ps(br, ti, _mm512_mul_ps(bi, tr));
                    __m512 or0 = _mm512_add_ps(ar, vr), oi0 = _mm512_add_ps(ai, vi);
                    __m512 or1 = _mm512_sub_ps(ar, vr), oi1 = _mm512_sub_ps(ai, vi);
                    _mm512_storeu_ps(lo + 2 * k, _mm512_permutex2var_ps(or0, kIntLo, oi0));
                    _mm512_storeu_ps(lo + 2 * k + 16, _mm512_permutex2var_ps(or0, kIntHi, oi0));
                    _mm512_storeu_ps(hi + 2 * k, _mm512_permutex2var_ps(or1, kIntLo, oi1));
                    _mm512_storeu_ps(hi + 2 * k + 16, _mm512_permutex2var_ps(or1, kIntHi, oi1));
                }
                for (; k < half; ++k) {        // scalar tail (early stages)
                    float ar = lo[2 * k], ai = lo[2 * k + 1];
                    float br = hi[2 * k], bi = hi[2 * k + 1];
                    float tr = str[k], ti = sti[k];
                    float vr = br * tr - bi * ti, vi = br * ti + bi * tr;
                    lo[2 * k] = ar + vr; lo[2 * k + 1] = ai + vi;
                    hi[2 * k] = ar - vr; hi[2 * k + 1] = ai - vi;
                }
            }
        }
    }
}
