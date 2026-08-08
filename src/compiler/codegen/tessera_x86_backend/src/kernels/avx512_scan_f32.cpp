// x86 elementwise inclusive prefix scan (f32) for the Tessera x86 backend.
//
// Inclusive scan of each row of a [rows, cols] f32 matrix along the last axis —
// the CPU analog of the ROCm block-scan lane. kind: 0 = cumsum, 1 = cumprod,
// 2 = cummax, 3 = cummin. A scalar reference is provided alongside for on-device
// validation.
//
// Sum/product use an in-register 16-lane Hillis-Steele prefix. A scalar carry
// joins adjacent chunks and handles the tail. On Zen 5 this wins for arithmetic
// scans; the four permute/NaN-check stages regress extrema scans, so max/min
// deliberately retain the scalar recurrence.

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
constexpr int kCumsum = 0;
constexpr int kCumprod = 1;
constexpr int kCummax = 2;
constexpr int kCummin = 3;

inline float identity(int kind) {
    switch (kind) {
    case kCumsum: return 0.0f;
    case kCumprod: return 1.0f;
    case kCummax: return -std::numeric_limits<float>::infinity();
    case kCummin: return std::numeric_limits<float>::infinity();
    default: return 0.0f;
    }
}

inline float combine(float a, float v, int kind) {
    switch (kind) {
    case kCumsum: return a + v;
    case kCumprod: return a * v;
    case kCummax:
        if (std::isnan(a) || std::isnan(v))
            return std::numeric_limits<float>::quiet_NaN();
        return a > v ? a : v;
    case kCummin:
        if (std::isnan(a) || std::isnan(v))
            return std::numeric_limits<float>::quiet_NaN();
        return a < v ? a : v;
    default: return v;
    }
}

inline __m512 combine_vector(__m512 a, __m512 b, int kind) {
    switch (kind) {
    case kCumsum: return _mm512_add_ps(a, b);
    case kCumprod: return _mm512_mul_ps(a, b);
    case kCummax: {
        const __mmask16 nan = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
        return _mm512_mask_mov_ps(
            _mm512_max_ps(a, b), nan,
            _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()));
    }
    case kCummin: {
        const __mmask16 nan = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
        return _mm512_mask_mov_ps(
            _mm512_min_ps(a, b), nan,
            _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()));
    }
    default: return b;
    }
}

inline __m512 prefix_stage(__m512 values, __m512i indices, __mmask16 active,
                           int kind) {
    const __m512 shifted = _mm512_maskz_permutexvar_ps(active, indices, values);
    const __m512 combined = combine_vector(shifted, values, kind);
    return _mm512_mask_mov_ps(values, active, combined);
}
}  // namespace

extern "C" void tessera_x86_reference_scan_f32(const float* X, int64_t rows,
                                               int64_t cols, float* out,
                                               int kind) {
    for (int64_t r = 0; r < rows; ++r) {
        float acc = identity(kind);
        const float* row = X + r * cols;
        float* orow = out + r * cols;
        for (int64_t c = 0; c < cols; ++c) {
            acc = combine(acc, row[c], kind);
            orow[c] = acc;
        }
    }
}

extern "C" void tessera_x86_avx512_scan_f32(const float* X, int64_t rows,
                                            int64_t cols, float* out, int kind) {
    if (kind == kCummax || kind == kCummin) {
        tessera_x86_reference_scan_f32(X, rows, cols, out, kind);
        return;
    }
    const __m512i shift1 = _mm512_setr_epi32(
        0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    const __m512i shift2 = _mm512_setr_epi32(
        0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    const __m512i shift4 = _mm512_setr_epi32(
        0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    const __m512i shift8 = _mm512_setr_epi32(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7);
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = X + r * cols;
        float* output = out + r * cols;
        float carry = identity(kind);
        int64_t c = 0;
        for (; c + 16 <= cols; c += 16) {
            __m512 values = _mm512_loadu_ps(row + c);
            values = prefix_stage(values, shift1, 0xfffe, kind);
            values = prefix_stage(values, shift2, 0xfffc, kind);
            values = prefix_stage(values, shift4, 0xfff0, kind);
            values = prefix_stage(values, shift8, 0xff00, kind);
            if (c != 0)
                values = combine_vector(_mm512_set1_ps(carry), values, kind);
            _mm512_storeu_ps(output + c, values);
            carry = output[c + 15];
        }
        for (; c < cols; ++c) {
            carry = combine(carry, row[c], kind);
            output[c] = carry;
        }
    }
}
