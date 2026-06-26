// AVX-512 row-wise reduction kernels (f32) for the Tessera x86 backend.
//
// Reduces each row of a [rows, cols] f32 matrix over the last axis, producing
// out[rows].  kind: 0 = sum, 1 = max, 2 = mean (= sum / cols).  This is the
// optimized CPU lane for the S-series reduction family (reduce_sum / mean / max
// over the last axis) — the AVX-512 analog of the AMX/AVX-512 GEMM lane, so the
// reduction primitives get a REAL vectorized CPU kernel rather than only the
// numpy reference.  A scalar reference is provided alongside for on-device
// validation (the test compares the two + a hand-computed expectation).
//
// 16 f32 lanes per __m512; the column tail (cols % 16) is handled scalar.
// Horizontal reduce via the AVX-512 `_mm512_reduce_{add,max}_ps` intrinsics.

#include <immintrin.h>
#include <cstdint>
#include <limits>

namespace {
constexpr int kSum = 0;
constexpr int kMax = 1;
constexpr int kMean = 2;
}  // namespace

extern "C" void tessera_x86_reference_reduce_f32(const float* X, int64_t rows,
                                                 int64_t cols, float* out,
                                                 int kind) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = X + r * cols;
        if (kind == kMax) {
            float acc = -std::numeric_limits<float>::infinity();
            for (int64_t c = 0; c < cols; ++c)
                acc = row[c] > acc ? row[c] : acc;
            out[r] = acc;
        } else {  // sum / mean
            float acc = 0.0f;
            for (int64_t c = 0; c < cols; ++c) acc += row[c];
            out[r] = (kind == kMean && cols > 0) ? acc / (float)cols : acc;
        }
    }
}

extern "C" void tessera_x86_avx512_reduce_f32(const float* X, int64_t rows,
                                              int64_t cols, float* out,
                                              int kind) {
    const int64_t vstep = 16;  // f32 lanes per __m512
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = X + r * cols;
        int64_t c = 0;
        if (kind == kMax) {
            __m512 vacc = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
            for (; c + vstep <= cols; c += vstep)
                vacc = _mm512_max_ps(vacc, _mm512_loadu_ps(row + c));
            float acc = _mm512_reduce_max_ps(vacc);
            for (; c < cols; ++c) acc = row[c] > acc ? row[c] : acc;
            out[r] = acc;
        } else {  // sum / mean
            __m512 vacc = _mm512_setzero_ps();
            for (; c + vstep <= cols; c += vstep)
                vacc = _mm512_add_ps(vacc, _mm512_loadu_ps(row + c));
            float acc = _mm512_reduce_add_ps(vacc);
            for (; c < cols; ++c) acc += row[c];
            out[r] = (kind == kMean && cols > 0) ? acc / (float)cols : acc;
        }
    }
}
