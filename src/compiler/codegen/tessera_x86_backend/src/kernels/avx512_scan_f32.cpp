// x86 elementwise inclusive prefix scan (f32) for the Tessera x86 backend.
//
// Inclusive scan of each row of a [rows, cols] f32 matrix along the last axis —
// the CPU analog of the ROCm block-scan lane. kind: 0 = cumsum, 1 = cumprod,
// 2 = cummax, 3 = cummin. A scalar reference is provided alongside for on-device
// validation.
//
// Scan is inherently sequential along the row, so this is a straight scalar
// recurrence (numpy's np.cumsum/cumprod/maximum.accumulate is likewise
// sequential); an AVX-512 Hillis-Steele SIMD prefix is a perf follow-up. The
// outer rows are embarrassingly parallel.

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
    case kCummax: return a > v ? a : v;   // numpy maximum.accumulate
    case kCummin: return a < v ? a : v;
    default: return v;
    }
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

// Same recurrence; kept as the "avx512" entry so the runtime binds one symbol.
// (The row recurrence is scalar; rows parallelize. SIMD prefix = perf follow-up.)
extern "C" void tessera_x86_avx512_scan_f32(const float* X, int64_t rows,
                                            int64_t cols, float* out, int kind) {
    tessera_x86_reference_scan_f32(X, rows, cols, out, kind);
}
