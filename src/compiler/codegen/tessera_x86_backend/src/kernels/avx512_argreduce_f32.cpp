// x86 elementwise row arg-reduction (f32 -> i32 index) for the Tessera x86
// backend — the CPU analog of the ROCm warp-shuffle arg-reduce lane.
//
// For each row of a [rows, cols] f32 matrix, returns the index of the row max
// (argmax, kind 0) or row min (argmin, kind 1). numpy first-occurrence
// tie-break: a STRICT comparison only updates on a strictly better value, so an
// equal value never displaces the earlier index. A scalar reference is provided
// alongside (the row recurrence is scalar — a horizontal-index SIMD reduce is a
// perf follow-up; the outer rows parallelize).

#include <cstdint>
#include <limits>

namespace {
constexpr int kArgmax = 0;
constexpr int kArgmin = 1;

inline int32_t arg_row(const float* row, int64_t cols, int kind) {
    float best = (kind == kArgmax) ? -std::numeric_limits<float>::infinity()
                                   : std::numeric_limits<float>::infinity();
    int32_t bestIdx = 0;
    for (int64_t c = 0; c < cols; ++c) {
        float v = row[c];
        bool better = (kind == kArgmax) ? (v > best) : (v < best);  // strict
        if (better) {
            best = v;
            bestIdx = (int32_t)c;
        }
    }
    return bestIdx;
}
}  // namespace

extern "C" void tessera_x86_reference_argreduce_f32(const float* X, int64_t rows,
                                                    int64_t cols, int32_t* out,
                                                    int kind) {
    for (int64_t r = 0; r < rows; ++r)
        out[r] = arg_row(X + r * cols, cols, kind);
}

extern "C" void tessera_x86_avx512_argreduce_f32(const float* X, int64_t rows,
                                                 int64_t cols, int32_t* out,
                                                 int kind) {
    tessera_x86_reference_argreduce_f32(X, rows, cols, out, kind);
}
