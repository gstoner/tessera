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
#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
constexpr int kMax = 1;
constexpr int kMean = 2;
constexpr int kMin = 3;
constexpr int kProd = 4;
// NaN must PROPAGATE in reduce_max to match the reference (numpy `np.amax`):
// a row containing a NaN reduces to NaN. Plain MAXPS / ordered `>` would drop
// it. We track NaN explicitly (the f32 self-inequality test) and force NaN out.
inline bool is_nan_f32(float v) { return v != v; }
const float kQNaN = std::numeric_limits<float>::quiet_NaN();

struct WelfordState {
    int64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;
};

inline void welford_update(WelfordState& state, float value) {
    ++state.count;
    const double sample = static_cast<double>(value);
    const double delta = sample - state.mean;
    state.mean += delta / static_cast<double>(state.count);
    state.m2 += delta * (sample - state.mean);
}

inline void welford_merge(WelfordState& dst, const WelfordState& src) {
    if (src.count == 0) return;
    if (dst.count == 0) {
        dst = src;
        return;
    }
    const double delta = src.mean - dst.mean;
    const int64_t count = dst.count + src.count;
    dst.m2 += src.m2 + delta * delta *
                           (static_cast<double>(dst.count) *
                            static_cast<double>(src.count) /
                            static_cast<double>(count));
    dst.mean += delta * (static_cast<double>(src.count) /
                         static_cast<double>(count));
    dst.count = count;
}
}  // namespace

extern "C" void tessera_x86_reference_reduce_f32(const float* X, int64_t rows,
                                                 int64_t cols, float* out,
                                                 int kind) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = X + r * cols;
        if (kind == kMax || kind == kMin) {
            float acc = (kind == kMax)
                            ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::infinity();
            bool nan = false;
            for (int64_t c = 0; c < cols; ++c) {
                float v = row[c];
                if (is_nan_f32(v)) nan = true;
                else if (kind == kMax ? (v > acc) : (v < acc)) acc = v;
            }
            out[r] = nan ? kQNaN : acc;
        } else if (kind == kProd) {  // product — `*` propagates NaN
            float acc = 1.0f;
            for (int64_t c = 0; c < cols; ++c) acc *= row[c];
            out[r] = acc;
        } else {  // sum / mean — `+` already propagates NaN
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
        if (kind == kMax || kind == kMin) {
            const float ident = (kind == kMax)
                                     ? -std::numeric_limits<float>::infinity()
                                     : std::numeric_limits<float>::infinity();
            __m512 vacc = _mm512_set1_ps(ident);
            bool nan = false;
            for (; c + vstep <= cols; c += vstep) {
                __m512 v = _mm512_loadu_ps(row + c);
                // any NaN lane in this chunk? (v != v, unordered with itself)
                if (_mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q)) nan = true;
                vacc = (kind == kMax) ? _mm512_max_ps(vacc, v)
                                      : _mm512_min_ps(vacc, v);
            }
            float acc = (kind == kMax) ? _mm512_reduce_max_ps(vacc)
                                       : _mm512_reduce_min_ps(vacc);
            for (; c < cols; ++c) {
                float v = row[c];
                if (is_nan_f32(v)) nan = true;
                else if (kind == kMax ? (v > acc) : (v < acc)) acc = v;
            }
            out[r] = nan ? kQNaN : acc;
        } else if (kind == kProd) {  // product
            __m512 vacc = _mm512_set1_ps(1.0f);
            for (; c + vstep <= cols; c += vstep)
                vacc = _mm512_mul_ps(vacc, _mm512_loadu_ps(row + c));
            float acc = _mm512_reduce_mul_ps(vacc);
            for (; c < cols; ++c) acc *= row[c];
            out[r] = acc;
        } else {  // sum / mean — `+` already propagates NaN
            __m512 vacc = _mm512_setzero_ps();
            for (; c + vstep <= cols; c += vstep)
                vacc = _mm512_add_ps(vacc, _mm512_loadu_ps(row + c));
            float acc = _mm512_reduce_add_ps(vacc);
            for (; c < cols; ++c) acc += row[c];
            out[r] = (kind == kMean && cols > 0) ? acc / (float)cols : acc;
        }
    }
}

// Population variance over each row using mergeable Welford states. Full
// vectors are loaded with AVX-512, then accumulated into one independent state
// per SIMD lane. Merging those states avoids the cancellation in E[x*x]-E[x]^2
// while retaining the native [rows, cols] package ABI and arbitrary-axis fold
// performed by the runtime.
extern "C" void tessera_x86_avx512_welford_f32(const float* X, int64_t rows,
                                               int64_t cols, float* out) {
    constexpr int64_t kLanes = 16;
    alignas(64) float values[kLanes];
    for (int64_t r = 0; r < rows; ++r) {
        const float* row = X + r * cols;
        WelfordState lanes[kLanes];
        int64_t c = 0;
        for (; c + kLanes <= cols; c += kLanes) {
            _mm512_store_ps(values, _mm512_loadu_ps(row + c));
            for (int64_t lane = 0; lane < kLanes; ++lane)
                welford_update(lanes[lane], values[lane]);
        }
        for (; c < cols; ++c)
            welford_update(lanes[c % kLanes], row[c]);

        WelfordState total;
        for (const WelfordState& lane : lanes) welford_merge(total, lane);
        out[r] = total.count > 0
                     ? static_cast<float>(std::max(
                           total.m2 / static_cast<double>(total.count), 0.0))
                     : kQNaN;
    }
}
