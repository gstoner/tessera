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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../../../../solvers/spectral/lib/TargetHooks/Common/FFTPlan.h"

namespace {

inline uint32_t bit_reverse(uint32_t x, int bits) {
    uint32_t r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (x & 1u); x >>= 1; }
    return r;
}

struct FFTStagePlan {
    int64_t len = 0;
    int64_t half = 0;
    std::vector<float> twiddle_re;
    std::vector<float> twiddle_im;
};

struct FFTExecutionPlan {
    int bits = 0;
    std::vector<int32_t> gather_offsets;
    std::vector<FFTStagePlan> stages;
};

struct MixedStagePlan {
    int radix = 0;
    int64_t length = 0;
    int64_t groups = 0;
    std::vector<float> twiddle_re;
    std::vector<float> twiddle_im;
    std::vector<float> dft_re;
    std::vector<float> dft_im;
};

struct MixedExecutionPlan {
    bool supported = false;
    std::vector<MixedStagePlan> stages;
};

struct SixStepPlan {
    int64_t n1 = 0;
    int64_t n2 = 0;
    std::vector<float> twiddle_re;
    std::vector<float> twiddle_im;
};

struct RealFFTPlan {
    std::vector<float> forward_re;
    std::vector<float> forward_im;
};

const RealFFTPlan& real_fft_plan(int64_t n) {
    thread_local std::unordered_map<int64_t, std::shared_ptr<RealFFTPlan>> cache;
    if (auto found = cache.find(n); found != cache.end()) return *found->second;
    if (cache.size() >= 8) cache.erase(cache.begin());
    auto plan = std::make_shared<RealFFTPlan>();
    const int64_t m = n / 2;
    plan->forward_re.resize(static_cast<size_t>(m + 1));
    plan->forward_im.resize(static_cast<size_t>(m + 1));
    for (int64_t k = 0; k <= m; ++k) {
        const double angle = -2.0 * M_PI * static_cast<double>(k) / n;
        plan->forward_re[static_cast<size_t>(k)] = static_cast<float>(std::cos(angle));
        plan->forward_im[static_cast<size_t>(k)] = static_cast<float>(std::sin(angle));
    }
    auto inserted = cache.emplace(n, std::move(plan));
    return *inserted.first->second;
}

// Per-thread and deliberately bounded: callers commonly execute several rows
// of one shape, so a shared global cache would add locking to the hot path.
// Eight plans cover the normal benchmark/autotune working set while preventing
// an unbounded accumulation of O(N) twiddle and permutation tables.
const FFTExecutionPlan& execution_plan(int64_t n, bool inverse) {
    thread_local std::unordered_map<uint64_t, std::shared_ptr<FFTExecutionPlan>> cache;
    const uint64_t key = (static_cast<uint64_t>(n) << 1) | (inverse ? 1u : 0u);
    if (auto found = cache.find(key); found != cache.end()) return *found->second;
    if (cache.size() >= 8) cache.erase(cache.begin());

    auto plan = std::make_shared<FFTExecutionPlan>();
    while ((int64_t(1) << plan->bits) < n) ++plan->bits;
    plan->gather_offsets.resize(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        const uint32_t reversed = bit_reverse(static_cast<uint32_t>(i), plan->bits);
        plan->gather_offsets[static_cast<size_t>(i)] =
            static_cast<int32_t>(2u * reversed);
    }

    const double sign = inverse ? 1.0 : -1.0;
    for (int64_t len = 2; len <= n; len <<= 1) {
        FFTStagePlan stage;
        stage.len = len;
        stage.half = len / 2;
        stage.twiddle_re.resize(static_cast<size_t>(stage.half));
        stage.twiddle_im.resize(static_cast<size_t>(stage.half));
        for (int64_t k = 0; k < stage.half; ++k) {
            const double angle = sign * 2.0 * M_PI * static_cast<double>(k) /
                                 static_cast<double>(len);
            stage.twiddle_re[static_cast<size_t>(k)] = static_cast<float>(std::cos(angle));
            stage.twiddle_im[static_cast<size_t>(k)] = static_cast<float>(std::sin(angle));
        }
        plan->stages.push_back(std::move(stage));
    }
    auto inserted = cache.emplace(key, std::move(plan));
    return *inserted.first->second;
}

const MixedExecutionPlan& mixed_execution_plan(int64_t n, bool inverse) {
    thread_local std::unordered_map<uint64_t, std::shared_ptr<MixedExecutionPlan>> cache;
    const uint64_t key = (static_cast<uint64_t>(n) << 1) | (inverse ? 1u : 0u);
    if (auto found = cache.find(key); found != cache.end()) return *found->second;
    if (cache.size() >= 8) cache.erase(cache.begin());

    auto result = std::make_shared<MixedExecutionPlan>();
    if (n <= 0 || n > INT32_MAX) {
        auto inserted = cache.emplace(key, std::move(result));
        return *inserted.first->second;
    }
    const tessera::spectral::FFTPlan arithmetic =
        tessera::spectral::fft_plan(static_cast<int>(n));
    if (arithmetic.kind != tessera::spectral::kFFTMixedRadix) {
        auto inserted = cache.emplace(key, std::move(result));
        return *inserted.first->second;
    }

    const double sign = inverse ? 1.0 : -1.0;
    int64_t length = 1;
    for (int index = 0; index < arithmetic.stage_count; ++index) {
        MixedStagePlan stage;
        stage.radix = arithmetic.stages[index];
        stage.length = length;
        stage.groups = n / (stage.radix * length);
        stage.twiddle_re.resize(static_cast<size_t>(stage.radix) * length);
        stage.twiddle_im.resize(static_cast<size_t>(stage.radix) * length);
        for (int q = 0; q < stage.radix; ++q)
            for (int64_t j = 0; j < length; ++j) {
                const double angle = sign * 2.0 * M_PI * static_cast<double>(j) * q /
                                     (static_cast<double>(stage.radix) * length);
                const size_t at = static_cast<size_t>(q) * length + j;
                stage.twiddle_re[at] = static_cast<float>(std::cos(angle));
                stage.twiddle_im[at] = static_cast<float>(std::sin(angle));
            }
        stage.dft_re.resize(static_cast<size_t>(stage.radix) * stage.radix);
        stage.dft_im.resize(static_cast<size_t>(stage.radix) * stage.radix);
        for (int p = 0; p < stage.radix; ++p)
            for (int q = 0; q < stage.radix; ++q) {
                const double angle = sign * 2.0 * M_PI * ((p * q) % stage.radix) /
                                     stage.radix;
                const size_t at = static_cast<size_t>(p) * stage.radix + q;
                stage.dft_re[at] = static_cast<float>(std::cos(angle));
                stage.dft_im[at] = static_cast<float>(std::sin(angle));
            }
        result->stages.push_back(std::move(stage));
        length *= arithmetic.stages[index];
    }
    result->supported = length == n;
    auto inserted = cache.emplace(key, std::move(result));
    return *inserted.first->second;
}

const SixStepPlan& six_step_plan(int64_t n, bool inverse) {
    thread_local std::unordered_map<uint64_t, std::shared_ptr<SixStepPlan>> cache;
    const uint64_t key = (static_cast<uint64_t>(n) << 1) | (inverse ? 1u : 0u);
    if (auto found = cache.find(key); found != cache.end()) return *found->second;
    if (cache.size() >= 8) cache.erase(cache.begin());
    auto plan = std::make_shared<SixStepPlan>();
    int bits = 0;
    while ((int64_t(1) << bits) < n) ++bits;
    plan->n1 = int64_t(1) << (bits / 2);
    plan->n2 = n / plan->n1;
    plan->twiddle_re.resize(static_cast<size_t>(n));
    plan->twiddle_im.resize(static_cast<size_t>(n));
    const double sign = inverse ? 1.0 : -1.0;
    for (int64_t j = 0; j < plan->n2; ++j)
        for (int64_t k1 = 0; k1 < plan->n1; ++k1) {
            const double angle = sign * 2.0 * M_PI * static_cast<double>(j) * k1 / n;
            const size_t at = static_cast<size_t>(j) * plan->n1 + k1;
            plan->twiddle_re[at] = static_cast<float>(std::cos(angle));
            plan->twiddle_im[at] = static_cast<float>(std::sin(angle));
        }
    auto inserted = cache.emplace(key, std::move(plan));
    return *inserted.first->second;
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

inline void load_complex16(const float* source, __m512& real, __m512& imag) {
    const __m512 low = _mm512_loadu_ps(source);
    const __m512 high = _mm512_loadu_ps(source + 16);
    real = _mm512_permutex2var_ps(low, kEven, high);
    imag = _mm512_permutex2var_ps(low, kOdd, high);
}

inline void store_complex16(float* destination, __m512 real, __m512 imag) {
    _mm512_storeu_ps(destination, _mm512_permutex2var_ps(real, kIntLo, imag));
    _mm512_storeu_ps(destination + 16,
                     _mm512_permutex2var_ps(real, kIntHi, imag));
}

void execute_mixed_stage(const float* source, float* destination,
                         const MixedStagePlan& stage) {
    const int radix = stage.radix;
    const int64_t length = stage.length;
    const int64_t groups = stage.groups;
    for (int64_t k = 0; k < groups; ++k) {
        int64_t j = 0;
        // Once a prior stage has produced at least sixteen contiguous lanes,
        // execute sixteen independent radix-r codelets together.  Odd radix
        // stages normally occur late in the shared plan, so this is their hot
        // path; early tiny-L stages use the scalar tail below.
        for (; j + 16 <= length; j += 16) {
            __m512 value_re[17], value_im[17];
            for (int q = 0; q < radix; ++q) {
                const int64_t input_index = k * length + j + q * length * groups;
                __m512 xr, xi;
                load_complex16(source + 2 * input_index, xr, xi);
                const size_t twiddle_index = static_cast<size_t>(q) * length + j;
                const __m512 wr = _mm512_loadu_ps(stage.twiddle_re.data() + twiddle_index);
                const __m512 wi = _mm512_loadu_ps(stage.twiddle_im.data() + twiddle_index);
                value_re[q] = _mm512_fmsub_ps(xr, wr, _mm512_mul_ps(xi, wi));
                value_im[q] = _mm512_fmadd_ps(xr, wi, _mm512_mul_ps(xi, wr));
            }
            for (int p = 0; p < radix; ++p) {
                __m512 acc_re = _mm512_setzero_ps();
                __m512 acc_im = _mm512_setzero_ps();
                for (int q = 0; q < radix; ++q) {
                    const size_t coefficient = static_cast<size_t>(p) * radix + q;
                    const __m512 wr = _mm512_set1_ps(stage.dft_re[coefficient]);
                    const __m512 wi = _mm512_set1_ps(stage.dft_im[coefficient]);
                    acc_re = _mm512_add_ps(
                        acc_re, _mm512_fmsub_ps(value_re[q], wr,
                                               _mm512_mul_ps(value_im[q], wi)));
                    acc_im = _mm512_add_ps(
                        acc_im, _mm512_fmadd_ps(value_re[q], wi,
                                               _mm512_mul_ps(value_im[q], wr)));
                }
                const int64_t output_index = k * (radix * length) + j + p * length;
                store_complex16(destination + 2 * output_index, acc_re, acc_im);
            }
        }
        for (; j < length; ++j) {
            float value_re[17], value_im[17];
            for (int q = 0; q < radix; ++q) {
                const int64_t input_index = k * length + j + q * length * groups;
                const float xr = source[2 * input_index];
                const float xi = source[2 * input_index + 1];
                const size_t twiddle_index = static_cast<size_t>(q) * length + j;
                const float wr = stage.twiddle_re[twiddle_index];
                const float wi = stage.twiddle_im[twiddle_index];
                value_re[q] = xr * wr - xi * wi;
                value_im[q] = xr * wi + xi * wr;
            }
            for (int p = 0; p < radix; ++p) {
                float acc_re = 0.0f, acc_im = 0.0f;
                for (int q = 0; q < radix; ++q) {
                    const size_t coefficient = static_cast<size_t>(p) * radix + q;
                    const float wr = stage.dft_re[coefficient];
                    const float wi = stage.dft_im[coefficient];
                    acc_re += value_re[q] * wr - value_im[q] * wi;
                    acc_im += value_re[q] * wi + value_im[q] * wr;
                }
                const int64_t output_index = k * (radix * length) + j + p * length;
                destination[2 * output_index] = acc_re;
                destination[2 * output_index + 1] = acc_im;
            }
        }
    }
}

}  // namespace

extern "C" void tessera_x86_fft_c2c_f32(float* data, int64_t batch, int64_t n,
                                        int inverse) {
    if (n <= 1) return;
    if ((n & (n - 1)) != 0) return;
    const FFTExecutionPlan& plan = execution_plan(n, inverse != 0);
    // One reusable workspace per worker thread.  The bit-reversal permutation
    // is a generated AVX-512 gather into sequential storage, replacing the
    // previous branchy scalar random-swap loop.  Keeping it outside the plan
    // preserves plan immutability while avoiding warm-call allocation.
    thread_local std::vector<float> permutation_workspace;
    if (permutation_workspace.size() < static_cast<size_t>(2 * n))
        permutation_workspace.resize(static_cast<size_t>(2 * n));

    for (int64_t b = 0; b < batch; ++b) {
        float* x = data + b * 2 * n;
        float* permuted = permutation_workspace.data();
        int64_t i = 0;
        const __m512i one = _mm512_set1_epi32(1);
        for (; i + 16 <= n; i += 16) {
            const __m512i offsets = _mm512_loadu_si512(
                static_cast<const void*>(plan.gather_offsets.data() + i));
            const __m512 re = _mm512_i32gather_ps(offsets, x, 4);
            const __m512 im = _mm512_i32gather_ps(_mm512_add_epi32(offsets, one), x, 4);
            _mm512_storeu_ps(permuted + 2 * i,
                             _mm512_permutex2var_ps(re, kIntLo, im));
            _mm512_storeu_ps(permuted + 2 * i + 16,
                             _mm512_permutex2var_ps(re, kIntHi, im));
        }
        for (; i < n; ++i) {
            const int32_t offset = plan.gather_offsets[static_cast<size_t>(i)];
            permuted[2 * i] = x[offset];
            permuted[2 * i + 1] = x[offset + 1];
        }
        std::memcpy(x, permuted, static_cast<size_t>(2 * n) * sizeof(float));
        // All immutable plan work is cached. Execution now performs only the
        // permutation and butterflies; no allocation or transcendental work.
        for (const FFTStagePlan& stage : plan.stages) {
            const int64_t len = stage.len;
            const int64_t half = stage.half;
            const float* str = stage.twiddle_re.data();
            const float* sti = stage.twiddle_im.data();
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
                    __m512 tr = _mm512_loadu_ps(str + k);   // already deinterleaved
                    __m512 ti = _mm512_loadu_ps(sti + k);
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

// Candidate mixed-radix ABI.  It deliberately returns a status and declines
// Bluestein lengths; selector integration follows only after shape-specific
// evidence.  Inverse is unnormalized, matching tessera_x86_fft_c2c_f32.
extern "C" int tessera_x86_fft_mixed_c2c_f32(float* data, int64_t batch,
                                              int64_t n, int inverse) {
    if (batch <= 0 || n <= 0) return 2;
    const MixedExecutionPlan& plan = mixed_execution_plan(n, inverse != 0);
    if (!plan.supported) return 1;
    thread_local std::vector<float> first;
    thread_local std::vector<float> second;
    const size_t floats = static_cast<size_t>(2 * n);
    if (first.size() < floats) first.resize(floats);
    if (second.size() < floats) second.resize(floats);
    for (int64_t row = 0; row < batch; ++row) {
        float* output = data + row * 2 * n;
        std::memcpy(first.data(), output, floats * sizeof(float));
        float* source = first.data();
        float* destination = second.data();
        for (const MixedStagePlan& stage : plan.stages) {
            execute_mixed_stage(source, destination, stage);
            std::swap(source, destination);
        }
        std::memcpy(output, source, floats * sizeof(float));
    }
    return 0;
}

// Native Bailey six-step candidate.  The first and final permutations are
// blocked transposes; the middle transpose fuses twiddle multiplication so the
// candidate makes three matrix passes rather than Python's transpose + twiddle
// + transpose sequence.  It remains a separate ABI until large-shape evidence
// beats the production lane.
extern "C" int tessera_x86_fft_six_step_c2c_f32(float* data, int64_t batch,
                                                 int64_t n, int inverse) {
    if (batch <= 0 || n < 4 || (n & (n - 1)) != 0) return 1;
    const SixStepPlan& plan = six_step_plan(n, inverse != 0);
    thread_local std::vector<float> first;
    thread_local std::vector<float> second;
    const size_t floats = static_cast<size_t>(2 * n);
    if (first.size() < floats) first.resize(floats);
    if (second.size() < floats) second.resize(floats);
    constexpr int64_t tile = 16;

    for (int64_t row = 0; row < batch; ++row) {
        float* output = data + row * 2 * n;
        // [n1,n2] -> [n2,n1].
        for (int64_t i0 = 0; i0 < plan.n1; i0 += tile)
            for (int64_t j0 = 0; j0 < plan.n2; j0 += tile)
                for (int64_t i = i0; i < std::min(i0 + tile, plan.n1); ++i)
                    for (int64_t j = j0; j < std::min(j0 + tile, plan.n2); ++j) {
                        const int64_t source = i * plan.n2 + j;
                        const int64_t destination = j * plan.n1 + i;
                        first[2 * destination] = output[2 * source];
                        first[2 * destination + 1] = output[2 * source + 1];
                    }

        tessera_x86_fft_c2c_f32(first.data(), plan.n2, plan.n1, inverse);

        // [n2,n1] -> [n1,n2], with the Bailey twiddle fused into the copy.
        for (int64_t j0 = 0; j0 < plan.n2; j0 += tile)
            for (int64_t k0 = 0; k0 < plan.n1; k0 += tile)
                for (int64_t j = j0; j < std::min(j0 + tile, plan.n2); ++j)
                    for (int64_t k1 = k0; k1 < std::min(k0 + tile, plan.n1); ++k1) {
                        const int64_t source = j * plan.n1 + k1;
                        const int64_t destination = k1 * plan.n2 + j;
                        const size_t twiddle = static_cast<size_t>(source);
                        const float xr = first[2 * source];
                        const float xi = first[2 * source + 1];
                        const float wr = plan.twiddle_re[twiddle];
                        const float wi = plan.twiddle_im[twiddle];
                        second[2 * destination] = xr * wr - xi * wi;
                        second[2 * destination + 1] = xr * wi + xi * wr;
                    }

        tessera_x86_fft_c2c_f32(second.data(), plan.n1, plan.n2, inverse);

        // [n1,n2] -> output order k = k1 + n1*k2, i.e. [n2,n1].
        for (int64_t k0 = 0; k0 < plan.n1; k0 += tile)
            for (int64_t j0 = 0; j0 < plan.n2; j0 += tile)
                for (int64_t k1 = k0; k1 < std::min(k0 + tile, plan.n1); ++k1)
                    for (int64_t k2 = j0; k2 < std::min(j0 + tile, plan.n2); ++k2) {
                        const int64_t source = k1 * plan.n2 + k2;
                        const int64_t destination = k2 * plan.n1 + k1;
                        output[2 * destination] = second[2 * source];
                        output[2 * destination + 1] = second[2 * source + 1];
                    }
    }
    return 0;
}

namespace {

std::shared_ptr<std::vector<float>> spectral_workspace(
    const char* digest, const char* slot, size_t floats) {
    thread_local std::map<std::string, std::shared_ptr<std::vector<float>>> cache;
    std::string key = std::string(digest ? digest : "") + ":" + slot;
    if (cache.size() >= 32 && cache.find(key) == cache.end()) cache.erase(cache.begin());
    std::shared_ptr<std::vector<float>>& value = cache[key];
    if (!value) value = std::make_shared<std::vector<float>>();
    if (value->size() < floats) value->resize(floats);
    return value;
}

bool valid_digest(const char* digest) {
    if (!digest || std::strlen(digest) != 64) return false;
    for (const char* at = digest; *at; ++at)
        if (!((*at >= '0' && *at <= '9') || (*at >= 'a' && *at <= 'f')))
            return false;
    return true;
}

void scale_complex(float* data, int64_t count, float scale) {
    int64_t i = 0;
    const __m512 factor = _mm512_set1_ps(scale);
    for (; i + 16 <= 2 * count; i += 16)
        _mm512_storeu_ps(data + i, _mm512_mul_ps(_mm512_loadu_ps(data + i), factor));
    for (; i < 2 * count; ++i) data[i] *= scale;
}

int execute_any_fft(float* data, int64_t batch, int64_t n, bool inverse,
                    const char* digest) {
    if (batch <= 0 || n <= 0) return 1;
    if (n == 1) return 0;
    if ((n & (n - 1)) == 0) {
        tessera_x86_fft_c2c_f32(data, batch, n, inverse ? 1 : 0);
        if (inverse) scale_complex(data, batch * n, 1.0f / static_cast<float>(n));
        return 0;
    }
    if (tessera_x86_fft_mixed_c2c_f32(data, batch, n, inverse ? 1 : 0) == 0) {
        if (inverse) scale_complex(data, batch * n, 1.0f / static_cast<float>(n));
        return 0;
    }

    int64_t m = 1;
    while (m < 2 * n - 1) m <<= 1;
    auto firstOwner = spectral_workspace(digest, "bluestein_a",
                                         static_cast<size_t>(2 * m));
    auto secondOwner = spectral_workspace(digest, "bluestein_b",
                                          static_cast<size_t>(2 * m));
    std::vector<float>& first = *firstOwner;
    std::vector<float>& second = *secondOwner;
    const double sign = inverse ? 1.0 : -1.0;
    for (int64_t row = 0; row < batch; ++row) {
        std::fill(first.begin(), first.begin() + 2 * m, 0.0f);
        std::fill(second.begin(), second.begin() + 2 * m, 0.0f);
        float* source = data + 2 * row * n;
        for (int64_t k = 0; k < n; ++k) {
            const double angle = sign * M_PI * static_cast<double>(k) * k / n;
            const float cr = static_cast<float>(std::cos(angle));
            const float ci = static_cast<float>(std::sin(angle));
            const float xr = source[2 * k], xi = source[2 * k + 1];
            first[2 * k] = xr * cr - xi * ci;
            first[2 * k + 1] = xr * ci + xi * cr;
            second[2 * k] = cr;
            second[2 * k + 1] = -ci;
            if (k) {
                second[2 * (m - k)] = cr;
                second[2 * (m - k) + 1] = -ci;
            }
        }
        tessera_x86_fft_c2c_f32(first.data(), 1, m, 0);
        tessera_x86_fft_c2c_f32(second.data(), 1, m, 0);
        for (int64_t k = 0; k < m; ++k) {
            const float ar = first[2 * k], ai = first[2 * k + 1];
            const float br = second[2 * k], bi = second[2 * k + 1];
            first[2 * k] = ar * br - ai * bi;
            first[2 * k + 1] = ar * bi + ai * br;
        }
        tessera_x86_fft_c2c_f32(first.data(), 1, m, 1);
        const float convolutionScale = 1.0f / static_cast<float>(m);
        const float transformScale = inverse ? 1.0f / static_cast<float>(n) : 1.0f;
        for (int64_t k = 0; k < n; ++k) {
            const double angle = sign * M_PI * static_cast<double>(k) * k / n;
            const float cr = static_cast<float>(std::cos(angle));
            const float ci = static_cast<float>(std::sin(angle));
            const float xr = first[2 * k] * convolutionScale;
            const float xi = first[2 * k + 1] * convolutionScale;
            source[2 * k] = (xr * cr - xi * ci) * transformScale;
            source[2 * k + 1] = (xr * ci + xi * cr) * transformScale;
        }
    }
    return 0;
}

}  // namespace

// Packed even-length real transforms.  The physical FFT is N/2 complex
// values: z[j] = x[2j] + i*x[2j+1].  The pre/post processing stays inside the
// native package so the Python executor cannot accidentally reconstruct an
// N-point complex lane while advertising an r2c/c2r artifact.
extern "C" int tessera_x86_fft_r2c_packed_f32(
    const char* digest, const float* input, float* output, int64_t batch,
    int64_t n) {
    if (!valid_digest(digest) || !input || !output || batch <= 0 || n < 2 ||
        (n & 1))
        return 1;
    const int64_t m = n / 2;
    const RealFFTPlan& realPlan = real_fft_plan(n);
    auto packedOwner = spectral_workspace(
        digest, "r2c_packed", static_cast<size_t>(2 * batch * m));
    float* packed = packedOwner->data();
    for (int64_t row = 0; row < batch; ++row)
        for (int64_t j = 0; j < m; ++j) {
            packed[2 * (row * m + j)] = input[row * n + 2 * j];
            packed[2 * (row * m + j) + 1] = input[row * n + 2 * j + 1];
        }
    if ((m & (m - 1)) == 0)
        tessera_x86_fft_c2c_f32(packed, batch, m, 0);
    else if (tessera_x86_fft_mixed_c2c_f32(packed, batch, m, 0) != 0)
        return 2;

    for (int64_t row = 0; row < batch; ++row) {
        const float* z = packed + 2 * row * m;
        float* spectrum = output + 2 * row * (m + 1);
        for (int64_t k = 0; k <= m; ++k) {
            const int64_t aIndex = k == m ? 0 : k;
            const int64_t bIndex = (m - k) % m;
            const float ar = z[2 * aIndex], ai = z[2 * aIndex + 1];
            const float br = z[2 * bIndex], bi = -z[2 * bIndex + 1];
            const float sumr = ar + br, sumi = ai + bi;
            const float diffr = ar - br, diffi = ai - bi;
            const float wr = realPlan.forward_re[static_cast<size_t>(k)];
            const float wi = realPlan.forward_im[static_cast<size_t>(k)];
            const float tr = wr * diffr - wi * diffi;
            const float ti = wr * diffi + wi * diffr;
            spectrum[2 * k] = 0.5f * (sumr + ti);
            spectrum[2 * k + 1] = 0.5f * (sumi - tr);
        }
        spectrum[1] = 0.0f;
        spectrum[2 * m + 1] = 0.0f;
    }
    return 0;
}

extern "C" int tessera_x86_fft_c2r_packed_f32(
    const char* digest, const float* input, float* output, int64_t batch,
    int64_t n) {
    if (!valid_digest(digest) || !input || !output || batch <= 0 || n < 2 ||
        (n & 1))
        return 1;
    const int64_t m = n / 2;
    const RealFFTPlan& realPlan = real_fft_plan(n);
    auto packedOwner = spectral_workspace(
        digest, "c2r_packed", static_cast<size_t>(2 * batch * m));
    float* packed = packedOwner->data();
    for (int64_t row = 0; row < batch; ++row) {
        const float* spectrum = input + 2 * row * (m + 1);
        float* z = packed + 2 * row * m;
        for (int64_t k = 0; k < m; ++k) {
            const int64_t mirror = m - k;
            const float xr = spectrum[2 * k], xi = spectrum[2 * k + 1];
            const float yr = spectrum[2 * mirror];
            const float yi = -spectrum[2 * mirror + 1];
            const float sumr = xr + yr, sumi = xi + yi;
            const float diffr = xr - yr, diffi = xi - yi;
            const float wr = realPlan.forward_re[static_cast<size_t>(k)];
            const float wi = -realPlan.forward_im[static_cast<size_t>(k)];
            const float tr = wr * diffr - wi * diffi;
            const float ti = wr * diffi + wi * diffr;
            z[2 * k] = 0.5f * (sumr - ti);
            z[2 * k + 1] = 0.5f * (sumi + tr);
        }
    }
    if ((m & (m - 1)) == 0)
        tessera_x86_fft_c2c_f32(packed, batch, m, 1);
    else if (tessera_x86_fft_mixed_c2c_f32(packed, batch, m, 1) != 0)
        return 2;
    const float scale = 1.0f / static_cast<float>(m);
    for (int64_t row = 0; row < batch; ++row)
        for (int64_t j = 0; j < m; ++j) {
            output[row * n + 2 * j] = packed[2 * (row * m + j)] * scale;
            output[row * n + 2 * j + 1] = packed[2 * (row * m + j) + 1] * scale;
        }
    return 0;
}

extern "C" const char* tessera_x86_spectral_composite_package_abi() {
    return "tessera.x86.spectral_composite.v5";
}

extern "C" int tessera_x86_spectral_filter_f32(
    const char* digest, const float* a, const float* b, float* output,
    int64_t elements) {
    if (!valid_digest(digest) || !a || !b || !output || elements <= 0) return 1;
    for (int64_t i = 0; i < elements; ++i) {
        const float ar = a[2 * i], ai = a[2 * i + 1];
        const float br = b[2 * i], bi = b[2 * i + 1];
        output[2 * i] = ar * br - ai * bi;
        output[2 * i + 1] = ar * bi + ai * br;
    }
    return 0;
}

extern "C" int tessera_x86_dct_f32(const char* digest, const float* input,
                                    float* output, int64_t batch, int64_t n,
                                    float output_scale) {
    if (!valid_digest(digest) || !input || !output || batch <= 0 || n <= 0) return 1;
    auto mirroredOwner = spectral_workspace(
        digest, "dct", static_cast<size_t>(4 * batch * n));
    std::vector<float>& mirrored = *mirroredOwner;
    for (int64_t row = 0; row < batch; ++row)
        for (int64_t column = 0; column < 2 * n; ++column) {
            const int64_t source = column < n ? column : 2 * n - 1 - column;
            mirrored[2 * (row * 2 * n + column)] = input[row * n + source];
            mirrored[2 * (row * 2 * n + column) + 1] = 0.0f;
        }
    int rc = execute_any_fft(mirrored.data(), batch, 2 * n, false, digest);
    if (rc) return rc;
    for (int64_t row = 0; row < batch; ++row)
        for (int64_t column = 0; column < n; ++column)
            output[row * n + column] =
                mirrored[2 * (row * 2 * n + column)] * output_scale;
    return 0;
}

extern "C" int tessera_x86_spectral_conv_f32(
    const char* digest, const float* input, int64_t input_n,
    const float* kernel, int64_t kernel_n, float* output, int64_t batch,
    int64_t fft_n) {
    const int64_t output_n = input_n + kernel_n - 1;
    if (!valid_digest(digest) || !input || !kernel || !output || batch <= 0 ||
        input_n <= 0 || kernel_n <= 0 || fft_n < output_n || fft_n < 2 ||
        (fft_n & 1)) return 1;
    const int64_t bins = fft_n / 2 + 1;
    const size_t realFloats = static_cast<size_t>(batch * fft_n);
    const size_t spectrumFloats = static_cast<size_t>(2 * batch * bins);
    auto xOwner = spectral_workspace(digest, "conv_x_real", realFloats);
    auto wOwner = spectral_workspace(digest, "conv_w_real", realFloats);
    auto xSpectrumOwner = spectral_workspace(
        digest, "conv_x_spectrum", spectrumFloats);
    auto wSpectrumOwner = spectral_workspace(
        digest, "conv_w_spectrum", spectrumFloats);
    auto inverseOwner = spectral_workspace(
        digest, "conv_inverse_real", realFloats);
    std::vector<float>& x = *xOwner;
    std::vector<float>& w = *wOwner;
    std::vector<float>& xSpectrum = *xSpectrumOwner;
    std::vector<float>& wSpectrum = *wSpectrumOwner;
    std::vector<float>& inverse = *inverseOwner;
    std::fill(x.begin(), x.begin() + realFloats, 0.0f);
    std::fill(w.begin(), w.begin() + realFloats, 0.0f);
    for (int64_t row = 0; row < batch; ++row) {
        std::memcpy(x.data() + row * fft_n, input + row * input_n,
                    static_cast<size_t>(input_n) * sizeof(float));
        std::memcpy(w.data() + row * fft_n, kernel + row * kernel_n,
                    static_cast<size_t>(kernel_n) * sizeof(float));
    }
    if (tessera_x86_fft_r2c_packed_f32(
            digest, x.data(), xSpectrum.data(), batch, fft_n) ||
        tessera_x86_fft_r2c_packed_f32(
            digest, w.data(), wSpectrum.data(), batch, fft_n)) return 2;
    for (int64_t i = 0; i < batch * bins; ++i) {
        const float ar = xSpectrum[2 * i], ai = xSpectrum[2 * i + 1];
        const float br = wSpectrum[2 * i], bi = wSpectrum[2 * i + 1];
        xSpectrum[2 * i] = ar * br - ai * bi;
        xSpectrum[2 * i + 1] = ar * bi + ai * br;
    }
    if (tessera_x86_fft_c2r_packed_f32(
            digest, xSpectrum.data(), inverse.data(), batch, fft_n)) return 3;
    for (int64_t row = 0; row < batch; ++row)
        std::memcpy(output + row * output_n, inverse.data() + row * fft_n,
                    static_cast<size_t>(output_n) * sizeof(float));
    return 0;
}

extern "C" int tessera_x86_stft_f32(
    const char* digest, const float* input, const float* window, float* output,
    int64_t batch, int64_t samples, int64_t win, int64_t hop, int64_t frames,
    float output_scale) {
    if (!valid_digest(digest) || !input || !window || !output || batch <= 0 ||
        samples <= 0 || win < 2 || hop <= 0 || frames <= 0) return 1;
    const int64_t rows = batch * frames, bins = win / 2 + 1;
    if (win & 1) {
        auto fullOwner = spectral_workspace(
            digest, "stft_odd_full", static_cast<size_t>(2 * rows * win));
        std::vector<float>& full = *fullOwner;
        for (int64_t row = 0; row < batch; ++row)
            for (int64_t frame = 0; frame < frames; ++frame)
                for (int64_t i = 0; i < win; ++i) {
                    const int64_t at = (row * frames + frame) * win + i;
                    full[2 * at] =
                        input[row * samples + frame * hop + i] * window[i];
                    full[2 * at + 1] = 0.0f;
                }
        if (execute_any_fft(full.data(), rows, win, false, digest)) return 2;
        for (int64_t row = 0; row < rows; ++row)
            for (int64_t i = 0; i < 2 * bins; ++i)
                output[2 * row * bins + i] =
                    full[2 * row * win + i] * output_scale;
        return 0;
    }
    auto framesOwner = spectral_workspace(
        digest, "stft_frames_real", static_cast<size_t>(rows * win));
    std::vector<float>& framed = *framesOwner;
    for (int64_t row = 0; row < batch; ++row)
        for (int64_t frame = 0; frame < frames; ++frame)
            for (int64_t i = 0; i < win; ++i) {
                const int64_t at = (row * frames + frame) * win + i;
                framed[at] = input[row * samples + frame * hop + i] * window[i];
            }
    if (tessera_x86_fft_r2c_packed_f32(
            digest, framed.data(), output, rows, win)) return 2;
    if (output_scale != 1.0f) scale_complex(output, rows * bins, output_scale);
    return 0;
}

extern "C" int tessera_x86_istft_f32(
    const char* digest, const float* input, const float* window, float* output,
    int64_t batch, int64_t frames, int64_t win, int64_t hop,
    float output_scale) {
    if (!valid_digest(digest) || !input || !window || !output || batch <= 0 ||
        frames <= 0 || win < 2 || hop <= 0) return 1;
    const int64_t rows = batch * frames, bins = win / 2 + 1;
    const int64_t samples = (frames - 1) * hop + win;
    if (win & 1) {
        auto fullOwner = spectral_workspace(
            digest, "istft_odd_full", static_cast<size_t>(2 * rows * win));
        std::vector<float>& full = *fullOwner;
        for (int64_t row = 0; row < rows; ++row) {
            for (int64_t i = 0; i < bins; ++i) {
                full[2 * (row * win + i)] = input[2 * (row * bins + i)];
                full[2 * (row * win + i) + 1] =
                    input[2 * (row * bins + i) + 1];
            }
            for (int64_t i = bins; i < win; ++i) {
                full[2 * (row * win + i)] =
                    input[2 * (row * bins + win - i)];
                full[2 * (row * win + i) + 1] =
                    -input[2 * (row * bins + win - i) + 1];
            }
        }
        if (execute_any_fft(full.data(), rows, win, true, digest)) return 2;
        for (int64_t row = 0; row < batch; ++row)
            for (int64_t sample = 0; sample < samples; ++sample) {
                float sum = 0.0f, weight = 0.0f;
                for (int64_t frame = 0; frame < frames; ++frame) {
                    const int64_t local = sample - frame * hop;
                    if (local < 0 || local >= win) continue;
                    const float w = window[local];
                    sum += full[2 * ((row * frames + frame) * win + local)] * w;
                    weight += w * w;
                }
                output[row * samples + sample] =
                    (sum / std::max(weight, 1.0e-12f)) * output_scale;
            }
        return 0;
    }
    auto framesOwner = spectral_workspace(
        digest, "istft_frames_real", static_cast<size_t>(rows * win));
    std::vector<float>& framed = *framesOwner;
    if (tessera_x86_fft_c2r_packed_f32(
            digest, input, framed.data(), rows, win)) return 2;
    for (int64_t row = 0; row < batch; ++row)
        for (int64_t sample = 0; sample < samples; ++sample) {
            float sum = 0.0f, weight = 0.0f;
            for (int64_t frame = 0; frame < frames; ++frame) {
                const int64_t local = sample - frame * hop;
                if (local < 0 || local >= win) continue;
                const float w = window[local];
                sum += framed[(row * frames + frame) * win + local] * w;
                weight += w * w;
            }
            output[row * samples + sample] =
                (sum / std::max(weight, 1.0e-12f)) * output_scale;
        }
    return 0;
}

namespace {

float load_spectral_storage(const void* input, int64_t index, int storage) {
    if (storage == 0) return static_cast<const float*>(input)[index];
    const uint16_t bits = static_cast<const uint16_t*>(input)[index];
    if (storage == 1) return _cvtsh_ss(bits);
    uint32_t wide = uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

void store_spectral_storage(void* output, int64_t index, int storage,
                            float value) {
    if (storage == 0) {
        static_cast<float*>(output)[index] = value;
        return;
    }
    if (storage == 1) {
        static_cast<uint16_t*>(output)[index] = _cvtss_sh(
            value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        return;
    }
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fff + ((bits >> 16) & 1);
    static_cast<uint16_t*>(output)[index] = uint16_t((bits + rounding) >> 16);
}

bool valid_spectral_storage(int storage) {
    return storage >= 0 && storage <= 2;
}

void unpack_spectral_storage(const void* input, float* output, int64_t elements,
                             int storage) {
    for (int64_t i = 0; i < elements; ++i)
        output[i] = load_spectral_storage(input, i, storage);
}

void pack_spectral_storage(const float* input, void* output, int64_t elements,
                           int storage) {
    for (int64_t i = 0; i < elements; ++i)
        store_spectral_storage(output, i, storage, input[i]);
}

}  // namespace

extern "C" int tessera_x86_dct_storage(
    const char* digest, const void* input, void* output, int64_t batch,
    int64_t n, int storage, float output_scale) {
    if (!valid_spectral_storage(storage) || !input || !output) return 10;
    if (storage == 0)
        return tessera_x86_dct_f32(digest, static_cast<const float*>(input),
                                   static_cast<float*>(output), batch, n,
                                   output_scale);
    const int64_t elements = batch * n;
    auto unpackedOwner = spectral_workspace(
        digest, "dct_storage_input", static_cast<size_t>(elements));
    auto packedOwner = spectral_workspace(
        digest, "dct_storage_output", static_cast<size_t>(elements));
    std::vector<float>& unpacked = *unpackedOwner;
    std::vector<float>& packed = *packedOwner;
    unpack_spectral_storage(input, unpacked.data(), elements, storage);
    int rc = tessera_x86_dct_f32(digest, unpacked.data(), packed.data(), batch,
                                  n, output_scale);
    if (!rc) pack_spectral_storage(packed.data(), output, elements, storage);
    return rc;
}

extern "C" int tessera_x86_spectral_conv_storage(
    const char* digest, const void* input, int64_t input_n,
    const void* kernel, int64_t kernel_n, void* output, int64_t batch,
    int64_t fft_n, int storage) {
    if (!valid_spectral_storage(storage) || !input || !kernel || !output)
        return 10;
    if (storage == 0)
        return tessera_x86_spectral_conv_f32(
            digest, static_cast<const float*>(input), input_n,
            static_cast<const float*>(kernel), kernel_n,
            static_cast<float*>(output), batch, fft_n);
    const int64_t input_elements = batch * input_n;
    const int64_t kernel_elements = batch * kernel_n;
    const int64_t output_elements = batch * (input_n + kernel_n - 1);
    auto xOwner = spectral_workspace(
        digest, "conv_storage_input", static_cast<size_t>(input_elements));
    auto wOwner = spectral_workspace(
        digest, "conv_storage_kernel", static_cast<size_t>(kernel_elements));
    auto yOwner = spectral_workspace(
        digest, "conv_storage_output", static_cast<size_t>(output_elements));
    std::vector<float>& x = *xOwner;
    std::vector<float>& w = *wOwner;
    std::vector<float>& y = *yOwner;
    unpack_spectral_storage(input, x.data(), input_elements, storage);
    unpack_spectral_storage(kernel, w.data(), kernel_elements, storage);
    int rc = tessera_x86_spectral_conv_f32(
        digest, x.data(), input_n, w.data(), kernel_n, y.data(), batch, fft_n);
    if (!rc) pack_spectral_storage(y.data(), output, output_elements, storage);
    return rc;
}

extern "C" int tessera_x86_stft_storage(
    const char* digest, const void* input, const void* window, float* output,
    int64_t batch, int64_t samples, int64_t win, int64_t hop, int64_t frames,
    int storage, float output_scale) {
    if (!valid_spectral_storage(storage) || !input || !window || !output)
        return 10;
    if (storage == 0)
        return tessera_x86_stft_f32(
            digest, static_cast<const float*>(input),
            static_cast<const float*>(window), output, batch, samples, win, hop,
            frames, output_scale);
    auto xOwner = spectral_workspace(
        digest, "stft_storage_input", static_cast<size_t>(batch * samples));
    auto wOwner = spectral_workspace(
        digest, "stft_storage_window", static_cast<size_t>(win));
    std::vector<float>& x = *xOwner;
    std::vector<float>& w = *wOwner;
    unpack_spectral_storage(input, x.data(), batch * samples, storage);
    unpack_spectral_storage(window, w.data(), win, storage);
    return tessera_x86_stft_f32(digest, x.data(), w.data(), output, batch,
                                 samples, win, hop, frames, output_scale);
}

extern "C" int tessera_x86_istft_storage(
    const char* digest, const float* input, const void* window, void* output,
    int64_t batch, int64_t frames, int64_t win, int64_t hop, int storage,
    float output_scale) {
    if (!valid_spectral_storage(storage) || !input || !window || !output)
        return 10;
    if (storage == 0)
        return tessera_x86_istft_f32(
            digest, input, static_cast<const float*>(window),
            static_cast<float*>(output), batch, frames, win, hop,
            output_scale);
    const int64_t samples = (frames - 1) * hop + win;
    auto wOwner = spectral_workspace(
        digest, "istft_storage_window", static_cast<size_t>(win));
    auto yOwner = spectral_workspace(
        digest, "istft_storage_output", static_cast<size_t>(batch * samples));
    std::vector<float>& w = *wOwner;
    std::vector<float>& y = *yOwner;
    unpack_spectral_storage(window, w.data(), win, storage);
    int rc = tessera_x86_istft_f32(digest, input, w.data(), y.data(), batch,
                                    frames, win, hop, output_scale);
    if (!rc) pack_spectral_storage(y.data(), output, batch * samples, storage);
    return rc;
}

namespace {

size_t spectral_storage_bytes(int storage) { return storage == 0 ? 4 : 2; }

void pack_axis_storage(const void* input, void* output, int64_t outer,
                       int64_t axis_extent, int64_t inner, size_t bytes) {
    const auto* source = static_cast<const unsigned char*>(input);
    auto* destination = static_cast<unsigned char*>(output);
    for (int64_t o = 0; o < outer; ++o)
        for (int64_t j = 0; j < inner; ++j)
            for (int64_t i = 0; i < axis_extent; ++i) {
                int64_t source_index = (o * axis_extent + i) * inner + j;
                int64_t destination_index = (o * inner + j) * axis_extent + i;
                std::memcpy(destination + destination_index * bytes,
                            source + source_index * bytes, bytes);
            }
}

void unpack_axis_storage(const void* input, void* output, int64_t outer,
                         int64_t axis_extent, int64_t inner, size_t bytes) {
    pack_axis_storage(input, output, outer, inner, axis_extent, bytes);
}

}  // namespace

extern "C" int tessera_x86_dct_strided_storage(
    const char* digest, const void* input, void* output, int64_t outer,
    int64_t n, int64_t inner, int storage, float output_scale) {
    if (!valid_spectral_storage(storage) || outer <= 0 || inner <= 0)
        return 20;
    size_t bytes = spectral_storage_bytes(storage);
    size_t total = size_t(outer * n * inner) * bytes;
    std::vector<unsigned char> packed_input(total), packed_output(total);
    pack_axis_storage(input, packed_input.data(), outer, n, inner, bytes);
    int rc = tessera_x86_dct_storage(
        digest, packed_input.data(), packed_output.data(), outer * inner, n,
        storage, output_scale);
    if (!rc)
        unpack_axis_storage(packed_output.data(), output, outer, n, inner, bytes);
    return rc;
}

extern "C" int tessera_x86_spectral_conv_strided_storage(
    const char* digest, const void* input, int64_t input_n,
    const void* kernel, int64_t kernel_n, void* output, int64_t outer,
    int64_t inner, int64_t fft_n, int storage) {
    if (!valid_spectral_storage(storage) || outer <= 0 || inner <= 0)
        return 20;
    int64_t output_n = input_n + kernel_n - 1;
    int64_t batch = outer * inner;
    size_t bytes = spectral_storage_bytes(storage);
    std::vector<unsigned char> packed_input(size_t(batch * input_n) * bytes);
    std::vector<unsigned char> packed_kernel(size_t(batch * kernel_n) * bytes);
    std::vector<unsigned char> packed_output(size_t(batch * output_n) * bytes);
    pack_axis_storage(input, packed_input.data(), outer, input_n, inner, bytes);
    pack_axis_storage(kernel, packed_kernel.data(), outer, kernel_n, inner, bytes);
    int rc = tessera_x86_spectral_conv_storage(
        digest, packed_input.data(), input_n, packed_kernel.data(), kernel_n,
        packed_output.data(), batch, fft_n, storage);
    if (!rc)
        unpack_axis_storage(packed_output.data(), output, outer, output_n, inner,
                            bytes);
    return rc;
}

extern "C" int tessera_x86_stft_strided_storage(
    const char* digest, const void* input, const void* window, float* output,
    int64_t outer, int64_t samples, int64_t inner, int64_t win, int64_t hop,
    int64_t frames, int storage, float output_scale) {
    if (!valid_spectral_storage(storage) || outer <= 0 || inner <= 0)
        return 20;
    int64_t batch = outer * inner;
    int64_t bins = win / 2 + 1;
    size_t bytes = spectral_storage_bytes(storage);
    std::vector<unsigned char> packed_input(size_t(batch * samples) * bytes);
    std::vector<float> packed_output(size_t(2 * batch * frames * bins));
    pack_axis_storage(input, packed_input.data(), outer, samples, inner, bytes);
    int rc = tessera_x86_stft_storage(
        digest, packed_input.data(), window, packed_output.data(), batch,
        samples, win, hop, frames, storage, output_scale);
    if (!rc)
        unpack_axis_storage(packed_output.data(), output, outer, frames * bins,
                            inner, sizeof(float) * 2);
    return rc;
}

extern "C" int tessera_x86_istft_strided_storage(
    const char* digest, const float* input, const void* window, void* output,
    int64_t outer, int64_t frames, int64_t bins, int64_t inner, int64_t win,
    int64_t hop, int storage, float output_scale) {
    if (!valid_spectral_storage(storage) || outer <= 0 || inner <= 0 ||
        bins != win / 2 + 1)
        return 20;
    int64_t batch = outer * inner;
    int64_t samples = (frames - 1) * hop + win;
    size_t bytes = spectral_storage_bytes(storage);
    std::vector<float> packed_input(size_t(2 * batch * frames * bins));
    std::vector<unsigned char> packed_output(size_t(batch * samples) * bytes);
    pack_axis_storage(input, packed_input.data(), outer, frames * bins, inner,
                      sizeof(float) * 2);
    int rc = tessera_x86_istft_storage(
        digest, packed_input.data(), window, packed_output.data(), batch,
        frames, win, hop, storage, output_scale);
    if (!rc)
        unpack_axis_storage(packed_output.data(), output, outer, samples, inner,
                            bytes);
    return rc;
}
