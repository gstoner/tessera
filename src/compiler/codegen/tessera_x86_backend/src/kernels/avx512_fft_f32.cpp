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
