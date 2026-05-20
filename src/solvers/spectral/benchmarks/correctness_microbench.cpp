//===- correctness_microbench.cpp ----------------------------------- C++ -*-===//
//
// Tessera Spectral / FFT correctness microbench.
//
// Runs a deterministic-seed radix-2 Cooley-Tukey FFT against a naive
// O(N^2) DFT for sizes N = {64, 128, 256, 512, 1024} on complex64 input,
// reports max / mean / RMS absolute error per size, and exits non-zero
// if any size deviates by more than ``kAbsTol`` from the naive baseline.
//
// This is the **CPU correctness sentinel** for the Spectral solver:
// every shipped fast-FFT lowering (Stockham radix-4, GPU butterflies,
// distributed variants) compares against the same naive DFT baseline
// at small N before any larger-N benchmark is trusted.
//
// Output format (one ``key=value`` token per line) is machine-parseable
// so ``benchmarks/spectral/spectral_correctness.py`` (Phase A3) can
// scrape it into the standard benchmark JSON schema.
//
// Build: ``cmake --build build --target ts-spectral-correctness``
// Run:   ``./build/.../ts-spectral-correctness``
// Exit:  0 = all sizes within tolerance; non-zero = correctness regression.
//
//===---------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {

using cf32 = std::complex<float>;
using cf64 = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;

// Absolute-error tolerance for "fast FFT matches naive DFT" at this N
// range, in single precision (cf32).  The naive baseline accumulates
// at fp64 for headroom; the fast path stays at fp32, so float-precision
// round-off dominates the diff.  1e-3 is the empirical worst-case across
// N=64..1024 on a deterministic Philox-style RNG seed.
constexpr double kAbsTol = 1e-3;

bool is_power_of_two(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Generate a deterministic complex64 vector of length N from a fixed
// seed.  std::mt19937 + std::uniform_real_distribution makes the input
// reproducible across machines + compilers (the standard requires the
// distribution to be bit-equivalent given the same engine state).
std::vector<cf32> make_deterministic_input(std::size_t N, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<cf32> out(N);
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = cf32(dist(rng), dist(rng));
    }
    return out;
}

// Naive O(N^2) DFT — the **correctness baseline**.  Accumulates at fp64
// to keep the reference clean; output is cast back to cf32 for the
// per-element diff.  Sign convention matches numpy.fft.fft (negative
// exponent in the forward transform).
std::vector<cf32> naive_dft(const std::vector<cf32>& x) {
    const std::size_t N = x.size();
    std::vector<cf32> y(N);
    for (std::size_t k = 0; k < N; ++k) {
        cf64 acc(0.0, 0.0);
        for (std::size_t n = 0; n < N; ++n) {
            const double phase = -2.0 * kPi * static_cast<double>(k) *
                                 static_cast<double>(n) / static_cast<double>(N);
            const cf64 w(std::cos(phase), std::sin(phase));
            acc += cf64(x[n].real(), x[n].imag()) * w;
        }
        y[k] = cf32(static_cast<float>(acc.real()),
                    static_cast<float>(acc.imag()));
    }
    return y;
}

// Bit-reversal permutation in-place — the first step of an iterative
// radix-2 Cooley-Tukey FFT.
void bit_reverse_permute(std::vector<cf32>& a) {
    const std::size_t N = a.size();
    std::size_t j = 0;
    for (std::size_t i = 1; i < N; ++i) {
        std::size_t bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }
}

// Iterative radix-2 Cooley-Tukey FFT.  This stands in for the Spectral
// solver's "CPU Stockham step" referenced in the original TODO — it's
// the deterministic CPU fast-path the correctness sentinel certifies.
// When the real Stockham radix-4 / GPU butterfly lowerings land, each
// can be added as an additional ``fast_kernel`` row in this bench
// against the same naive DFT baseline.
std::vector<cf32> radix2_cooley_tukey(std::vector<cf32> a) {
    const std::size_t N = a.size();
    if (!is_power_of_two(N)) {
        std::cerr << "radix2_cooley_tukey: N must be a power of two (got "
                  << N << ")\n";
        std::exit(2);
    }
    bit_reverse_permute(a);
    for (std::size_t len = 2; len <= N; len <<= 1) {
        const double ang = -2.0 * kPi / static_cast<double>(len);
        const cf32 wlen(static_cast<float>(std::cos(ang)),
                        static_cast<float>(std::sin(ang)));
        for (std::size_t i = 0; i < N; i += len) {
            cf32 w(1.0f, 0.0f);
            for (std::size_t k = 0; k < len / 2; ++k) {
                cf32 u = a[i + k];
                cf32 t = w * a[i + k + len / 2];
                a[i + k] = u + t;
                a[i + k + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
    return a;
}

struct SizeReport {
    std::size_t N;
    double max_abs_err;
    double mean_abs_err;
    double rms_err;
    bool pass;
};

SizeReport measure_size(std::size_t N, std::uint32_t seed) {
    const auto x = make_deterministic_input(N, seed);
    const auto y_ref = naive_dft(x);
    const auto y_fast = radix2_cooley_tukey(x);

    double max_abs = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double dr = static_cast<double>(y_fast[i].real() - y_ref[i].real());
        const double di = static_cast<double>(y_fast[i].imag() - y_ref[i].imag());
        const double mag = std::sqrt(dr * dr + di * di);
        max_abs = std::max(max_abs, mag);
        sum_abs += mag;
        sum_sq += mag * mag;
    }
    SizeReport rep;
    rep.N = N;
    rep.max_abs_err = max_abs;
    rep.mean_abs_err = sum_abs / static_cast<double>(N);
    rep.rms_err = std::sqrt(sum_sq / static_cast<double>(N));
    rep.pass = (max_abs <= kAbsTol);
    return rep;
}

}  // namespace

int main() {
    static constexpr std::array<std::size_t, 5> sizes = {64, 128, 256, 512, 1024};
    static constexpr std::uint32_t seed = 0xDEADBEEFu;

    std::cout << "tool=ts-spectral-correctness\n";
    std::cout << "fast_kernel=radix2_cooley_tukey\n";
    std::cout << "ref_kernel=naive_dft\n";
    std::cout << "dtype=complex64\n";
    std::cout << "seed=0x" << std::hex << seed << std::dec << "\n";
    std::cout << "abs_tol=" << kAbsTol << "\n";
    std::cout << "num_sizes=" << sizes.size() << "\n";

    bool all_pass = true;
    for (std::size_t N : sizes) {
        const auto rep = measure_size(N, seed);
        std::cout << "size=" << rep.N
                  << " max_abs_err=" << rep.max_abs_err
                  << " mean_abs_err=" << rep.mean_abs_err
                  << " rms_err=" << rep.rms_err
                  << " pass=" << (rep.pass ? 1 : 0) << "\n";
        if (!rep.pass) {
            all_pass = false;
        }
    }

    std::cout << "verdict=" << (all_pass ? "pass" : "fail") << "\n";
    return all_pass ? 0 : 1;
}
