// On-device test for the AVX-512 elementwise binary-arithmetic kernel (f32).
//
// Validates tessera_x86_avx512_binary_f32 against the scalar reference AND a
// hand-computed expectation, across kinds (sub/div/maximum/minimum) and lengths
// incl. non-multiple-of-16 tails, plus NaN-propagation for maximum/minimum.
// Runs natively on the AVX-512 host — the "tested + running on the key device"
// proof for the CPU binary lane.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_binary_f32(const float*, const float*,
                                                 int64_t, float*, int);
extern "C" void tessera_x86_avx512_binary_f32(const float*, const float*,
                                              int64_t, float*, int);

static int g_fail = 0;

static double want_binary(double a, double b, int kind) {
    switch (kind) {
    case 0: return a - b;
    case 1: return a / b;
    case 2: return (std::isnan(a) || std::isnan(b)) ? NAN : (a > b ? a : b);
    case 3: return (std::isnan(a) || std::isnan(b)) ? NAN : (a < b ? a : b);
    default: return a;
    }
}

static void check(const char* name, int kind, int64_t n) {
    std::mt19937 rng(2024 + (unsigned)(n * 131 + kind * 7));
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> a((size_t)n), b((size_t)n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) {
        v = dist(rng);
        // keep the divisor away from zero for div
        if (kind == 1 && std::fabs(v) < 0.25f) v += (v < 0 ? -0.5f : 0.5f);
    }

    std::vector<float> ref(n), avx(n);
    tessera_x86_reference_binary_f32(a.data(), b.data(), n, ref.data(), kind);
    tessera_x86_avx512_binary_f32(a.data(), b.data(), n, avx.data(), kind);

    for (int64_t i = 0; i < n; ++i) {
        double want = want_binary(a[i], b[i], kind);
        float tol = 2e-5f * (1.0f + std::fabs((float)want));
        if (std::fabs(avx[i] - ref[i]) > tol ||
            std::fabs(avx[i] - (float)want) > tol) {
            std::printf("FAIL %s kind=%d n=%lld i=%lld: a=%g b=%g avx=%g ref=%g "
                        "want=%g\n", name, kind, (long long)n, (long long)i,
                        a[i], b[i], avx[i], ref[i], (double)want);
            ++g_fail;
            return;
        }
    }
    std::printf("ok   %s kind=%d n=%lld\n", name, kind, (long long)n);
}

// maximum/minimum must propagate NaN (numpy semantics), in both the vector body
// and the scalar tail — unlike the SSE max/min intrinsics.
static void check_nan_propagation() {
    std::vector<float> a = {1.0f, NAN, 3.0f, -1.0f, NAN, 2.0f};
    std::vector<float> b = {2.0f, 5.0f, NAN, -2.0f, NAN, 1.0f};
    int64_t n = (int64_t)a.size();
    for (int kind = 2; kind <= 3; ++kind) {
        std::vector<float> avx(n), ref(n);
        tessera_x86_avx512_binary_f32(a.data(), b.data(), n, avx.data(), kind);
        tessera_x86_reference_binary_f32(a.data(), b.data(), n, ref.data(), kind);
        for (int64_t i = 0; i < n; ++i) {
            double want = want_binary(a[i], b[i], kind);
            bool ok = std::isnan(want)
                          ? (std::isnan(avx[i]) && std::isnan(ref[i]))
                          : (avx[i] == (float)want && ref[i] == (float)want);
            if (!ok) {
                std::printf("FAIL nan kind=%d i=%lld: a=%g b=%g avx=%g ref=%g "
                            "want=%g\n", kind, (long long)i, a[i], b[i], avx[i],
                            ref[i], want);
                ++g_fail;
                return;
            }
        }
        std::printf("ok   nan_propagation kind=%d\n", kind);
    }
}

int main() {
    for (int kind = 0; kind <= 3; ++kind) {
        check("aligned", kind, 64);    // multiple of 16
        check("tail", kind, 70);       // n % 16 != 0
        check("small", kind, 5);       // n < 16 (all scalar tail)
        check("wide", kind, 1024);     // many vector steps
        check("one", kind, 1);         // degenerate
    }
    check_nan_propagation();
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
