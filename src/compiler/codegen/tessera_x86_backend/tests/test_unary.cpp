// On-device test for the AVX-512 elementwise unary-math kernel (f32).
//
// Validates tessera_x86_avx512_unary_f32 against the scalar reference AND a
// hand-computed expectation, across kinds (sqrt/rsqrt/reciprocal/abs/neg/sign)
// and lengths incl. non-multiple-of-16 tails. Runs natively on the AVX-512
// host — the "tested + running on the key device" proof for the CPU unary lane.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_unary_f32(const float*, int64_t, float*,
                                                int);
extern "C" void tessera_x86_avx512_unary_f32(const float*, int64_t, float*,
                                             int);

static int g_fail = 0;

static double want_unary(double v, int kind) {
    switch (kind) {
    case 0: return std::sqrt(v);
    case 1: return 1.0 / std::sqrt(v);
    case 2: return 1.0 / v;
    case 3: return std::fabs(v);
    case 4: return -v;
    case 5: return (v > 0) ? 1.0 : (v < 0 ? -1.0 : 0.0);
    default: return v;
    }
}

static void check(const char* name, int kind, int64_t n) {
    std::mt19937 rng(2024 + (unsigned)(n * 131 + kind * 7));
    // sqrt/rsqrt/reciprocal need a positive domain; abs/neg/sign take any sign.
    bool pos = (kind == 0 || kind == 1 || kind == 2);
    std::uniform_real_distribution<float> dist(pos ? 0.05f : -3.0f, 3.0f);
    std::vector<float> x((size_t)n);
    for (auto& v : x) v = dist(rng);

    std::vector<float> ref(n), avx(n);
    tessera_x86_reference_unary_f32(x.data(), n, ref.data(), kind);
    tessera_x86_avx512_unary_f32(x.data(), n, avx.data(), kind);

    for (int64_t i = 0; i < n; ++i) {
        double want = want_unary(x[i], kind);
        float tol = 2e-5f * (1.0f + std::fabs((float)want));
        if (std::fabs(avx[i] - ref[i]) > tol ||
            std::fabs(avx[i] - (float)want) > tol) {
            std::printf("FAIL %s kind=%d n=%lld i=%lld: x=%g avx=%g ref=%g "
                        "want=%g\n", name, kind, (long long)n, (long long)i,
                        x[i], avx[i], ref[i], (double)want);
            ++g_fail;
            return;
        }
    }
    std::printf("ok   %s kind=%d n=%lld\n", name, kind, (long long)n);
}

// sign() must yield exactly 0 at +/-0 and propagate the right sign elsewhere,
// in both the vector body and the scalar tail.
static void check_sign_zero() {
    std::vector<float> x = {-2.0f, -0.0f, 0.0f, 3.0f, -1e-9f, 1e-9f};
    int64_t n = (int64_t)x.size();
    std::vector<float> avx(n), ref(n);
    tessera_x86_avx512_unary_f32(x.data(), n, avx.data(), 5);
    tessera_x86_reference_unary_f32(x.data(), n, ref.data(), 5);
    float expect[] = {-1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f};
    for (int64_t i = 0; i < n; ++i) {
        if (avx[i] != expect[i] || ref[i] != expect[i]) {
            std::printf("FAIL sign_zero i=%lld: x=%g avx=%g ref=%g want=%g\n",
                        (long long)i, x[i], avx[i], ref[i], expect[i]);
            ++g_fail;
            return;
        }
    }
    std::printf("ok   sign_zero\n");
}

int main() {
    for (int kind = 0; kind <= 5; ++kind) {
        check("aligned", kind, 64);    // multiple of 16
        check("tail", kind, 70);       // n % 16 != 0
        check("small", kind, 5);       // n < 16 (all scalar tail)
        check("wide", kind, 1024);     // many vector steps
        check("one", kind, 1);         // degenerate
    }
    check_sign_zero();
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
