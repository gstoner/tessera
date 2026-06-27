// On-device test for the AVX-512 fp-grid quantization kernel vs the scalar
// mantissa-snap reference, across the fp4/fp6/fp8 grids.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_avx512_fpquant_f32(const float*, int64_t, float, int,
                                               float*);

static int g_fail = 0;

static float ref(float v, float maxn, int m) {
    float s = v < 0.0f ? -1.0f : 1.0f;
    float ax = std::fmin(std::fabs(v), maxn);
    if (ax <= 0.0f) return 0.0f;
    float e = std::floor(std::log2(ax));
    float ulp = std::ldexp(1.0f, (int)e - m);
    float r = std::fmin(std::nearbyint(ax / ulp) * ulp, maxn);
    return s * r;
}

static void check(const char* nm, float maxn, int m, int64_t n) {
    std::mt19937 rng(11 + m + (unsigned)n);
    std::uniform_real_distribution<float> d(-maxn * 1.2f, maxn * 1.2f);
    std::vector<float> x(n), out(n);
    for (auto& v : x) v = d(rng);
    x[0] = 0.0f;  // exercise the ax==0 path
    tessera_x86_avx512_fpquant_f32(x.data(), n, maxn, m, out.data());
    for (int64_t i = 0; i < n; ++i) {
        float w = ref(x[i], maxn, m);
        if (std::fabs(out[i] - w) > 1e-6f * std::fmax(1.0f, std::fabs(w))) {
            std::printf("FAIL %s n=%lld i=%lld x=%g got=%g want=%g\n", nm,
                        (long long)n, (long long)i, x[i], out[i], w);
            ++g_fail; return;
        }
    }
    std::printf("ok   %-10s n=%lld\n", nm, (long long)n);
}

int main() {
    for (int64_t n : {16, 70, 5, 1024, 1}) {
        check("fp4_e2m1", 6.0f, 1, n);
        check("fp6_e2m3", 7.5f, 3, n);
        check("fp6_e3m2", 28.0f, 2, n);
        check("fp8_e4m3", 448.0f, 3, n);
        check("fp8_e5m2", 57344.0f, 2, n);
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
