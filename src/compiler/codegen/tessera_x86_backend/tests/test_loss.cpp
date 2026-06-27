// On-device test for the AVX-512 pointwise-loss kernel (per-element) vs scalar.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_avx512_pointwise_loss_f32(const float*, const float*,
                                                      int64_t, int, float,
                                                      float*);

static int g_fail = 0;
static const float kLn2 = 0.6931471805599453f;

static float ref(float p, float t, int kind, float prm) {
    float e = p - t, a = std::fabs(e);
    switch (kind) {
    case 0: return e * e;
    case 1: return a;
    case 2: return a <= prm ? 0.5f*a*a : prm*(a-0.5f*prm);
    case 3: return a < prm ? 0.5f*a*a/prm : a-0.5f*prm;
    case 4: return e + std::log1p(std::exp(-2.0f*e)) - kLn2;
    default: return 0;
    }
}

static void check(const char* nm, int kind, float prm, int64_t n) {
    std::mt19937 rng(7 + kind*13 + (unsigned)n);
    std::uniform_real_distribution<float> d(-4.0f, 4.0f);
    std::vector<float> p(n), t(n), out(n);
    for (auto& v : p) v = d(rng);
    for (auto& v : t) v = d(rng);
    tessera_x86_avx512_pointwise_loss_f32(p.data(), t.data(), n, kind, prm,
                                          out.data());
    for (int64_t i = 0; i < n; ++i) {
        float w = ref(p[i], t[i], kind, prm);
        float tol = 2e-5f + 2e-5f * std::fabs(w);
        if (std::fabs(out[i]-w) > tol) {
            std::printf("FAIL %s n=%lld i=%lld got=%g want=%g\n", nm,
                        (long long)n, (long long)i, out[i], w);
            ++g_fail; return;
        }
    }
    std::printf("ok   %-10s n=%lld\n", nm, (long long)n);
}

int main() {
    for (int64_t n : {16, 70, 5, 1024, 1}) {
        check("mse", 0, 0, n);
        check("mae", 1, 0, n);
        check("huber", 2, 1.0f, n);
        check("smooth_l1", 3, 1.0f, n);
        check("log_cosh", 4, 0, n);
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
