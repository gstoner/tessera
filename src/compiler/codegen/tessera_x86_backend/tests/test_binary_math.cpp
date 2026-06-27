// On-device test for the transcendental-backed binary kernels (f32):
// pow(a,b)=a^b (positive base) and silu_mul(a,b)=a*sigmoid(a)*b.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_avx512_pow_f32(const float*, const float*, int64_t,
                                           float*);
extern "C" void tessera_x86_avx512_silu_mul_f32(const float*, const float*,
                                                int64_t, float*);

static int g_fail = 0;

static void check_pow(int64_t n) {
    std::mt19937 rng(7 + (unsigned)n);
    std::uniform_real_distribution<float> base(0.05f, 8.0f);   // positive base
    std::uniform_real_distribution<float> exp_(-3.0f, 3.0f);
    std::vector<float> a((size_t)n), b((size_t)n), out(n);
    for (int64_t i = 0; i < n; ++i) { a[i] = base(rng); b[i] = exp_(rng); }
    tessera_x86_avx512_pow_f32(a.data(), b.data(), n, out.data());
    for (int64_t i = 0; i < n; ++i) {
        float want = std::pow(a[i], b[i]);
        float tol = 2e-5f + 2e-5f * std::fabs(want);
        if (std::fabs(out[i] - want) > tol) {
            std::printf("FAIL pow n=%lld i=%lld: a=%g b=%g got=%g want=%g\n",
                        (long long)n, (long long)i, a[i], b[i], out[i], want);
            ++g_fail; return;
        }
    }
    std::printf("ok   pow      n=%lld\n", (long long)n);
}

static void check_silu_mul(int64_t n) {
    std::mt19937 rng(99 + (unsigned)n);
    std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
    std::vector<float> a((size_t)n), b((size_t)n), out(n);
    for (int64_t i = 0; i < n; ++i) { a[i] = dist(rng); b[i] = dist(rng); }
    tessera_x86_avx512_silu_mul_f32(a.data(), b.data(), n, out.data());
    for (int64_t i = 0; i < n; ++i) {
        float want = (a[i] / (1.0f + std::exp(-a[i]))) * b[i];
        float tol = 2e-5f + 2e-5f * std::fabs(want);
        if (std::fabs(out[i] - want) > tol) {
            std::printf("FAIL silu_mul n=%lld i=%lld: a=%g b=%g got=%g want=%g\n",
                        (long long)n, (long long)i, a[i], b[i], out[i], want);
            ++g_fail; return;
        }
    }
    std::printf("ok   silu_mul n=%lld\n", (long long)n);
}

int main() {
    for (int64_t n : {64, 70, 5, 1024, 1}) { check_pow(n); check_silu_mul(n); }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
