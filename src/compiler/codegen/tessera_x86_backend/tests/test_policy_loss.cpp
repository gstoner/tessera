// On-device test for the AVX-512 RL policy-loss kernel (ppo / cispo) vs scalar.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_avx512_policy_loss_f32(const float*, const float*,
                                                   const float*, int64_t, int,
                                                   float, float*);

static int g_fail = 0;

static float ref(float ln, float lo, float adv, int kind, float clip) {
    float r = std::exp(ln - lo);
    if (kind == 0) {
        float c = std::fmin(std::fmax(r, 1.0f - clip), 1.0f + clip);
        return -std::fmin(r * adv, c * adv);
    }
    float w = std::fmin(r, clip);
    return -(w * adv * ln);
}

static void check(const char* nm, int kind, float clip, int64_t n) {
    std::mt19937 rng(17 + kind*5 + (unsigned)n);
    std::uniform_real_distribution<float> dlp(-1.0f, 1.0f);  // log-prob diffs
    std::uniform_real_distribution<float> dadv(-3.0f, 3.0f);
    std::vector<float> ln(n), lo(n), adv(n), out(n);
    for (int64_t i = 0; i < n; ++i) {
        lo[i] = dlp(rng); ln[i] = lo[i] + dlp(rng); adv[i] = dadv(rng);
    }
    tessera_x86_avx512_policy_loss_f32(ln.data(), lo.data(), adv.data(), n, kind,
                                       clip, out.data());
    for (int64_t i = 0; i < n; ++i) {
        float w = ref(ln[i], lo[i], adv[i], kind, clip);
        float tol = 2e-5f + 2e-5f * std::fabs(w);
        if (std::fabs(out[i]-w) > tol) {
            std::printf("FAIL %s n=%lld i=%lld got=%g want=%g\n", nm,
                        (long long)n, (long long)i, out[i], w);
            ++g_fail; return;
        }
    }
    std::printf("ok   %-7s n=%lld\n", nm, (long long)n);
}

int main() {
    for (int64_t n : {16, 70, 5, 1024, 1}) {
        check("ppo", 0, 0.2f, n);
        check("cispo", 1, 5.0f, n);
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
