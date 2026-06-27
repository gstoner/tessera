// On-device test for the AVX-512 row-wise norm/softmax kernels (f32) vs a
// scalar reference, at atol/rtol 2e-5.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_avx512_rmsnorm_f32(const float*, int64_t, int64_t,
                                               float, float*);
extern "C" void tessera_x86_avx512_layernorm_f32(const float*, int64_t, int64_t,
                                                 float, float*);
extern "C" void tessera_x86_avx512_softmax_f32(const float*, int64_t, int64_t,
                                               float*);

static int g_fail = 0;
static const float kEps = 1e-5f;

static bool close(float a, float b) {
    return std::fabs(a - b) <= 2e-5f + 2e-5f * std::fabs(b);
}

static void check_rmsnorm(int64_t M, int64_t D) {
    std::mt19937 rng(1 + (unsigned)(M * 131 + D));
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> x(M * D), out(M * D);
    for (auto& v : x) v = dist(rng);
    tessera_x86_avx512_rmsnorm_f32(x.data(), M, D, kEps, out.data());
    for (int64_t m = 0; m < M; ++m) {
        double ss = 0;
        for (int64_t d = 0; d < D; ++d) ss += (double)x[m*D+d]*x[m*D+d];
        float inv = 1.0f / std::sqrt((float)(ss / D) + kEps);
        for (int64_t d = 0; d < D; ++d)
            if (!close(out[m*D+d], x[m*D+d]*inv)) {
                std::printf("FAIL rmsnorm M=%lld D=%lld\n", (long long)M,
                            (long long)D); ++g_fail; return;
            }
    }
    std::printf("ok   rmsnorm   M=%-4lld D=%lld\n", (long long)M, (long long)D);
}

static void check_layernorm(int64_t M, int64_t D) {
    std::mt19937 rng(2 + (unsigned)(M * 131 + D));
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> x(M * D), out(M * D);
    for (auto& v : x) v = dist(rng);
    tessera_x86_avx512_layernorm_f32(x.data(), M, D, kEps, out.data());
    for (int64_t m = 0; m < M; ++m) {
        double mean = 0;
        for (int64_t d = 0; d < D; ++d) mean += x[m*D+d];
        mean /= D;
        double var = 0;
        for (int64_t d = 0; d < D; ++d) var += (x[m*D+d]-mean)*(x[m*D+d]-mean);
        var /= D;
        float inv = 1.0f / std::sqrt((float)var + kEps);
        for (int64_t d = 0; d < D; ++d)
            if (!close(out[m*D+d], (x[m*D+d]-(float)mean)*inv)) {
                std::printf("FAIL layernorm M=%lld D=%lld\n", (long long)M,
                            (long long)D); ++g_fail; return;
            }
    }
    std::printf("ok   layernorm M=%-4lld D=%lld\n", (long long)M, (long long)D);
}

static void check_softmax(int64_t M, int64_t D) {
    std::mt19937 rng(3 + (unsigned)(M * 131 + D));
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
    std::vector<float> x(M * D), out(M * D);
    for (auto& v : x) v = dist(rng);
    tessera_x86_avx512_softmax_f32(x.data(), M, D, out.data());
    for (int64_t m = 0; m < M; ++m) {
        float mx = -INFINITY;
        for (int64_t d = 0; d < D; ++d) mx = x[m*D+d] > mx ? x[m*D+d] : mx;
        double sum = 0;
        for (int64_t d = 0; d < D; ++d) sum += std::exp(x[m*D+d] - mx);
        for (int64_t d = 0; d < D; ++d)
            if (!close(out[m*D+d], (float)(std::exp(x[m*D+d]-mx)/sum))) {
                std::printf("FAIL softmax M=%lld D=%lld\n", (long long)M,
                            (long long)D); ++g_fail; return;
            }
    }
    std::printf("ok   softmax   M=%-4lld D=%lld\n", (long long)M, (long long)D);
}

int main() {
    for (int64_t D : {16, 33, 64, 128, 5, 257}) {
        check_rmsnorm(4, D);
        check_layernorm(4, D);
        check_softmax(4, D);
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
