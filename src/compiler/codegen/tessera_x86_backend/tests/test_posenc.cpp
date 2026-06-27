// On-device test for the AVX-512 rope / alibi kernels vs a scalar reference.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_avx512_rope_f32(const float*, const float*, int64_t,
                                            int64_t, float*);
extern "C" void tessera_x86_avx512_alibi_f32(const float*, int64_t, int64_t,
                                             float*);

static int g_fail = 0;

static bool close(float a, float b) {
    return std::fabs(a - b) <= 2e-5f + 2e-5f * std::fabs(b);
}

static void check_rope(int64_t M, int64_t D) {
    std::mt19937 rng((unsigned)(M * 7 + D));
    std::uniform_real_distribution<float> dx(-2.0f, 2.0f);
    std::uniform_real_distribution<float> da(-6.2f, 6.2f);
    std::vector<float> x(M * D), th(M * D), out(M * D);
    for (auto& v : x) v = dx(rng);
    for (auto& v : th) v = da(rng);
    tessera_x86_avx512_rope_f32(x.data(), th.data(), M, D, out.data());
    for (int64_t m = 0; m < M; ++m)
        for (int64_t p = 0; p < D; p += 2) {
            float e = x[m*D+p], o = x[m*D+p+1], a = th[m*D+p];
            float we = e*std::cos(a) - o*std::sin(a);
            float wo = e*std::sin(a) + o*std::cos(a);
            if (!close(out[m*D+p], we) || !close(out[m*D+p+1], wo)) {
                std::printf("FAIL rope M=%lld D=%lld p=%lld\n", (long long)M,
                            (long long)D, (long long)p); ++g_fail; return;
            }
        }
    std::printf("ok   rope  M=%-3lld D=%lld\n", (long long)M, (long long)D);
}

static void check_alibi(int64_t H, int64_t S) {
    std::vector<float> slopes(H), out(H * S * S);
    for (int64_t h = 0; h < H; ++h)
        slopes[h] = std::pow(2.0f, -8.0f * (float)(h + 1) / (float)H);
    tessera_x86_avx512_alibi_f32(slopes.data(), H, S, out.data());
    for (int64_t h = 0; h < H; ++h)
        for (int64_t i = 0; i < S; ++i)
            for (int64_t j = 0; j < S; ++j) {
                float want = slopes[h] * (float)(j - i);
                if (!close(out[(h*S+i)*S+j], want)) {
                    std::printf("FAIL alibi H=%lld S=%lld\n", (long long)H,
                                (long long)S); ++g_fail; return;
                }
            }
    std::printf("ok   alibi H=%-3lld S=%lld\n", (long long)H, (long long)S);
}

int main() {
    for (int64_t D : {2, 8, 32, 64, 34, 130}) check_rope(4, D);
    for (int64_t S : {1, 4, 16, 17, 33}) check_alibi(3, S);
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
