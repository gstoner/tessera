// On-device test for the x86 inclusive prefix scan kernel (f32).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_scan_f32(const float*, int64_t, int64_t,
                                               float*, int);
extern "C" void tessera_x86_avx512_scan_f32(const float*, int64_t, int64_t,
                                            float*, int);

static int g_fail = 0;

static void check(int kind, int64_t rows, int64_t cols) {
    std::mt19937 rng(7 + (unsigned)(rows * 131 + cols * 7 + kind));
    // cumprod over many >1 overflows; keep near 1 for kind 1.
    std::uniform_real_distribution<float> dist(kind == 1 ? 0.85f : -3.0f,
                                               kind == 1 ? 1.15f : 3.0f);
    std::vector<float> x((size_t)rows * cols);
    for (auto& v : x) v = dist(rng);
    std::vector<float> ref(x.size()), avx(x.size());
    tessera_x86_reference_scan_f32(x.data(), rows, cols, ref.data(), kind);
    tessera_x86_avx512_scan_f32(x.data(), rows, cols, avx.data(), kind);
    for (int64_t r = 0; r < rows; ++r) {
        double acc = kind == 0 ? 0.0 : kind == 1 ? 1.0 : kind == 2 ? -1e30 : 1e30;
        for (int64_t c = 0; c < cols; ++c) {
            double v = x[(size_t)r * cols + c];
            acc = kind == 0 ? acc + v
                  : kind == 1 ? acc * v
                  : kind == 2 ? (v > acc ? v : acc)
                              : (v < acc ? v : acc);
            float want = (float)acc;
            size_t i = (size_t)r * cols + c;
            float tol = 1e-4f * (1.0f + std::fabs(want));
            if (std::fabs(avx[i] - ref[i]) > tol || std::fabs(avx[i] - want) > tol) {
                std::printf("FAIL kind=%d [%lld,%lld] r%lld c%lld: avx=%g want=%g\n",
                            kind, (long long)rows, (long long)cols, (long long)r,
                            (long long)c, avx[i], want);
                ++g_fail; return;
            }
        }
    }
    std::printf("ok   kind=%d [%lld,%lld]\n", kind, (long long)rows, (long long)cols);
}

int main() {
    for (int kind = 0; kind <= 3; ++kind) {
        check(kind, 4, 50);
        check(kind, 8, 300);   // long row
        check(kind, 1, 1);
        check(kind, 16, 7);
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
