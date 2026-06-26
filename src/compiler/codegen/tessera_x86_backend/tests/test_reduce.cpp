// On-device test for the AVX-512 row-reduction kernel (f32).
//
// Validates tessera_x86_avx512_reduce_f32 against the scalar reference AND a
// hand-computed expectation, across kinds (sum/max/mean) and shapes incl.
// non-multiple-of-16 column tails. Runs natively on the AVX-512 host — this is
// the "tested + running on the key device" proof for the CPU reduction lane.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_reduce_f32(const float*, int64_t, int64_t,
                                                 float*, int);
extern "C" void tessera_x86_avx512_reduce_f32(const float*, int64_t, int64_t,
                                              float*, int);

static int g_fail = 0;

static void check(const char* name, int kind, int64_t rows, int64_t cols) {
    std::mt19937 rng(1234 + (unsigned)(rows * 131 + cols * 7 + kind));
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> x((size_t)rows * cols);
    for (auto& v : x) v = dist(rng);

    std::vector<float> ref(rows), avx(rows);
    tessera_x86_reference_reduce_f32(x.data(), rows, cols, ref.data(), kind);
    tessera_x86_avx512_reduce_f32(x.data(), rows, cols, avx.data(), kind);

    for (int64_t r = 0; r < rows; ++r) {
        // hand-computed expectation (independent of both kernels)
        double want = (kind == 1) ? -1e30 : 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            double v = x[(size_t)r * cols + c];
            if (kind == 1) want = v > want ? v : want;
            else want += v;
        }
        if (kind == 2 && cols > 0) want /= (double)cols;

        float tol = 1e-3f * (1.0f + std::fabs((float)want));
        if (std::fabs(avx[r] - ref[r]) > tol ||
            std::fabs(avx[r] - (float)want) > tol) {
            std::printf("FAIL %s kind=%d [%lld,%lld] row %lld: avx=%g ref=%g "
                        "want=%g\n", name, kind, (long long)rows,
                        (long long)cols, (long long)r, avx[r], ref[r],
                        (double)want);
            ++g_fail;
            return;
        }
    }
    std::printf("ok   %s kind=%d [%lld,%lld]\n", name, kind, (long long)rows,
                (long long)cols);
}

int main() {
    for (int kind = 0; kind <= 2; ++kind) {
        check("aligned", kind, 4, 64);     // cols multiple of 16
        check("tail", kind, 8, 70);        // cols % 16 != 0
        check("small", kind, 3, 5);        // cols < 16 (all scalar tail)
        check("wide", kind, 2, 1024);      // many vector steps
        check("onecol", kind, 5, 1);       // degenerate
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
