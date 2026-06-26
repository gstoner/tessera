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
        // hand-computed expectation (independent of both kernels). NaN must
        // propagate (numpy semantics) — for any kind, a NaN in the row => NaN.
        bool row_nan = false;
        double want = (kind == 1) ? -1e30 : 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            double v = x[(size_t)r * cols + c];
            if (std::isnan(v)) { row_nan = true; continue; }
            if (kind == 1) want = v > want ? v : want;
            else want += v;
        }
        if (kind == 2 && cols > 0) want /= (double)cols;

        if (row_nan) {
            // every kind must yield NaN from both kernels
            if (!std::isnan(avx[r]) || !std::isnan(ref[r])) {
                std::printf("FAIL %s kind=%d [%lld,%lld] row %lld: NaN not "
                            "propagated: avx=%g ref=%g\n", name, kind,
                            (long long)rows, (long long)cols, (long long)r,
                            avx[r], ref[r]);
                ++g_fail;
                return;
            }
            continue;
        }

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

// NaN propagation: a row with a NaN must reduce to NaN for every kind, in both
// the vector body and the scalar tail (matches numpy np.amax/np.sum semantics).
static void check_nan(int kind, int64_t cols, int64_t nan_col) {
    const int64_t rows = 3;
    std::vector<float> x((size_t)rows * cols, 1.5f);
    std::vector<float> avx(rows), ref(rows);
    for (int64_t r = 0; r < rows; ++r)
        x[(size_t)r * cols + nan_col] = std::nanf("");
    tessera_x86_avx512_reduce_f32(x.data(), rows, cols, avx.data(), kind);
    tessera_x86_reference_reduce_f32(x.data(), rows, cols, ref.data(), kind);
    for (int64_t r = 0; r < rows; ++r) {
        if (!std::isnan(avx[r]) || !std::isnan(ref[r])) {
            std::printf("FAIL nan kind=%d cols=%lld nan_col=%lld row %lld: "
                        "avx=%g ref=%g\n", kind, (long long)cols,
                        (long long)nan_col, (long long)r, avx[r], ref[r]);
            ++g_fail;
            return;
        }
    }
    std::printf("ok   nan kind=%d cols=%lld nan_col=%lld\n", kind,
                (long long)cols, (long long)nan_col);
}

int main() {
    for (int kind = 0; kind <= 2; ++kind) {
        check("aligned", kind, 4, 64);     // cols multiple of 16
        check("tail", kind, 8, 70);        // cols % 16 != 0
        check("small", kind, 3, 5);        // cols < 16 (all scalar tail)
        check("wide", kind, 2, 1024);      // many vector steps
        check("onecol", kind, 5, 1);       // degenerate
        check_nan(kind, 64, 5);            // NaN in the vector body
        check_nan(kind, 70, 67);           // NaN in the scalar tail
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
