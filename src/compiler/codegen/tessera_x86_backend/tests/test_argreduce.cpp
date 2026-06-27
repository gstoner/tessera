// On-device test for the x86 row arg-reduction kernel (f32 -> i32 index).
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_argreduce_f32(const float*, int64_t,
                                                    int64_t, int32_t*, int);
extern "C" void tessera_x86_avx512_argreduce_f32(const float*, int64_t, int64_t,
                                                 int32_t*, int);

static int g_fail = 0;

static int32_t want_arg(const float* row, int64_t cols, int kind) {
    int32_t bi = 0;
    float best = row[0];
    for (int64_t c = 1; c < cols; ++c) {
        bool better = (kind == 0) ? (row[c] > best) : (row[c] < best);  // strict
        if (better) { best = row[c]; bi = (int32_t)c; }
    }
    return bi;
}

static void check(int kind, int64_t rows, int64_t cols) {
    std::mt19937 rng(11 + (unsigned)(rows * 131 + cols * 7 + kind));
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> x((size_t)rows * cols);
    for (auto& v : x) v = dist(rng);
    std::vector<int32_t> ref(rows), avx(rows);
    tessera_x86_reference_argreduce_f32(x.data(), rows, cols, ref.data(), kind);
    tessera_x86_avx512_argreduce_f32(x.data(), rows, cols, avx.data(), kind);
    for (int64_t r = 0; r < rows; ++r) {
        int32_t want = want_arg(x.data() + r * cols, cols, kind);
        if (avx[r] != ref[r] || avx[r] != want) {
            std::printf("FAIL kind=%d [%lld,%lld] r%lld: avx=%d want=%d\n", kind,
                        (long long)rows, (long long)cols, (long long)r, avx[r], want);
            ++g_fail; return;
        }
    }
    std::printf("ok   kind=%d [%lld,%lld]\n", kind, (long long)rows, (long long)cols);
}

// equal values: the FIRST occurrence must win (strict comparison).
static void check_tie() {
    std::vector<float> x = {1.0f, 5.0f, 5.0f, 2.0f, 5.0f};
    int32_t avx = 0;
    tessera_x86_avx512_argreduce_f32(x.data(), 1, 5, &avx, 0);
    if (avx != 1) { std::printf("FAIL tie: argmax=%d want 1\n", avx); ++g_fail; return; }
    std::printf("ok   tie (first-occurrence)\n");
}

int main() {
    for (int kind = 0; kind <= 1; ++kind) {
        check(kind, 4, 50);
        check(kind, 8, 300);
        check(kind, 16, 7);
        check(kind, 1, 1);
    }
    check_tie();
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
