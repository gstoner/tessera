// On-device test for the AVX-512 ternary-select (where) kernel (f32).
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_where_f32(const uint8_t*, const float*,
                                                const float*, int64_t, float*);
extern "C" void tessera_x86_avx512_where_f32(const uint8_t*, const float*,
                                             const float*, int64_t, float*);

static int g_fail = 0;

static void check(const char* name, int64_t n) {
    std::mt19937 rng(2024 + (unsigned)n);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::uniform_int_distribution<int> ci(0, 3);  // mix 0 and nonzero
    std::vector<uint8_t> c((size_t)n);
    std::vector<float> a((size_t)n), b((size_t)n), ref(n), avx(n);
    for (int64_t i = 0; i < n; ++i) {
        c[i] = (uint8_t)ci(rng); a[i] = dist(rng); b[i] = dist(rng);
    }
    tessera_x86_reference_where_f32(c.data(), a.data(), b.data(), n, ref.data());
    tessera_x86_avx512_where_f32(c.data(), a.data(), b.data(), n, avx.data());
    for (int64_t i = 0; i < n; ++i) {
        float want = c[i] ? a[i] : b[i];
        if (avx[i] != ref[i] || avx[i] != want) {
            std::printf("FAIL %s i=%lld: c=%u a=%g b=%g avx=%g want=%g\n", name,
                        (long long)i, c[i], a[i], b[i], avx[i], want);
            ++g_fail; return;
        }
    }
    std::printf("ok   %s n=%lld\n", name, (long long)n);
}

int main() {
    check("aligned", 64);
    check("tail", 70);
    check("small", 5);
    check("wide", 1024);
    check("one", 1);
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
