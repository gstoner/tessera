// On-device test for the AVX-512 elementwise bitwise kernel (i32).
//
// Validates tessera_x86_avx512_bitwise_i32 against the scalar reference AND a
// hand-computed expectation, across kinds (and/or/xor/not) and lengths incl.
// non-multiple-of-16 tails. Runs natively on the AVX-512 host — the "tested +
// running on the key device" proof for the CPU bitwise lane.

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_bitwise_i32(const int32_t*, const int32_t*,
                                                  int64_t, int32_t*, int);
extern "C" void tessera_x86_avx512_bitwise_i32(const int32_t*, const int32_t*,
                                               int64_t, int32_t*, int);

static int g_fail = 0;

static int32_t want_bitwise(int32_t a, int32_t b, int kind) {
    switch (kind) {
    case 0: return a & b;
    case 1: return a | b;
    case 2: return a ^ b;
    case 3: return ~a;
    default: return a;
    }
}

static void check(const char* name, int kind, int64_t n) {
    std::mt19937 rng(2024 + (unsigned)(n * 131 + kind * 7));
    std::uniform_int_distribution<int32_t> dist(-1000000, 1000000);
    std::vector<int32_t> a((size_t)n), b((size_t)n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);

    std::vector<int32_t> ref(n), avx(n);
    tessera_x86_reference_bitwise_i32(a.data(), b.data(), n, ref.data(), kind);
    tessera_x86_avx512_bitwise_i32(a.data(), b.data(), n, avx.data(), kind);

    for (int64_t i = 0; i < n; ++i) {
        int32_t want = want_bitwise(a[i], b[i], kind);
        if (avx[i] != ref[i] || avx[i] != want) {
            std::printf("FAIL %s kind=%d n=%lld i=%lld: a=%d b=%d avx=%d ref=%d "
                        "want=%d\n", name, kind, (long long)n, (long long)i,
                        a[i], b[i], avx[i], ref[i], want);
            ++g_fail;
            return;
        }
    }
    std::printf("ok   %s kind=%d n=%lld\n", name, kind, (long long)n);
}

int main() {
    for (int kind = 0; kind <= 3; ++kind) {
        check("aligned", kind, 64);    // multiple of 16
        check("tail", kind, 70);       // n % 16 != 0
        check("small", kind, 5);       // n < 16 (all scalar tail)
        check("wide", kind, 1024);     // many vector steps
        check("one", kind, 1);         // degenerate
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
