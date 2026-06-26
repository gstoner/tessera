// On-device test for the AVX-512 elementwise logical kernel (i8 bool).
//
// Validates tessera_x86_avx512_logical_i8 against the scalar reference AND a
// hand-computed expectation, across kinds (and/or/xor/not) and lengths incl.
// non-multiple-of-64 tails, plus nonzero-normalization (any nonzero -> true).
// Runs natively on the AVX-512 host — the "tested + running on the key device"
// proof for the CPU logical lane.

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_logical_i8(const uint8_t*, const uint8_t*,
                                                 int64_t, uint8_t*, int);
extern "C" void tessera_x86_avx512_logical_i8(const uint8_t*, const uint8_t*,
                                              int64_t, uint8_t*, int);

static int g_fail = 0;

static uint8_t want_logical(uint8_t a, uint8_t b, int kind) {
    bool ba = a != 0, bb = b != 0;
    switch (kind) {
    case 0: return (ba && bb) ? 1 : 0;
    case 1: return (ba || bb) ? 1 : 0;
    case 2: return (ba != bb) ? 1 : 0;
    case 3: return ba ? 0 : 1;
    default: return 0;
    }
}

static void check(const char* name, int kind, int64_t n) {
    std::mt19937 rng(2024 + (unsigned)(n * 131 + kind * 7));
    // mix 0/1 with arbitrary nonzero values to exercise normalization
    std::uniform_int_distribution<int> dist(0, 4);
    std::vector<uint8_t> a((size_t)n), b((size_t)n);
    for (auto& v : a) v = (uint8_t)dist(rng);
    for (auto& v : b) v = (uint8_t)dist(rng);

    std::vector<uint8_t> ref(n), avx(n);
    tessera_x86_reference_logical_i8(a.data(), b.data(), n, ref.data(), kind);
    tessera_x86_avx512_logical_i8(a.data(), b.data(), n, avx.data(), kind);

    for (int64_t i = 0; i < n; ++i) {
        uint8_t want = want_logical(a[i], b[i], kind);
        if (avx[i] != ref[i] || avx[i] != want) {
            std::printf("FAIL %s kind=%d n=%lld i=%lld: a=%u b=%u avx=%u ref=%u "
                        "want=%u\n", name, kind, (long long)n, (long long)i,
                        a[i], b[i], avx[i], ref[i], want);
            ++g_fail;
            return;
        }
    }
    std::printf("ok   %s kind=%d n=%lld\n", name, kind, (long long)n);
}

int main() {
    for (int kind = 0; kind <= 3; ++kind) {
        check("aligned", kind, 128);   // multiple of 64
        check("tail", kind, 140);      // n % 64 != 0
        check("small", kind, 5);       // n < 64 (all scalar tail)
        check("wide", kind, 4096);     // many vector steps
        check("one", kind, 1);         // degenerate
    }
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
