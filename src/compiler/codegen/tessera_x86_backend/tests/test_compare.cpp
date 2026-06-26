// On-device test for the AVX-512 elementwise comparison kernel (f32 -> bool).
//
// Validates tessera_x86_avx512_compare_f32 against the scalar reference AND a
// hand-computed expectation, across kinds (eq/ne/lt/le/gt/ge) and lengths incl.
// non-multiple-of-16 tails, plus NaN semantics (ordered everywhere except ne).
// Runs natively on the AVX-512 host — the "tested + running on the key device"
// proof for the CPU comparison lane.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_reference_compare_f32(const float*, const float*,
                                                  int64_t, uint8_t*, int);
extern "C" void tessera_x86_avx512_compare_f32(const float*, const float*,
                                               int64_t, uint8_t*, int);

static int g_fail = 0;

static uint8_t want_compare(float a, float b, int kind) {
    switch (kind) {
    case 0: return a == b ? 1 : 0;
    case 1: return a != b ? 1 : 0;
    case 2: return a < b ? 1 : 0;
    case 3: return a <= b ? 1 : 0;
    case 4: return a > b ? 1 : 0;
    case 5: return a >= b ? 1 : 0;
    default: return 0;
    }
}

static void check(const char* name, int kind, int64_t n) {
    std::mt19937 rng(2024 + (unsigned)(n * 131 + kind * 7));
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> a((size_t)n), b((size_t)n);
    for (auto& v : a) v = dist(rng);
    for (int64_t i = 0; i < n; ++i)
        // half the time make b == a so eq/le/ge actually fire
        b[i] = (rng() & 1) ? a[i] : dist(rng);

    std::vector<uint8_t> ref(n), avx(n);
    tessera_x86_reference_compare_f32(a.data(), b.data(), n, ref.data(), kind);
    tessera_x86_avx512_compare_f32(a.data(), b.data(), n, avx.data(), kind);

    for (int64_t i = 0; i < n; ++i) {
        uint8_t want = want_compare(a[i], b[i], kind);
        if (avx[i] != ref[i] || avx[i] != want) {
            std::printf("FAIL %s kind=%d n=%lld i=%lld: a=%g b=%g avx=%u ref=%u "
                        "want=%u\n", name, kind, (long long)n, (long long)i,
                        a[i], b[i], avx[i], ref[i], want);
            ++g_fail;
            return;
        }
    }
    std::printf("ok   %s kind=%d n=%lld\n", name, kind, (long long)n);
}

// NaN: ordered predicates yield 0; ne yields 1 — in both vector body and tail.
static void check_nan() {
    const float qnan = std::nanf("");
    std::vector<float> a = {1.0f, qnan, qnan, 2.0f, qnan, 5.0f};
    std::vector<float> b = {1.0f, 1.0f, qnan, 3.0f, qnan, 5.0f};
    int64_t n = (int64_t)a.size();
    for (int kind = 0; kind <= 5; ++kind) {
        std::vector<uint8_t> avx(n), ref(n);
        tessera_x86_avx512_compare_f32(a.data(), b.data(), n, avx.data(), kind);
        tessera_x86_reference_compare_f32(a.data(), b.data(), n, ref.data(), kind);
        for (int64_t i = 0; i < n; ++i) {
            uint8_t want = want_compare(a[i], b[i], kind);
            if (avx[i] != want || ref[i] != want) {
                std::printf("FAIL nan kind=%d i=%lld: a=%g b=%g avx=%u ref=%u "
                            "want=%u\n", kind, (long long)i, a[i], b[i],
                            avx[i], ref[i], want);
                ++g_fail;
                return;
            }
        }
        std::printf("ok   nan kind=%d\n", kind);
    }
}

int main() {
    for (int kind = 0; kind <= 5; ++kind) {
        check("aligned", kind, 64);
        check("tail", kind, 70);
        check("small", kind, 5);
        check("wide", kind, 1024);
        check("one", kind, 1);
    }
    check_nan();
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
