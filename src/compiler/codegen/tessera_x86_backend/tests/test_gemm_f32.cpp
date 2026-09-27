// On-device test for the AVX-512 f32 GEMM microkernel vs a scalar triple-loop
// reference (same accumulation, exact match) across square + rectangular +
// tail-N shapes.
//
// X86-GEMM-ALIGN-1: the kernel packs B into 64-byte-aligned K-blocked panels. check_bitwise()
// places B at every 4-byte offset within a cache line and requires the output to be
// bitwise identical to `oracle_unpacked` -- the pre-2026-09-27 kernel, kept here as
// the declared oracle (Decision #31) -- for every offset, on the packed (M > 1) and
// direct (M == 1) paths, full and tail strips.
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" void tessera_x86_avx512_gemm_f32(const float*, const float*, int64_t,
                                            int64_t, int64_t, float*);

static int g_fail = 0;

// The kernel as it was before X86-GEMM-ALIGN-1: one unaligned load of B per FMA,
// row-major, one accumulator per 16-column strip, k in order.
static void oracle_unpacked(const float* A, const float* B, int64_t M, int64_t N,
                            int64_t K, float* C) {
    for (int64_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        float* c = C + m * N;
        for (int64_t n = 0; n < N; n += 16) {
            const int64_t w = N - n < 16 ? N - n : 16;
            const __mmask16 mask = w == 16 ? (__mmask16)0xffffu
                                           : (__mmask16)((1u << (unsigned)w) - 1u);
            __m512 acc = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; ++k)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]),
                                      _mm512_maskz_loadu_ps(mask, B + k * N + n), acc);
            _mm512_mask_storeu_ps(c + n, mask, acc);
        }
    }
}

static void check_bitwise(int64_t M, int64_t N, int64_t K) {
    std::mt19937 rng((unsigned)(M * 131 + N * 17 + K * 7 + 5));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(M * K), Bsrc(K * N), want(M * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : Bsrc) v = dist(rng);
    oracle_unpacked(A.data(), Bsrc.data(), M, N, K, want.data());
    const size_t bytes = (size_t)K * N * sizeof(float);
    char* raw = static_cast<char*>(std::aligned_alloc(64, (bytes + 128 + 63) / 64 * 64));
    std::vector<float> C(M * N);
    for (int off = 0; off < 64; off += 4) {
        float* B = reinterpret_cast<float*>(raw + off);
        std::memcpy(B, Bsrc.data(), bytes);
        std::fill(C.begin(), C.end(), -7.0f);
        tessera_x86_avx512_gemm_f32(A.data(), B, M, N, K, C.data());
        if (std::memcmp(C.data(), want.data(), C.size() * sizeof(float)) != 0) {
            std::printf("FAIL bitwise M=%lld N=%lld K=%lld B%%64=%d\n", (long long)M,
                        (long long)N, (long long)K, off);
            ++g_fail;
            std::free(raw);
            return;
        }
    }
    std::free(raw);
    std::printf("ok   bitwise M=%-4lld N=%-4lld K=%-4lld at B%%64 = 0..60\n",
                (long long)M, (long long)N, (long long)K);
}

static void check(int64_t M, int64_t N, int64_t K) {
    std::mt19937 rng((unsigned)(M * 911 + N * 31 + K));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A(M * K), B(K * N), C(M * N), ref(M * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    tessera_x86_avx512_gemm_f32(A.data(), B.data(), M, N, K, C.data());
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; ++k) acc += A[m*K+k] * B[k*N+n];
            ref[m*N+n] = acc;
        }
    float worst = 0.0f;
    for (int64_t i = 0; i < M * N; ++i) {
        float err = std::fabs(C[i] - ref[i]);
        float tol = 1e-4f + 1e-4f * std::fabs(ref[i]);
        if (err > tol) {
            std::printf("FAIL gemm M=%lld N=%lld K=%lld i=%lld: got=%g want=%g\n",
                        (long long)M, (long long)N, (long long)K, (long long)i,
                        C[i], ref[i]);
            ++g_fail; return;
        }
        worst = std::fmax(worst, err);
    }
    std::printf("ok   gemm M=%-4lld N=%-4lld K=%-4lld worst=%.2e\n",
                (long long)M, (long long)N, (long long)K, worst);
}

int main() {
    check(1, 1, 1);
    check(4, 16, 8);
    check(8, 17, 33);      // tail N
    check(16, 64, 64);
    check(7, 5, 9);
    check(32, 128, 256);
    check(3, 1, 100);
    check(2, 300, 40);     // two panels, second a 3-strip tail block
    // bitwise vs the pre-align kernel at every B offset
    check_bitwise(1, 1, 1);
    check_bitwise(1, 250, 33);   // M == 1: direct path, tail strip
    check_bitwise(2, 16, 5);
    check_bitwise(7, 129, 17);   // one full 8-strip panel + a 1-wide tail panel
    check_bitwise(16, 256, 64);
    check_bitwise(33, 100, 3);   // 7-strip block, last strip 4 wide
    check_bitwise(4, 16, 0);     // K == 0: C must be the empty sum (zeros)
    check_bitwise(2, 16, 512);   // exactly one K block
    check_bitwise(2, 16, 513);   // a one-row second K block (continues through C)
    check_bitwise(3, 130, 1100); // two full K blocks + a partial one, tail panel
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
