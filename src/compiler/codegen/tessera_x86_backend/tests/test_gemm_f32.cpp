// On-device test for the AVX-512 f32 GEMM microkernel vs a scalar triple-loop
// reference (same accumulation, exact match) across square + rectangular +
// tail-N shapes.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

extern "C" void tessera_x86_avx512_gemm_f32(const float*, const float*, int64_t,
                                            int64_t, int64_t, float*);

static int g_fail = 0;

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
    std::printf(g_fail ? "\n%d FAILED\n" : "\nALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
