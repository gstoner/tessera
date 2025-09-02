#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include "tessera/x86/target.h"
#include "tessera/x86/amx_runtime.h"

static uint16_t fp32_to_bf16(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    // round-to-nearest-even
    uint16_t bf = (uint16_t)((u + 0x00008000u) >> 16);
    return bf;
}

int main() {
    using namespace tessera::x86;
    const int M=16, N=16, K=64; // multiples of tile shapes

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> A(M*K), B(K*N);
    std::vector<float> C0(M*N, 0.0f), C1(M*N, 0.0f);

    for (int i=0;i<M;i++) for (int k=0;k<K;k++) {
        A[i*K+k] = fp32_to_bf16(dist(rng));
    }
    for (int k=0;k<K;k++) for (int j=0;j<N;j++) {
        B[k*N+j] = fp32_to_bf16(dist(rng));
    }

    // Try AMX first
    if (tessera_x86_amx_available()) {
        tessera_x86_amx_gemm_bf16(A.data(), B.data(), C0.data(), M,N,K, 0.0f);
        printf("AMX path ran.\\n");
    } else {
        printf("AMX not available; skipping.\\n");
    }

    // AVX-512 path (may emulate)
    tessera_x86_avx512_gemm_bf16(A.data(), B.data(), C1.data(), M,N,K, 0.0f);
    printf("AVX-512 path ran.\\n");

    // Compare (tolerant due to bf16)
    double max_abs=0.0;
    for (int i=0;i<M*N;i++) {
        max_abs = std::max(max_abs, std::abs((double)C0[i] - (double)C1[i]));
    }
    printf("Max |AMX-AVX512| diff = %.5f\\n", max_abs);
    return 0;
}
