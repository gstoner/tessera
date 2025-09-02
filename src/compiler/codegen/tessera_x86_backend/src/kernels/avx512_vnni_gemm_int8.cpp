#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>

// u8*s8 -> s32 accumulate
extern "C" void tessera_x86_avx512_vnni_gemm_u8s8_s32(const uint8_t* A, const int8_t* B, int32_t* C,
                                                      int M, int N, int K, int beta)
{
#if !defined(__AVX512VNNI__)
    // Minimal scalar fallback if VNNI not available
    if (beta == 0) std::memset(C, 0, sizeof(int32_t)*(size_t)M*(size_t)N);
    else if (beta != 1) for (int i=0;i<M*N;i++) C[i] *= beta;
    for (int m=0;m<M;m++) {
        for (int n=0;n<N;n++) {
            int32_t acc = 0;
            for (int k=0;k<K;k++) acc += int32_t(A[m*K+k]) * int32_t(B[k*N+n]);
            C[m*N+n] += acc;
        }
    }
    return;
#else
    if (beta == 0) std::memset(C, 0, sizeof(int32_t)*(size_t)M*(size_t)N);
    else if (beta != 1) for (int i=0;i<M*N;i++) C[i] *= beta;

    const int BN = 16;
    for (int m=0; m<M; ++m) {
        for (int n=0; n<N; n+=BN) {
            int nb = std::min(BN, N - n);
            __m512i acc = _mm512_setzero_si512();
            int k=0;
            for (; k+63 < K; k+=64) {
                // Load 64 A bytes and broadcast in 4x lanes of 16 u8 each
                __m512i a0 = _mm512_loadu_si512((const void*)(A + m*K + k));
                // Process B 64xnb
                for (int j=0; j<nb; j+=16) {
                    __m512i b0 = _mm512_loadu_si512((const void*)(B + (k+0)*N + n + j));
                    __m512i b1 = _mm512_loadu_si512((const void*)(B + (k+16)*N + n + j));
                    __m512i b2 = _mm512_loadu_si512((const void*)(B + (k+32)*N + n + j));
                    __m512i b3 = _mm512_loadu_si512((const void*)(B + (k+48)*N + n + j));
                    __m512i acc0 = _mm512_setzero_si512();
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_extracti64x2_epi64(a0, 0), b0);
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_extracti64x2_epi64(a0, 1), b1);
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_extracti64x2_epi64(a0, 2), b2);
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_extracti64x2_epi64(a0, 3), b3);
                    // Store/accumulate
                    int32_t* cptr = C + m*N + n + j;
                    __m512i old = _mm512_loadu_si512((const void*)cptr);
                    __m512i sum = _mm512_add_epi32(old, acc0);
                    _mm512_storeu_si512((void*)cptr, sum);
                }
            }
            // Remainder K cleanup (scalar)
            for (; k<K; ++k) {
                for (int j=0;j<nb;j++) {
                    C[m*N + n + j] += int32_t(A[m*K+k]) * int32_t(B[k*N + n + j]);
                }
            }
        }
    }
#endif
}
