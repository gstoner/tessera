#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>

static inline __m512 bf16_to_fp32_16(__m256i v_bf16) {
    // Expand 16x u16 -> 16x u32
    __m512i v32 = _mm512_cvtepu16_epi32(v_bf16);
    // Shift left by 16 to place bf16 bits in the high part of fp32
    v32 = _mm512_slli_epi32(v32, 16);
    // Reinterpret as float
    return _mm512_castsi512_ps(v32);
}

// Simple AVX-512 BF16 GEMM for multiples of 16x16 tiles.
// A: MxK (bf16), B: KxN (bf16), C: MxN (fp32)
extern "C" void tessera_x86_avx512_gemm_bf16(const uint16_t* A, const uint16_t* B, float* C,
                                             int M, int N, int K, float beta)
{
    // Initialize C *= beta
    if (beta == 0.0f) {
        std::memset(C, 0, sizeof(float)*size_t(M)*size_t(N));
    } else {
        for (int i=0;i<M*N;i++) C[i] *= beta;
    }

#if defined(__AVX512BF16__)
    // Use VDPBF16PS if available
    const int BM = 16;
    const int BN = 16;
    const int BK = 32; // process 32 K at a time
    for (int m=0;m<M;m+=BM) {
        for (int n=0;n<N;n+=BN) {
            __m512 acc[BM][BN/16];
            for (int i=0;i<BM;i++) for (int j=0;j<BN/16;j++) acc[i][j] = _mm512_setzero_ps();
            for (int k=0;k<K;k+=BK) {
                for (int kk=0; kk<BK; kk+=32) {
                    for (int i=0;i<BM;i++) {
                        const uint16_t* a_ptr = A + (size_t)(m+i)*K + k + kk;
                        // Load 32 bf16 as __m512bh (reinterpret from epi16 load)
                        __m512bh a0 = (__m512bh)_mm512_loadu_epi16((const void*)a_ptr);
                        for (int j=0;j<BN;j+=16) {
                            const uint16_t* b_ptr = B + (size_t)(k+kk)*N + (n+j);
                            __m512bh b0 = (__m512bh)_mm512_loadu_epi16((const void*)b_ptr);
                            acc[i][j/16] = _mm512_dpbf16_ps(acc[i][j/16], a0, b0);
                        }
                    }
                }
            }
            // Store
            for (int i=0;i<BM;i++) {
                for (int j=0;j<BN;j+=16) {
                    float* cptr = C + (size_t)(m+i)*N + (n+j);
                    _mm512_storeu_ps(cptr, acc[i][j/16]);
                }
            }
        }
    }
#else
    // BF16 emulation: convert bf16 -> fp32 then FMAs
    const int BM = 8, BN = 16, BK = 32;
    for (int m=0;m<M;m+=BM) {
        for (int n=0;n<N;n+=BN) {
            __m512 acc[BM][BN/16];
            for (int i=0;i<BM;i++) for (int j=0;j<BN/16;j++) acc[i][j] = _mm512_setzero_ps();
            for (int k=0;k<K;k+=BK) {
                for (int kk=0; kk<BK; kk+=16) {
                    for (int i=0;i<BM;i++) {
                        const uint16_t* a_ptr = A + (size_t)(m+i)*K + k + kk;
                        __m256i a_bf16 = _mm256_loadu_si256((const __m256i*)a_ptr); // 16x u16
                        __m512 a_f32 = bf16_to_fp32_16(a_bf16);
                        for (int j=0;j<BN;j+=16) {
                            const uint16_t* b_ptr = B + (size_t)(k+kk)*N + (n+j);
                            __m256i b_bf16 = _mm256_loadu_si256((const __m256i*)b_ptr);
                            __m512 b_f32 = bf16_to_fp32_16(b_bf16);
                            acc[i][j/16] = _mm512_fmadd_ps(a_f32, b_f32, acc[i][j/16]);
                        }
                    }
                }
            }
            for (int i=0;i<BM;i++) {
                for (int j=0;j<BN;j+=16) {
                    float* cptr = C + (size_t)(m+i)*N + (n+j);
                    _mm512_storeu_ps(cptr, acc[i][j/16]);
                }
            }
        }
    }
#endif
}
