// AVX-512 f32 GEMM microkernel for the Tessera x86 backend — the ctypes-loadable
// C = A[M,K] @ B[K,N] (row-major, f32) that the runtime matmul-family lane
// builds on (batched_gemm / linear_general / einsum / attention all compose
// around this 2D GEMM in Python, mirroring the ROCm WMMA-family executor).
//
// The bf16/AMX GEMM lives in the static backend lib; this is the pure-f32
// AVX-512 path exposed in libtessera_x86_elementwise.so. Vectorizes over N (16
// f32 lanes/__m512), accumulating over K with a broadcast of A[m,k] and an FMA
// — a real vectorized GEMM (not a scalar triple loop). N % 16 tail via mask.
// f32 accumulate; matches numpy matmul to a K-scaled tolerance.

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include "tessera/Rank2Index.h"

using tessera::layout::linearIndex2D;
using tessera::layout::Rank2Order;

// X86-GEMM-ALIGN-1 (2026-09-27). The kernel used to issue one 64-byte
// `_mm512_loadu_ps` of B per FMA straight from the caller's buffer. When B's
// address is not 64-byte aligned every such load spans two cache lines, and at
// 256^3 the kernel ran ~1.5x slower (X86-MATMUL-BIMODAL-1). numpy guarantees
// only 16-byte alignment, so the level was decided by the caller's heap.
//
// The kernel now owns B's alignment. For each block of up to kStrips 16-column
// strips and each block of up to kKBlock rows it copies B[k0:k0+kc, n0:n0+16*S]
// into a 64-byte-aligned kc x (16*S) panel (row k contiguous, so each B row
// segment is read once, and the panel stays L2-resident), and every FMA reads
// the panel with an aligned load. The S strips of a panel row are kept in S
// independent accumulators; between K blocks an accumulator continues through
// C (an exact fp32 store and reload). Per output element the FMA sequence is
// therefore unchanged -- acc = 0, then acc = fma(A[m,k], B[k,n], acc) for
// k = 0..K-1 in order, masked-off lanes read as zero and are never stored --
// so the result is bitwise identical to the previous kernel for every
// alignment. C must not alias A or B (it is read back between K blocks).
//
// M == 1 reads B directly (no pack): a GEMV never reuses the panel, so packing
// is pure copy overhead there (measured slower than the direct read at every B
// alignment). If the panel cannot be allocated the direct path runs too --
// same result, only slower -- so the kernel never fails for want of scratch
// memory. Evidence: benchmarks/baselines/x86_gemm_align_20260927/.
namespace {

constexpr int kStrips = 8;         // 16-float strips per panel row (128 floats)
constexpr int64_t kPanelRowFloats = 16 * kStrips;
constexpr int64_t kKBlock = 512;   // panel rows per K block (256 KiB panel)

inline __mmask16 stripMask(int64_t width) {
    return width >= 16 ? static_cast<__mmask16>(0xffffu)
                       : static_cast<__mmask16>((1u << static_cast<unsigned>(width)) - 1u);
}

// C[:, n0 : n0 + 16*S] += A[:, k0 : k0 + kc] @ src for every row of A, where
// `src` row k (k < kc) starts at src + k*ld. Packed: `src` is the aligned
// panel (tail lanes zero-filled). Direct: `src` is B + k0*N + n0 and masked
// loads zero the lanes past N. The first K block (k0 == 0) starts from zero.
template <int S, bool Packed>
void blockRows(const float* A, const float* src, int64_t ld, int64_t M,
               int64_t N, int64_t K, float* C, int64_t n0, int64_t k0,
               int64_t kc, const __mmask16* mask) {
    for (int64_t m = 0; m < M; ++m) {
        const float* a = A + linearIndex2D<Rank2Order::RowMajor>(m, k0, K);
        float* c = C + linearIndex2D<Rank2Order::RowMajor>(m, n0, N);
        __m512 acc[S];
        for (int j = 0; j < S; ++j)
            acc[j] = k0 == 0 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(mask[j], c + 16 * j);
        for (int64_t k = 0; k < kc; ++k) {
            const __m512 av = _mm512_set1_ps(a[k]);
            const float* row = src + linearIndex2D<Rank2Order::RowMajor>(k, 0, ld);
            for (int j = 0; j < S; ++j)
                acc[j] = _mm512_fmadd_ps(
                    av,
                    Packed ? _mm512_load_ps(row + 16 * j)
                           : _mm512_maskz_loadu_ps(mask[j], row + 16 * j),
                    acc[j]);
        }
        for (int j = 0; j < S; ++j) _mm512_mask_storeu_ps(c + 16 * j, mask[j], acc[j]);
    }
}

// S is a template parameter so acc[] stays in registers (a runtime strip count
// spilled the accumulators and ran ~4x slower at 32^3 in the design sweep).
template <bool Packed>
void dispatchStrips(int S, const float* A, const float* src, int64_t ld,
                    int64_t M, int64_t N, int64_t K, float* C, int64_t n0,
                    int64_t k0, int64_t kc, const __mmask16* mask) {
    switch (S) {
    case 1: blockRows<1, Packed>(A, src, ld, M, N, K, C, n0, k0, kc, mask); break;
    case 2: blockRows<2, Packed>(A, src, ld, M, N, K, C, n0, k0, kc, mask); break;
    case 3: blockRows<3, Packed>(A, src, ld, M, N, K, C, n0, k0, kc, mask); break;
    case 4: blockRows<4, Packed>(A, src, ld, M, N, K, C, n0, k0, kc, mask); break;
    case 5: blockRows<5, Packed>(A, src, ld, M, N, K, C, n0, k0, kc, mask); break;
    case 6: blockRows<6, Packed>(A, src, ld, M, N, K, C, n0, k0, kc, mask); break;
    case 7: blockRows<7, Packed>(A, src, ld, M, N, K, C, n0, k0, kc, mask); break;
    default: blockRows<kStrips, Packed>(A, src, ld, M, N, K, C, n0, k0, kc, mask); break;
    }
}

} // namespace

extern "C" void tessera_x86_avx512_gemm_f32(const float* A, const float* B,
                                            int64_t M, int64_t N, int64_t K,
                                            float* C) {
    if (M <= 0 || N <= 0) return;
    const int64_t kBlock = std::min<int64_t>(kKBlock, K);
    // K <= 0 takes the direct path, which writes C = 0 (the empty sum) as the
    // previous kernel did.
    float* panel = nullptr;
    if (M > 1 && K > 0)
        panel = static_cast<float*>(std::aligned_alloc(
            64, static_cast<size_t>(kBlock) * sizeof(float) * kPanelRowFloats));
    for (int64_t n0 = 0; n0 < N; n0 += kPanelRowFloats) {
        const int S = static_cast<int>(std::min<int64_t>(kStrips, (N - n0 + 15) / 16));
        __mmask16 mask[kStrips];
        for (int j = 0; j < kStrips; ++j)
            mask[j] = j < S ? stripMask(N - n0 - 16 * j) : static_cast<__mmask16>(0);
        if (!panel) {
            dispatchStrips<false>(S, A, B + n0, N, M, N, K, C, n0, 0, K, mask);
            continue;
        }
        const int64_t ld = 16 * S;
        for (int64_t k0 = 0; k0 < K; k0 += kBlock) {
            const int64_t kc = std::min<int64_t>(kBlock, K - k0);
            for (int64_t k = 0; k < kc; ++k) {
                const float* row = B + linearIndex2D<Rank2Order::RowMajor>(k0 + k, n0, N);
                float* dst = panel + linearIndex2D<Rank2Order::RowMajor>(k, 0, ld);
                for (int j = 0; j < S; ++j)
                    _mm512_store_ps(dst + 16 * j, _mm512_maskz_loadu_ps(mask[j], row + 16 * j));
            }
            dispatchStrips<true>(S, A, panel, ld, M, N, K, C, n0, k0, kc, mask);
        }
    }
    std::free(panel);
}

// T1 evidence ABI: the same AVX-512 microkernel with explicit compiler-owned
// M/N/K blocking.  This is intentionally separate from the promoted ABI until
// measured rank correlation shows that the cache model can select among these
// physically distinct loop nests.  BN is rounded to whole AVX-512 vectors;
// ragged N remains mask-safe.
extern "C" int tessera_x86_avx512_gemm_f32_tiled(
    const float* A, const float* B, int64_t M, int64_t N, int64_t K, float* C,
    int64_t BM, int64_t BN, int64_t BK) {
    if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0 || BM <= 0 || BN <= 0 ||
        BK <= 0)
        return 1;
    BN = ((BN + 15) / 16) * 16;
    for (int64_t m0 = 0; m0 < M; m0 += BM) {
        const int64_t mEnd = std::min(m0 + BM, M);
        for (int64_t n0 = 0; n0 < N; n0 += BN) {
            const int64_t nEnd = std::min(n0 + BN, N);
            for (int64_t m = m0; m < mEnd; ++m) {
                const float* a =
                    A + linearIndex2D<Rank2Order::RowMajor>(m, 0, K);
                float* c =
                    C + linearIndex2D<Rank2Order::RowMajor>(m, 0, N);
                int64_t n = n0;
                for (; n + 16 <= nEnd; n += 16) {
                    __m512 acc = _mm512_setzero_ps();
                    for (int64_t k0 = 0; k0 < K; k0 += BK) {
                        const int64_t kEnd = std::min(k0 + BK, K);
                        for (int64_t k = k0; k < kEnd; ++k)
                            acc = _mm512_fmadd_ps(
                                _mm512_set1_ps(a[k]),
                                _mm512_loadu_ps(
                                    B + linearIndex2D<Rank2Order::RowMajor>(k, n, N)),
                                acc);
                    }
                    _mm512_storeu_ps(c + n, acc);
                }
                if (n < nEnd) {
                    const unsigned width = static_cast<unsigned>(nEnd - n);
                    const __mmask16 tail =
                        width == 16 ? static_cast<__mmask16>(0xffffu)
                                    : static_cast<__mmask16>((1u << width) - 1u);
                    __m512 acc = _mm512_setzero_ps();
                    for (int64_t k0 = 0; k0 < K; k0 += BK) {
                        const int64_t kEnd = std::min(k0 + BK, K);
                        for (int64_t k = k0; k < kEnd; ++k)
                            acc = _mm512_fmadd_ps(
                                _mm512_set1_ps(a[k]),
                                _mm512_maskz_loadu_ps(
                                    tail,
                                    B + linearIndex2D<Rank2Order::RowMajor>(k, n, N)),
                                acc);
                    }
                    _mm512_mask_storeu_ps(c + n, tail, acc);
                }
            }
        }
    }
    return 0;
}
