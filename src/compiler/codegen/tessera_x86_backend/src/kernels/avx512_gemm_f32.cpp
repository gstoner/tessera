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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
// alignment. The blocked loop needs C disjoint from A and B; the entry point
// enforces that (an overlapping C is computed through scratch, see below).
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

// Measurement hooks: benchmarks/baselines/x86_gemm_align_20260927/run_path_crossover.sh
// rewrites these two lines to build forced-path variants. Production leaves them false.
constexpr bool kForcePath = false;     // PATH-PROBE
constexpr bool kForcedPacked = false;  // PATH-PROBE

// Whether packing B pays for itself for this shape. PROVISIONAL (M > 1).
bool packedPathWins(int64_t M, int64_t N, int64_t K) {
    (void)N;
    if (kForcePath) return kForcedPacked;
    return M > 1 && K > 0;
}

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
        // Every j loop is fully unrolled so acc[] lives in S zmm registers. At
        // -O2 GCC 15 left the S >= 4 loops rolled and kept acc[] on the stack
        // (a load + FMA + store per step), which made N = 64 slower than the
        // pre-fix kernel; see the evidence README.
        __m512 acc[S];
#pragma GCC unroll 16
        for (int j = 0; j < S; ++j)
            acc[j] = k0 == 0 ? _mm512_setzero_ps() : _mm512_maskz_loadu_ps(mask[j], c + 16 * j);
        for (int64_t k = 0; k < kc; ++k) {
            const __m512 av = _mm512_set1_ps(a[k]);
            const float* row = src + linearIndex2D<Rank2Order::RowMajor>(k, 0, ld);
#pragma GCC unroll 16
            for (int j = 0; j < S; ++j)
                acc[j] = _mm512_fmadd_ps(
                    av,
                    Packed ? _mm512_load_ps(row + 16 * j)
                           : _mm512_maskz_loadu_ps(mask[j], row + 16 * j),
                    acc[j]);
        }
#pragma GCC unroll 16
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

// The blocked GEMM. Requires C not to overlap A or B (see the entry point).
void gemmNoAlias(const float* A, const float* B, int64_t M, int64_t N,
                 int64_t K, float* C) {
    const int64_t kBlock = std::min<int64_t>(kKBlock, K);
    // K <= 0 takes the direct path, which writes C = 0 (the empty sum) as the
    // previous kernel did.
    float* panel = nullptr;
    if (K > 0 && packedPathWins(M, N, K))
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

// Byte extent of a dense row-major rows x cols f32 operand; UINT64_MAX when
// it does not fit in 64 bits (then it overlaps everything: the safe answer).
uint64_t operandBytes(int64_t rows, int64_t cols) {
    if (rows <= 0 || cols <= 0) return 0;
    const unsigned __int128 bytes = static_cast<unsigned __int128>(rows) *
                                    static_cast<unsigned __int128>(cols) * sizeof(float);
    return bytes > std::numeric_limits<uint64_t>::max()
               ? std::numeric_limits<uint64_t>::max()
               : static_cast<uint64_t>(bytes);
}

// Whether half-open byte ranges [p, p + pBytes) and [q, q + qBytes) intersect.
bool rangesOverlap(const void* p, uint64_t pBytes, const void* q, uint64_t qBytes) {
    if (pBytes == 0 || qBytes == 0) return false;
    const uint64_t pa = reinterpret_cast<uintptr_t>(p);
    const uint64_t qa = reinterpret_cast<uintptr_t>(q);
    const uint64_t kMax = std::numeric_limits<uint64_t>::max();
    const uint64_t pEnd = pBytes > kMax - pa ? kMax : pa + pBytes;
    const uint64_t qEnd = qBytes > kMax - qa ? kMax : qa + qBytes;
    return pa < qEnd && qa < pEnd;
}

} // namespace

// Whether C's bytes overlap A's or B's for C[M,N] = A[M,K] @ B[K,N] in the
// ABI's only layout (dense, row-major). The GEMM entry point consumes it; it
// is exported so the rule is testable on its own.
extern "C" int tessera_x86_avx512_gemm_f32_operands_overlap(
    const float* A, const float* B, int64_t M, int64_t N, int64_t K,
    const float* C) {
    const uint64_t cBytes = operandBytes(M, N);
    return rangesOverlap(C, cBytes, A, operandBytes(M, K)) ||
           rangesOverlap(C, cBytes, B, operandBytes(K, N));
}

// C = A @ B. The blocked loop writes C while A and B are still being read and
// reads C back between K blocks, so an overlapping C (an in-place C = A @ B,
// or any view overlap) would return a wrong result. The entry point therefore
// checks overlap and, when found, computes the product of the inputs' values
// at entry into an aligned scratch C and copies it out: the result equals the
// non-aliased product bit for bit. (The pre-2026-09-27 kernel was wrong for
// almost every overlap too -- it was right only by accident, e.g. C == A with
// N == K <= 16 -- so computing through scratch is strictly more correct than
// anything a caller could have relied on.) If the scratch cannot be allocated
// the call fails closed: C is filled with quiet NaN and the reason goes to
// stderr, never a silently wrong product.
extern "C" void tessera_x86_avx512_gemm_f32(const float* A, const float* B,
                                            int64_t M, int64_t N, int64_t K,
                                            float* C) {
    if (M <= 0 || N <= 0) return;
    if (!tessera_x86_avx512_gemm_f32_operands_overlap(A, B, M, N, K, C)) {
        gemmNoAlias(A, B, M, N, K, C);
        return;
    }
    const uint64_t bytes = operandBytes(M, N);
    float* scratch = nullptr;
    if (bytes <= std::numeric_limits<size_t>::max() - 63)
        scratch = static_cast<float*>(std::aligned_alloc(
            64, static_cast<size_t>((bytes + 63) / 64 * 64)));
    if (!scratch) {
        std::fprintf(stderr,
                     "tessera_x86_avx512_gemm_f32: C overlaps A or B and the %llu-byte "
                     "scratch for a non-aliased product could not be allocated; C is "
                     "filled with NaN\n",
                     static_cast<unsigned long long>(bytes));
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (int64_t i = 0; i < M * N; ++i) C[i] = nan;
        return;
    }
    gemmNoAlias(A, B, M, N, K, scratch);
    std::memcpy(C, scratch, static_cast<size_t>(bytes));
    std::free(scratch);
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
