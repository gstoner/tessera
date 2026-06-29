// Bitonic sort kernel (f32 key + i64 index) for the Tessera x86 backend — the
// sort lane (P9 of S_SERIES_GAP_CLOSURE_PLAN) backing sort / argsort / top_k.
//
// A bitonic sorting network is data-independent (the compare-exchange schedule
// depends only on n, never on the data), which is exactly why it is the
// GPU-friendly choice the plan calls for and why the same schedule runs on the
// ROCm in-LDS kernel. Here it sorts ONE row of length `n` (which the host has
// padded to a power of two with +INF sentinels so the real elements land in the
// first L slots) ascending, carrying the parallel index array so the same swaps
// realize argsort / top_k. `descending` is handled host-side by flipping the
// ascending result (matching the numpy reference's flip semantics), so this
// kernel only ever sorts ascending — the flag is kept for ABI symmetry.
//
// Vectorization: for the wide stages (stride j >= 16) a whole 16-lane block
// [b, b+16) lies entirely within one monotone segment (k >= 2j >= 32, so the
// direction bit i&k is constant across the block) and its partner block is
// b+j; those compare-exchanges run as AVX-512 min/max + masked index blend.
// The narrow stages (j < 16) and any n < 16 fall to the scalar network tail.

#include <cstdint>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {

inline void cexg(float* k, int64_t* x, int64_t i, int64_t j, bool up) {
    // Scalar compare-exchange of lanes i and i^… (caller passes the partner as
    // `j` already = i^stride). `up` ascending: keep min at i, max at the partner.
    const bool swap = up ? (k[i] > k[j]) : (k[i] < k[j]);
    if (swap) {
        const float tk = k[i]; k[i] = k[j]; k[j] = tk;
        const int64_t tx = x[i]; x[i] = x[j]; x[j] = tx;
    }
}

} // namespace

extern "C" void tessera_x86_bitonic_sort_kv_f32(float* keys, int64_t* idx,
                                                int64_t n, int /*descending*/) {
    for (int64_t k = 2; k <= n; k <<= 1) {
        for (int64_t j = k >> 1; j > 0; j >>= 1) {
#if defined(__AVX512F__)
            if (j >= 16) {
                // Wide stage: 16-lane vector compare-exchange between aligned
                // block b and its partner b+j (only the lower side, b&j == 0).
                for (int64_t b = 0; b < n; b += 16) {
                    if (b & j) continue;
                    const int64_t pb = b + j;
                    const bool up = ((b & k) == 0);
                    __m512 a = _mm512_loadu_ps(keys + b);
                    __m512 c = _mm512_loadu_ps(keys + pb);
                    const __mmask16 sw =
                        up ? _mm512_cmp_ps_mask(a, c, _CMP_GT_OQ)
                           : _mm512_cmp_ps_mask(a, c, _CMP_LT_OQ);
                    _mm512_storeu_ps(keys + b, _mm512_mask_blend_ps(sw, a, c));
                    _mm512_storeu_ps(keys + pb, _mm512_mask_blend_ps(sw, c, a));
                    // Indices follow the same mask, 8 lanes (i64) at a time.
                    __m512i a0 = _mm512_loadu_si512(idx + b);
                    __m512i c0 = _mm512_loadu_si512(idx + pb);
                    __m512i a1 = _mm512_loadu_si512(idx + b + 8);
                    __m512i c1 = _mm512_loadu_si512(idx + pb + 8);
                    const __mmask8 s0 = (__mmask8)(sw & 0xFF);
                    const __mmask8 s1 = (__mmask8)(sw >> 8);
                    _mm512_storeu_si512(idx + b,
                                        _mm512_mask_blend_epi64(s0, a0, c0));
                    _mm512_storeu_si512(idx + pb,
                                        _mm512_mask_blend_epi64(s0, c0, a0));
                    _mm512_storeu_si512(idx + b + 8,
                                        _mm512_mask_blend_epi64(s1, a1, c1));
                    _mm512_storeu_si512(idx + pb + 8,
                                        _mm512_mask_blend_epi64(s1, c1, a1));
                }
                continue;
            }
#endif
            for (int64_t i = 0; i < n; ++i) {
                const int64_t l = i ^ j;
                if (l > i)
                    cexg(keys, idx, i, l, (i & k) == 0);
            }
        }
    }
}
