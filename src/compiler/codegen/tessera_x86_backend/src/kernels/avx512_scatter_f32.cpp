// Scatter kernel (f32) for the Tessera x86 backend — the 0-reduce / indexed-
// store lane (P8 of S_SERIES_GAP_CLOSURE_PLAN). The companion to the P4 gather
// kernel: where gather is an indexed LOAD (out[i] = src[idx[i]]), scatter is an
// indexed STORE into a destination indexed by `idx`, reducing duplicate targets:
//
//   mode 0 (set):  out[idx[i]]  = src[i]
//   mode 1 (add):  out[idx[i]] += src[i]            (scatter_add)
//   mode 2 (min):  out[idx[i]]  = min(out, src[i])  (scatter_reduce "amin")
//   mode 3 (max):  out[idx[i]]  = max(out, src[i])  (scatter_reduce "amax")
//
// Row-wise: each index addresses a contiguous `row_len`-element row, so the
// reduction over a row vectorizes (AVX-512 over row_len + scalar tail). The
// host moves the scatter axis to 0 and flattens the trailing dims into row_len;
// `out` is preloaded with the base tensor (untouched rows keep their value).
// On the single-thread CPU the scatter is sequential, so the accumulation needs
// no atomics — the ROCm lane (one thread per element) uses atomic_rmw instead.
// f32. Matches numpy scatter / np.add.at / np.minimum.at / np.maximum.at.

#include <cstdint>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace {

inline void row_set(float* dst, const float* s, int64_t n) {
    int64_t c = 0;
#if defined(__AVX512F__)
    for (; c + 16 <= n; c += 16)
        _mm512_storeu_ps(dst + c, _mm512_loadu_ps(s + c));
#endif
    for (; c < n; ++c) dst[c] = s[c];
}

inline void row_add(float* dst, const float* s, int64_t n) {
    int64_t c = 0;
#if defined(__AVX512F__)
    for (; c + 16 <= n; c += 16)
        _mm512_storeu_ps(dst + c, _mm512_add_ps(_mm512_loadu_ps(dst + c),
                                                _mm512_loadu_ps(s + c)));
#endif
    for (; c < n; ++c) dst[c] += s[c];
}

inline void row_min(float* dst, const float* s, int64_t n) {
    int64_t c = 0;
#if defined(__AVX512F__)
    for (; c + 16 <= n; c += 16)
        _mm512_storeu_ps(dst + c, _mm512_min_ps(_mm512_loadu_ps(dst + c),
                                                _mm512_loadu_ps(s + c)));
#endif
    for (; c < n; ++c) dst[c] = s[c] < dst[c] ? s[c] : dst[c];
}

inline void row_max(float* dst, const float* s, int64_t n) {
    int64_t c = 0;
#if defined(__AVX512F__)
    for (; c + 16 <= n; c += 16)
        _mm512_storeu_ps(dst + c, _mm512_max_ps(_mm512_loadu_ps(dst + c),
                                                _mm512_loadu_ps(s + c)));
#endif
    for (; c < n; ++c) dst[c] = s[c] > dst[c] ? s[c] : dst[c];
}

} // namespace

// out: [out_rows, row_len] (preloaded with the base tensor).
// src: [n_idx, row_len].   idx: [n_idx] row targets into out.
extern "C" void tessera_x86_scatter_f32(float* out, int64_t out_rows,
                                        const float* src, const int64_t* idx,
                                        int64_t n_idx, int64_t row_len,
                                        int mode) {
    for (int64_t i = 0; i < n_idx; ++i) {
        const int64_t r = idx[i];
        if (r < 0 || r >= out_rows) continue;   // out-of-range index is skipped
        float* dst = out + r * row_len;
        const float* s = src + i * row_len;
        switch (mode) {
        case 1: row_add(dst, s, row_len); break;
        case 2: row_min(dst, s, row_len); break;
        case 3: row_max(dst, s, row_len); break;
        default: row_set(dst, s, row_len); break;
        }
    }
}
