// AVX-512 mixture-of-experts compute kernel (f32) — the optimized CPU lane for
// the S-series `moe` family's COMPUTE part (the routed expert matmuls).  Each
// token does a GEMV with its routed expert's weight (top-1):
//
//   out[t, :] = x[t, :] · experts[route[t]]        experts[e] is [in_dim, out_dim]
//
// Routing (argmax of scores / round-robin) is resolved on the host; this kernel
// runs the FLOP-heavy per-token expert matmuls.  The out_dim axis is the SIMD
// dimension (contiguous AXPY).  A scalar reference is provided alongside.

#include <immintrin.h>
#include <cstdint>

namespace {
// acc[:len] += f * row[:len]
inline void axpy(float f, const float* row, float* acc, int64_t len) {
    __m512 vf = _mm512_set1_ps(f);
    int64_t c = 0;
    for (; c + 16 <= len; c += 16)
        _mm512_storeu_ps(acc + c, _mm512_fmadd_ps(vf, _mm512_loadu_ps(row + c),
                                                  _mm512_loadu_ps(acc + c)));
    for (; c < len; ++c) acc[c] += f * row[c];
}
}  // namespace

extern "C" void tessera_x86_reference_moe_f32(
    const float* x, const float* experts, const int32_t* route, int64_t tokens,
    int64_t in_dim, int64_t out_dim, float* out) {
    for (int64_t t = 0; t < tokens; ++t) {
        const float* xt = x + t * in_dim;
        const float* w = experts + (int64_t)route[t] * in_dim * out_dim;
        float* o = out + t * out_dim;
        for (int64_t j = 0; j < out_dim; ++j) o[j] = 0.0f;
        for (int64_t i = 0; i < in_dim; ++i)
            for (int64_t j = 0; j < out_dim; ++j) o[j] += xt[i] * w[i * out_dim + j];
    }
}

extern "C" void tessera_x86_moe_f32(
    const float* x, const float* experts, const int32_t* route, int64_t tokens,
    int64_t in_dim, int64_t out_dim, float* out) {
    for (int64_t t = 0; t < tokens; ++t) {
        const float* xt = x + t * in_dim;
        const float* w = experts + (int64_t)route[t] * in_dim * out_dim;
        float* o = out + t * out_dim;
        int64_t j = 0;
        for (; j + 16 <= out_dim; j += 16) _mm512_storeu_ps(o + j, _mm512_setzero_ps());
        for (; j < out_dim; ++j) o[j] = 0.0f;
        for (int64_t i = 0; i < in_dim; ++i)
            axpy(xt[i], w + i * out_dim, o, out_dim);
    }
}
