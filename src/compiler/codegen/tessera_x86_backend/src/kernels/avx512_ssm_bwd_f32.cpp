// AVX-512 backward pass for the Mamba2 selective_ssm scan (f32).
//
// Reverse-mode adjoint of avx512_ssm_f32's forward scan, matching the numpy VJP
// (autodiff/vjp.py::vjp_selective_ssm) exactly. Forward (per batch b, dropping b):
//   A_bar[t,d,n] = exp(delta[t,d]·A[d,n]) ; B_bar[t,d,n] = delta[t,d]·B[t,n]
//   h[t,d,n]     = A_bar·h[t-1,d,n] + B_bar·x[t,d]
//   y[t,d]       = Σ_n C[t,n]·h[t,d,n]
//
// The kernel first fills the forward trajectory h_traj (h_traj[0] = the caller's
// initial state, h_traj[t+1] = h after step t) into a caller-provided scratch,
// then walks t = S-1 → 0 accumulating (dx, dA2d, dB, dC, ddelta). The scan state
// dh_curr[n] carries across t for a fixed (b,d), so the loop nest is b → d → t;
// the n-loop is the SIMD dimension. dx/ddelta are unique per (b,t,d) (written
// once); dC/dB accumulate over d and dA2d over (b,t) — all races-free because the
// scan is single-threaded (the GPU lane uses atomics instead). Caller zeros the
// five outputs and applies the gate to dy = dout·gate before the call.

#include <immintrin.h>
#include <cmath>
#include <cstdint>

namespace {
inline __m512 exp512(__m512 x) {
    const __m512 one = _mm512_set1_ps(1.0f);
    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));
    __m512 fx = _mm512_fmadd_ps(x, _mm512_set1_ps(1.44269504088896341f),
                                _mm512_set1_ps(0.5f));
    fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 z = _mm512_mul_ps(r, r);
    __m512 y = _mm512_set1_ps(1.9875691500E-4f);
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.3981999507E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(8.3334519073E-3f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(4.1665795894E-2f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(1.6666665459E-1f));
    y = _mm512_fmadd_ps(y, r, _mm512_set1_ps(5.0000001201E-1f));
    y = _mm512_fmadd_ps(y, z, _mm512_add_ps(r, one));
    return _mm512_scalef_ps(y, fx);
}
}  // namespace

// h_traj: scratch [(S+1), B, D, N]; h_traj[0] pre-set to the initial state (or
// zeros). dx/dA2d/dB/dC/ddelta: outputs, pre-zeroed by the caller. dy = dout·gate.
extern "C" void tessera_x86_selective_ssm_bwd_f32(
    const float* x, const float* A2d, const float* B, const float* C,
    const float* delta, const float* dy, int64_t Bsz, int64_t S, int64_t D,
    int64_t N, float* h_traj, float* dx, float* dA2d, float* dB, float* dC,
    float* ddelta) {
    const int64_t BDN = Bsz * D * N;

    // ── forward: fill h_traj[t+1] = A_bar·h_traj[t] + B_bar·x ──
    for (int64_t b = 0; b < Bsz; ++b)
        for (int64_t t = 0; t < S; ++t)
            for (int64_t d = 0; d < D; ++d) {
                float dt = delta[(b * S + t) * D + d];
                float xt = x[(b * S + t) * D + d];
                __m512 vdt = _mm512_set1_ps(dt), vxt = _mm512_set1_ps(xt);
                const float* arow = A2d + d * N;
                const float* brow = B + (b * S + t) * N;
                const float* hprev = h_traj + t * BDN + (b * D + d) * N;
                float* hnext = h_traj + (t + 1) * BDN + (b * D + d) * N;
                int64_t n = 0;
                for (; n + 16 <= N; n += 16) {
                    __m512 ab = exp512(_mm512_mul_ps(vdt, _mm512_loadu_ps(arow + n)));
                    __m512 bb = _mm512_mul_ps(vdt, _mm512_loadu_ps(brow + n));
                    _mm512_storeu_ps(hnext + n,
                        _mm512_fmadd_ps(ab, _mm512_loadu_ps(hprev + n),
                                        _mm512_mul_ps(bb, vxt)));
                }
                for (; n < N; ++n)
                    hnext[n] = std::exp(dt * arow[n]) * hprev[n]
                               + dt * brow[n] * xt;
            }

    // ── reverse: t = S-1 → 0, dh_curr[n] carried across t per (b,d) ──
    for (int64_t b = 0; b < Bsz; ++b)
        for (int64_t d = 0; d < D; ++d) {
            // dh_curr in a small stack/heap buffer (N lanes); reuse across t.
            // Kept on the stack via VLA-free fixed loop with a scratch alloca is
            // avoided — use the h_traj tail? Simpler: a heap-free approach with a
            // register-blocked n-loop needs dh across t, so alloc once per (b,d).
            float dh_stack[1024];
            float* dh = (N <= 1024) ? dh_stack : new float[N];
            for (int64_t n = 0; n < N; ++n) dh[n] = 0.0f;
            for (int64_t t = S - 1; t >= 0; --t) {
                float dt = delta[(b * S + t) * D + d];
                float xt = x[(b * S + t) * D + d];
                float dyt = dy[(b * S + t) * D + d];
                const float* arow = A2d + d * N;
                const float* brow = B + (b * S + t) * N;
                const float* crow = C + (b * S + t) * N;
                const float* hcur = h_traj + t * BDN + (b * D + d) * N;       // h[t-1]
                const float* hnxt = h_traj + (t + 1) * BDN + (b * D + d) * N; // h[t]
                float* dArow = dA2d + d * N;
                float* dBrow = dB + (b * S + t) * N;
                float* dCrow = dC + (b * S + t) * N;
                __m512 vdt = _mm512_set1_ps(dt), vxt = _mm512_set1_ps(xt);
                __m512 vdyt = _mm512_set1_ps(dyt);
                __m512 vdx = _mm512_setzero_ps(), vddl = _mm512_setzero_ps();
                int64_t n = 0;
                for (; n + 16 <= N; n += 16) {
                    __m512 a = _mm512_loadu_ps(arow + n);
                    __m512 bv = _mm512_loadu_ps(brow + n);
                    __m512 cv = _mm512_loadu_ps(crow + n);
                    __m512 abar = exp512(_mm512_mul_ps(vdt, a));
                    __m512 bbar = _mm512_mul_ps(vdt, bv);
                    __m512 dhc = _mm512_fmadd_ps(cv, vdyt, _mm512_loadu_ps(dh + n));
                    // dC += h[t]·dy   (accumulate over d)
                    _mm512_storeu_ps(dCrow + n,
                        _mm512_fmadd_ps(_mm512_loadu_ps(hnxt + n), vdyt,
                                        _mm512_loadu_ps(dCrow + n)));
                    __m512 dAbar = _mm512_mul_ps(dhc, _mm512_loadu_ps(hcur + n));
                    __m512 dhprev = _mm512_mul_ps(dhc, abar);
                    __m512 dBbar = _mm512_mul_ps(dhc, vxt);
                    vdx = _mm512_fmadd_ps(dhc, bbar, vdx);
                    // dB += dBbar·delta   (accumulate over d)
                    _mm512_storeu_ps(dBrow + n,
                        _mm512_fmadd_ps(dBbar, vdt, _mm512_loadu_ps(dBrow + n)));
                    vddl = _mm512_fmadd_ps(dBbar, bv, vddl);
                    __m512 dz = _mm512_mul_ps(dAbar, abar);
                    // dA2d += delta·dz   (accumulate over b,t)
                    _mm512_storeu_ps(dArow + n,
                        _mm512_fmadd_ps(vdt, dz, _mm512_loadu_ps(dArow + n)));
                    vddl = _mm512_fmadd_ps(dz, a, vddl);
                    _mm512_storeu_ps(dh + n, dhprev);
                }
                float dxacc = _mm512_reduce_add_ps(vdx);
                float ddlacc = _mm512_reduce_add_ps(vddl);
                for (; n < N; ++n) {
                    float abar = std::exp(dt * arow[n]);
                    float bbar = dt * brow[n];
                    float dhc = dh[n] + crow[n] * dyt;
                    dCrow[n] += hnxt[n] * dyt;
                    float dAbar = dhc * hcur[n];
                    float dBbar = dhc * xt;
                    dxacc += dhc * bbar;
                    dBrow[n] += dBbar * dt;
                    ddlacc += dBbar * brow[n];
                    float dz = dAbar * abar;
                    dArow[n] += dt * dz;
                    ddlacc += dz * arow[n];
                    dh[n] = dhc * abar;
                }
                dx[(b * S + t) * D + d] = dxacc;
                ddelta[(b * S + t) * D + d] = ddlacc;
            }
            if (N > 1024) delete[] dh;
        }
}
