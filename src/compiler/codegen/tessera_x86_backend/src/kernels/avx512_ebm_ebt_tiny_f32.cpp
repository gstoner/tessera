// EBM EBT-tiny fused pipeline (f32) for the Tessera x86 backend — the
// energy-based-transformer inference step. For B batches of K candidate
// trajectories (y0, grad row-major [B*K, D]) it fuses, per batch b:
//
//   refinement:  y_T[b,k,:] = y0[b,k,:] - (T*eta) * grad[b,k,:]   (closed form)
//   energy:      e[b,k]     = Σ_d y_T[b,k,d]^2                    (squared norm)
//   argmin:      k*         = argmin_k e[b,k]                     (first-min)
//   gather:      out[b,:]   = y_T[b,k*,:]
//
// (the T inner refinement steps with a fixed gradient collapse to the single
// affine y0 - T*eta*grad). Output is [B, D] best candidates. This is
// `tessera.ebm.ebt_tiny`. Energies accumulate in double so the argmin agrees
// with the numpy reference; ties break toward the lower candidate index (strict
// `<`), matching numpy argmin. ROCm analog: generate-rocm-ebm-ebt-tiny-kernel.
// Apple analog: tessera_apple_gpu_ebm_ebt_tiny_* (single Metal dispatch).

#include <cstdint>

extern "C" void tessera_x86_ebm_ebt_tiny_f32(
        const float* y0, const float* grad, float eta, int32_t T,
        int64_t B, int64_t K, int64_t D, float* out) {
    const double scale = static_cast<double>(eta) * static_cast<double>(T);
    for (int64_t b = 0; b < B; ++b) {
        int64_t kstar = 0;
        double best = 1.0e308;
        for (int64_t k = 0; k < K; ++k) {
            const int64_t off = (b * K + k) * D;
            double e = 0.0;
            for (int64_t d = 0; d < D; ++d) {
                double v = static_cast<double>(y0[off + d])
                           - scale * static_cast<double>(grad[off + d]);
                e += v * v;
            }
            if (e < best) { best = e; kstar = k; }
        }
        const int64_t so = (b * K + kstar) * D;
        for (int64_t d = 0; d < D; ++d) {
            out[b * D + d] = static_cast<float>(
                static_cast<double>(y0[so + d])
                - scale * static_cast<double>(grad[so + d]));
        }
    }
}
