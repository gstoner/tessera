// EBM quadratic-energy kernel (f32) for the Tessera x86 backend — the dominant
// EBT / diffusion energy form. Computes the per-row quadratic energy
//
//   out[b] = 0.5 * Σ_d (x[b,d] - y[b,d])^2
//
// over B rows of D elements each (x, y row-major [B, D]; higher-rank callers
// flatten the trailing dims into D). This is `tessera.ebm.energy_quadratic`
// (½‖x−y‖² per row — reconstruction loss / Gaussian log-likelihood up to a
// constant), the concrete energy that `ebm_energy` / `ebm_energy_quadratic`
// share. A dedicated fused row reduction (diff → square → sum in one pass),
// accumulated in double so the result agrees with the numpy reference to f32
// epsilon. ROCm analog: generate-rocm-ebm-energy-quadratic-kernel. Apple analog:
// tessera_apple_gpu_ebm_energy_quadratic_f32.

#include <cstdint>

extern "C" void tessera_x86_ebm_energy_quadratic_f32(
        const float* x, const float* y, int64_t B, int64_t D, float* out) {
    for (int64_t b = 0; b < B; ++b) {
        const float* xr = x + b * D;
        const float* yr = y + b * D;
        double s = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            double diff = static_cast<double>(xr[d]) - static_cast<double>(yr[d]);
            s += diff * diff;
        }
        out[b] = static_cast<float>(0.5 * s);
    }
}
