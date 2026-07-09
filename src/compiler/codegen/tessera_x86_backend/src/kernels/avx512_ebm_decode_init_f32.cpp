// EBM decode-init noise-apply kernel (f32) for the Tessera x86 backend — the
// initialization half of the DFlash / EBM speculative-decode lane. When
// `tessera.ebm.decode_init(init_strategy="noise", mean=…)` seeds K candidate
// trajectories, it draws unit-variance Gaussian noise on the host and combines
// it with a per-element base (mean) offset:
//
//   out[i] = base[i] + std * noise[i]
//
// One flat pass over the (B·K·event) elements. `base` and `noise` are already
// broadcast to the full trajectory shape by the caller, so this is a pure
// elementwise affine combine — no reduction, no RNG (the host draws the noise so
// the fast path and the numpy reference share identical Gaussian samples). The
// combine accumulates in double so the result agrees with the numpy reference
// (mean_arr + std*noise) to f32 epsilon. ROCm analog:
// generate-rocm-ebm-decode-init-kernel. Apple analog: ebm_decode_init bridge
// (_try_apple_gpu_decode_init_noise_apply_f32).

#include <cstdint>

extern "C" void tessera_x86_ebm_decode_init_noise_apply_f32(
        const float* base, const float* noise, int64_t n, float std,
        float* out) {
    const double s = static_cast<double>(std);
    for (int64_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(static_cast<double>(base[i])
                                    + s * static_cast<double>(noise[i]));
    }
}
