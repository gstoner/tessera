// EBM exact-partition kernel (f32) for the Tessera x86 backend — the EBM3
// partition-function lane. Computes the stable log-sum-exp partition value
//
//   Z = Σ_i exp(-E_i / T)
//
// over per-state energies E (n elements) at temperature T > 0, via the numerically
// stable form  Z = exp(m) · Σ_i exp(-E_i/T - m),  m = max_i(-E_i/T).  The reduction
// accumulates in double so the result agrees with the numpy reference
// (tessera.ebm.partition_exact_from_energies) to f32 epsilon. ROCm analog:
// generate-rocm-ebm-partition-kernel. Apple analog:
// tessera_apple_gpu_ebm_partition_exact_f32.

#include <cstdint>
#include <cmath>

extern "C" void tessera_x86_ebm_partition_exact_f32(
        const float* E, int64_t n, float temperature, float* out) {
    if (n <= 0) { out[0] = 0.0f; return; }
    const double inv_t = 1.0 / static_cast<double>(temperature);
    // Pass 1 — max of the negative scaled energies (the log-sum-exp shift).
    double max_neg = -INFINITY;
    for (int64_t i = 0; i < n; ++i) {
        double neg = -static_cast<double>(E[i]) * inv_t;
        if (neg > max_neg) max_neg = neg;
    }
    // Pass 2 — shifted exponential sum.
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += std::exp(-static_cast<double>(E[i]) * inv_t - max_neg);
    }
    out[0] = static_cast<float>(std::exp(max_neg + std::log(s)));
}
