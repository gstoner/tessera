// EBM Langevin step with on-device Philox noise (f32) for the Tessera x86
// backend — the SAMPLING half of the P7 EBM follow-up. One Langevin step that
// DRAWS its own Gaussian noise from counter-based Philox-4x32-10 (the same
// Salmon-et-al-2011 generator as the P6 device RNG), so the noise never has to
// be supplied from the host:
//
//   z[i]   = sqrt(-2 ln u0) * cos(2π u1)               (Box-Muller, first lobe)
//   out[i] = y[i] - eta * grad[i] + noise_scale * z[i]
//
// where (u0, u1) are the first two Philox outputs for the per-element counter
// (counter0 + i, counter1, counter2, counter3) under key (k0, k1), mapped to
// (0,1) by `(x + 0.5) * 2^-32`. This mirrors `tessera.ebm.langevin_step_philox`
// byte-for-byte: the counter layout, the `+0.5` uniform map, and the float64
// Box-Muller all match the numpy reference (which the Apple-GPU MSL kernel also
// matches). The transcendentals run in double so the result agrees with the
// reference to f32 epsilon. CPU analog of generate-rocm-ebm-langevin-kernel.

#include <cstdint>
#include <cmath>

namespace {
constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;
constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;

inline void philox4x32_10(uint32_t c[4], uint32_t k0, uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        if (r > 0) { k0 += PHILOX_W0; k1 += PHILOX_W1; }
        uint64_t p0 = static_cast<uint64_t>(PHILOX_M0) * c[0];
        uint64_t p1 = static_cast<uint64_t>(PHILOX_M1) * c[2];
        uint32_t hi0 = static_cast<uint32_t>(p0 >> 32), lo0 = static_cast<uint32_t>(p0);
        uint32_t hi1 = static_cast<uint32_t>(p1 >> 32), lo1 = static_cast<uint32_t>(p1);
        uint32_t n0 = hi1 ^ c[1] ^ k0;
        uint32_t n2 = hi0 ^ c[3] ^ k1;
        c[0] = n0; c[1] = lo1; c[2] = n2; c[3] = lo0;
    }
}
}  // namespace

extern "C" void tessera_x86_ebm_langevin_philox_f32(
        const float* y, const float* grad, int64_t n, float eta,
        float noise_scale, uint32_t k0, uint32_t k1,
        uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, float* out) {
    const double kInv = 1.0 / 4294967296.0;     // 2^-32
    const double two_pi = 2.0 * 3.14159265358979323846;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t c[4] = {c0 + static_cast<uint32_t>(i), c1, c2, c3};
        philox4x32_10(c, k0, k1);
        double u0 = (static_cast<double>(c[0]) + 0.5) * kInv;
        double u1 = (static_cast<double>(c[1]) + 0.5) * kInv;
        double r = std::sqrt(-2.0 * std::log(u0));
        double z = r * std::cos(two_pi * u1);
        out[i] = static_cast<float>(static_cast<double>(y[i])
                                    - static_cast<double>(eta) * grad[i]
                                    + static_cast<double>(noise_scale) * z);
    }
}
