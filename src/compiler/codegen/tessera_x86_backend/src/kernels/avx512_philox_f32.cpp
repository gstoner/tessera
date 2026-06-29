// Counter-based Philox-4x32-10 RNG for the Tessera x86 backend (P6 of
// S_SERIES_GAP_CLOSURE_PLAN).
//
// Device RNG can't bit-match numpy's Generator (numpy uses Philox-4x64 + its own
// Lemire/ziggurat transforms), so Tessera's device lane uses the STANDARD
// counter-based Philox-4x32-10 (Salmon et al. 2011; the same algorithm JAX and
// cuRAND use). It is embarrassingly parallel — output element i depends only on
// (key, counter=i), never on its neighbors — and reproducible: the numpy
// reference `_philox4x32_ref` in tessera.rng_device runs the identical rounds,
// so the device output is validated BIT-EXACTLY (uniform) against it.
//
// Element layout: counter block b = i/4 packs (b, 0, 0, 0); the 4 round outputs
// fill elements [4b, 4b+1, 4b+2, 4b+3]. key = (seed_lo, seed_hi). uniform =
// out_u32 * 2^-32 ∈ [0, 1).

#include <cstdint>

namespace {
constexpr uint32_t PHILOX_M0 = 0xD2511F53u;
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57u;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9u;  // golden ratio
constexpr uint32_t PHILOX_W1 = 0xBB67AE85u;  // sqrt(3) - 1

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

// Fill `out[0..n)` with uniform f32 in [0, 1) from Philox-4x32-10 keyed by
// `seed` (split into two 32-bit halves). `counter_base` offsets the counter
// block so independent draws (split streams) don't overlap.
extern "C" void tessera_x86_philox_uniform_f32(uint64_t seed,
                                               uint64_t counter_base,
                                               int64_t n, float* out) {
    const uint32_t k0 = static_cast<uint32_t>(seed);
    const uint32_t k1 = static_cast<uint32_t>(seed >> 32);
    const float kInv = 1.0f / 4294967296.0f;  // 2^-32
    int64_t nblocks = (n + 3) / 4;
    for (int64_t b = 0; b < nblocks; ++b) {
        uint64_t blk = counter_base + static_cast<uint64_t>(b);
        uint32_t c[4] = {static_cast<uint32_t>(blk),
                         static_cast<uint32_t>(blk >> 32), 0u, 0u};
        philox4x32_10(c, k0, k1);
        for (int j = 0; j < 4; ++j) {
            int64_t idx = b * 4 + j;
            if (idx < n) out[idx] = static_cast<float>(c[j]) * kInv;
        }
    }
}
