# gfx1201 scheduled matmul cache recheck — 2026-10-02

**Host:** Tajasaurus RX 9070 XT, live/configured architecture gfx1201  
**Route:** Graph IR → Schedule IR → Tile IR → ROCm Target IR → HSACO  
**Compiler SHA-256:** `31b835d5c05e1863ab272b98eb52efb932b95d5ea37d6cb2cbbab66b9f66ee7b`  
**Source revision:** `58b848ccbc76`; source worktree dirty  
**Promotion:** diagnostic only; no selector change.

Three static f16 register-matmul shapes reused one HSACO digest
(`4ce8b11e9f1c66f648908701ce73d2d4dbd59e8119895f678d1b8cbc5a3dc13e`)
and one entry symbol, while retaining distinct schedule digests and shape
guards. Each output matched the independent fp32 NumPy product; maximum
absolute errors were 1.19e-7, 3.58e-7, and 1.79e-7. Compile states were cold,
warm-cache, warm-cache. HIP-event medians were 8.64/14.60/15.36 us;
end-to-end medians were 2.20/2.37/2.45 ms. The first cold end-to-end
distribution contains a 39.6 ms outlier. Schedule and package construction
were recorded separately. The event interval covers only the native kernel;
end-to-end includes runtime launch overhead and transfers.

This recheck proves image reuse only for the bounded static f16,
`k_unroll=1`, `split_k=1`, unfused envelope. It does not close dynamic,
split-K, fused-epilogue, LDS-staged, or broader ROCm matmul cache-key coverage.

[Raw packet](gfx1201_recheck_20261002.json).


## Representative-shape extension — 2026-10-02

A source-matched rerun expanded the shape matrix to 16³, 32³, 48x32x16,
128³, 256³, and 512³ (M,K,N). All six cases reused one HSACO and entry;
only the first was a cold compile. Every launch matched the fp32 NumPy oracle.
Maximum absolute error ranged from 1.19e-7 to 1.10e-5. HIP-event medians ranged
from 8.60 to 28.76 us, and end-to-end medians from 2.20 to 6.51 ms. The packet
records seven samples per shape, along with Schedule and package times. These
values characterize the tested route and are not selector-promotion evidence.

The final recorder SHA-256 is
4f9be86888c3b28eca5b948b8d613619bae741fefe6992ac313c52e6e5a3cea3.
[Representative packet](gfx1201_representative_recheck_final_20261002.json).


## Warmed sample refresh — 2026-10-02

A refreshed run adds five correctness-checked launches per shape before timing,
then records seven HIP-event and end-to-end samples. All six shapes reused one
HSACO and entry, with one cold package followed by five warm-cache packages.
Maximum absolute fp32-oracle error was 1.10e-5. Kernel medians ranged from
8.64 to 24.96 us; event-time CV ranged from 0.74% to 1.45%. End-to-end
medians ranged from 2.14 to 5.85 ms. Warmup launches are excluded from both
sample sets. These values remain diagnostic attribution, with no selector
promotion.

[Warmed raw packet](matmul_cache_reuse_warmed.json).
