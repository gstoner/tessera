# Large-capacity H-bounded MLP with the global gate panel on gfx1201 (2026-10-30)

**Sync key:** `GFX1201-BOUNDED-GLOBAL-PANEL-20261030`. **Owner:** ROCM-NVFP4-INGEST-1.

**Host:** Tajasarus, an RX 9070 XT (gfx1201) on Ubuntu 26.04 under WSL2, with ROCm 10.0 / HIP 7.15. The compiler is the assertions-ON LLVM/MLIR 23.1.1 build (`build-assertions/`). Everything ran unsandboxed on the device box.

**Recorded by:** `benchmarks/rocm/benchmark_k_bounded_quantized_mlp.py --h-bound --bounds 256 2048 --kn 4096 256` (`paired.json`) and `member_attribution.py` (`members.json`), both at commit `9df652d5f` on a clean worktree.

## What changed

Before this commit, a K-, N- or H-bounded program whose capacity reached an LDS gate panel did not compile. At M = 256 and gate N = 2·2048, the static rule takes LDS `[128, 128]`, and no runtime-K/N image exists in LDS form. The product member now requests `tessera.rocm.blockscale_staging = "global"`. The selector skips the LDS rules and takes the register panel at the admitted minimum M, which is `[16, 32]` for M >= 1.

## Result (median ms per invocation, the method of `../gfx1201_h_bounded_quantized_mlp_20261029/`)

| Active (M, H) | Bounded owner | Static at shape | Bounded / static | Bitwise equal |
|---|---|---|---|---|
| 16 × 512 | 0.0874 | 0.0848 | 1.032 | yes |
| 16 × 2048 | 0.1967 | 0.1921 | 1.024 | yes |
| 100 × 1024 | 0.2187 | 0.1960 | 1.116 | yes |
| 256 × 512 | 0.1281 | 0.1282 | 0.999 | yes |
| 256 × 2048 | 0.9649 | 0.3509 | **2.750** | yes |

The program now compiles, and it is correct at every shape. One bounded compile took 1.16 s; its owner holds 18 device buffers and 50.8 MB at capacity.

## Where the cost is (`members.json`, median µs per member invocation)

| (M, H) | Arm | gate quantizer / gate/up expansion / gate product / SiLU / down quantizer / down expansion / down product |
|---|---|---|
| 16 × 2048 | bounded | 4.11, 110.44, **43.65**, 2.72, 4.26, 4.51, 21.87 |
| 16 × 2048 | static | 4.12, 110.15, **42.65**, 1.95, 4.07, 4.39, 20.86 |
| 256 × 2048 | bounded | 11.85, 111.59, **826.02**, 11.45, 6.17, 4.55, 24.17 |
| 256 × 2048 | static | 12.86, 110.86, **197.96**, 7.28, 6.32, 4.46, 23.30 |

- **The gate product's panel carries it.** At 256 × 2048 the global `[16, 32]` gate product takes 4.2× static's LDS `[128, 128]`. At M = 16 the two match (static also takes `[16, 32]` there).
- **For comparison (diagnostic).** An earlier uncommitted override measured a `[32, 32]` register panel at 1.35× LDS for this static shape, so the 16-row choice made for small M costs about 3× at the capacity end.
- **What it implies.** No single panel serves M = 16 to 256 at this width: LDS at capacity costs about 2–3× at small M, and the small-M register panel costs about 4× at capacity. Selecting a product image per active M is the open item this measures.

## Limits

- **Diagnostic only.** HIP events on WSL2 are not an admitted kernel-side witness, and nothing here is a promotion.
- **One K and N, gfx1201 only.** gfx1151 has no evidence and none transfers.
- **Bitwise equality with the static program was observed at these shapes and in the device tests.** It is not asserted in general.
