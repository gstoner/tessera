# gfx1201 FP8 / MXFP8 / folded MXFP4 evaluation

Owner ROCM-FP8-BLOCKSCALE-1 and ROCM-MXFP4-W4A8-1; sync ROCM-THREE-FORMATS-2026-10-03.

## Question and execution boundary

Evaluate three required numerical formats before choosing short/long-K or persistent strategies. Every arm uses compiler-owned typed Graph -> Schedule -> Tile -> ROCm Target -> ROCDL/LLVM -> HSACO, then checked runtime.launch. Device replay borrows the same image and descriptor buffer order through a resident diagnostic adapter; its geometry is recorded explicitly.

Live gfx1201 RX 9070 XT is queried and required by the recorder. LLVM 23.1.1 and the actual compiler binary SHA are recorded. This is Tajasaurus evidence only.

## Matched inputs and independent numerical proof

Each shape uses the same seeded floating-point A and B across all arms. B has Gaussian values with independently varying K32/column power-of-two magnitudes. SHA256s bind both logical source arrays. FP8 K128/N128 uses amax/448 fp32 scales; the FP8 K32/N1 control changes granularity. MXFP8 uses K32/per-column scales rounded upward to standard E8M0 powers of two. Folded MXFP4 uses per-token fp32 activation scales, nearest-even E2M1 K32 weights and explicit opt-in row-reference folding into expanded E4M3 storage. No packed-four-bit bandwidth claim is made.

Independent f64 decoded-operand matmuls separate quantized ideal output error, actual native output error, pre-fold FP4 error, folding weight error and folding output error. Every native output must be finite and satisfy an elementwise conservative f32 accumulation/scaling forward error bound plus BF16 final-store rounding, before timing and after replay. Host tests check E2M1 ties, standard E8M0 zero-code semantics, folding loss separately from quantization, and rejection of corrupted native outputs.

These are distinct numerical policies and native schedules. The K32 FP8 control exposes granularity dependence but does not isolate E8M0 load/arithmetic cost: native selectors can choose different panels and source quantization differs. This synthetic workload is not model accuracy evidence.

## Native copy repair discovered by the sweep

The existing FP8 K32, M=200/N=2048/K=2048 route selected a 128x64 eight-wave LDS tile, then refused serialization because its 128 RHS copy vectors did not divide evenly among 256 threads. The native generator now uses ceiling copy rounds and masks the final incomplete round. Inactive threads perform neither global loads nor LDS stores; barriers remain outside the mask. Evenly divided copy rounds retain their prior emitted IR. The physical Schedule, scaling order, ABI and lifetime contract are unchanged. K16 tests use explicit scale_group_panels=1; the existing auto default of 2 for a one-panel group remains a separate open selector-policy obligation.

## Timing

Five rotated/reversed windows per shape and process, at least 20 ms each. GPU clock markers are witnessed by HIP events within 5%. Device numbers include graph dispatch and kernel execution; they do not isolate instruction phases. Checked end-to-end timings separately include host staging, transfers, module/descriptor work and synchronization. Quantization, compilation and graph capture are outside steady-state timings. Resident inputs stay allocated through replay; graphs close before modules/buffers.

First-process medians below are microseconds; evaluation-repeat.json records an independent repeat. All four arms for all seven shapes passed numerical gates in both processes.

| M,N,K | FP8 K128/N128 | FP8 K32/N1 control | MXFP8 K32/N1 | Folded MXFP4 |
| --- | ---: | ---: | ---: | ---: |
| 200,256,128 | 17.438 | 18.362 | 7.635 | 14.628 |
| 200,256,1024 | 346.555 | 440.393 | 33.100 | 25.381 |
| 200,4096,1536 | 29.345 | 42.861 | 120.310 | 33.044 |
| 200,2048,2048 | 25.032 | 41.418 | 104.061 | 32.672 |
| 200,8192,1024 | 41.030 | 68.645 | 138.779 | 39.583 |
| 256,1024,1024 | 20.202 | 26.969 | 19.886 | 21.221 |
| 256,4096,5120 | 78.641 | 135.667 | 489.764 | 77.488 |

## Evidence-based next steps

FP8/MXFP8 source-output relative RMS errors are approximately 3.3-3.8%; folded MXFP4 is 11.4-11.9%. Additional folding output error is 0.025-0.084% relative to pre-fold FP4 on these inputs. This separates approximation cost from the larger FP4 quantization error.

MXFP8's one-wave seed wins on the tiny grids but is 4.1-6.2x slower than FP8 on wide/long rows. Next: integrate E8M0 byte scales with an architecture-owned LDS Schedule/Tile path and prove scale-group isolation and partial-copy safety before performance promotion. The FP8 global fallback at M=200/N=256/K=1024 is also disproportionately slow and needs attribution. Folded MXFP4 and FP8 are close on two wide rows, but their quality differs materially. No format, panel, persistence strategy or global selector is promoted.

The M=256 per-column gap to Radiance remains open: this packet has no Radiance arm, fixed-N slope scan or counters. Previous C-LDS/prologue/fragment experiments remain measured diagnostics. Source-model quality, broader K tails, cache-cold/rotating-copy characterization and sibling architecture consumers remain separate obligations.

## Validation

Fresh host WSL compiler build, then 523 focused tests passed with no skips:
48 exact-device partial-copy cases cover K16/K32, f32/BF16, regular/ragged bounds,
and prefetch modes 0/1/2; five host quality-oracle cases; existing FP8/MXFP8/folded
package and diagnostic/pass-registry regressions. Typed generated-IR FileCheck
and native target lowering pass. A preserved pre-change compiler and the retained
compiler emit byte-identical FP8, MXFP8 and folded MXFP4 whole-copy images,
with identical Tile/Target digests and ABIs (whole-copy-reference.json versus
whole-copy-candidate.json). The reference binary's stale-generator warning is
expected and preserved; its actual binary and generator source SHA are bound
in validation.json.

## Files and reproducibility

evaluation.json and evaluation-repeat.json bind logical inputs, per-arm source/image/IR/ISA digests, geometry, correctness/error metrics, all clock windows and checked E2E samples. The matching .s files retain native WMMA disassembly. pre_copy_fix_partial.txt/json preserve the original failure; smoke.txt/json are the initial row before the copy fix. final-tests.txt contains the final focused device/unit regression result. whole_copy_images.py records reference/candidate hashes for existing whole-copy FP8, MXFP8 and folded MXFP4 packages.

Run from the matching Tessera checkout in host WSL after sourcing scripts/_rocm_env.sh and selecting the matching LLVM/toolchain, TESSERA_OPT and TESSERA_ROCM_CHIP=gfx1201. Set TESSERA_GFX1201_DEVICE_PROOF=1 for the focused device suite. Recorder invocation:

    python benchmarks/rocm/benchmark_gfx1201_three_formats.py --compiler "$TESSERA_OPT" --llvm-bin "$LLVM_BIN" --shapes '200,256,128;200,256,1024;200,4096,1536;200,2048,2048;200,8192,1024;256,1024,1024;256,4096,5120' --output benchmarks/baselines/rocm_three_formats_20261003/evaluation.json

Graphify is unavailable on the owning WSL host; graph maintenance must run on a host with Graphify installed. This does not substitute for numerical or compiler validation.
