# Current-source gfx1201 compiler and execution revalidation

Owner ROCM-NVFP4-INGEST-1 / E2E-REAL-6.
Synchronization key GFX1201-CURRENT-SOURCE-2026-10-07. Publication pending.

The coordinated compiler/Python delta was synchronized from Super-Bear to
Tajasaurus. Previous differing files were preserved in a scratch archive.
All 141 archived source paths match the recorded coordinated bytes; 53
needed updating. No deleted source paths occurred. Matching core tessera-opt
and tessera-rocm-opt were rebuilt against LLVM/MLIR 23.1.1. build.txt records
successful targets; source-snapshot.json pins both compiler binaries and
every transferred source file. Exact owning GPU is RX 9070 XT / gfx1201;
rocminfo and each recorder's HIP identity queries establish the architecture.

425 tests passed, eight skipped (all gfx1151-owned cases) in tests.txt.
Includes native NVFP4 ingest leaf and Graph/Schedule/Tile execution,
ordinary JIT conversion, resident/native ownership and injected lifecycle
failures, paged-KV compiler spine and public movement, diagnostic/pass/op and
dtype-attribute registry gates. One warning reports the missing pytest-timeout
plugin; numerical assertions and compiler ancestry gates ran.

nvfp4-jit.json: six static three-stage ordinary JIT/portable rows, three shapes
with both argument orders. Five independent windows per lane; converter,
lossless storage, scaled-matmul consumer and combined HIP graph windows stay
separate from allocating warm JIT and portable replay wall costs. Bitwise
conversion/storage, independent conversion statistics and float64 folded
output are checked before/after timing. Warm JIT medians span 3.26–4.64 ms.
This establishes current-source static execution, not dynamic/model-quality
acceptance or a matched performance change.

movement.json: three paged-KV shapes, nine rotating paired host trials,
20 launches per trial, separate preloaded resident HIP-event windows.
Read results are bit-exact before/after timing. Resident event medians:
0.003925, 0.006172 and 0.020663 ms. Native pooled host medians:
0.571518, 1.592296 and 2.916119 ms. Event windows include dispatch gaps;
host wall includes launch/binding/copy/lifecycle costs. These timings are
different domains and establish no isolated kernel speedup.

Both packet source/compiled-binary fingerprints were verified against the
rebuilt checkout (packet-verification.txt). No competing tests/benchmarks or
compiler build ran on the owning GPU during characterization.

Open: general paged-KV layouts, dynamic/composed NVFP4/AD, model-quality
acceptance, broad cache-key families, W8A8 short/ragged gaps, MXFP4 M256
Radiance attribution and gfx1151 current-source synchronization. No sibling
physical execution or full-suite closure follows from this receipt.

Reproduce in scratch/next-five-compiler-slices-rocm after sourcing
.build-gfx1201-current/validation-env.sh, with .venv-movement-capture on PATH:
python -m benchmarks.rocm.benchmark_jit_nvfp4_program --compiler "$TESSERA_OPT" --windows 5 --output nvfp4-jit.json
python -m benchmarks.rocm.benchmark_native_movement --architecture gfx1201 --repeats 20 --output movement.json

## Packed-format current-source follow-through

209 FP8/W8A8, MXFP8, MXFP4 package and image-identity regressions passed,
without skips (packed-format-tests.txt). three-formats.json records four
shapes and 20 correctness-checked arms, with three resident timing windows
and separate checked host walls per arm. It uses identical seeded original
f32 sources but different declared format quantization/scaling policies;
cross-format timing is not an equal-numeric-policy speedup comparison.
Packet source and owning compiler hashes match the synchronized snapshot.

At M200/N1024/K1536 the FP8 K128/N128 arm records 565.004 us, versus
20.247 us for M256/N1024/K1024. Shapes differ, so this is a prioritization
signal, not an attributed regression. Artifact metadata identifies global
register 16x32 / one-wave runtime-shape materialization at M200, versus
32x32 / one-wave at M256. The long M256/N4096/K5120 arm uses eight-wave
128x128 LDS staging. Next action is a matched M200 schedule experiment
that preserves blockscale and masked-row semantics. No AITER/Radiance
reference or route-promotion evidence is present in this packet.
