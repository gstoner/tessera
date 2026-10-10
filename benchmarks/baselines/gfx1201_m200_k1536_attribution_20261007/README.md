# GFX1201 M200/K1536 projected-image attribution

Owner: ROCM-FP8-BLOCKSCALE-1 / E2E-REAL-6.
Synchronization key: GFX1201-PROJECTION-ATTRIBUTION-2026-10-07.

This is an attribution and candidate receipt, not program closure or selector
promotion. All GPU measurements execute on Tajasaurus RX 9070 XT / gfx1201.
Packages follow Graph MLIR -> Schedule MLIR -> Tile MLIR -> ROCm Target IR ->
native LLVM/AMDGPU image -> checked runtime ABI. Python supplies diagnostic
operands, numerical oracles and measurement harnesses; it does not construct
production kernel bodies.

## Findings

The original projected shape-free register image takes about 560 us for
M200/N1024/K1536, versus about 40 us for the static image of the same schedule.
M200/N1024/K1024 similarly takes about 350 us projected versus 31 us static.
These are separate image strategies, not an AITER comparison. The original
static/projected images and disassemblies are retained in this directory.

Code-object notes report zero private segment and zero spills for both
original images. The static image uses 90 SGPR / 188 VGPR; projected uses
50 SGPR / 85 VGPR. Spill attribution is therefore unsupported. Buffering
one/two/four/eight scale panels did not remove the projected-image penalty.
LDS alternatives improve the ragged row in this diagnostic envelope but do
not establish a universal schedule or cache policy.

A native LLVM assumption candidate preserves the checked positive dimensions
and whole scale-group K facts during projection. It does not truncate K,
change scale-zero semantics, reorder numeric accumulation, or change the ABI.
The broad transform affected both K128 and K32. Its final shared-allocation,
40 ms / 21-window M256 test reproduced a 10.18% FP8 K32 penalty (paired
ratio 1.1018), while K128 and MXFP8 controls in that run were near parity. The revised candidate applies this compiler optimization
to K128 groups; K32 FP8 and MXFP8 native images are byte-identical to reference
in all nine-profile paired controls. This is a measured compiler optimization
choice, not a restriction of the supported input envelope.

## Timing scope and limitations

Original confirmation uses nine interleaved windows and reports approximately
6% lower projected image times for M200 and M197 ragged rows. The two named
M200 LDS regression shapes are effectively unchanged. The static/projected
gap remains much larger than this improvement.

The newer recorder alternates candidate/reference images and validates every
window against the same FP64 decoded-operand oracle. Its final version borrows
identical resident allocation addresses for both images and closes graphs and
the borrowed image before freeing the owner allocation. Timings include device
execution plus dispatch through resident HIP graph replay; they exclude host
transfer and checked-launch staging.

Shared-allocation K128 median paired candidate/reference ratios were 0.9293
for M200/N1024/K1536, 0.9827 for M256/N1024/K1024 and 0.9280 for
M197/N1056/K1536. These are observed ratios, not promotion evidence: even
byte-identical MXFP8 control images show large timing swings. Clock and HIP
event witnesses agree on those swings, so attributing them to code generation
would be incorrect. Preserve all windows; do not discard slow samples.

Longer shared-allocation windows (40 ms target, 21 trials) put unchanged K32
control medians within about 1% of parity. K128 paired ratios were 0.9532 for
M200/N1024/K1536, 0.9867 for M256/N1024/K1024 and 0.9408 for
M197/N1056/K1536. Individual windows remain noisy, so these observed medians
do not close the remaining projected/static-image gap.

The sequential three-format packets include FP8, MXFP8, expanded folded MXFP4
and native packed MXFP4. All arms passed their numerical bounds. Sequential
compiler runs and different format strategies do not isolate an instruction,
scale-consumer cost or dtype speedup. No AITER/Radiance performance closure,
cross-architecture transfer, or general MXFP4 M256 attribution is claimed.

## Evidence versions

- `sweep.json`, `buffering-sweep.json`, `static.*`, `projected.*`:
  original compiler, before assumptions.
- `assumption-ab.json`, `assumption-confirmation.json`,
  `three-formats-{candidate,reference}.json`: broad assumption candidate
  and preserved original compiler; historical evidence.
- `interleaved-formats-broad-assume.json`: broad candidate, separate
  resident allocations.
- `interleaved-formats-profile.json`: revised K128 candidate, separate
  resident allocations.
- `interleaved-formats-shared-allocation.json`: revised K128 candidate,
  shared physical allocation addresses.
- `interleaved-formats-long-windows.json`: revised candidate, shared
  allocations, 21 trials with 40 ms target windows.
- `interleaved-broad-shared-long.json`: historical broad candidate,
  shared allocations, M256 regression confirmation.
- `reference-generator.cpp`, `broad-assume-generator.cpp`,
  `recorder-separate-allocation-source.txt`: source versions for historical packets.
- `source-snapshot.json`, `packet-verification.txt`: current and preserved
  compiler/source hashes; exact-device labels and K32 native image equality.
- `candidate-tests.txt`: historical broad candidate, 501 passed.
- `mxfp8-device-tests.txt`: historical broad candidate, 119 passed after
  correcting the descriptor rejection message without weakening its test.
- `profile-build.txt`, `profile-tests.txt`: revised candidate build and
  focused format/device/identity/registry validation: 669 passed and one stale
  K64 rejection fixture failed. The admitted K64 slab now has distinct-key,
  runtime-K reuse and partial-slab refusal checks; the owning identity/K64
  follow-up lane passed 88 tests in `profile-identity-followup-tests.txt`.
  This is focused evidence, not a fresh green full unit suite.

The preserved reference compiler intentionally warns that it predates the
generator edit. Its binary and original source are pinned; this warning does
not authorize using stale compiler evidence for current-source claims.

## Reproduction

On the owning gfx1201 WSL checkout, source the matching validation environment
and activate the existing validation virtualenv, then run:

```sh
python benchmarks/rocm/record_gfx1201_interleaved_compiler_formats.py \
  --compiler "$TESSERA_OPT" \
  --reference /home/angstorms/scratch/fp8-projection-reference-20261007/tessera-opt \
  --llvm-bin /home/angstorms/scratch/gfx1201-scheduled-norm-edge/.toolchain/llvm-23/bin \
  --windows 15 \
  --output benchmarks/baselines/gfx1201_m200_k1536_attribution_20261007/interleaved-formats-shared-allocation.json
```

## Documentation gates

The coordinated source hashes match the owning-device snapshot. Ruff passed;
audit-document tests passed 11/11; freshness tests passed 12/12; baseline and
recorder citation gates passed 4/4. The older NVIDIA JIT owner recorder is now
named from its owning queue. These gates do not imply a green full unit suite.
A fresh batching/transpose closure check still fails for `scaled_matmul`
(two failed, four passed; `batching-closure-tests.txt`).

## Remaining work

Obtain stable control timing before promotion, identify the remaining native
projected-image cost, and rerun broader short/ragged W8A8 coverage against the
owning reference. Further profiling must preserve FP8/MXFP8/MXFP4 numerical
contracts. General paged-KV, other image-key families, gfx1151 synchronization,
and the full-suite batching/transpose failures remain separate obligations.
