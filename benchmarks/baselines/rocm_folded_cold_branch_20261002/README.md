# gfx1201 folded native scale-recovery branch likelihood

Owner: ROCM-MXFP4-W4A8-1. Sync: ROCM-FOLDED-COLD-BRANCH-2026-10-02.

The native folded Tile consumer now emits llvm.expect(true) for the regular
finite/nonzero combined scale condition. The f32 normal expression and ordered
FP64 overflow/underflow recovery are unchanged. This is a code-layout hint,
not a numerical assumption or removal of the exceptional path. It remains in
active native code after correctness and repeat interleaved timing proof.

[device-tests.txt](device-tests.txt): 101 compiler/device/verifier tests passed,
including overflow, underflow and zero partials, ragged M/N, K64 boundaries,
runtime-M/N/K image reuse and capacity/geometry checks.
[llvm-gate.txt](llvm-gate.txt): one focused gate confirms the hint survives to
native pre-serialization LLVM; do not add overlapping test totals.

[pressure/pressure.json](pressure/pressure.json) records the surviving optimized
branch-weight node (expected 2000:1), and diagnostic selected instructions match
the actual native HSACO exactly. The native stream changed, but scheduled
virtual peak remains 165 VGPRs and physical allocation 177 VGPRs/29 SGPRs,
25600 LDS bytes, no scratch/spills. This is not a register-reduction claim.

## Interleaved repeat measurements

The native reference is compiled from the restored producer without the hint.
Its selected instruction stream is checked against the
[restored pressure packet](../rocm_folded_terminal_prefetch_20261002/README.md);
the candidate stream matches the pressure receipt. Both compiler fingerprints
are retained. The historical-reference stale-source warning is expected:
it is deliberately the older compiler, not a mistaken candidate build.

Every candidate/reference/HIP full-output hash matches and an independent
sampled oracle passes before timing. Three rotating resident copies and nine
alternating device-clock windows are used in [gfx1201.json](gfx1201.json).
Device windows include host dispatch gaps; public runtime wall time, including
allocation/transfers/module lifecycle/completion, is separately recorded.
No isolated epilogue duration, hardware profiler counter or Radiance comparison
is claimed.

| MxNxK | Candidate device-window us | Native reference us | Candidate/reference | Candidate/HIP |
|---|---:|---:|---:|---:|
| 256x4096x1024 | 23.610 | 25.912 | 0.9112 | 1.1134 |
| 256x4096x2048 | 34.650 | 36.768 | 0.9424 | 1.0644 |
| 256x4096x5120 | 78.085 | 77.689 | 1.0051 | 1.0755 |
| 256x8192x5120 | 150.255 | 150.869 | 0.9959 | 1.0232 |
| 256x16384x5120 | 286.766 | 288.509 | 0.9940 | 1.0318 |

[gfx1201_initial.json](gfx1201_initial.json) retains the initial seven-trial
four-case run: short-K candidate/reference ratios 0.9118/0.9224, long-K
1.0186/1.0082. The nine-trial repeat confirms short-K improvements of 5.8–8.9%;
three long-K median ratios are within about 0.6% of the reference. Individual
paired windows are noisy, including occasional ratios above one. Do not infer
a universal gain or close the larger performance envelope from these medians.

The [K16 read/WMMA grouping experiment](../rocm_folded_panel_group_20261002/README.md)
reduced virtual pressure but regressed two N4096 rows in interleaved timing;
it remains removed from active code.

## Remaining obligations

The measured native/HIP gap is still 1.0232–1.1134 in this packet. Exact per-K32
native migration, short/ragged-K widening, wider layouts/model-quality proof
and pure device-kernel versus host dispatch separation remain open. The LLVM
hint belongs to the internal ROCm folded consumer; no sibling execution parity
or gfx1151 FP8 WMMA support follows.

## Reproduce

Build the current native compiler and a preserved no-hint reference compiler
from the reconstructed prior source, then use the owning gfx1201 WSL host.
Run benchmarks/rocm/record_gfx1201_folded_native_package.py with:

    --tessera-opt "$CANDIDATE_OPT" --native-reference-opt "$REFERENCE_OPT"
    --llvm-bin "$LLVM_BIN" --trials 9
    --case prefill:256x4096x1024 --case prefill:256x4096x2048
    --case prefill:256x4096x5120 --case prefill:256x8192x5120
    --case prefill:256x16384x5120
    --output benchmarks/baselines/rocm_folded_cold_branch_20261002/gfx1201.json

The reference must omit only the retained hint in TileToROCM.cpp and retain
the same native folded numerical and ABI contracts. Source and compiler
fingerprints keep both measurements revision-bound.
