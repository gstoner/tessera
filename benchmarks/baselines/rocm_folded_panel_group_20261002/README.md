# gfx1201 folded K16 read/WMMA grouping experiment

Owner: ROCM-MXFP4-W4A8-1. Sync: ROCM-FOLDED-PANEL-GROUP-2026-10-02.

The native producer grouped each K16 panel's DS reads and WMMA chain with
ROCDL scheduling intrinsics, scoped to folded prefetch mode. It retained
ordered full-K accumulation, shared-memory barriers and the checked ABI.
[device-tests.txt](device-tests.txt): 96 compiler/device checks passed.

[pressure/pressure.json](pressure/pressure.json) matches the candidate's actual
native instruction stream and resources. LLVM maximum scheduled VGPR pressure
fell from 165 to 154; the maximum moved from a K-loop WMMA to an epilogue global
scale load. Physical allocation stayed 177 VGPRs/29 SGPRs, no scratch/spills.
The category classifier uses defining instructions; its epilogue "other" set
must not be interpreted as all addresses. There are no hardware counters.

## Interleaved correctness-gated timings

The recorder now admits an independently built native reference compiler.
Frontend and native projector use the same executable for each arm, with
environment restored afterward. The physical schedule must match. All output
hashes match across candidate, reference and HIP; an independent sampled oracle
precedes seven alternating windows with three rotating resident copies.
Public launch wall time is recorded separately. Reference selected instructions
match the [restored producer](../rocm_folded_terminal_prefetch_20261002/README.md);
candidate instructions match the pressure packet. Both compiler fingerprints
are recorded in [gfx1201.json](gfx1201.json).

| MxNxK | Candidate/reference device ratio | Candidate/HIP |
|---|---:|---:|
| 256x4096x1024 | 0.9927 | 1.2334 |
| 256x4096x2048 | 1.0440 | 1.1989 |
| 256x4096x5120 | 1.0342 | 1.0581 |
| 256x8192x5120 | 0.9679 | 0.9909 |

The candidate improved one larger-N case while regressing two N4096 cases.
It was removed from the default producer; [candidate.patch](candidate.patch)
preserves the narrow change. Lower scheduled virtual pressure did not improve
the full measured envelope or reduce physical allocation. Do not promote this
global schedule or infer broader N/M closure from one winning row.

## Remaining work

Native epilogue layout/liveness, short-K overhead, exact per-K32 migration and
wider layout/model-quality evidence remain open. gfx1151 FP8 WMMA is not
applicable under RDNA3.5; no NVIDIA, Apple or x86 physical parity follows.

## Reproduce

Apply candidate.patch and build a candidate tessera-opt; preserve a separately
built reference compiler from the restored source. Run
benchmarks/rocm/record_gfx1201_folded_native_package.py with:

    --tessera-opt "$CANDIDATE_OPT" --native-reference-opt "$REFERENCE_OPT"
    --llvm-bin "$LLVM_BIN" --trials 7
    --case prefill:256x4096x1024 --case prefill:256x4096x2048
    --case prefill:256x4096x5120 --case prefill:256x8192x5120
    --output benchmarks/baselines/rocm_folded_panel_group_20261002/gfx1201.json

Use the owning gfx1201 WSL environment. This packet is revision-bound
experimental evidence; source/default producer has the grouping removed.
