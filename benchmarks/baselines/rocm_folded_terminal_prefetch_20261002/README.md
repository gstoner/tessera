# Native folded compiler liveness and terminal-prefetch experiment

Owner: ROCM-MXFP4-W4A8-1. Sync: ROCM-FOLDED-LIVENESS-2026-10-02.

This packet identifies scheduled register liveness for the actual native
runtime-M/N/K folded gfx1201 instruction stream and records one rejected
compiler experiment. The selected instruction hashes from both diagnostic
LLVM objects match their corresponding native HSACO streams exactly; resources
also match. The restored stream additionally matches the prior
[default runtime-K timing packet](../rocm_folded_native_runtime_k_20261002/README.md).
The recorder extracts native pre-serialization LLVM/ROCDL IR, retains its data
layout, optimizes with explicit gfx1201 target context and requests LLVM's
maximum register-pressure def/use report after scheduling. Generic opt without
the target context did not match native instructions and is not used here.

## Restored production evidence

[restored-pressure/pressure.json](restored-pressure/pressure.json) reports a
165-VGPR virtual live peak at a WMMA in the K loop, versus physical allocation
of 177 VGPRs/29 SGPRs. The raw LLVM report and optimized IR are compressed beside
the receipt. Classifications use defining instructions in the reported live set:

| Listed live set | VGPRs |
|---|---:|
| WMMA accumulator tuples | 64 |
| LDS fragment vectors | 48 |
| Global next-slab vectors | 20 |
| Other address/state tuples | 25 |
| Instruction-point difference not assigned to a source category | 8 |
| LLVM maximum | 165 |

The report locates the overlap; it does not prove that removing a category
saves an equal number of physical registers or improves timing. Virtual
after-scheduler pressure and final physical allocation are different metrics.
There are no hardware performance counters or Radiance claims.

## Rejected terminal peel

The original native prefetch loop stages one terminal slab that will never be
consumed; the matched HIP control skips its last fetch. The experiment peeled
the final native stage so it computes already-published LDS without another
register fetch or LDS handoff. K64 remains valid because the final compute
exists even when the prefetched loop is empty. Full-K accumulation and scale
semantics were unchanged; [candidate-tests.txt](candidate-tests.txt) has 96
passing compiler/device tests.

[candidate.patch](candidate.patch) reconstructs the exact narrow C++ change.
[candidate-pressure/pressure.json](candidate-pressure/pressure.json) matches its
native instructions and resources. Virtual peak rose from 165 to 187; the
listed other address/state set rose from 25 to 47, while the listed accumulators,
LDS fragments and prefetch vectors stayed 64/48/20. Physical allocation rose to
180 VGPRs/29 SGPRs, no scratch/spills, versus restored 177/29.

[gfx1201.json](gfx1201.json) contains four correctness-gated matched native/HIP
cases with seven alternating windows, rotating inputs and separately recorded
public-launch wall time. Native/HIP device ratios were 1.2485, 1.1396, 1.0977 and
1.0300 at M256/N4096/K1024, K2048, K5120 and M256/N8192/K5120. Candidate
runtime/static ratios were 1.0450, 0.9888, 1.0379 and 1.0043. These are separate
windows from the earlier packet, not interleaved candidate/restored timings;
they do not establish a consistent improvement. The peel was removed from
active code and the native compiler rebuilt. [restored-tests.txt](restored-tests.txt)
has 96 passing tests; do not add the overlapping suites.

## Next engineering obligation

Reduce native LDS-fragment/address overlap using a correctness-preserving
compiler schedule, then require instruction/resource attribution and paired
device timing. Exact per-K32 native migration, wider layouts and model quality
also remain open. Do not infer gfx1151, NVIDIA, Apple or x86 parity from this
gfx1201 FP8 WMMA experiment.

## Reproduce pressure attribution

Run benchmarks/rocm/record_gfx1201_folded_register_pressure.py on actual gfx1201
with the matching compiler and LLVM tool directory:

    --tessera-opt "$TESSERA_OPT" --llvm-bin "$LLVM_BIN"
    --output-dir benchmarks/baselines/rocm_folded_terminal_prefetch_20261002/restored-pressure

Apply candidate.patch and rebuild tessera-opt to reproduce candidate-pressure;
restore the producer and rebuild before production execution. Source/compiler
fingerprints and raw reports remain revision-bound in the receipts.
