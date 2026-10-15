# Shared SM120 compiler changes: gfx1201 parity and proof integration

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1.
Synchronization key: W11-TENSOR-RHS-JVP-20261015.

## Source and device scope

The native compiler source is 264e1a0f74d853c948ed28a466f71c6e35929d01.
Tajasaurus rebuilt core and ROCm tools against LLVM/MLIR 23.1.1 with assertions.
The matching tools are pinned in parity-start.json. The GPU is the owning
RX9070XT/gfx1201, verified live before running the lane. Existing ROCm native
runtime libraries are unchanged by this NVIDIA compiler/owner slice.

463 combined shared-registry and gfx1201 regression cases pass. The gate
includes the NVFP4 frontend program, semantic Graph, native ingest, package,
lifetime, owner, prepared program, ordinary JIT and resident suites, plus
native scaled-program projection and diagnostic/pass/op/dtype drift guards.
This is an owning sibling regression certificate for those existing ROCm
routes, not a claim that ROCm executes the new SM120 paired program or a
performance comparison. The raw log retains the full case count.

## Integration repair

A frozen-source NVIDIA unit sweep exposed two missing integration records:
ReLU was absent from the derivative proof matrix, and the new paired-tensor
specification lacked its date marker. The three minimal failures reproduce
on Tajasaurus before the repair (registry-red.txt). ReLU now declares the
strict-positive diagonal JVP/VJP mask, zero selection at both signed zeros
and NaN, and finite-difference applicability away from the kink. A new witness
runs actual compiler-generated f64 JVP and reverse SSA, compares against
independent central differences, and checks the adjoint inner-product identity.
Existing storage/boundary tests retain f16/BF16/f32/f64 coverage.

All 53 proof/native-derivative/spec-sync/freshness cases pass after the repair.
No compiler arithmetic, native ABI, physical mapping or selector changes in
this repair. No test is weakened and no undated-document exception is added.
The original broad NVIDIA sweep remains a separate source/environment-bound
run; its terminal results must be inspected before claiming that lane green.

## Sibling assessment

NVIDIA owns the executable paired program and its separately recorded RTX5070
numerical/timing receipts. This packet adds shared proof and actual gfx1201
regression evidence. Apple and x86 receive the same mathematical registry/date
repair; neither receives new physical execution evidence or ABI changes.
Active-source/A+B, composed reverse, symbolic/resident roots and portable
paired replay remain open, as do the broader five-slice requirements.

## LLVM companion environment

The Super-Bear source-built LLVM prefix contains the MLIR tools but lacks
clang/llc/llvm-link. An isolated gated-heap case reproduces FileNotFoundError
before GPU execution. A scratch-only tool overlay retains the primary MLIR
executables and resolves missing companions to the installed LLVM 23.1.1
package. llvm-companions.json pins every actual executable path and hash.
All 21 gated-heap cases pass with that complete environment. No compiler
source, test expectation or numerical policy is changed for this host repair.
The still-running broad sweep keeps its original environment and source;
failed materialization cases need explicit revalidation after it terminates.

Final focused delivery gates pass 72 cases; all 33 generated documents are in
sync. Ruff, the one-module type gate, compiler-plan links, claims and the AST
knowledge-graph refresh pass. Archive extractor warnings remain in graphify.txt.
