# Native tensor Target controller

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Sync: NATIVE-TENSOR-PROGRAM-PIPELINE-20261019.

The registered tessera_nvidia.program_invoke command now consumes the complete
Tile host controller. Its typed memref SSA retains input order, initialized
reads, fresh outputs, exact private retirement and owned result lifetime.
Tile and Target invoke the same native kernel-binding/ownership verifier.
Device members lower through the existing NVIDIA Target/NVVM route; host
commands remain explicit for the checked native-frame C++ consumer.

## Evidence

- 44 compiler-route cases pass: original producer preservation, reversed roots,
  complete Target conversion, Target member projection, lifetime/geometry/
  architecture tampering, semantic-member rejection, pending AD intent.
- 538 focused host gates pass, including those 44 and diagnostics, pass metadata,
  operator foundation, dtype/attribute audit and NVIDIA fragment/compiler gates.
- Six original-source cases compile to complete Target programs and all 14 stage
  bindings retain byte-identical PTX relative to the earlier Tile controls.
  The eight distinct images remain in
  ../native_tensor_program_pipeline_20261019/images/; no shader copies here.
- 45 owning RTX5070 / SM120 device tests pass with the new Target-projected
  frames, existing native provider and identical images. This is a rerun of the
  runtime cohort, not 45 additional unique numerical envelopes. It includes
  delayed producer streams, retained allocations, invalid spans/contexts,
  changed inputs and completed independent host results.

index.json binds the actual compiler hash, original Graph source hashes and
full Target program hashes. The compressed .target.mlir.gz files retain exact
compiler output bytes, including printer whitespace; frame files are projected
from the actual verified controller. compile-target-controller.py records the
compilation/equivalence procedure; its paths reflect the owning scratch run.
Original Graphs remain in the preceding pipeline packet.

## Scope still open

This increment does not connect public JIT or portable reconstruction.
The retained original Graph must be replayed against the admitted whole
program before trusting a portable package. General producer/layout/dynamic/AD
coverage and all broader five-slice completion obligations remain open.
No physical schedule or performance policy is promoted.

## Timings

target-controller-timings.json retains nine shuffled rounds of 32 repetitions
per six original M17/N11/K32 programs, correctness-gated before and after timing.
Completed adapter medians span 0.209534–0.219506 ms; identical prebound native
C ABI medians span 0.173829–0.230445 ms. Captured full-program device medians
span 0.003092–0.005164 ms. Stage windows are independent, not additive.
These are new-route latency observations, not a claimed improvement over the
earlier run or evidence of model throughput. No arithmetic/schedule changed.
The recorder and checked native provider are those of the native-frame packet.
