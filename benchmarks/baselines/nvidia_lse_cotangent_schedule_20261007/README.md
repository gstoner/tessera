# Explicit LSE cotangent through native Graph, Schedule and Tile

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync NVIDIA-LSE-COTANGENT-SCHEDULE-2026-10-07. Publication pending.

The authored typed Graph consumer carries dO,Q,K,V,O,[bias],LSE,dLSE.
Graph and attention-dialect verification enforce a boolean enabled policy,
saved checkpoint and a matching f32 [B,Hq,Sq] row cotangent. Native
Graph-to-Schedule seals the operand roles and policy in its content hash;
Schedule-to-Tile carries the extra pointer and typed policy into the proved
SM120 consumer. No Python production Schedule or Tile constructor is added.
Caller Graph objects remain unchanged. Missing/incorrect seed rank, dtype,
policy and changed sealed Schedule fields are rejected.

The Schedule artifact decoder retains and verifies this role against both
Graph-to-Schedule and Schedule-to-Tile replay. False/absent use keeps existing
binding counts and contract identities. The generated native symbol names
identify the new input role. The old C bridge refuses these symbols before
allocation/launch: its existing arity inference would otherwise misclassify
the row cotangent as a score bias. This guard is temporary integration safety;
it is not the completed checked ABI. Host and resident refusal are tested.

Matching LLVM/MLIR 23.1.1 core/NVIDIA compiler and launch runtime builds pass
on Super-Bear with RTX 5070 / SM120. contract-device-tests.txt records the
combined native scheduling, existing checkpoint and device lane. There are
36 new Graph-generated numerical cases in addition to the existing physical
leaf cases; tests include output-only, LSE-only and mixed signed cotangents,
GQA, batching, both rectangular directions, causal/full masks and score bias.
The explicit Graph consumer produces Q/K/V; the earlier physical fixture also
checks bias derivatives. Counts overlap earlier receipts.

Recorder: benchmarks/nvidia/record_lse_cotangent_leaf.py --scheduled.
rtx5070.json has 36 generated-route rows and 180 poisoned CUDA-event windows,
all checked against independent FP64 derivatives. Legacy zero-seed controls,
enabled zero-seed controls and mixed-seed arms remain separate. Timing uses
resident buffers and includes diagnostic driver gaps; allocation, upload,
readback and oracle are outside the event windows. No background Graphify
or generated-doc job was active during the final run. Every source file,
both compiler binaries and the runtime library match packet SHA256 values.
Graph/Schedule/Tile/Target/image identities are recorded. This is native
Graph-to-device proof through a controlled diagnostic launch, not checked
public-package or automatic multi-result AD execution proof.

Registry gates: 322 passed for diagnostics, pass metadata, operator and tensor
dtype audit. Mypy for the changed Schedule decoder and Ruff are clean.

Remaining: checked package ABI/descriptor projection, common-runtime and
portable replay, native multi-result AD seed plumbing, private residual owner
integration, invalid-extent/alias/stream checks and separate end-to-end timing.
The five-slice goal remains open. ROCm rejects the physical Tile feature;
Apple/x86 have no admitted physical consumer, so no sibling device proof is
inferred from shared Graph verification.
