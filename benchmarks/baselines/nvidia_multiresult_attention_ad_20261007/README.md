# Native differentiation of semantic attention O/LSE results

Owner NVIDIA-LSE-1 / AD-RESIDUAL-EVAL-1 / FRONTEND-IR-MEDIUM-1 / E2E-REAL-6.
Synchronization key NVIDIA-MULTIRESULT-ATTENTION-AD-2026-10-07.
Publication pending.

Native reverse AD admits the saved two-result semantic tessera.flash_attn
Graph contract with f32 static rank-four Q/K/V and optional broadcast bias.
It carries O and LSE result cotangents into the verified checkpoint product,
including an explicit row seed. Paired AD replaces both original forward
SSA results with their saved checkpoint results, retains both residuals and
exports the checked seeded backward package. Missing inactive output seeds
are materialized as typed zeros in the adjoint; the exact-device package
cases pass explicit zero/output-only/LSE-only/mixed seeds. Reverse opt-in is
separate from the existing one-result tangent contract.

Physical policy validation now reads the executable's own sealed native
contract. Embedded paired lineage can contain a seeded backward sibling in
a forward artifact; it must not supply the forward executable's seed,
activity, bias shape, mapping or scale metadata. Native Graph/Schedule and
Schedule/Tile replay remain mandatory before package projection.

Matching LLVM/MLIR 23.1.1 core/NVIDIA builds pass. RTX 5070 numerical proof:
72 cases start with a semantic Graph O/LSE op, derive forward/backward with
native paired AD, package through Schedule/Tile/Target/PTX, round-trip JSON,
execute both on resident buffers and check backward resident-event outputs.
Coverage: three rectangular/batched GQA shapes, causal/full masks, plain or
broadcast bias, three cotangent modes and complete/compact requested roles.
The oracle is independent FP64, including physical bias reduction. Owning
checkpoint/compact/broadcast and registry gate: 491 passed. Schedule decoder
Mypy and changed Python test/recorder Ruff checks pass. Counts overlap earlier
receipts; full-suite and all-backend closure are not established.

Recorder: benchmarks/nvidia/record_multiresult_attention_ad.py.
It reuses the checked package timing engine. rtx5070.json has 72 forward and
72 backward stage rows, 720 CUDA-event and 720 host end-to-end windows, all
poisoned and checked after timing. The backward recorder executes and checks
the generated native forward producer before consuming its actual saved
outputs. Stage costs and timing domains remain separate; this is not a
paired public-JIT timing or a default performance promotion. No graph/docs
refresh or test job was active during characterization. Recorded source and
matching core/NVIDIA compiler/runtime hashes are verified.

Remaining: private residual ownership and public Python JIT multi-result AD
capture/backward integration, wider composition/dynamic/batching contracts.
The current numerical proof invokes generated packages explicitly; it does
not imply caller mutation or frame-lifetime safety from a public AD API.
ROCm/Apple/x86 share semantic AD changes but have no new seeded physical
consumer proof. The five-slice objective remains open.
