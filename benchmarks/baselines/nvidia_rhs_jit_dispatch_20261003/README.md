# NVIDIA ordinary JIT RHS native program

Owner W1.1; frontend integration sibling FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-RHS-JIT-DISPATCH-2026-10-03.

## Exact-device result

Super-Bear: NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0. Ten FP16/BF16 rows passed ordinary JIT and
serialized checked runtime artifact replay, with independent float64 numerical
oracles. Maximum pipeline absolute error: 0.00125699691.
Focused runtime, artifact, execution matrix, compile report and device tests:
60 passed, 8 skipped in 10.14s.

The complete typed semantic Graph is verified by the native MLIR compiler
before partitioning. The producer and consumer each follow Graph → Schedule →
Tile → NVIDIA Target IR → native PTX image with checked runtime ABI. A portable
manifest binds Graph lineage, frontend argument order, descriptor/image hashes
and component contracts. The runtime owns one CUDA stream and private
intermediate/output allocations. Invalid parent/component contracts and an
external launch stream are rejected before CUDA allocation. Warm ordinary calls
reuse the compiled packages; tests disable eager execution and recompilation.

## Timing and reproducibility

[Packet](packet.json) contains exact GPU identity, compiler and source SHA256s,
cold and warm ordinary-call wall samples, serialized replay wall samples and
separate resident producer/consumer CUDA-event dispatch windows. Wall timing
includes staging, module dispatch and download. Event windows are not isolated
kernel-only measurements. Correctness is checked before and after timing.
No speedup or default strategy promotion is claimed.

## Required decision gates

**FP8, MXFP8 and MXFP4 are independent mandatory correctness and performance
evaluation points before selecting a final/default strategy.** This FP16/BF16
program does not prove those formats. Each format needs its own physical scale,
packing, numerical policy, ragged/short/long shape and exact-device evidence.
gfx1201 results do not establish sm_120 parity or vice versa.

## Remaining scope

This route covers one static, primal, affine-free RMSNorm RHS into unfused
FP32-output matmul with FP16/BF16 storage. Dynamic shapes, gamma, fused
epilogues, arbitrary composed graphs and composed AD remain open. Apple,
ROCm and x86 require architecture-owned producer consumers and execution
proof; the shared executor protocol does not establish their parity.

Additional audit/diagnostic/pass/execution-matrix gates: 318 passed in 2.03s. Compiler plan, generated-document regeneration, scoped Ruff and diff checks passed. Graphify is unavailable on Super-Bear (command not found); no fresh graph claim.
