# LayerNorm RHS native producer integration

Owner W1.1; frontend sibling FRONTEND-IR-MEDIUM-1.
Sync NVIDIA-LAYERNORM-RHS-2026-10-03.

## Architecture and result

Exact host: Super-Bear WSL, NVIDIA GeForce RTX 5070, GPU-cba12639-821a-7a10-4cd3-f918f9c0a545, 610.88, 12.0.
Ten FP16/BF16 complete/ragged producer-to-matmul cases passed ordinary JIT and
portable runtime replay. Maximum pipeline absolute error: 0.00347955432
Regression including RMSNorm, runtime artifacts, checkpoint packages and
attention forward/backward: 104 passed, 8 skipped in 40.15s.

The typed semantic Graph is verified by tessera-opt before partitioning.
Native Schedule/Tile owns the affine-free LayerNorm computation and typed
transposed B fragments for the row-major RHS matmul. The two packages carry
checked native PTX images and runtime descriptors. This extends an actual
tensor-producing operation through the compiler, with no eager arithmetic or
Python-generated kernel body during execution.

The normalization manifest v2 binds the Graph operation kind to the producer
image entry and descriptor. RMSNorm v1 artifacts remain executable; LayerNorm
cannot masquerade as that legacy schema. Mutation checks reject an altered
normalization kind before CUDA allocation even with recomputed metadata
digests. Each execution owns one stream and private intermediate/output
buffers. Padded host views are explicitly compacted before upload; resident
pitched layout support is a separate envelope.

## Numerical and timing protocol

The float64 oracle subtracts the row mean, computes centered population
variance, normalizes, rounds the intermediate to the declared FP16/BF16
storage, then performs float64 matmul. Separate producer and consumer checks
compare resident outputs against their own independent references. Tests
include default epsilon, large-offset rows, constant rows, both argument
orders, cached calls with eager execution/recompilation disabled, portable
replay and unsupported affine/axis attributes.

[Packet](packet.json) records compiler/source hashes, GPU UUID and driver,
cold/warm JIT wall samples, portable replay wall samples, and separate resident
producer/consumer event dispatch windows. Correctness precedes and follows
timing. Device windows include driver dispatch gaps; no isolated-kernel-only,
speedup or default strategy claim is made.

## Required gates and open scope

FP8, MXFP8 and MXFP4 are mandatory separate correctness/performance evaluation
points before a final strategy or default promotion. Half normalization proof
does not establish their scale, packing or arithmetic contracts.

General producers, affine gamma/beta, composed AD, dynamic shapes and fused
consumer epilogues remain open. Apple, gfx1151/gfx1201 ROCm and x86 need their
own physical producer/view routes and exact-architecture evidence; CUDA
execution does not establish sibling parity.

## Final validation

348 passed in 3.50s. The final representable BF16 large-offset fixture and LayerNorm device lane: 20 passed in 12.28s. Scoped Ruff, compiler-plan, diff and generated-document regeneration gates passed. Graphify query/update was unavailable in WSL (command not found); no fresh knowledge graph evidence is claimed.
