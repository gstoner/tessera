---
last_updated: 2026-10-15
---

# Native normalization AD accumulator evidence

Owner AD-FWD-CORE-1 / AD-HIGHER-1 / NUMPOL-CARRIER-1.
Synchronization key: NORM-AD-ACCUM-20261015.

The standard CMake-built integrated compiler passes 693 focused cases with
seven environment skips on Super-Bear WSL. The 97 normalization cases include
native LLVM object execution against an independent FP64 oracle, dynamic seed
assertion failures, permuted affine arguments, explicit unrealizable precision
refusals, default working-type preservation, mixed signs and offset cancellation.
Registry and existing matmul/attention AD consumers are included.
Documentation/freshness/lifecycle gates pass 33 cases; Ruff, compiler-plan and
public/all-surface claim gates pass. Compressed raw logs and exact source/tool
hashes are adjacent in source-and-proof.json.

The original compiler fails the explicit FP32 accumulator numerical regression.
The preserved baseline red log belongs to that original compiler hash. The
489-case isolated build receipt is a separate earlier epoch. Neither is
relabeled as integrated-build evidence.

This proves generic native Graph/linalg/LLVM CPU arithmetic. It provides no new
GPU Schedule/Tile producer package, no exact-device GPU numerical certificate
and no physical performance promotion. Reverse-mode dead-primal elimination
can still leave conservative NUMPOL-CARRIER-1 provenance debt.

Contract: docs/spec/ir/normalization_ad_accumulator_policy.md.
The original five-slice aggregate remains open.
