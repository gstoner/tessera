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


## Frozen full-unit gate

The clean committed source 861f4ad6bd5cf693f2249572fbce4f1feb5fe2b4 passes the
full non-slow tests/unit gate on Super-Bear WSL: 26,392 passed, 7,676 skipped,
876 deselected and 11 warnings in 1983.33 seconds. LLVM/MLIR is 23.1.1 with
assertions; core and matching ROCm optimizer hashes are in full-unit-repair.json.
Raw logs and both source/environment fingerprints are retained beside it.

The initial run had one failing pytest gate: all configured ROCm lit fixtures
invoked a missing tessera-rocm-opt. Building that target fixed the command;
the exact gate passed and the complete suite was rerun. This also enabled
additional compiler coverage. Source and tests were unchanged between runs.

This certificate covers the committed arithmetic fix. The later isolated
explicit-policy Schedule extension and active-source outline have separate
source/compiler epochs. It is host/compiler evidence, with no new sibling
GPU execution or physical performance promotion.
