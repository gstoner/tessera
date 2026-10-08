# Current-head W1.1 numerical and timing proof

Owner W1.1 / E2E-REAL-6. Sync NVIDIA-W11-CURRENT-HEAD-2026-10-08.

At source 57aba3b8e00e3ca49c723cd24cd0544f1955c756, 147 owning RTX 5070 sm_120 tests pass with no skips. These cover resident and host fused epilogues, legacy tensor producer migration, canonical tensor loops, prepared macro paths, bounded single-producer shapes, and native partition lifetime/ancestry guards.

The 24 correctness-gated benchmark rows cover f16/bf16, fp16/fp32 outputs, static/bounded M, and three capacity profiles. Every row measures producer and consumer separately with seven CUDA-event samples, 1000 repetitions and 20 warmups. End-to-end wall time includes allocation/upload/producer/consumer/synchronization/cleanup and excludes compilation/download. Timing domains are not interchangeable; no default promotion or speedup claim.

| Domain | Minimum median ms | Maximum median ms |
| --- | ---: | ---: |
| producer | 0.008874 | 0.012720 |
| consumer | 0.008766 | 0.016403 |
| end_to_end | 2.362328 | 2.808102 |

Packets bind source, compiler, target tool, runtime, images and descriptor provenance. Tests bind current source and validate changed-input reuse and retained allocations. This is named-envelope proof, not arbitrary composition, dynamic multi-producer admission, external asynchronous ownership, generic scaled-matmul closure, or sibling-device proof.

## Current CI diagnosis

PR895 compiler-route job 113500847703 completed with 110 passed / 235 skipped. Its execution gate rejects ROCm image toolchain skips on the GitHub runner. Those tests require AMD clang/device libraries; LLVM/MLIR 23.1.1 and the portable compiler build succeeded. Supplying the genuine image toolchain remains open; the no-skip gate and coverage flags are unchanged. Aggregate unit CI still has two generic scaled-matmul batching/transpose failures.
