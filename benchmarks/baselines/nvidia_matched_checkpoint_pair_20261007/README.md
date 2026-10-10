# Matched complete checkpoint pairs

Owner NVIDIA-LSE-1 / E2E-REAL-6 / AD-RESIDUAL-EVAL-1; sync NVIDIA-MATCHED-PAIR-2026-10-07. Publication pending.

Recorder: benchmarks/nvidia/record_lse_checkpoint.py, schema v6. --paired-only measures the same checked host-buffer launcher for both saved and recompute. Each sample alternates lane order. Complete calls contain forward then backward; upload/download and synchronization are included, while host allocations, poisoning and independent FP64 oracle checks are excluded. This scope differs from the resident-frame packet and must not be combined into a speedup.

Native paired AD already saves O/LSE: src/compiler/ir/AdjointInterface.cpp and src/transforms/lib/AutodiffPairedPass.cpp generate saved checkpoints. The old standalone recorder selector_default=recompute field was misleading and is replaced by explicit native-paired and comparator policy fields. No production selection policy is changed by this diagnostic recorder.

RTX 5070 / SM120 host WSL. Plain and exact-bias packets cover three shapes, five samples, --adaptive-window-ms 20, --reps 20. Every window poisons O/LSE and gradients, launches forward before its corresponding backward, then checks the actual outputs. All 60 paired windows pass. Complete ancestry and matching native Target hashes are checked before timing. The Graphify process is suspended for final timings and resumed afterward.

| Shape | Bias | Saved pair ms | Recompute pair ms | Saved residual bytes |
| --- | --- | ---: | ---: | ---: |
| 1x2x1x3x4x4x3 | False | 2.3030 | 2.2002 | 96 |
| 1x4x2x127x131x64x64 | False | 5.9405 | 311.9342 | 132080 |
| 1x4x2x256x256x64x64 | False | 15.1849 | 1930.2595 | 266240 |
| 1x2x1x3x4x4x3 | True | 2.3899 | 2.2746 | 96 |
| 1x4x2x127x131x64x64 | True | 6.2004 | 323.9575 | 132080 |
| 1x4x2x256x256x64x64 | True | 17.0209 | 2031.3977 | 266240 |

Validation: window-tests.txt has 15 passed; mypy.txt and lint.txt are clean. Source and both compiler/runtime binary fingerprints match both packets.

Open: native explicit-LSE cotangents, general dynamic/composed AD, general scaled-matmul batching/transpose gates and publication. Standalone recompute kernels remain valid comparators; native paired AD already uses saved residuals. No sibling physical evidence is inferred.
