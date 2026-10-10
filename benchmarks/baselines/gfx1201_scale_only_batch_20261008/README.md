# gfx1201 independently owned scaled batch axes

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Sync SCALE-ONLY-BATCH-2026-10-08.

RX 9070 XT / gfx1201: 13 frontend/device tests pass, one existing missing-timeout-plugin configuration warning. Ordinary traced FP8 scaled_matmul runs through the native Graph/Schedule/Tile/ROCm/LLVM program; no replacement Python Graph or numerical backend is introduced.

Each of A, B, lhs scale and rhs scale independently supplies batch prefix (2,3), with other inputs rank two. Matrix dimensions M=3,N=5,K=35, block scale [3,32], include ragged K and columns. Four primal, four scale-JVP and four scale-VJP profiles pass float64/finite-difference numerical checks before and after timing. Adjoint outputs retain the supplying scale prefix and reduce unmapped axes for shared scales. Warm compiler-free public calls pass. Maximum absolute benchmark error 1.15684470559e-07.

Recorder: benchmarks/rocm/benchmark_scale_only_batch.py.
Sources and compiler/runtime hashes are recorded in benchmark.json; all recorded source hashes match the coordinated checkout. Public cold/warm wall times and three native HIP program-event windows of 32 repetitions remain separate. This is characterization, not isolated kernel gain or selector promotion.

An initial combined collection collided on unit/device module names; renaming the device module resolved collection without weakening checks. The 13-case owning selection was rerun.

Generic dynamic/nested/composed batching, matrix derivatives and other numeric policies remain open. This evidence does not close primitive batching/transpose statuses or substitute for Apple, x86, gfx1151 or NVIDIA physical proof.
