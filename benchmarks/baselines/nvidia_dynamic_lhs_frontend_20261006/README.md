# SM120 bounded frontend producer-to-matmul programs

Owner W1.1 / FRONTEND-IR-MEDIUM-1.
Synchronization key NVIDIA-DYNAMIC-LHS-2026-10-06.

## Compiler contract

compile_native_lhs_matmul(..., dynamic_axes=("M","N","K")) admits independently
bounded M/N/K axes using the traced input shapes as explicit maximum capacities.
Any distinct subset of these three axes is supported. Packaging copies the
caller Graph. The retained semantic Graph carries dynamic source/intermediate/
RHS/output types, native shape_bounds and correctly mapped optional epilogue
arguments. Both the original trace and projected full Graph are verified by
the matching tessera-opt.

The shape-preserving producer is compiled at its declared capacity, then bound
to checked row/column prefixes by the existing producer ABI. The consumer is
lowered through native Graph -> Schedule -> Tile -> NVIDIA Target -> NVVM/
LLVM/PTX using runtime leading dimensions. A native Schedule gate now permits
explicit column-major RHS storage for that dynamic strided ABI. Dynamic
row-major RHS still requires a separate physical ABI and is refused.

Portable v1 manifests retain the complete dynamic Graph and semantic
certificate; dynamics are recovered from identity-bound component guards.
Replay checks independent source/RHS contraction extents, positive capacities,
epilogue extents, ABI, target and parent Graph before allocating a device session.
Compiler-free replay retains the same images across active shapes. Producer/
consumer intermediate lifetime is owned by the resident session through stream
completion/readback/close. No intermediate numerical work runs in Python.

The static C++ prepared owner is not used for dynamic frames. These frames
execute the existing checked dynamic resident packages. This is explicit
frontend compilation and portable replay, not ordinary JIT shape-cache
coalescing, a new dynamic C++ owner or general composed Graph support.

## Validation and attribution

Super-Bear's GPU name, UUID, driver and SM120 compute capability are captured.
766 focused checks pass, 49 skip, including 202 device cases. The new suite
has 97 device cases and seven host argument/ABI guards. Four fresh-process
replays pass with compiler entry points disabled. Positive native IR FileCheck
proves dynamic column-major Tile views; a negative fixture retains the
dynamic row-major ABI boundary. Device tests cover all seven axis subsets, three producers, FP16/BF16,
plain/fused epilogues, ragged/one-element/boundary/repeated active shapes,
unchanged caller Graph, guards before device allocation, permuted named
arguments and fresh compiler-disabled replay.

The benchmark covers 48 independent numerical rows: all producers/storage/
epilogues with active M,K,N = 128,1024,64; 17,35,19; 1,1,1; and 63,511,31.
Each group reuses one 128,1024,64-capacity pair. The analyzer rejects image,
Schedule, ABI, semantic-capacity or source drift across reuse. Compile wall,
complete synchronous replay wall and separate resident producer/consumer
CUDA-event dispatch windows are recorded. All timing follows numerical checks.
Dispatch events include driver gaps; they are not isolated kernel-only time.
The refreshed packet passes identity/freshness checks. Maximum absolute
oracle error across 48 rows is 0.015625 under the unchanged FP16/BF16 tolerance.
Median replay wall spans 3.951108–5.787530 ms. Producer dispatch-event medians
span 0.008455–0.250648 ms; consumer medians span 0.008446–0.020638 ms.
These ranges combine distinct shapes/producers and are attribution records,
not cross-case performance comparisons.
No speedup or physical strategy promotion is claimed.

## Reproduction and open work

On Super-Bear WSL, source .build-sm120-w1-1/validation-env.sh.
Run tests/device/nvidia/test_dynamic_lhs_frontend.py and the captured focused
regression lane. Set TESSERA_DYNAMIC_LHS_PACKET to a JSON path and run
benchmarks/nvidia/benchmark_dynamic_lhs_frontend.py. Run this directory's
analyze.py from the repository root.

General composed producer/AD, dynamic row-major RHS, ordinary JIT bounded-shape
selection, a native dynamic prepared owner, asynchronous ownership, wider
formats and sibling exact-device consumers remain open. Full five-slice
closure remains unproven.
