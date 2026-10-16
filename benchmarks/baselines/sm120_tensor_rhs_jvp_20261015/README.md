# Native SM120 paired tensor RHS JVP

Owner W1.1 / FRONTEND-IR-MEDIUM-1 / AD-FWD-CORE-1.
Synchronization key: W11-TENSOR-RHS-JVP-20261015.

The public frontend RMSNorm -> ReLU -> matmul RHS-only JVP now executes
through compiler-owned paired Graph SSA, native Schedule/Tile members,
NVIDIA Target IR/LLVM/NVVM/PTX, and a distinct sealed C++ program owner.
Python constructs no derivative Graph/Tile arithmetic and submits no
individual member kernels. The coefficient edge is materialized once and
retained through the primal and tangent consumers. Two f32 outputs are
independently owned; both host writes commit after native completion.

The v7 tensor manifest records the original Graph, actual paired Graph,
role permutation, per-buffer bytes/ownership/lifetimes, returned IDs and
member Graphs. Primal v1-v6 replay and the existing one-output owner are
unchanged. The old one-output portable validator deliberately does not
accept v7; this increment does not claim portable paired replay.

## Proof

- 666 shared/owning-device regressions pass against the rebuilt tools/provider.
- A subsequent 26-case program gate additionally covers every registered row
  coefficient and serial/cooperative mapping in f16/BF16.
- Four public dtype/role-permutation cases pass after adding reversed frontend
  argument order (24 unrelated program cases deselected in that focused run).
- Named matrix envelopes include M/K/N = 1/16/1, 7/32/9, 17/48/19, 129/256/33.
- Independent stored-coefficient matmul oracles check primal and tangent.
  Retained outputs, warm scratch allocation reuse, close/metadata guards and
  shared-arena stale-lease rejection are covered. Public tests forbid the
  Python per-member NVIDIA launcher.

`timing.json` pins source and compiler/provider hashes, exact RTX5070 UUID,
full native manifests, IR/image/descriptor digests and six correctness-gated
public cases. Captured program medians are 6.46-51.85 us; completed public
call medians are 0.725-0.884 ms. First compile/prepare calls are 1.00-1.92 s.
Maximum absolute oracle error is below 4.8e-6. Each packet row separates
four member timings, complete captured program, profile-call host overhead,
first compile/prepare and completed warm frontend-call timing.
These characterize this route; there is no older native paired route to
support a speedup comparison. Default ReLU remains serial.

## Remaining work

Active-source/A+B derivative member lowering, composed reverse, symbolic
capacities, borrowed resident roots, portable paired replay, and wider
encoded storage/layout/epilogue envelopes remain open. Direct ReLU Graph
TangentInterface support alone still does not prove these target consumers.
No Apple, x86, gfx1151 or gfx1201 physical or timing evidence is transferred.
The original five-slice goal is not declared complete.

Reproduce with the matching LLVM/MLIR23.1.1 tools and updated native provider:
`python benchmarks/nvidia/benchmark_tensor_rhs_jvp.py --output timing.json`.
See docs/spec/ir/sm120_paired_tensor_program_contract.md for the checked ABI.
