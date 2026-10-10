# Direct broadcast scaled frontend: gfx1201

## Direct broadcast scaled frontend — owning gfx1201 integration

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1.
Sync DIRECT-SCALED-BROADCAST-2026-10-08.
Ordinary traced scaled_matmul with batching="broadcast" now infers its result
from all four right-aligned matrix/scale prefixes. Known incompatible or zero
extents are rejected; named packed physical contracts cannot be reinterpreted.
No replacement Python Graph, Tile constructor or numerical backend is added.
Native existing independent address maps consume the actual frontend Graph.
The named source prefixes (2,1), (3), (), (1,3) join to output (2,3).
FP8/MXFP8 primal plus FP32-scale JVP/VJP cover all four matrix transpose flags
at M=3,N=5,K=35 (ragged K32 groups). Scale adjoints reduce to each scale's own
prefix, including singleton and absent axes. Warm compiler-free calls pass.
Owning RX 9070 XT/gfx1201 lane: 16 passed; focused frontend/dtype/op/diagnostic/
pass gates: 988 passed. Sixteen benchmark profiles check independent float64
primal/JVP and finite-difference VJP before/after timing; maximum output error
1.01959069e-07. Public cold/warm wall and prepared HIP program-event windows
are separate. No isolated kernel speedup or default-selector promotion.
Compiler SHA256: 65126f8c651e5ecc0473239e311e9e1e81a812b5a59b348b0b75d1c9d982e4b2.
All recorder source hashes match the sealed source. Encoded scale byte
annotations retain their existing explicit planned/gated token.
Generic dynamic/composed batching, other group widths/storage types,
matrix derivatives and sibling owning-device proof remain open.
Evidence: benchmarks/baselines/gfx1201_direct_broadcast_scaled_20261008/README.md.
Recorder: benchmarks/rocm/benchmark_direct_broadcast_scaled.py.

summary.json records timing ranges. direct-broadcast-scaled-gfx1201-benchmark-20261008.json binds GPU UUID, compiler/runtime, source and member images. Numerical oracles are reference checks only. The earlier recorder failed because its private prepared-package path omitted explicit ROCm target/architecture metadata; the corrected path copies target metadata without altering semantic operations. Device-test annotation repairs preserved dimension equality checks and the existing gated-byte policy.

After sealing the benchmark, four additional shape-role guards were added: a prefix supplied by either matrix or either scale must appear in the result. The complete direct frontend file passes 16 checks. Benchmark production sources and callable test case/oracle definitions are unchanged; its test-file fingerprint refers to the earlier twelve-check snapshot. Current test-source SHA256: 0ad15882b1b4e7e5a38b56441567b814d4851217d6a41621adbbc96033f0942b
