# Native logical-prefix row pullback prerequisite

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: NATIVE-ATTENTION-SOURCE-REVERSE-20261018.
Base: PR987 merged as 9e63d8c71 after all required CI and exact-head review.

The native scalarizer retains logical rank-three/four tensors and last-axis
statistics/broadcasts, then forms compact [product(prefix), last_axis] physical
Tile views. No derivative is reauthored in Python. Existing pointer ABIs are
unchanged; source hashes retain logical rank. ROCm admission remains rank two.

## Evidence and scope

The unchanged compiler rejects six valid rank-three/four cases. A matching
LLVM/MLIR 23.1.1 assertions compiler with all six backends passes 89 focused
cases; the expanded twelve-case rank suite adds two unique multi-workgroup
cases, for 91 unique compiler cases with zero skips. All twenty retained
rank-two Graph/Schedule/Tile/Target/native stages remain byte-identical.

RTX5070 / SM120 passes 25 initial native/adjacent cases plus an expanded eight
ranked cases, for 27 unique device cases with zero skips. Random, tiny and zero
rows, 35/257 columns, multiple workgroups, wrong row/column guards and tail
sentinels are checked against an independent float64 oracle. This is raw
normalization/ReLU pullback member execution and adjacent existing source-VJP
regression; it is not composed attention reverse or public rank-four VJP proof.

Recorder: benchmarks/nvidia/benchmark_ranked_reverse_row.py.
sm120.json compares eight logical profiles with rank-two controls on identical
resident operands, alternating arm order. Each arm retains 21 samples of 32
launches with numerical checks before and after. Native event dispatch windows
can include enqueue gaps; compilation cost is separate. No public-call,
pure-kernel, speedup or selector promotion is claimed. Wide 257-column ranked
medians are about 0.039-0.0945 ms; short profiles vary more.

manifest.json binds measured source, compiler, actual native session runtime
and exact GPU identities. Compressed logs retain individual case identities;
unique counts are the union across the initial and expanded JUnit records.
Full checked image fingerprints are not compared across compiler builds.

## Remaining work

Native outlining of the original paired attention source, retained normalized
Q/O/LSE ownership through backward, public/portable source gradients and full
composed timing remain open. All five aggregate programs remain open.
Contract: docs/spec/NATIVE_ROW_LOGICAL_PREFIX.md.
