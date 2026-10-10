# Bounded physical broadcast-bias integration — 2026-10-08

Owner E2E-REAL-6 / AD-RESIDUAL-EVAL-1.
Sync NVIDIA-ATTENTION-DYNAMIC-BIAS-2026-10-08.

The public isolated attention VJP accepts sequence_bounds=(SqMax,SkMax).
Python records capacities; native paired AD projects typed Graph dimensions,
derives saved O/LSE and adjoints, then exports sealed Schedule/Tile products.
Fixed batch/head/width and the traced bias broadcast policy remain verified.

Symbolic physical bias now carries four actual extent scalars through Tile to
native NVIDIA indexing and physical bias-gradient reduction. Static carriers
retain their previous operand layouts. Positive capacities, symbolic policy,
actual physical shape and pitches are checked together. Native identity keeps
symbolic metadata; allocation and execution use concrete actual dimensions.
Malformed physical scalars fail before loading CUDA.

Matching core/NVIDIA compiler builds succeed. Six native Graph/Schedule/Tile/
Target/PTX cases cover query, key and both symbolic physical axes in forward
and backward. Two negative native tests reject missing extent operands.
Eight host tests prove positive capacity guards and malformed-pitch rejection.
The initial negative fixture failures used incorrect command/printer syntax;
the original logs are retained without labeling those runs green.

668 affected host WSL AD/package/residual/registry/lifecycle tests pass.
The combined native/owning lane passes 36 tests: sixteen host native/guard
tests and twenty exact RTX 5070 public replay cases. The latter cover six
actual sequence shapes, no/full/key/query/both bias, compact/noncompact
gradients and synchronous/asynchronous capture. Programs serialize and replay
with subprocess compilation forbidden; output/LSE, selected adjoints and
physical reduced bias gradients match the independent oracle. Changed seeds
and different-shape live generations preserve older gradients.

Twenty recorder arms pass numerical checks before and after timing. One image
pair per physical policy is reused across two actual shapes and both capture
modes. Maximum gradient error: 9.685754776000977e-8.
Completed capture medians: 3.864272–5.732103 ms.
Completed backward host medians: 0.158383–0.256270 ms.
Complete-owner CUDA event medians: 0.155567–0.254464 ms.

The event metric includes allocations, seed copies, kernels, frees and driver
enqueue gaps. Capture includes allocations, module loading, snapshots and
completion. These are characterization measurements, not isolated kernel
timings or counterbalanced speedup claims.

packet.json records actual RTX 5070 UUID, SM12, source and runtime fingerprints.
Core compiler SHA256: 2daee6b16be18a340302bd5cefc1521092f31cdf6a6f20fdafcb6bac2e4d7b47
NVIDIA compiler SHA256: 34cc11367b156953c52a997d5706b03db8f57b15f68d63372056c025a39cb1bf

Open: dynamic JVP, arbitrary composition/nested AD, dynamic batch/head/width,
runtime changes to the traced broadcast policy, automatic external-consumer
lifetime tracking, sibling backend physical parity, fresh aggregate validation
and focused publication. The five-slice goal is not closed.

## Static/residual regression follow-up

92 additional owning RTX 5070 tests pass existing static tuple-result AD,
asynchronous residual ownership and explicit bounded package routes.
Eleven final audit-document tests pass; the focused tracked diff has no
whitespace errors. fingerprints.log confirms every packet source/tool/runtime
hash matches authoritative current bytes. These focused results do not claim
a new aggregate full-suite green result.
