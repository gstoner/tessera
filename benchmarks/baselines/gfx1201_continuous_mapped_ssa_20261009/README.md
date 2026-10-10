# Mapped continuous product SSA — gfx1201

Owner: FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / LAYOUT-ALG-1.
Sync: CONTINUOUS-SCALED-MAPPED-SSA-20261009.

The frontend projects semantic types per SSA value. A fully shared producer
retains its scalar shape; a later consumer can acquire a mapped prefix from
another root. Computed FP32 edges retain native allocation/lifetime ownership.
No Python intermediate numerical computation or Tile construction is added.
Encoded FP8 computed edges remain outside this continuous contract.

Validation: all 127 nonempty root policies across primal/JVP/VJP; nested
result-axis certificates; twelve native package checks; 60 owning gfx1201
numerical cases with independent per-plane oracles, shared gradient sums,
changed values/seeds, retained outputs and compiler/eager-free warm replay.
Adjacent host frontend/AD: 554 passed, 13 skipped. Mypy and Ruff pass.

The packet benchmarks masks 1 (producer mapped), 16 (late consumer mapped),
127 (all roots mapped), nested prefix [2,3], outer out_axes=-1. Scalar dimensions
M/K/N/P are 2/9/5/3. Each actual executed package is reused through the checked
native program ABI for seven 128-repeat windows. Output is independently
checked before and after every window. JVP uses the persisted artifact manifest.

Native program medians: primal 6.588–8.460 us; JVP 45.261–46.954 us;
reverse 34.728–60.964 us. Public compilation-warm wall times:
primal 1.971–2.096 ms; JVP 3.987–4.194 ms; reverse 2.631–2.724 ms.
These tiny frames characterize overhead, not throughput or a speedup.
Grouped member samples include HIP graph dispatch and are not additive
interleaved program timings. Device UUID, tools, sources and images are bound
in gfx1201.json. The public wall measurement includes preparation/allocation.

Remaining: broadcast sums between scalar and mapped intermediate results,
aliases, dynamic shapes, residual saving policies and general scaled operation
batching/transpose closure. No sibling device or format proof is inferred.

Full CPU gate: 20,859 passed, 9,604 skipped, 11 warnings; two failures in test_batching_rule_closure (generic scaled_matmul batching/transpose), 384.96 seconds. No tests or contract states were weakened.
