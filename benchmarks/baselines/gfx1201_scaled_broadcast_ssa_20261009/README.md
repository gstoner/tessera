# Native broadcasted scaled-result SSA — gfx1201

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync NATIVE-SCALED-BROADCAST-SSA-20261009.

The frontend now inserts explicit typed Broadcast Graph nodes when a fully
shared computed result participates in a mapped sum. Add retains its strict
same-shape contract. Native MLIR outlines and materializes that broadcast;
native reverse AD's sum-axis reductions are outlined with seed lineage.
A named serial_tensor_carrier Schedule/Tile/Target algorithm has one f32 input,
one f32 output and checked count. Existing serial/wave scaled reductions retain
their four-input contract. All buffers retain SSA first-write/last-read ownership.

Broadcast lowering performs direct loads/stores, with no arithmetic on copied
values. Sum reductions use the existing compensated FP32 structured lane.
No Python numerical backend, intermediate allocation, Tile constructor or
eager execution route is added. The differential certificate remains an
independent reference-only check.

Validation:
- Matching LLVM/MLIR 23.1.1 core/ROCm build passed.
- 24 frontend/native package checks passed for both operand orders, map depths
  one/two and primal/JVP/VJP.
- Three forged axis/algorithm/count packages with internally consistent
  witnesses were rejected by the carrier ABI validator.
- 915 adjacent native/registry/audit checks passed; Ruff and zero-error mypy pass.
- Twelve owning RX 9070 XT / gfx1201 cases passed independent encoded-product
  oracles, shared-gradient accumulation, changed inputs/seeds, retained results
  and compiler/eager-free warm replay. The owning pytest warns that its timeout
  option is unavailable; no timeout enforcement is claimed.

The six-row final packet uses nested [2,3] maps, M/N/K=3/5/37, encoded E4M3
matrices and FP32 scales. Each executed persisted package is reused for seven
128-repeat HIP windows, with numerical checks before/after timing.
Primal native medians: 30.020–35.068 us; JVP 104.756–105.151 us;
reverse 647.870–648.017 us. Public compilation-warm wall medians:
primal 2.052–2.096 ms; JVP 3.457–3.542 ms; reverse 3.006–3.027 ms.
These tiny cases characterize overhead; no speedup, selector or throughput claim.
Grouped member windows include graph dispatch and are not additive ordinary
program times. UUID, tools, images and actual source hashes are in gfx1201.json.

Remaining: dynamic/alias envelopes, singleton mapped-prefix reshapes,
broader residual save policies, sibling native consumers and generic scaled
batching/transpose closure. The original five-slice goal remains open.

Final CPU gate: 20,871 passed, 9,619 skipped, 11 warnings; only the two existing generic scaled_matmul batching/transpose closure failures remain (393.21 seconds). No test or contract state was weakened.
Final tools also pass all 27 new cases and a direct authored -2 reduction-axis
package, normalized to axis 0. This negative-axis check is compiler/package
evidence, not a separate owning-device numerical claim.
