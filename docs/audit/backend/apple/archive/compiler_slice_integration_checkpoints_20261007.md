---
audit_role: reference
last_updated: 2026-10-07
---

# Superseded integration checkpoints

Original intermediate notes below are superseded by proved named profiles in ../todo.md.

## Independent-prefix transposed-A integration — active

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Native gfx1201 Schedule orientation,
scaled Tile carrier, column-major A tile.view, serialized program orientation
and public leading-map admission are being integrated. Sixty host frontend
projection cases pass. Native tests found and drove repairs to Schedule
verification and scaled Tile orientation propagation; matching rebuild is
active. This entry does not claim native/device numerical or timing proof.
Partial scale groups, wider layouts/composition and generic closure remain open.
Follow-up required for apple physical parity; the new gfx1201 gather recipe and HIP program orientation provide no sibling execution evidence.


## Independent-prefix partial scale groups — active integration

Owner FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / ROCM-FP8-BLOCKSCALE-1;
sync INDEPENDENT-SCALE-BATCH-2026-10-07. Native static independent-prefix
register lowering now derives ceiling scale-group counts and bounds trailing
WMMA fragment loads. Aligned recipes and other physical families retain their
existing contracts. 120 host projection cases pass. Matching native build,
image checks, gfx1201 numerical proof and timing are still being validated;
this entry is not an execution or performance claim.
Follow-up required for apple-owned partial-group materialization and exact-device parity; no gfx1201 WMMA schedule or HIP proof transfers.


## Typed primal single-image integration — active
Owner FRONTEND-IR-MEDIUM-1 / ROCM-FP8-BLOCKSCALE-1;
sync TYPED-PRIMAL-SINGLE-IMAGE-2026-10-07. Scalar and coupled static FP8/MXFP8
packages now bind their actual native Graph-derived member image, entry and
geometry, removing the separate legacy Tile image compilation. 388 focused
host WSL gates pass, including all 16 scalar/coupled orientation/format image
consistency cases. Matching gfx1201 numerical replay and alternating cold
package-cost measurements are running; no performance or closure claim yet.
Not applicable to Metal package construction: this adapter emits only gfx1201 HSACO. Independent-scale Metal execution remains follow-up required.



## Scalar typed bounded planes — active integration
Owner FRONTEND-IR-MEDIUM-1 / ROCM-FP8-BLOCKSCALE-1;
sync SCALAR-SCALED-PLANE-2026-10-07. A shared native helper derives the bounded
plane profile for rank-two typed FP8 transposed-A or partial scale groups.
Schedule and serialized native program export consume the same derivation;
the semantic Graph retains rank-two operands with no batching attribute.
32 frontend projection cases pass. Matching compiler build, native artifact
checks and exact gfx1201 numerical/timing proof remain pending.
Dynamic/nonleading/composed/storage AD, generic closure and delivery remain open.
Shared semantic assessment: physical execution parity requires this backend's own implementation and exact-device evidence; gfx1201 WMMA schedules do not transfer.



## ROCm version-query metadata reuse — active
Owner FRONTEND-IR-MEDIUM-1 / E2E-REAL-6;
sync ROCM-VERSION-METADATA-2026-10-07. Repeated compiler/driver version queries
now use a bounded 32-entry metadata cache keyed by resolved executable,
device/inode/size/mtime/ctime and loader/toolchain environment. Replacement,
rewrite, symlink, environment, failure and capacity tests pass. Device-library
discovery and content fingerprints remain independently checked.
The focused WSL lane passes 49 tests; 22 ROCm device tests skip on NVIDIA.
Exact gfx1201 scalar regressions and alternating real-package metadata A/B
are running. No warm speedup or wider family closure is claimed yet.
Not applicable to this backend's packaging: only ROCm compiler/driver metadata querying changes; shared artifact contracts are preserved.
