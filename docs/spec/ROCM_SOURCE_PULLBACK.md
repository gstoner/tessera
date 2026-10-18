---
audit_role: reference
last_updated: 2026-10-18
---

# Native ROCm source pullback

Owner: W1.1 / FRONTEND-IR-MEDIUM-1 / AD-RESIDUAL-EVAL-1 / E2E-REAL-6.
Sync: ROCM-SOURCE-PULLBACK-20261018.

## Semantic and native boundary

The static affine-free FP16/BF16 RMSNorm → ReLU → FP32-output matmul
source VJP consumes the paired AD graph. Native SSA outlining copies the
compiler-derived mixed contraction and row expression, retaining original
root indices, seed storage, private intermediate and return lineage.
No derivative formula is rebuilt by packaging.

The cotangent stays FP32 through the ordered contraction. Its result stores
to the half type selected by the AD graph. The row body preserves explicit
Graph casts, FP32 accumulation, normalization epsilon and activation behavior.

Graph MLIR → paired AD → native member Graphs → Schedule MLIR → typed-view
Tile MLIR → ROCm Target → native GPU/ROCDL/LLVM → HSACO → checked C++ owner
is the production package route.

## Schemas and frame ownership

- Existing SM120 row schemas 1 and 6 retain their meanings.
- Row schema 7 is a single-output static gfx1151/gfx1201 FP16/BF16 row body.
- Attention schemas, including saved-LSE plus bounded schema 5, are unrelated
  namespaces and are unchanged.
- The compiler exports scalar and row frames with physical storage, shapes,
  entry identity, scalar arguments and geometry.
- The row adapter checks compact memref sizes, zero offsets, unit strides,
  dimensions and workgroup size before cloning the carried numerical SSA.
- Native physical-image identity excludes the scalar launch frame.
- Both members execute under the existing native HIP owner. Original roots
  are readonly; member zero writes private scratch, member one reads it and
  writes a separate returned generation.

The checked source package retains native source/paired/member Graphs, every
lowering stage, compiler/toolchain identities, byte hashes and member ABI
records. Portable validation uses those records without compiler access.
Integrity hashes detect mutation; they do not authenticate arbitrary code.

## Public call and replay ownership

Ordinary native_backward on exact rocm_gfx1151 or rocm_gfx1201 JIT targets
uses the shared composite family dispatch and differential frontend certificate.
The native source Graph determines the derivative and both member frames.
A retained owner updates all three original roots and completes invoke/read;
warm calls neither compile nor deserialize and prepare a new owner.
The common RuntimeArtifact dispatch borrows that live owner when available;
after closure, a validated portable package prepares a temporary native owner.
Parent Graph, exact target, argument names, placement and image integrity are
checked before provider access. Forked or closed retained owners are refused.
Returned host arrays own independent snapshots.

The input contract is two compactable host FP16/BF16 roots and one FP32 seed.
This route does not admit a generic rocm alias, resident roots, dynamic shapes,
arbitrary layouts, fused matmul epilogues or encoded operands.

## Current proof boundary

Each owning AMD host passes 39 device cases: 27 ordinary public and 12 package
cases. Coverage includes both storage types and root permutations, changed
roots, retained outputs, FP32 seed cancellation, valid-root malformed-frame
refusals, mutated parent refusal and fresh-process replay with missing compiler
paths. Static M/N/K profiles include 17/19/35, 1/1/1, 3/7/1, 129/33/65 and
5/67/257. Finite Gaussian inputs avoid storage-underflow activation ambiguity;
this is not a general nonfinite or exhaustive-rounding AD certificate.

The final device gate passes 106 cases per AMD host (39 composed/public
and 67 scalar regressions). The full CPU lane passes 30,706 with 3,724 skips;
the production compiler lane passes 1,983 with zero skips. RTX5070 parity
passes 167 cases and retains forty identical native member-stage digests.
Ruff, the zero-error mypy ratchet, all five surface audits and claim lint pass.

The source-bound packet is
[rocm_source_pullback_20261018](../../benchmarks/baselines/rocm_source_pullback_20261018/README.md).
It retains eight compressed portable programs, per-element FP64 error-budget
checks, raw compiler/library identities and distinct timing domains.
Named-profile warm public calls measure 1.20–1.58 ms, prepared calls
0.38–0.46 ms, and portable calls 11.54–13.41 ms. Native event windows measure
9.66–17.71 us and may include enqueue gaps. First public calls follow image
priming. No isolated-kernel or speedup promotion follows these values.

Public CI/review remain required. Broader numerical regimes and AD,
dynamic/resident/arbitrary layouts, representative-model quality and all five
aggregate slices remain open.
