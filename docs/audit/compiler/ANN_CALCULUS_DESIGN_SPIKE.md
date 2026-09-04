---
audit_role: plan
plan_state: landing
last_updated: 2026-09-04
---

# MSW-9 — ANN-calculus design spike

Navigation: [README.md](README.md). Global sequencing remains in
[INTEGRATED_COMPILER_PLAN.md](INTEGRATED_COMPILER_PLAN.md).

Owner: [MATH_SOURCE_WORKSTREAM.md](MATH_SOURCE_WORKSTREAM.md), MSW-9.
The spike is an executable reference oracle plus a concrete integration plan.
It does not promote ANN laws to native evidence or claim a fusion gate exists.

## Source and mathematical scope

[Jentzen, Kuckuck and von Wurstemberger, v3](https://arxiv.org/pdf/2310.20360v3),
Chapter 2: composition (Definition 2.1.1), parallelization (2.2.1), ReLU
identity networks (2.2.6), extension (2.2.9), and same-length sums (§2.4).
The prototype implements affine chains with a common componentwise ReLU or
Swish activation between layers and an affine final layer. Composition merges
two adjacent affine maps; parallelization uses block-diagonal affine maps on
independent input blocks. Shared-input sums explicitly duplicate the input and
sum output blocks. Identity-based extension is restricted to ReLU here.

## Executable spike and findings

Run `python3 tools/ann_calculus_spike.py` (NumPy required). This independent
oracle checks realization under composition, associativity, parallelization,
sums, ReLU identity/extension, and dense parameter-slot arithmetic. It rejects
an unsupported activation and detects mutations to a fused bias and a reported
parameter count. It is intentionally outside the runtime and compiler passes.

The useful result is a distinction that must precede integration: three counts
are different. Mathematical dense parameter slots include zero entries in
block matrices; unique trainable storage excludes shared/frozen buffers; a
fused kernel's embedded constants and live inputs are yet another inventory.
MSW-9 must compare like with like. It cannot require all three counts to agree.
For widths d0,...,dL, dense slots are sum((d[l-1]+1)*d[l]); composition removes
the two boundary affine layers and substitutes their matrix product and bias.
The prototype checks this replacement arithmetic against constructed shapes.

## Integration decision

The current `evaluator.metamorphic_equivalence` runs the **same callable** on
two input tuples and returns inconclusive unless both runs are native. ANN
composition instead compares **two programs**. Do not weaken that evidence
boundary or report the NumPy spike as a passing native metamorphic check.

1. Extract a feed-forward fragment from the existing Graph IR: affine matmul
   and bias, one declared activation, static shapes, immutable parameter
   identity and explicit sharing. Reject attention, normalization, RNG,
   control-flow regions, side effects, mixed activation chains, unknown
   parameter ownership and numerical policies that forbid reassociation.
   This is a view over Graph IR, not a second graph/tracer or an ODS operation.
2. Add a program-pair relation adapter around the existing evaluator machinery.
   Keep native provenance on both outputs. Register a separate reference law
   result for host-free algebra, using the MSW-4 `LawResult` pattern. A native
   result must still be inconclusive if either side is reference-only.
3. Attach the extracted inventory to the actual fusion transformation boundary.
   Compare the post-transform affine shapes and buffer provenance to the
   derived replacement inventory before accepting the transform. A changed
   bias, dropped activation, wrong dimension, stale parameter count, or
   duplicated shared parameter must fail its corresponding mutation test.
4. Start with composition and identity extension; then parallelization and
   sums. Keep identity/enlargement proofs scoped to their supported activation.
   Real-arithmetic equivalence is an oracle; target numeric policy still owns
   finite-precision tolerance and legal reassociation.

## Estimate and acceptance

Three independently reviewable implementation slices: (A) fragment extraction
and inventory, (B) program-pair relation and law registry, (C) fusion consumer
and mutation suite. A/B are host-free contracts; C needs each affected backend's
existing exact-device lane before native promotion. Estimate by these slices,
not a blanket claim that the algebra prototype closes MSW-9. No new lowering
registry, backend candidate, or production execution path is introduced here.

## 2026-09-04 engineering integration

Slice B now has a production evaluator entry point:
`evaluator.program_pair_equivalence` runs distinct programs through the
existing `run_native` boundary. The original same-program metamorphic API
delegates to it. Either reference/fallback side remains inconclusive; finite
nonnegative tolerances are required. Composition bias mutation and all four
native/reference combinations are covered by synthetic-provenance tests,
which prove adapter behavior rather than hardware execution.

`ann_calculus` is now a separate reference law family in `run_law_sweep`,
covering affine composition and ReLU identity extension across three width
sets. Generated law evidence labels these as reference algebra. Slice A
(Graph IR fragment and immutable/shared parameter inventory) and slice C
(fusion-boundary consumer and exact-device mutation evidence) remain open.
The spike itself has not become a product execution path.
