---
last_updated: 2026-09-07
audit_role: reference
---

# FORGE assessment — consolidated ownership

This is a historical assessment, not an independent implementation queue.
The full mathematical derivation and negative examples are preserved below.

- W1/W2 locality and residency use LAYOUT-ALG-1 and the existing Schedule/Tile
  ownership/materialization machinery; do not introduce a parallel lattice.
- W3/W4/W7/W8 stateful fusion, clipping/routing guards and producer-consumer
  epilogues belong to foundation F3 / W5.1–W5.2. Native IR must preserve layout,
  numeric policy, effects, optimizer-state order and full reduction semantics.
- W5 precision realizability belongs to NUMPOL-CARRIER-1 / FA-1.
- W6 affine reduce-into-state belongs to DIST-NATIVE-1 with multi-rank proof.

Algebraic equivalence does not make admission trivial: rounding, optimizer
mutation order, aliasing, tied weights and distributed partials must be checked
before target-specific measurement. Archiving the proposal does not close any
of those residuals.

Global order and acceptance gates: [integrated math/foundation reconciliation](INTEGRATED_COMPILER_PLAN.md#math-audit-foundation-reconciliation--2026-09-07).

[Read the historical assessment](archive/FORGE_ASSESSMENT.md).
