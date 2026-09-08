---
last_updated: 2026-09-07
audit_role: reference
---

# Capability-plan status and sequencing — August 2026

Historical record, not a current queue or execution claim. Current ownership is
[in the integrated plan](../INTEGRATED_COMPILER_LOG.md#2026-09-07--capability-plan-reconciliation). Mathematical contracts retained here
are not new device or performance evidence.

## Block AttnRes original introduction

**Status:** plan (2026-08-12). **Source paper:** Attention Residuals, arXiv 2603.15031
(Kimi Team / MoonshotAI). Official repo ships no code; no faithful public
implementation of Block AttnRes exists (survey in §5 of this doc's provenance
review, summarized in Appendix B). Gaps the paper leaves open are filled here
and marked **[GAP-n]** with the choice justified.

**Why ROCm first:** core compiler work is routed to the Strix Halo box
(`INTEGRATED_COMPILER_PLAN.md` §6a), which is the only fleet machine with an
executing non-Apple GPU lane (gfx1151 matmul + flash-attention family via
`runtime.launch()`), and ROCm is a lead performance target whose ceiling shared
infra must not cap (Decision #28). The mathematical contract lands host-free
first (Phases 0–2 run anywhere); the first hardware proof lands on gfx1151.


## EGGROLL proposed queue row

## 8. Suggested `MASTER_AUDIT.md` queue row

> **[P1] Gradient-free / Evolution-Strategies track (EGGROLL).** New primitive
> `es_low_rank_correction` + reference tier + moment-free optimizer. Op contract
> proof-backed (`EGGROLL_SUPPORT_PLAN.md`); oracles green. Drives P0 operator
> wins O1 (numeric_policy carrier), O2 (saturating requantize), O3 (shared-operand
> batched GEMM). W1 host-free first; W4 distributed is Phase G/H.

## Game theory original first-build advice

## 10. What to build first

If only one thing lands: **G1's `subset_zeta`/`subset_mobius` +
`semivalue`**, with oracles 1–6. It is self-contained, needs no hardware, is
exactly testable, gives `transpose_rule` a real consumer, and produces the
butterfly region class that G5's arbiter lane and G6's sharding both build on.
`boltzmann_value` is the natural second, because it is the one that reuses the
online-softmax emitter and therefore reaches an executing GPU lane soonest.

**G1b is the highest-value item that outlives game theory**, and it is the one
piece here that would still be worth building if the game-theory surface were
cancelled tomorrow: it consolidates butterfly tiling for the spectral FFT lane
too. But it is deliberately *second*, not first — the Decision #31 ordering
caveat says do not collapse a duplication before the surviving path can carry
what the deleted one carried, and until G1 exists there is only one butterfly
consumer and therefore nothing to consolidate.

---
