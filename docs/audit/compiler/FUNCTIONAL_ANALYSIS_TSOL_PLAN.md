---
last_updated: 2026-09-06
audit_role: reference
---

# Functional-analysis contracts — historical routing

Remaining FA-1–FA-7 work, dependencies and acceptance gates now live in the
[integrated compiler queue](INTEGRATED_COMPILER_PLAN.md#numpol-carrier-1).
Adjoint-law follow-ups belong to [the active AD plan](AUTODIFF_EXECUTION_PLAN.md);
recurrence stability belongs to [the sequence-mixer plan](SEQUENCE_MIXER_ENGINEERING_PLAN.md).

The [original mathematical design](archive/FUNCTIONAL_ANALYSIS_TSOL_PLAN.md)
preserves models M1–M7 and the verification log. Archiving supersedes the old
queue; it does not declare numerical error contracts or native execution complete.

## Acceptance contracts

Carried forward from the integrated reconciliation before archival. This table
retains mathematical gates and scoped ownership; it does not order delivery.
Current numerical-consumer work is NUMPOL-CARRIER-1 in the live queue.

| Item and owner | Remaining work and dependencies | Acceptance gate |
|---|---|---|
| **FA-1 — numerical legality / evaluator and arbiter** | Extend the landed bounded reduction and ANN consumers to broader native candidates. Extend NUMPOL-CARRIER-1 semantics with explicit norm, admissible domain, shape/reduction length and absolute error budget; preserve analysis facts across native MLIR boundaries. No independent Python production lowering stack or unused coverage axis. | A real candidate is accepted/refused using the budget. Composition checks perturbed intermediate-domain containment and uses justified downstream Lipschitz constants. Reduction bounds enforce their accumulation-algorithm, Ku < 1 and under/overflow assumptions. Unknown bounds fail closed for budget-based promotion. Oracle evidence, analytic bounds and device measurements stay distinct; norm-to-tolerance conversions are explicit. |
| **FA-2 — AD-LAW / AD-CLOSEOUT-1** | Reuse the implemented adjoint and canonical-forward laws; carry only public debug adapter, norm-aware tolerance and derivative-coverage evidence integration into AUTODIFF_EXECUTION_PLAN.md. Does not wait on invention of FA-1 or a new harness. | Planted incorrect and matched-incorrect derivative pairs remain detected; coverage changes cite the actual law result and evidence tier. Public adapter tests exercise the existing engine. Amend the coverage contract explicitly before tightening automatic status transitions. |
| **FA-3 — spectral family / numerical legality** | Audit existing spectral laws, then fill normalization, window/domain and multiplier-bound gaps. Error-budget consumption depends on FA-1; independent oracle coverage does not. | FFT normalization and adjoints agree; STFT/ISTFT round trips declare window/overlap assumptions. A spectral transformation consumes a justified multiplier bound, with negative legality cases and native-boundary preservation. Existing adjoint tests alone do not close this consumer. |
| **FA-4 — sequence-mixer stability** | Sequence-mixer plan owns recurrence/discretization-specific certificates and eventual op wiring. No new generic certificate registry without a recurrence consumer. | Domain and timestep assumptions are checked; nonnormal transient amplification and discretization-specific stability are covered. Reference oracle and native-device execution remain separate evidence. |
| **FA-5 — functional-calculus admission, deferred** | Require a named workload and a specific missing operation before reopening broad admission. Reuse spectral/solver owners; not a prerequisite for FA-1/2/3/6. | An admission proposal names semantics, domain, derivative rules, native producer/consumer and measurable benefit. An abstract common interface is insufficient. |
| **FA-6 — approximation legality / arbiter, consumer-gated** | After FA-1, attach norm-specific truncation bounds to an actual low-rank substitution candidate. Preserve the distinction between exact factorization and approximation. | Candidate selection respects the caller's budget and composition domain; over-budget substitutions reject. Sampled estimates cannot masquerade as certified upper bounds, and no automatic tolerance relaxation is allowed. |
| **FA-7 — PDE/forms, deferred** | Reopen under PDE_STENCIL_CAPABILITY_PLAN.md only when a solver or rewrite needs coercivity. | Concrete discrete operator, boundary/domain hypotheses and a certificate-consuming solver or transformation. |

Order: FA-1's bounded consumer first; FA-3/FA-6 numerical promotion builds on
it. Existing AD and spectral law improvements proceed independently. FA-4 is
sequenced by its domain owner; FA-5/FA-7 are explicitly deferred. Introduce
registry/schema changes together with their first consumer and focused drift
gates. CUDA, ROCm, Apple and x86 each require their own lowering-preservation
and exact-device evidence before hardware promotion; this consolidation changes
no backend support state or physical schedule.
