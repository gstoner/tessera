# Mixed aliases and asynchronous owned state

Sync key: MIXED-ALIAS-ASYNC-STATE-2026-09-09.
Owners: W4-PRODUCT-1, W2.4a, AD-RESIDUAL-EVAL-1.

Independent SM120 and gfx1151 packets record three asynchronous owned-state
updates, stable allocation identity and refusal of reads while an update is
pending. The computation completes and validates before event-ordered copyback;
`poll_step()` transfers the independent result frame only after the copy event
completes. Callers must keep the supplied stream alive until completion or
successful close. Close/failure cleanup may synchronize. This is correctness
proof for one state input and one writer, not performance promotion.

CPU/compiler validation: 97 focused source, state, public-result, status and
retirement tests passed using the assertions-enabled compiler and native CPU
JIT on Super-Bear. Plain object fields, read-only overlapping inputs beside
disjoint writable state, cached reuse, functional next-state VJP, and pending/
failed-copyback retention are covered. Package mypy passed.

`vjp` on declared state returns public results followed by next-state results,
and takes corresponding explicit cotangents; it does not copy back into inputs.
Mutable input aliases, object AD and exception transport remain excluded.
Full dynamic exception semantics, writable overlap and arbitrary custom
accessors remain architectural follow-ups in the scoped AD and integrated plans.
