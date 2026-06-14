---
name: validate-correctness
description: >
  Validate that a code change preserves training correctness by comparing
  per-step loss curves between two branches (or two runs). Use after a system
  optimization or refactor — "validate correctness", "compare loss curves",
  "did my change break training", "check parity before merge".
triggers:
  - "validate correctness"
  - "compare loss curves"
  - "did this change preserve training"
  - "check training parity before merge"
---

# Skill: validate-correctness

Compare per-step training loss between a **base** and a **feature** run and emit
a reproducible PASS/FAIL verdict — not a self-assessment. This is the natural
follow-up after any throughput/system optimization: prove the loss curve is
unchanged within tolerance.

## Specific scope

In scope: running N training steps on each of two branches/configs, logging the
per-step loss, and diffing the curves with a numeric tolerance. Out of scope:
downstream-accuracy eval, multi-node runs.

## Prerequisites (fail fast if missing)

1. A short, deterministic training entrypoint that writes one loss value per
   step to a log file (one float per line, or `step<TAB>loss`). The
   `tessera.train.loop` losses are deterministic under a fixed seed.
2. Determinism pinned: `@jit(deterministic=True, seed=...)` and a fixed data
   seed, so base-vs-base is bitwise/near-bitwise identical (sanity check below).
3. Both branches checked out (use a git worktree so the base tree is untouched).

## Procedure

1. **Worktree the base branch** so its tree is isolated:
   `git worktree add ../tessera-base <base-ref>`.
2. **Run N steps on each branch**, writing logs:
   - base:    `... > base.log`
   - feature: `... > feat.log`
   (Each line: a single loss float, or `step<TAB>loss`.)
3. **Sanity-check determinism**: a base-vs-base run must compare clean at a
   tight tolerance, or the comparison is meaningless — fix nondeterminism first.
4. **Compare**:
   ```bash
   PYTHONPATH=python python3 \
     python/tessera/train/skills/validate-correctness/scripts/compare.py \
     base.log feat.log --rtol 2e-2 --atol 1e-3
   ```
5. **Clean up**: `git worktree remove ../tessera-base`.

## Verifiable success

`compare.py` exits **0** and prints `PASS` iff every step's losses agree within
tolerance (and the two logs have equal length). It exits **1** / prints
`FAIL: step <i> ...` at the first divergence. Report the script's verdict.

## Anti-patterns

- ❌ Eyeballing two curves and declaring "looks the same".
- ❌ Comparing runs with different seeds or nondeterministic kernels.
- ❌ Loosening `--rtol` until it passes — investigate the divergence instead.
