<!-- ===== MERGE_START Tessera Empirical Software Agent ===== -->
# Overview

This document defines a Tessera-flavored version of the paper’s system:
an **LLM-driven program rewriter** inside a **Tree Search (TS)** loop, optimizing a **quality score**
for a **scorable task**.

- **Scorable Task**: any task with a measurable, machine-evaluable metric (public leaderboard, held-out set, or bespoke scientific criterion).
- **Candidate Program**: runnable artifact (script / module / MLIR pass pipeline) that yields a score.
- **Tree Search**: balances **exploration/exploitation** over program variants (nodes).
- **Literature Integration**: injects research ideas from papers/books/search into promptable “patch plans.”

The Tessera mapping adds:
- **IR hooks**: generate/patch **Tile IR** kernels; lower to **Target IR** backends; autotune the kernels.
- **Profiler**: attach rooflines + Perfetto traces to each candidate evaluation.
- **Pass hook**: `-tessera-empirical-search` to embed the loop in `tessera-opt` for IR-level searches.

<!-- ===== MERGE_END Tessera Empirical Software Agent ===== -->
