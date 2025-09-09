<!-- ===== MERGE_START Tessera Empirical Software Agent ===== -->
# Architecture

```mermaid
flowchart LR
    A[Task Spec + Metric] --> B[Idea Bank (papers/search/user notes)]
    A --> C[Seed Program]
    B --> D[LLM Rewriter]
    C --> D
    D --> E[Tree Search Controller]
    E --> F[Sandbox Executor]
    F --> G[Scorer]
    G --> H{Score}
    H -->|UCB/PUCT| E
    F --> I[Artifacts: IR, logs, traces]
    I --> J[Tessera Autotuner + Profiler]
```

**Key modules**
- `TreeSearchController`: PUCT/UCB policy, beam width, novelty filters, dead-end pruning.
- `LLMRewriter`: prompt templates → patch plans → code edits (file-level + IR-level).
- `SandboxExecutor`: runs candidates hermetically; enforces wallclock/mem; captures artifacts.
- `Scorer`: computes standardized metrics; returns scalar score with metadata.
- `TesseraBridge`: emits IR variants; compiles/runs on target backends; collects profiler data.

<!-- ===== MERGE_END Tessera Empirical Software Agent ===== -->
