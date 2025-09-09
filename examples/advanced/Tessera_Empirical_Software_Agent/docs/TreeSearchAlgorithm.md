<!-- ===== MERGE_START Tessera Empirical Software Agent ===== -->
# Tree Search Algorithm (PUCT-Style)

- Nodes represent **program versions** (git-like state: code + config + IR patches).
- Edges are **rewrite actions** (e.g., change model, add regularizer, switch optimizer/kernel).
- Selection via **PUCT** with score normalization and novelty bonus.
- Expansion by sampling **K** candidate rewrites from the LLM (temperature ladder).
- Rollout/evaluation in the sandbox; backpropagate scalar scores.

Pseudo:
```python
while budget_left():
    n = select_node(root, policy="puct+novelty")
    proposals = llm.propose(n.state, top_k=K)
    for p in proposals:
        child = n.apply(p)
        score, meta = evaluator.run(child)
        backpropagate(child, score)
    prune_stale()
```

<!-- ===== MERGE_END Tessera Empirical Software Agent ===== -->
