<!-- ===== MERGE_START Tessera Empirical Software Agent ===== -->
# Benchmarks & Metrics

Standardize **score.json**:
```json
{
  "task": "integrals_solver",
  "score": 0.9123,
  "metric": "mean_log_frac_error",
  "seed": 17,
  "runtime_s": 42.7,
  "artifacts": ["reports/roofline.csv", "reports/perfetto.json"]
}
```
Include **stat sig** helpers (bootstrap CIs), and **leaderboard** aggregation for multi-task suites.

<!-- ===== MERGE_END Tessera Empirical Software Agent ===== -->
