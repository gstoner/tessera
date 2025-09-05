
# Metric Definitions (Stable Schema)

Every result row follows:
```json
{
  "bench_id": "micro.gemm",
  "variant": "f32_mxr_1024x1024",
  "tags": ["micro","compute","cpu"],
  "metrics": {
    "throughput_flops": 1.23e12,
    "latency_ms": 4.2,
    "bytes_per_s": 7.5e11,
    "p95_ms": 5.0
  },
  "numerics": {
    "max_abs_err": 0.0,
    "max_rel_err": 0.0
  },
  "env": { "...": "captured by runner" },
  "ok": true,
  "skip_reason": null
}
```

Key metrics:
- `throughput_flops` — floating‑point operations per second (FLOP/s).
- `bytes_per_s` — sustained memory/system bandwidth.
- `latency_ms`, `p95_ms`, `p99_ms` — latency statistics.
- `efficiency_pct` — achieved / theoretical peak × 100.
- `ok` — benchmark ran and produced results; `skip_reason` if not.
