
# Metric Definitions (Stable Schema)

Every result row follows:
```json
{
  "bench_id": "micro.gemm",
  "variant": "f32_mxr_1024x1024",
  "operator": "gemm",
  "dtype": "f32",
  "shape": "1024x1024x1024",
  "target": "cpu",
  "compiler_path": "tessera_jit_cpu",
  "runtime_status": "executable",
  "throughput_flops": 1.23e12,
  "latency_ms": 4.2,
  "bytes_per_s": 7.5e11,
  "correctness_max_error": 0.0,
  "artifact_graph": true,
  "artifact_schedule": true,
  "artifact_tile": true,
  "artifact_target": true,
  "telemetry": {
    "schema": "tessera.telemetry.v1",
    "source": "tessera_superbench",
    "op": "matmul",
    "status": "ok"
  },
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
- `compiler_path` — `tessera_jit_cpu`, `graph_ir_only`, `artifact_only`,
  `reference`, or another explicit path.
- `runtime_status` — `executable`, `artifact_only`, `mock`,
  `backend_unavailable`, `skipped`, or related backend status.
- `artifact_*` — Graph/Schedule/Tile/Target availability plus artifact hash.
- `telemetry` — shared `tessera.telemetry.v1` event consumed by reports and
  perf gates.

Suite-level `results.json` also includes `telemetry` and `telemetry_summary`
arrays built from row events.
