# Apple native backward package reports — 2026-09-07

Five complete-package versus direct-ABI reports for the native attention
backward package (`selection_scope: package_subgraph`,
`package_kind: native_image_launch_descriptor`) on the Apple M1 Max, with the
strict-v2 route ledger derived from them.

| File | Holds |
|---|---|
| `run-1.json` … `run-5.json` | Five `benchmark_native_backward_package.py` reports (`schema_version`, `selection_scope`, `package_kind`, `context` with device/OS/SDK and compiler/runtime fingerprints, `runs`). |
| `strict_ledger.json` | `tessera.apple.route-ledger.v2`: decisions, ineligible decisions, promotion rules, `measured_at`/`expires_at`, and the digests of the five source reports. |
| `summary.json` | Per-row summary with `package_promoted: false`, scope "native_image_launch_descriptor, not Metal ML package", `device_timing_claim: false`. |

Recorded by `benchmarks/apple_gpu/benchmark_native_backward_package.py`
(`run-N.json`). `strict_ledger.json` is the strict-v2 form that
`benchmarks/apple_gpu/seal_strict_route_ledger.py` seals from raw reports; the
packet does not record the sealing invocation. `summary.json`: recorder not
found in tree.

Cited by `docs/audit/compiler/INTEGRATED_COMPILER_LOG.md`
("package reports and ledger").
