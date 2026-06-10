---
title: Deep Compiler Audit — 2026-06-10
last_updated: 2026-06-10
scope: frontend/API · Graph IR · Schedule/Tile/Target IR · backend manifest · runtime ABI · Apple envelope · benchmark coverage · generated docs · test gates
status: source-backed; two headline gaps fixed in this pass, remainder scoped below
---

# Deep Compiler Audit — 2026-06-10

A full, source-backed pass over the compiler surface. Every claim below was
checked against code, not prose. The headline finding: **generated-doc drift is
clean, but semantic coverage gaps existed** where the audit *model* lagged the
real surface. Two of those were fixed in this pass; the rest are scoped as
prioritized follow-ons.

> **Counts are not repeated here.** The drift-gated dashboards under
> `docs/audit/generated/` are the count authority (Decision #26). This report
> links them and records *structure / coverage-model* findings only.

## Generated drift: clean ✅

All machine-generated dashboards are in sync at audit time:

- `scripts/check_generated_docs.sh` → **16 generated docs in sync**.
- `python -m tessera.compiler.audit support_table --check` → matches.
- `python -m tessera.compiler.audit runtime_abi --check` / `verifier_coverage --check` → matches.
- Apple `.inc` table (`apple_runtime_ops.inc`) in sync with the generator
  (`test_apple_runtime_ops_table_in_sync`); **now committed** (was gitignored —
  fixed 2026-06-09, see commit `685c167`).

**Generated drift ≠ semantic drift.** The dashboards faithfully render whatever
the *registries* say. The gaps below were in the registries / audit models
themselves — the renderer can't catch a stale model.

## Semantic findings (verified against source)

| # | Area | Finding | Owner source | Proof source | Status |
|---|------|---------|--------------|--------------|--------|
| 1 | Benchmark coverage | Support-table `bench` axis was a hard-coded GA/EBM-only frozenset; ignored every other runnable benchmark surface (GEMM, attention, fusion, collectives, Apple hot paths, MegaMoE) despite ≥7 ops carrying a first-class `benchmark_json`. | `audit.py::_BENCH_INVENTORY` | `benchmarks_manifest._ENTRIES` (18 surfaces); `backend_manifest.benchmark_json` | **FIXED** — `benchmark_coverage.py` reads live coverage; bench set 26→39. |
| 2 | Backend manifest | `grouped_gemm` / `moe_swiglu_block` are runtime-envelope ops (lanes exist) with fused MSL kernels + execute-compare fixtures, but had **no `_APPLE_GPU_KERNELS` manifest row**. | `backend_manifest._APPLE_GPU_KERNELS` | `apple_gpu_envelope.APPLE_GPU_LANE_BY_OP`; `_NUMERICAL_FIXTURES` | **FIXED** — both rows added (`fused`, fp32), + target-map symbol/dispatch. |
| 3 | Fused-chain naming | `matmul_softmax` / `matmul_gelu` / `matmul_rmsnorm` / `matmul_softmax_matmul` / `swiglu` are benchmarked but are **not callable ops** (absent from `OP_SPECS`, no standalone symbol). Ambiguous whether they should be manifest rows. | `op_catalog.OP_SPECS` | benchmark harnesses | **DECIDED** — benchmark-only aliases; never support-table rows; their constituents carry coverage. Enforced by `test_benchmark_coverage.py`. |
| 4 | Benchmark schema | Row schema is split across `compiler/benchmark_row.py` (canonical), `benchmarks/common/artifact_schema.py`, and Apple-specific ad-hoc JSON; no single extensibility field for hot-path metadata. | `benchmark_row.py` | `artifact_schema.py`; Apple bench JSON | **OPEN (P1)** — add `hot_path_metadata` to `BenchmarkRow` + `benchmark_metadata` to `BackendKernelEntry`/`AppleKernelDescriptor`. |
| 5 | Apple hot-path metadata | Perf ratchets exist (`perf_gate.py --ratchet`, `apple_gpu_hot_paths.json`) but `benchmark_json` is attached unevenly across manifest rows, descriptors, fused chains, decode-chain paths, and MegaMoE. | `backend_manifest.benchmark_json` | `benchmarks/baselines/apple_gpu_hot_paths.json` | **PARTIAL** — MoE rows now attached; decode-chain / fused-epilogue rows still uneven. |
| 6 | Descriptor consumption | `AppleKernelDescriptor` + `apple_gpu_envelope` are now consumed by runtime dispatch (closed by #61), but Target IR fusion recognition / runtime decisions still don't read a descriptor-backed *benchmark* contract. | `apple_kernel_descriptor.py` | `runtime.py` (lane table) | **OPEN (P2)** — extend descriptor with benchmark metadata; have perf gate read it. |
| 7 | Prose vs metadata | Some prose docs claim "done" beyond what compiler metadata proves (e.g. blanket "all hot paths ratcheted"). | `docs/apple_backend.md`, `distributed_megamoe.md`, bench README | generated dashboards | **OPEN (P2)** — soften prose to link dashboards. |

## What this pass fixed

1. **Bench-coverage accounting (finding #1, #3).** New
   `python/tessera/compiler/benchmark_coverage.py` — manifest-attached
   (`benchmark_json`, read live) ∪ GA/EBM harness inventory ∪ explicit real-op
   map (collectives / GEMM / MHA). `audit.py` now sources the bench axis from
   it. Fused-chain names are documented benchmark-only aliases. `support_table`
   regenerated; `test_benchmark_coverage.py` locks the three contracts.
2. **Manifest blind spot (finding #2).** `grouped_gemm` + `moe_swiglu_block`
   `_APPLE_GPU_KERNELS` rows (`fused`, fp32, runtime symbol, benchmark_json),
   the `moe_swiglu_block` execute-compare fixture, and the `apple_target_map`
   symbol + driver-dispatch entries. Status decision: `fused` (not
   `hardware_verified`, which is reserved for encode-session ops).

## Still open (prioritized)

### P1 — Benchmark schema extensibility (finding #4)
Add a single `hot_path_metadata: Mapping` field to `BenchmarkRow` and a parallel
`benchmark_metadata` to `BackendKernelEntry` + `AppleKernelDescriptor`, so
hot-path group / expected-latency / ratchet-bound ride one open slot instead of
ad-hoc JSON. **Risk:** low (optional fields, default None; thread through
`as_dict` only when present). **Proof:** `test_benchmark_row.py`,
`test_backend_capability_extension.py`, `test_apple_kernel_descriptor.py`.

### P1 — Even hot-path metadata (finding #5)
Attach `benchmark_json` (or the new `benchmark_metadata`) to the remaining hot
paths: fused matmul epilogues (matmul→gelu/rmsnorm/softmax), the decode chain,
and the MegaMoE overlap path. **Risk:** low. **Proof:** `perf_gate.py --ratchet`
coverage assertion + `test_apple_gpu_perf_ratchet.py`.

### P2 — Canonical benchmark rows for MegaMoE + an Apple hot-path smoke (finding #5)
Bring `benchmark_megamoe_overlap.py` into `BenchmarkRow` shape (it currently
emits a bespoke superset) while preserving its CLI; add a hot-path smoke harness
emitting canonical rows for matmul / epilogues / conv2d / decode chain / package
lane / MegaMoE — environment-aware (valid skip row on non-Metal hosts). CI should
assert *required names/metadata*, not fixed row totals.

### P2 — Descriptor-backed benchmark contract (finding #6)
Extend `AppleKernelDescriptor` with the benchmark metadata and have the perf gate
+ Target IR fusion recognizer read it, so benchmark coverage is descriptor-driven
like dispatch now is (post-#61).

### P2 — Prose ↔ metadata alignment (finding #7)
Soften "done" claims in `docs/apple_backend.md`, `distributed_megamoe.md`, and
the benchmark README to link the drift-gated dashboards rather than assert
coverage the metadata doesn't yet prove.

## Test gates exercised this pass

`test_backend_kernel_manifest` · `test_backend_capability_extension` ·
`test_apple_kernel_descriptor` · `test_apple_target_map` ·
`test_apple_gpu_envelope_dispatch` · `test_apple_gpu_tile_pass_status_matches_envelope` ·
`test_op_target_conformance` · `test_benchmark_coverage` (new) ·
`test_benchmark_row` · `test_operator_benchmarks_contract` · `test_surface_audit` ·
`test_benchmark_surface_repair` — all green; mypy/ruff clean; 16 generated docs +
support_table in sync.

## Method note (Decision #27)

Every gap above was confirmed by reading the owning source and the proof source —
not from memory or a single dashboard. The "generated drift clean / semantic gap
open" split is the core lesson: a green drift gate proves the renderer matches the
registry, not that the registry models reality.
