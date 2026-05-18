# Apple GPU benchmarks

Two benchmark drivers live here:

| Driver | Coverage |
|---|---|
| [`benchmark_fusion.py`](benchmark_fusion.py) | Phase 8.4.x MSL fusion sweep — `matmul → softmax`, SwiGLU MLP block. Fused-vs-sequential pairing. |
| [`benchmark_ga_ebm.py`](benchmark_ga_ebm.py) | GA + EBM end-to-end stack walk. All 17 GA primitives ship fused MSL kernels (`backend=apple_gpu`, `mode=fused`); 2 EBM primitives (`ebm_inner_step`, `ebm_refinement`) ship the first native EBM MSL kernels; the remaining 7 EBM ops run via the Python reference path (`backend=python_ref`, `mode=reference`, `apple_gpu_status=planned`). |

## GA + EBM benchmark — what it walks

For every primitive the driver exercises the full stack:

1. **Python API** — `tessera.ga.*` / `tessera.ebm.*` reference call on seeded input.
2. **Manifest / backend selection** — `backend_manifest.clifford_manifest_for` / `ebm_manifest_for` resolves the `apple_gpu` symbol name and confirms the dispatch slot is `fused`.
3. **Dialect / lowering artifact** — the MSL kernel inside [`apple_gpu_runtime.mm`](../../src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime.mm) referenced by the resolved symbol.
4. **Apple runtime symbol dispatch** — `ctypes` binding into the freshly-compiled dylib.
5. **Real hardware execution** — Metal compute encoder dispatch on the M-series GPU.
6. **Correctness** — max-abs-diff vs the Python reference, gated by per-op tolerance.
7. **Benchmark output** — JSON row, compatible with `tools/roofline_tools/`.

## Status split (2026-05-17)

- **GA**: full Apple GPU end-to-end benchmarked. 17/17 primitives `backend=apple_gpu`, `mode=fused`, each with a manifest-resolved C ABI symbol.
- **EBM**: 2/9 primitives natively benchmarked on Apple GPU (`ebm_inner_step` + `ebm_refinement`); 7/9 stay on the deterministic Python reference path, manifest-marked `apple_gpu_status=planned`.

## Usage

```bash
# Manual run (default 50 reps — enough samples for stable p10/p90):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --output /tmp/ga_ebm.json

# CI-friendly run (2 reps, used by the unit test):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci --output /tmp/ga_ebm.json

# Longer EBT refinement chain (T inner-step iterations on-device):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --refinement-T 32 \
    --output /tmp/ga_ebm.json
```

Skips cleanly on non-Darwin or when `clang++` / the runtime source isn't available — the report still emits the 7 Python-reference EBM rows with `skipped_apple_gpu` set.

## Timing methodology

- **Median is the headline** (`latency_ms`); the row also carries `p10_ms`, `p50_ms`, `p90_ms`, `min_ms`, `max_ms`, `stdev_ms` so consumers can see the spread, not just a single number.
- **One warm-up iteration** per primitive before timed runs — first dispatch pays the Metal pipeline cache miss, which would otherwise dominate small kernels.
- **Compile time is in the envelope, not in rows.** `compile_time_ms` reports the `clang++` wall-clock for the runtime dylib (typically 2–3 seconds on M-series); per-row `latency_ms` is pure dispatch time (sub-millisecond for most ops). Amortize compile cost across runs — do not add it to row latencies.
- **CI uses `--ci` (reps=2)** — small but enough to populate percentiles; manual runs default to 50 reps so p10/p90 carry signal.
- **Apple GPU latencies vary 2–3× run-to-run** at this scale (sub-ms kernels are dominated by host-side overhead). Treat absolute numbers as illustrative; what's stable is the *order of magnitude* and the *correctness verdict*.

## Sample report — reference schema

[`sample_ga_ebm_report.json`](sample_ga_ebm_report.json) is a checked-in snapshot from a single Apple Silicon run (M-series, 20 reps). Headline numbers from that run:

| Op | Backend | latency_ms (median) | p10–p90 |
|---|---|---:|---:|
| `clifford_geometric_product` | apple_gpu / fused | 0.22 | 0.21–0.27 |
| `clifford_rotor_sandwich` | apple_gpu / fused | 0.40 | 0.38–0.48 |
| `clifford_codiff` (3-stage MSL composition) | apple_gpu / fused | 0.69 | 0.65–0.79 |
| `ebm_inner_step` (native) | apple_gpu / fused | 0.24 | 0.22–0.55 |
| `ebm_refinement` (T=8, native) | apple_gpu / fused | 2.16 | 1.78–3.39 |
| `ebm_energy` (Python ref) | python_ref / reference | 0.002 | 0.002–0.003 |
| `ebm_partition_exact` (Python ref) | python_ref / reference | 0.030 | 0.029–0.030 |

These exact numbers will drift across machines and toolchain versions. The **schema** is the stable contract — `tests/unit/test_benchmark_ga_ebm.py` enforces it.

## Schema reference

Per-row fields:

| Field | Type | Notes |
|---|---|---|
| `backend` | str | `"apple_gpu"` for fused MSL, `"python_ref"` for reference |
| `namespace` | str | `"ga"` or `"ebm"` |
| `op` | str | Primitive name (e.g., `clifford_geometric_product`, `ebm_inner_step`) |
| `shape` | str | Human-readable shape descriptor |
| `dtype` | str | `"f32"` for v1 |
| `mode` | str | `"fused"` for native MSL, `"reference"` for Python |
| `reps` | int | Number of timed iterations |
| `latency_ms` | float | Median (= `p50_ms`) — headline |
| `stdev_ms`, `min_ms`, `max_ms` | float | Spread |
| `p10_ms`, `p50_ms`, `p90_ms` | float | Percentiles |
| `max_abs_err` | float | Max-abs-diff vs Python reference |
| `tolerance` | float | Per-op tolerance gate |
| `ok` | bool | True iff `max_abs_err <= tolerance` (and the manifest agrees for native rows) |
| `symbol` | str | C ABI symbol name (native rows only) |
| `apple_gpu_status` | str | Manifest-resolved status (`fused` / `planned`); EBM rows only |
| `device`, `tessera_version` | str | Provenance |

Envelope fields:

| Field | Notes |
|---|---|
| `runs` | List of row dicts |
| `ga_primitives_count` | Count of `clifford_*` rows (17 when GPU available, 0 otherwise) |
| `ebm_paths_count` | Count of `ebm_*` rows (9 when GPU available, 7 otherwise) |
| `ebm_native_apple_gpu_count` | Count of native EBM rows (2 when GPU available) |
| `native_ebm_ops` | Sorted list of EBM ops on `apple_gpu` |
| `compile_time_ms` | clang++ wall-clock for the runtime dylib |
| `skipped_apple_gpu` | Skip reason string (`null` when Apple GPU runs) |
| `device`, `tessera_version`, `reps` | Provenance |
