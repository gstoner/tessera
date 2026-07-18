# Apple GPU benchmarks

Benchmark drivers live here:

| Driver | Coverage |
|---|---|
| [`benchmark_fusion.py`](benchmark_fusion.py) | Phase 8.4.x MSL fusion sweep — `matmul → softmax`, SwiGLU MLP block. Fused-vs-sequential pairing. |
| [`benchmark_package_lane.py`](benchmark_package_lane.py) | **PK8e** — `package` lane (`@jit(dispatch_via_package=True)`, MTL4 ML-encoder dispatch of a Tessera-authored `.mtlpackage`) vs `live` lane (MPS/MSL envelope), same jitted fn. Steady-state row pair + `cold_author_ms`. See "Package-lane findings" below. |
| [`benchmark_route_characterization.py`](benchmark_route_characterization.py) | Fresh-process route evidence for comparable native pairs: MPSGraph vs handwritten MSL softmax, MPS vs `simdgroup_matrix` f32 matmul, and MPS vs cooperative-tensor fp16 matmul. Produces selector-compatible JSON. |
| [`benchmark_attention_backward.py`](benchmark_attention_backward.py) | APPLE-ATTN-BWD-1 paired native `serial_recompute`, `atomic`, and `split_reduced` dQ/dK/dV characterization over f32/f16/bf16 MHA/GQA/MQA, bias/window/softcap, ragged, and long-context cases. [`select_stable_attention_backward_routes.py`](select_stable_attention_backward_routes.py) builds the committed Apple7 two-domain ledger. |
| [`benchmark_resident_paged_kv.py`](benchmark_resident_paged_kv.py) | APPLE-PAGED-KV-1 paired `staged` (two resident gathers plus dense attention) versus single-dispatch `direct` physical-page-table MLA attention. Two independent runs retain GPU and end-to-end domains separately. |
| [`benchmark_resident_replay.py`](benchmark_resident_replay.py) | APPLE-REPLAY-1 paired stateless `fused_block` versus persistent `resident_ring` serving. The report requires native/correct rows in both runs before selecting independently in GPU and end-to-end domains. |
| [`benchmark_resident_replay_flush.py`](benchmark_resident_replay_flush.py) | APPLE-REPLAY-1 deterministic resident checkpoint fold versus the shared CPU oracle, with measured device and end-to-end flush domains kept separate. |
| [`benchmark_legacy_retune.py`](benchmark_legacy_retune.py) | APPLE-RETUNE-1 paired grouped-GEMM, MoE, reduction, KV-residency, MLA, and Replay-decode corpus. It emits the strict APPLE-ROUTE-1 ledger while preserving unavailable multi-dispatch/mapped-memory device intervals. |
| [`benchmark_encode_session.py`](benchmark_encode_session.py) | R2 command-buffer batching — a dependent bmm chain per-op (one sync/op) vs. batched into one command buffer. |
| [`benchmark_gumiho.py`](benchmark_gumiho.py) | Gumiho speculative decoding — algorithmic tokens-per-target-pass + resident vs per-op serial draft wall-clock. |
| [`benchmark_ga_ebm.py`](benchmark_ga_ebm.py) | GA + EBM end-to-end stack walk **plus workload mode**. 17 GA primitives + **9 native EBM primitives** (including `ebm_partition_exact`) + Python-reference comparison rows + **4 workload rows** (GA feature pipeline + EBT-tiny refinement, each in apple_gpu + python_ref variants), plus opt-in `--ebt-sweep`. |

## Package-lane findings (PK8e, 2026-06-02)

`benchmark_package_lane.py` answers "when does the authored-`.mtlpackage`
execution lane beat the live MPS/MSL lane?" — steady-state (warm), fp32,
50 reps, Apple Silicon / macOS 26. Ratio = package ÷ live latency (lower =
package wins):

| op | shape | live ms | package ms | package/live |
|---|---|--:|--:|--:|
| matmul | 64×64×64 | 0.27 | 0.38 | 1.43 |
| matmul | 256×256×256 | 0.33 | 0.93 | 2.81 |
| matmul_softmax | 8×16×32 | 0.58 | 0.36 | 0.62 |
| matmul_softmax | 64×64×64 | 1.05 | 0.38 | 0.36 |
| matmul_softmax | 256×256×256 | 12.76 | 0.93 | **0.07** |

**Decision guidance:**
- **Single matmul → use the `live` lane.** MPS GEMM is already optimal; the
  MTL4 ML-encoder adds ~1.3–2.8× per-dispatch overhead with no fusion to win
  it back.
- **Fused multi-op chains (e.g. `matmul → softmax`) → use the `package`
  lane**, increasingly so with size: the MPSGraph package fuses + optimizes
  the whole graph, while the live MSL softmax over a materialized score matrix
  scales poorly (≈**14× faster** at 256×256×256).
- **Cold cost:** authoring + compile + prepare is a one-time ~11 ms per shape
  (`cold_author_ms`), amortized across repeated same-shape calls via the
  per-shape pipeline cache. Worth it for hot, fixed-shape inner loops; not for
  one-shot calls.

(Sample: [`sample_package_lane_report.json`](sample_package_lane_report.json).)

## Characterization-driven route promotion

`benchmark_package_lane.py` now emits an evidence-bearing route record for
each warm measurement: exact op/shape/dtype/device, route, latency, native
dispatch proof, and numerical validation. To let generic
`@jit(target="apple_gpu")` package-auto routing consume a freshly measured
report, opt in with both:

```bash
TESSERA_APPLE_GPU_PACKAGE_AUTOROUTE=1 \
TESSERA_APPLE_GPU_ROUTE_CHARACTERIZATION=/absolute/path/apple_gpu_package_lane.json
```

The selector promotes only a fused chain whose exact-shape package row and
live incumbent row are both native and numerically validated, and only when
the package latency wins. An old snapshot without this proof schema is
intentionally advisory-only; it cannot change compilation.

## Gumiho speculative-decoding benchmark

[`benchmark_gumiho.py`](benchmark_gumiho.py) emits the standard JSON schema with
two op groups:

- `op="gumiho_decode"` — **tokens committed per target forward pass** for
  `vanilla` (1.0), `speculative_untrained`, and `speculative_trained` (rows
  carry `tokens_per_step`, `speedup_vs_vanilla`, `mean_accepted_length`,
  `tokens_per_sec`). This is the model-size-independent speculative win. On a
  recent M-series run: untrained ~1.86×, trained ~4.33× tokens/pass; trained
  also wins ~2.3× on wall-clock throughput because 4.3 tokens/step amortizes the
  verify.
- `op="gumiho_serial_draft"` — `resident` (one command buffer per token) vs.
  `per_op` serial-draft wall-clock; `speedup_vs_per_op` ~2.6× from removing the
  per-op CPU↔GPU sync. Headline numbers drift across machines/toolchains — the
  schema is the stable contract.

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

- **GA**: full Apple GPU end-to-end benchmarked. 17/17 primitives `backend=apple_gpu`, `mode=fused`.
- **EBM (native)**: **9/9 primitives** — `ebm_inner_step`, `ebm_refinement`, `ebm_langevin_step`, `ebm_decode_init`, `ebm_bivector_langevin`, `ebm_sphere_langevin`, `ebm_self_verify`, `ebm_energy` (quadratic specialization), and `ebm_partition_exact` (single-dispatch stable logsumexp, 2026-05-17). `ebm_bivector_langevin` reuses the `ebm_langevin_step` kernel on grade-projected inputs.
- **EBM (Python ref)**: every native EBM op also emits a `python_ref` row so the native-vs-reference speedup is visible per op.
- **Workloads**: 2 small composite chains — `ga_feature_pipeline` (`exp → rotor_sandwich → norm`) and `ebt_tiny_refinement` (fused single-dispatch kernel `ebm.ebt_tiny`). Both emit `apple_gpu` + `python_ref` rows; native rows carry a `dispatched_on_gpu` proof bit so a silent fallback can't masquerade as native.
- **EBT-tiny break-even sweep**: opt-in via `--ebt-sweep`. After the **streaming closed-form fused kernel** (no register-vector spill — works for any `D`, `K ≤ 256` is the only remaining bound): the sweep table includes a `status` per shape (`native_dispatched` vs `degraded_fallback`) and only computes `speedup` when the native attempt actually fired on-device. On a recent M-series run: **first native win at `B=16,K=32,D=128/T=8` (~1.1×); peak ~55× at `B=64,K=128,D=1024/T=256`**. Headline numbers will drift across machines + toolchain versions — the schema (with the dispatch proof + status) is the stable contract.
- **Integration via `tessera.ga.*` / `tessera.ebm.*`**: **17 / 17 GA + 9 / 9 native EBM** ops route through [`tessera._apple_gpu_dispatch`](../../python/tessera/_apple_gpu_dispatch.py) transparently — every user-visible call with a fused kernel takes the GPU path when its inputs match the manifest contract. The workload benchmarks all call public APIs.
- **JIT / compiler bridge**: [`tessera.compiler.jit_bridge`](../../python/tessera/compiler/jit_bridge.py) is the single hop between public APIs and the runtime. **All 26 fast paths** (17 GA + 9 native EBM) route through `dispatch_via_manifest` — manifest resolves `(op_name, target) → symbol`, the shared loader binds it, a `JitBridgeRoute(op, target, status, symbol, context, latency_ms)` row appends to the thread-local trace when tracing is on. The native EBM primitive rows + JIT-bridge benchmark rows use the trace as their proof-of-dispatch bit. `ebm_bivector_langevin` reuses the langevin_step kernel but tags itself in the trace via the helper's `bridge_op_name` kwarg.

## Workload mode — beyond per-primitive timing

The two workloads exist because per-primitive timing tells only half the story: small primitives are dominated by host overhead, so the win-or-lose verdict only becomes meaningful once a real pipeline strings several kernels together. The workload rows make this explicit:

| Workload | Stack | Headline behavior |
|---|---|---|
| `ga_feature_pipeline` | `clifford_exp → clifford_rotor_sandwich → clifford_norm` on a batch of 32 | Native wins decisively (an order of magnitude) — the chain is arithmetically dense + the Python reference goes through `Multivector` object construction per element. |
| `ebt_tiny_refinement` | `ebm.ebt_tiny` — one Metal dispatch that runs the entire T-step refinement, per-row energy, and K-way argmin in registers (with the 2026-05-17 streaming closed-form rewrite, any `D`; `K ≤ 256`). | At the default tiny shape `(B=4, K=8, D=6, T=8)` numpy wins (~0.2 ms dispatch floor beats microsecond closed-form numpy). The `--ebt-sweep` ladder shows the crossover — see the break-even sweep section. |

The native row's `dispatched_on_gpu` bit is the proof: if the public-API guard rejects the input (e.g., `K > 256`), the row is degraded to `backend="python_ref"` + `mode="reference_chain_fallback"` + `ok=False` rather than silently misleading consumers.

## Usage

```bash
# Manual run (50 reps default — enough samples for stable p10/p90):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --output /tmp/ga_ebm.json

# CI-friendly run (2 reps, what the unit test uses):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci --output /tmp/ga_ebm.json

# Workload-only mode (skip per-primitive rows):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --workloads-only

# Primitives-only mode (skip composite workloads):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --primitives-only

# Longer EBT refinement chain (T inner-step iterations on-device):
python benchmarks/apple_gpu/benchmark_ga_ebm.py --refinement-T 32

# Break-even sweep — adds 7 (B, K, D, T) sweep points + envelope summary:
python benchmarks/apple_gpu/benchmark_ga_ebm.py --ebt-sweep
```

Skips cleanly on non-Darwin or when `clang++` / the runtime source isn't available — the report still emits Python-reference EBM rows + Python-reference workload rows with `skipped_apple_gpu` set.

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
| `ebm_inner_step` (native) | apple_gpu / fused | 0.26 | 0.22–0.59 |
| `ebm_refinement` (T=8, native) | apple_gpu / fused | 2.16 | 1.78–3.39 |
| `ebm_langevin_step` (native) | apple_gpu / fused | 0.25 | 0.23–0.46 |
| `ebm_sphere_langevin` (native) | apple_gpu / fused | 0.27 | 0.24–0.42 |
| `ebm_self_verify` (native hard argmin) | apple_gpu / fused | see sample JSON | see sample JSON |
| `ebm_energy` (native quadratic specialization) | apple_gpu / fused | see sample JSON | see sample JSON |
| `ga_feature_pipeline` (workload, native) | apple_gpu / fused_chain | 0.73 | 0.69–0.81 |
| `ga_feature_pipeline` (workload, ref) | python_ref / reference_chain | 9.57 | 9.10–10.3 |
| `ebm_partition_exact` (native stable logsumexp) | apple_gpu / fused | see sample JSON | see sample JSON |
| `ebm_partition_exact` (Python ref) | python_ref / reference | see sample JSON | see sample JSON |

These exact numbers will drift across machines and toolchain versions. The **schema** is the stable contract — `tests/unit/test_benchmark_ga_ebm.py` enforces it.

## Schema reference

Per-row fields:

| Field | Type | Notes |
|---|---|---|
| `backend` | str | `"apple_gpu"` for fused MSL, `"python_ref"` for reference |
| `namespace` | str | `"ga"`, `"ebm"`, or `"workload"` |
| `op` | str | Primitive or workload name |
| `shape` | str | Human-readable shape descriptor |
| `dtype` | str | `"f32"` for v1 |
| `mode` | str | `"fused"` (single MSL kernel), `"reference"` (Python), `"fused_chain"` (multi-kernel workload), `"reference_chain"` (Python workload) |
| `reps` | int | Number of timed iterations |
| `latency_ms` | float | Median (= `p50_ms`) — headline |
| `stdev_ms`, `min_ms`, `max_ms` | float | Spread |
| `p10_ms`, `p50_ms`, `p90_ms` | float | Percentiles |
| `max_abs_err` | float | Max-abs-diff vs Python reference |
| `tolerance` | float | Per-op tolerance gate |
| `ok` | bool | True iff `max_abs_err <= tolerance` (and the manifest agrees for native rows) |
| `symbol` | str | C ABI symbol name (single-kernel native rows) |
| `symbols` | list[str] | Multi-kernel chain (workload rows) |
| `apple_gpu_status` | str | Manifest-resolved status (`fused` / `planned`); EBM rows only |
| `device`, `tessera_version` | str | Provenance |

Envelope fields:

| Field | Notes |
|---|---|
| `runs` | List of row dicts |
| `ga_primitives_count` | Count of `clifford_*` rows |
| `ebm_paths_count` | Count of `ebm_*` rows (native + python_ref) |
| `ebm_native_apple_gpu_count` | Count of native EBM rows (9 when GPU available; includes `ebm_partition_exact`) |
| `native_ebm_ops` | Sorted list of EBM ops on `apple_gpu` |
| `workload_count` | Count of `namespace=workload` rows |
| `ebt_sweep_count` | Count of opt-in `ebt_tiny_sweep` rows |
| `ebt_sweep_summary` | Break-even summary with per-shape speedup and `first_native_win_shape` |
| `compile_time_ms` | clang++ wall-clock for the runtime dylib |
| `skipped_apple_gpu` | Skip reason string (`null` when Apple GPU runs) |
| `device`, `tessera_version`, `reps` | Provenance |
