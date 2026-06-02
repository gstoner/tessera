---
status: Historical (superseded by docs/status/ga_ebm_milestone.md)
classification: Audit / Execution gap (frozen snapshot)
authority: Original framing of the GA + EBM native-execution gap; kept for context only
last_updated: 2026-05-17
---

> ⚠️ **Historical snapshot.** This page captures the original
> framing of the Apple GA + EBM native-execution gap (and the
> running evaluation as that gap was closed).  The **canonical
> current status** now lives at
> [`docs/status/ga_ebm_milestone.md`](../status/ga_ebm_milestone.md)
> — read that page for the up-to-date TL;DR, known non-claims, and
> next targets.  This document is kept as a record of how the
> milestone was reached; it is no longer updated as features ship.
>
> The roadmap [`docs/audit/ga_ebm_roadmap.md`](ga_ebm_roadmap.md)
> is also still maintained as the sprint sequence.

# Apple GA + EBM Native Execution Gap

This note originally evaluated the Apple CPU/GPU path against the proposed
milestone:

> One fully native, measured GA/EBM path from Python API to dialect to lowering
> to backend execution with a reproducible benchmark.

Current closure state: **Apple GPU has measured native GA + EBM benchmark
coverage.** GA has 17/17 registered Clifford primitives benchmarked through
native Apple GPU symbols. EBM has 9/9 native Apple GPU rows, including
`ebm_partition_exact` as a single-dispatch stable-logsumexp kernel. Composite
workload rows exist for `ga_feature_pipeline` and `ebt_tiny_refinement`, each
paired with a Python-reference row.

The remaining limitation is no longer per-primitive GA/EBM native dispatch. It
is broader compiler integration: keeping public API fast paths, `@tessera.jit`,
CompileReport rows, benchmark rows, and generated support tables aligned so
native, reference, and artifact-only claims cannot blur together.

## Local Verification

Reviewed on macOS Apple Silicon:

```text
Darwin ... RELEASE_ARM64_T6000 arm64
```

Commands run:

```bash
.venv/bin/python - <<'PY'
from tessera.compiler.gpu_smoke import run_matmul_smoke
print(run_matmul_smoke('apple_gpu', size=8).to_dict())
PY
```

Result: Apple GPU matmul smoke returned `runtime_status='ready'`, `ok=True`.

```bash
.venv/bin/python -m pytest tests/unit/test_apple_gpu_clifford_msl.py -q
```

Result: **5 passed in 2.98s**. This compiles `apple_gpu_runtime.mm`, loads it
through `ctypes`, dispatches the GA9 GA MSL symbols, and compares against the
Python GA reference.

Additional GA11 verification:

```bash
.venv/bin/python -m pytest tests/unit/test_apple_gpu_clifford_msl_ga11.py -q
```

Result: **16 passed in 2.61s**. This covers the final six Apple GPU GA symbols:

- `tessera_apple_gpu_clifford_exp_cl30_f32`
- `tessera_apple_gpu_clifford_log_cl30_f32`
- `tessera_apple_gpu_clifford_ext_deriv_cl30_f32`
- `tessera_apple_gpu_clifford_vec_deriv_cl30_f32`
- `tessera_apple_gpu_clifford_codiff_cl30_f32`
- `tessera_apple_gpu_clifford_integral_cl30_f32`

Manifest check:

```text
17 Clifford primitives; 0 missing Apple GPU entries; 0 non-fused Apple GPU entries.
```

Native GA + EBM benchmark health check:

```bash
.venv/bin/python benchmarks/apple_gpu/benchmark_ga_ebm.py --ci
```

Current closure contract:

- 17 GA Apple GPU primitive rows (`backend=apple_gpu`, `mode=fused`).
- 9 native EBM Apple GPU rows:
  `ebm_inner_step`, `ebm_refinement`, `ebm_langevin_step`, `ebm_decode_init`,
  `ebm_bivector_langevin`, `ebm_sphere_langevin`, `ebm_self_verify`,
  `ebm_energy` (quadratic specialization), and `ebm_partition_exact`
  (single-dispatch stable logsumexp).
- 4 workload rows:
  `ga_feature_pipeline` and `ebt_tiny_refinement`, each as `apple_gpu` and
  `python_ref`.
- Envelope separates `compile_time_ms` from per-row dispatch latency.

Focused benchmark contract tests:

```bash
.venv/bin/python -m pytest \
  tests/unit/test_benchmark_ga_ebm.py \
  tests/unit/test_ga_backend_manifest.py -q
```

Latest focused contract result: **88 benchmark tests** plus manifest coverage;
the broader reported suite is **3557 passed, 1 skipped, 0 failures**.

## Current Apple CPU Path

Apple CPU is wired for the tensor compiler path:

- `@tessera.jit(target="apple_cpu")` builds a runtime artifact for supported
  tensor ops.
- The runtime dispatches through `apple_cpu_runtime.cpp`.
- GEMM routes to Accelerate-style C ABI symbols such as
  `tessera_apple_cpu_gemm_f32`.
- Metadata uses `compiler_path="apple_cpu_accelerate"` and reports native CPU
  execution when launch succeeds.

For GA/EBM:

- GA `python/tessera/ga/` is Python/NumPy reference.
- The backend manifest marks x86 and Apple CPU as reference-first for GA.
- There is no Apple CPU Clifford runtime shim for `geo_product`,
  `rotor_sandwich`, or point-cloud invariant features.
- EBM primitives are Python/NumPy reference; EBM dialect passes are annotation
  and scheduling intent, not Apple CPU native execution.

Assessment: **Apple CPU is useful as a tensor execution baseline, but it is not
the shortest path to closing the GA/EBM native execution gap.**

## Current Apple GPU Path

Apple GPU is the strongest existing native backend path.

What is real today:

- `@tessera.jit(target="apple_gpu")` can produce executable runtime artifacts
  for supported tensor ops and fusions.
- Runtime metadata uses `compiler_path="apple_gpu_mps"`.
- `runtime.launch` and the JIT fast path dispatch through
  `_execute_apple_gpu_mps_metadata`.
- The tensor path supports native MPS/MSL dispatch for matmul, rope,
  flash-attention, softmax, GELU, matmul-softmax, matmul-softmax-matmul,
  matmul-GELU, matmul-RMSNorm, SwiGLU, and related envelope ops.
- Local smoke confirms `tessera.matmul` executes through Apple GPU runtime.

What is real for GA:

- `apple_gpu_runtime.mm` exports Clifford MSL symbols for all 17 registered GA
  primitives.
- Focused tests compile the runtime, load the dylib, call the GA symbols
  directly through `ctypes`, and compare against `tessera.ga`.
- `tests/unit/test_apple_gpu_clifford_msl.py` validates:
  - symbol export;
  - `clifford_geo_product_cl30_f32`;
  - `clifford_rotor_sandwich_cl30_f32`;
  - rotor-sandwich rotation against SO(3);
  - batched dispatch consistency.
- `tests/unit/test_apple_gpu_clifford_msl_full.py` validates the expanded GA10
  pointwise/binary surface, including fp16/bf16 ports for the headline ops.
- `tests/unit/test_apple_gpu_clifford_msl_ga11.py` validates the final six GA
  primitives:
  - `clifford_exp_cl30_f32` — closed-form pure-bivector path plus fallback;
  - `clifford_log_cl30_f32` — closed-form rotor path plus fallback;
  - `clifford_ext_deriv_cl30_f32` and `clifford_vec_deriv_cl30_f32` —
    3D-grid finite-difference ABI `(F, Out, D0, D1, D2, h0, h1, h2)`;
  - `clifford_codiff_cl30_f32` — sequential Hodge / exterior derivative /
    Hodge composition;
  - `clifford_integral_cl30_f32` — weighted Riemann reduction ABI
    `(field, weights, out, n)`.
- The GA11 tests include symbol-export probes, exp/log bitwise-vs-Python
  checks, `log(exp(B)) = B`, finite-difference interior-cell comparisons
  against `tessera.ga.calculus`, boundary-zero checks, and weighted-sum checks
  including Euclidean manifold weights.
- `backend_manifest.py` has `_CLIFFORD_APPLE_GPU_FUSED` entries for the full
  `EXPECTED_CLIFFORD_OPS` set; the test gate asserts
  `FUSED_APPLE_GPU_OPS == EXPECTED_CLIFFORD_OPS` and
  `PLANNED_APPLE_GPU_OPS == set()`.
- `benchmarks/apple_gpu/benchmark_ga_ebm.py --ci` emits reproducible GA rows
  with manifest-resolved symbols, max-abs-diff correctness checks, and timing
  percentiles.

What is missing:

- `@tessera.jit(target="apple_gpu")` does not appear to lower `tessera.ga`
  Python calls into `tessera_clifford` Graph IR.
- The `tessera_clifford` dialect and GA8 passes are validated through
  `ts-clifford-opt` / lit-style fixtures, but the Apple GPU runtime path does
  not consume those lowered Clifford artifacts.
- The JIT Apple GPU fast path recognizes tensor-op plans, not Clifford plans.
- `_execute_apple_gpu_mps_metadata` dispatches tensor op names such as
  `tessera.matmul`, `tessera.softmax`, and `tessera.flash_attn`; it does not
  dispatch `tessera_clifford.*` or `clifford_*` runtime metadata.

Assessment: **the Apple GPU GA kernel surface is complete and benchmarked for
the current 17-op GA primitive set.** The remaining GA gap is integration
through Python/JIT, Clifford dialect lowering, and normal Tessera runtime
dispatch.

## Current EBM Path

EBM is mature at the Python reference and dialect-intent layers:

- `python/tessera/ebm/` contains energy primitives, inner-loop refinement,
  self-verification, samplers, partition estimators, losses, and manifold-aware
  sampling.
- `src/solvers/ebm/` contains the EBM dialect, canonicalization, and EBM6
  annotation passes for energy-gradient fusion, checkpointing, and candidate
  pipelining.
- Conformance demos verify RBM-style training, EBT-tiny inner-loop refinement,
  and GA/EBM bivector sampling.
- `apple_gpu_runtime.mm` exports native EBM symbols for eight benchmarked rows:
  `tessera_apple_gpu_ebm_inner_step_f32`,
  `tessera_apple_gpu_ebm_refinement_f32`,
  `tessera_apple_gpu_ebm_langevin_step_f32`,
  `tessera_apple_gpu_ebm_decode_init_noise_apply_f32`,
  `tessera_apple_gpu_ebm_langevin_step_f32` reused for
  `ebm_bivector_langevin`, and
  `tessera_apple_gpu_ebm_sphere_langevin_step_f32`,
  `tessera_apple_gpu_ebm_self_verify_hard_argmin_f32`,
  `tessera_apple_gpu_ebm_energy_quadratic_f32`.
- `backend_manifest.py` marks those eight EBM rows as `apple_gpu=fused`.
- `benchmark_ga_ebm.py --ci` emits native EBM rows and Python-reference rows
  side by side so speedup and correctness are visible in one report.
- Workload mode benchmarks `ebt_tiny_refinement` using the fused
  single-dispatch `ebm.ebt_tiny` kernel (refinement + per-row energy +
  K-way argmin in one Metal pass; streaming closed-form so D is
  unbounded, K ≤ 256), paired with the Python reference chain.  Every
  native row carries a `dispatched_on_gpu` proof bit so silent numpy
  fallbacks (e.g., K > 256) are labeled `degraded_fallback` instead of
  being mistaken for native wins.
- `--ebt-sweep` records the break-even ladder.  After the kernel
  widening + proof-of-dispatch hardening the sweep tags each shape
  with `status="native_dispatched"` or `"degraded_fallback"` and only
  computes a `speedup` for the former.  Headline numbers from a recent
  M-series run: first native win at `B=16,K=32,D=128,T=8` (~1.1×),
  peak ~55× at `B=64,K=128,D=1024,T=256`.  These will drift across
  hosts; the schema + proof bit is the stable contract.

What is missing:

- The EBM dialect annotation passes do not yet lower to Apple runtime calls.
- `ebm_partition_monte_carlo` and `ebm_partition_ais` execute through
  Python/NumPy; `ebm_partition_exact` is now superseded by the native
  stable-logsumexp Apple GPU path tracked in
  [`docs/status/ga_ebm_milestone.md`](../status/ga_ebm_milestone.md).
- Arbitrary user-defined energy functions do not yet lower to native MSL; the
  current native `ebm_energy` row is the quadratic specialization.
- **All 26 fast paths route through the `jit_bridge`** (17 GA + 9
  native EBM, including `ebt_tiny`).  Every public-API GPU dispatch
  produces a `JitBridgeRoute` row in the thread-local trace when
  tracing is enabled, and the benchmark uses the trace as proof of
  dispatch for native EBM primitive rows + JIT-bridge benchmark
  rows.  `bivector_langevin` reuses the langevin_step kernel but
  tags itself as `ebm_bivector_langevin` in the trace via the
  helper's `bridge_op_name` kwarg.

Assessment: **EBM now has measured native Apple GPU coverage for the
inner-loop/sampler/decode/self-verify/quadratic-energy slice, plus a tiny
workload benchmark and break-even sweep.** The next EBM gap is not
proof-of-native-execution; it is compiler integration, broader public API
dispatch, and the remaining Python-only `partition_exact` row.

## Recommended Milestone

Choose one narrow integration path:

> `@tessera.jit(target="apple_gpu")` for a `Cl(3,0)` point-cloud invariant
> feature lowers to a Clifford op plan, dispatches native Apple GPU MSL kernels,
> validates against Python reference, and emits a reproducible benchmark row.

Suggested initial function:

```python
def point_cloud_feature(points):
    # points: [N, 8] Cl(3,0) multivector coefficients, grade-1 vectors.
    # Computes sum_{i<j} clifford_inner(points[i], points[j]).
    return scalar
```

Why this path:

- It matches the visible GA demo already added under `examples/conformance/`.
- It uses `clifford_inner_cl30_f32`, one of the simpler Apple GPU kernels.
- It produces a scalar invariant, so correctness is easy to assert.
- It avoids field ops, autodiff, EBM samplers, and exp/log corner cases.
- It creates a benchmarkable kernel with stable input size knobs: `N` points,
  repeats, dtype.
- It does not imply that `clifford_inner` is the only ready GA kernel; all 17
  registered GA primitives now have fused Apple GPU manifest status. `inner` is
  simply the smallest good vertical-slice target.

A second milestone can wrap that feature in the EBT-tiny loop:

> Run K candidates through T EBM inner steps where the energy contains the
> native GA point-cloud feature, then self-verify in Python initially. Move
> self-verify to Apple GPU only after the GA path is measured.

## Required Work Items

1. **Frontend capture**

   Add a minimal lowering path from a Python-visible GA demo call to a
   compiler-recognized Clifford op. It does not need full arbitrary
   `tessera.ga` tracing at first; a constrained `tessera.ga.point_cloud_inner`
   or `clifford_inner_sum` demo primitive is acceptable if documented as a
   demo envelope.

2. **Graph IR / dialect bridge**

   Ensure the frontend artifact contains `tessera_clifford.inner` or an
   explicit fused demo op with `algebra=[3,0,0]`.

3. **Lowering bridge**

   Connect the Clifford lowering result to Apple Target IR metadata that names
   the runtime symbol, for example:

   - `tessera_apple_gpu_clifford_inner_cl30_f32`
   - later: `tessera_apple_gpu_clifford_rotor_sandwich_cl30_f32`

4. **Runtime dispatch**

   Extend `python/tessera/runtime.py` with a tiny GA Apple GPU dispatcher, in
   the same style as `_apple_gpu_dispatch_softmax` and
   `_apple_gpu_dispatch_gelu`.

5. **Runtime-path benchmark row**

   Extend the existing deterministic GA + EBM benchmark so the first
   compiler/runtime-integrated path emits:

   - target: `apple_gpu`;
   - op: `clifford_point_cloud_inner_sum`;
   - input shape and dtype;
   - runtime status;
   - median/mean wall time across repeats;
   - correctness drift versus Python reference;
   - manifest status and runtime symbol.

6. **CI**

   Keep the hardware test skip-gated on non-Darwin, like the existing Apple GPU
   Clifford tests. Keep the Python/reference benchmark path runnable everywhere.

## Acceptance Criteria

The gap should be considered closed for the first GA/EBM Apple milestone when
all of the following are true:

- A user-facing Python demo entry point exists and runs without direct `ctypes`
  calls in the demo.
- The compiled artifact includes a Clifford dialect or explicit Clifford
  lowering marker, not only a hand-written runtime call.
- The Apple GPU runtime dispatch is reached through the normal Tessera runtime
  or JIT fast path.
- The native result matches the Python GA reference within fp32 tolerance.
- The existing reproducible benchmark command emits timing and correctness
  metadata for the compiler/runtime-integrated path.
- CI has:
  - non-hardware reference coverage;
  - Darwin/Apple GPU hardware-gated native coverage.

## Bottom Line

Do not add more primitive breadth next. The Apple GPU backend already has
complete fused-kernel coverage for the current 17-op Clifford surface and all
nine native EBM rows, including `ebm_partition_exact` as a stable-logsumexp
kernel over precomputed energies. The valuable next step is integration: route
the dialect annotation path and one broader `@tessera.jit` path through the
normal compiler/runtime stack.
