---
status: Active evaluation
classification: Audit / Execution gap
authority: Evaluates the Apple CPU/GPU path for the GA + EBM native execution milestone
last_updated: 2026-05-17
---

# Apple GA + EBM Native Execution Gap

This note evaluates the current Apple CPU/GPU path against the proposed
milestone:

> One fully native, measured GA/EBM path from Python API to dialect to lowering
> to backend execution with a reproducible benchmark.

The short answer: **Apple GPU has real native runtime execution today, and GA
has real native MSL kernels today, but GA/EBM do not yet have one continuous
Python API → dialect → lowering → backend → benchmark path.** The next
milestone should integrate an existing GA Apple GPU kernel into the normal JIT
and runtime path rather than adding more GA or EBM primitives.

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
- There is no reproducible GA/EBM benchmark row that records native Apple GPU
  timing, correctness, and metadata in the benchmark harness.

Assessment: **the Apple GPU GA kernel surface is complete for the current 17-op
GA primitive set, but the GA compiler path is not end-to-end native yet.** The
gap is not missing Apple GPU kernels; it is integration through Python/JIT,
Clifford dialect lowering, normal runtime dispatch, and benchmark reporting.

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

What is missing:

- The EBM dialect annotation passes do not yet lower to Apple runtime calls.
- `inner_step` / `self_verify` execute through Python/NumPy, not Apple GPU.
- The EBM "measured native" story should probably depend on a GA or tensor
  kernel first, because EBM itself is an orchestration pattern over repeated
  energy/gradient evaluations.

Assessment: **EBM should be the benchmark wrapper around a first native GA
kernel path, not the first backend integration target by itself.**

## Recommended Milestone

Choose one narrow path:

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

5. **Benchmark**

   Add a deterministic benchmark script that emits:

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
- A reproducible benchmark command emits timing and correctness metadata.
- CI has:
  - non-hardware reference coverage;
  - Darwin/Apple GPU hardware-gated native coverage.

## Bottom Line

Do not add more GA or EBM primitives next. The Apple GPU backend already has
complete fused-kernel coverage for the current 17-op Clifford surface. The
valuable next step is a single measured vertical slice: **Python GA demo →
Clifford IR → Apple lowering → native MSL runtime → benchmark report**.
