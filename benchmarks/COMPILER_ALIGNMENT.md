# Benchmark/compiler alignment review

Reviewed 2026-09-10 against source in the current working tree. This is a
benchmark navigation and cleanup record, not another compiler status registry.
[MASTER_AUDIT](../docs/audit/MASTER_AUDIT.md) owns status routing;
[the integrated plan](../docs/audit/compiler/INTEGRATED_COMPILER_PLAN.md) owns
sequencing. `linalg` was supplied twice and is reviewed once.

## What these suites actually measure

| Directory | Executed boundary | Disposition and missing proof |
|---|---|---|
| `math/` | Seven metadata-driven runtime probes for x86/ROCm; host-wall timing includes wrapper work. Direct x86 scan C ABIs form a separate comparison. | Keep as diagnostic regression probes. Require observed native execution, finite correctly shaped outputs, and positive repetitions. No fresh run may inherit selector eligibility from target name or old packets. Migrate probes to serialized package ancestry before using them for F2/F3 promotion. |
| `linalg/` | fp64 public `tessera.ops` cholesky, QR, SVD and triangular solve; NumPy/SciPy-style reference composition. | Keep as numerical oracle. Enforce residual bounds even in smoke mode and retain valid JSON stdout. This script's route says nothing about whether another backend has native factorization support. Add a separate exact-package comparison instead of silently changing the oracle. |
| `energy_core/` | Python EBM composition with host RNG, analytic gradient, annealing and partition work. Public primitives can opportunistically dispatch to Apple. | Keep the domain workload. Fixed `cpu` attribution was incorrect: report unattributed library composition until per-call route receipts are captured. Report logical-byte bandwidth as an estimate. Determinism is not a native execution certificate. |
| `clifford_core/` | Public multivector/rotor/grade operations, with optional Apple primitive fast paths; Python composition and NumPy oracle share implementation. | Keep as a composition oracle. Report unattributed library execution; no proof of whole-program MLIR compilation, independent per-op correctness or GPU kernel timing. |
| `autodiff/` | Eight different native/JIT/compiler-substrate probes; details below. | Keep all eight: their input models and correctness obligations differ. Index newer resident/public-AD work rather than pretending the older solver scripts cover it. Separate fresh-run eligibility from historical evidence. |
| `Tessera_Operator_Benchmarks/` | Seven C++ CPU reference groups; optional Python JIT artifact/CPU bridge; native C ABI mode explicitly unavailable. | Keep the runnable reference harness and slow bridge tests. A target-IR string or reference bridge result is not native backend proof. Static MLIR samples and split specs are design/fixture material, not production route authority. |

The surface manifest now includes `math/` and `autodiff/` as `compile_only`: its
CI commands check syntax, while owning-device execution remains a separate gate.

## Autodiff entry points

| Script | Compiler connection | Main coverage limit |
|---|---|---|
| `benchmark_native_jvp.py` | Public JIT JVP, x86/ROCm | Three bounded families; completed parent host-wall timing, no calibrated device clock. |
| `benchmark_solver_ift.py` | Scheduled diagonal implicit-function solver | Bounded residual/linear-solve/cotangent pipeline; host-wall timing alone cannot promote. |
| `benchmark_solver_children.py` | NVIDIA unary/comparison/reduction native child ABIs | Primitive children, not automatic whole-program AD. |
| `benchmark_solver_krylov.py` | NVIDIA packaged diagonal Krylov solve | Device-resident solve with host-wrapper timing, not a comparative performance packet. |
| `benchmark_solver_dense_krylov.py` | NVIDIA dense CG/GMRES packages | A historical performance packet path cannot certify a newly compiled run. |
| `benchmark_general_solver.py` | Explicit GraphIRModule to physical general solver | Substrate test of a hand-built residual; not Python frontend capture proof. |
| `benchmark_general_products.py` | Hand-built nonlinear/reduction/matmul/mixed/predicate Graph IR plus spectral product | Valuable boundary coverage; `general` names do not imply arbitrary source CFG or general public AD. |
| `benchmark_w4_region_products.py` | Paired MLIR fixtures and bounded structured-region products | Explicit dynamic-if/hybrid-while envelopes, not unrestricted Python control flow. |

Newer complementary workloads live outside this directory:
[`record_snapshot_public_ad.py`](record_snapshot_public_ad.py) exercises public
resident SSD VJP, scoped readers and frame retirement;
[`record_async_pool_ad.py`](record_async_pool_ad.py) exercises composed resident
adjoints; [`record_ssd_gpu.py`](record_ssd_gpu.py) measures the native SSD family;
[`record_ssd_calibrated_pairs.py`](record_ssd_calibrated_pairs.py) preflights
clean-host paired calibration. These complement solver and source-product probes;
they do not supersede their numerical contracts.

## Cleanup and next engineering work

1. **EVIDENCE-PACKET-1 / TPROF-NATIVE-1:** use a shared evidence envelope for
   observed route, exact artifact/compiler identity, correctness, timing domain,
   environment and sample provenance. Retain existing readers through adapters;
   do not globally rename historical schemas. New runs are not promotion eligible
   without a validated comparison bound to those exact artifacts.
2. **F0/F2:** replace math's manufactured `RuntimeArtifact.metadata` declarations
   with package consumers, and collect per-call routes for GA/EBM. Until then,
   retain their diagnostic/unattributed labels. Public JIT and native package
   comparisons should use matched inputs and separate compilation, transfer,
   launch preparation and kernel time.
3. **AD / F3:** pair the existing direct-IR tests with automatic public frontend
   entry points for the same workloads, including saved-product ownership and
   failure paths. Graph constructors stay until differential proof covers their
   replacement; do not delete them merely to reduce file count.
4. **Operator harness:** add target-aware package adapters incrementally. The
   current bridge is CPU-oriented, launches once and has no general device timing
   or repetition contract. Preserve C++ reference cases as independent oracles.
5. **Evidence retention:** preserve old baseline JSON unchanged. Old eligibility
   assertions describe those recorded runs, not current compiler readiness. No
   timing was remeasured or performance promoted by this review.

No executable suite is archived: each has active consumers/tests or a distinct
oracle. Consolidation here is navigation and interpretation. The two operator
spec parts remain linked design references; their runnable instructions belong
to that suite's README. Archive a probe only after identifying its replacement,
migrating callers and preserving the historical evidence links.

## Additional suite review — 2026-09-10

| Suite | Actual boundary and action |
|---|---|
| `autodiff/` | The first-pass map still applies: eight distinct compiler probes, with public resident SSD ownership benchmarks outside this directory. Keep direct-IR tests until automatic frontend counterparts prove equivalence. |
| `dlop_longtail_core/` | Portable metamorphic composite checks plus a separate synthesized-fusion path. Catalog decomposition counts are static estimates, not measured GPU launches; rows now say so explicitly and cannot promote. The CLI now consumes and records its requested seed. Keep the CV long-tail gap and fused candidate names, then add profiler dispatch receipts and matched native package measurements. |
| `Tessera_SuperBench/` | GEMM's CPU JIT path, attention/conv reference timing with artifacts, and default mock collectives are different evidence categories. Retired the unused sleep-based attention placeholder so it cannot emit fake timings. Keep `_stub.py` files as compatibility forwarding entry points; they execute the current implementations rather than placeholders. Native package adapters and real distributed measurements remain separate work. |
| `DeepScholar-Bench/` | Deterministic CPU JIT scoring smoke; its oracle repeats the public operator chain. Keep STATUS.md as the scope authority. LOTUS integration and the research workflow roadmap are optional/design work, not current compiler or retrieval-quality proof. One host-wall smoke call is not a performance comparison. |
| `grid_ai_core/` | Library composition of stencil, attention, conv, RNG and mock halo transport; a separate MLIR fixture checks visibility. Rows now report unattributed library execution and estimated logical bandwidth. Neither mock halo nor a mesh-region fixture establishes native distributed execution. |
| `lattice_reasoning_core/` | NumPy references, public primitive rows, explicit Apple runtime-marker checks and integrated-step artifacts coexist. Apple-call perf_counter measurements now populate cpu_wall_ms, never kernel_elapsed_ms. No device clock or performance promotion is inferred from metal_runtime. Integrated LDT fusion and shared SSD-family substitution require additional lineage and execution proof. |
| `visual_complex_core/` | GA/EBM library composition inherits optional Apple primitive dispatch. Rows now report unattributed execution; deterministic composition and the IR fixture do not prove fused whole-program execution. Keep as a cross-domain layout/semantic regression. |
| `rl/` | Policy losses distinguish Python/decomposed references from optional Apple value-IR execution; GLM serving pressure is a scaled CPU reference and large-shape planning probe. Keep both. Apple policy timing now requires native status and finite output on every submission; full serving throughput still needs real KV/MTP/distributed execution rather than extrapolation. |
| `e2e_spine/` | Architecture-specific native packet recorders and a deliberately synthetic validation-overhead benchmark. Keep both, clearly separated. Packet sealing validates provenance and bounded families; it is not whole-compiler coverage or universal performance eligibility. Runtime/compiler changes require fresh owning-device measurements, never a fingerprint-only update. |

Prioritize profiler-backed DLOP comparisons and exact-package SuperBench adapters
under F2/F3, scoped public-AD comparisons under W5.2f, and route/timing attribution
under EVIDENCE-PACKET-1 / TPROF-NATIVE-1. Keep E2E-SPINE's sealing protocol and
historical packets intact. No new hardware measurement or performance promotion
was performed by this review. Archive neither active oracles nor compatibility
entry points until their callers and evidence links have been migrated.

DLOP, lattice reasoning, RL and E2E spine are now registered in the surface
manifest with syntax-only CI commands. This does not run hardware lanes or
reseal evidence. Their focused execution tests remain separate.

## Native adapter increment — 2026-09-10

SuperBench now has explicit CUDA/HIP ANN package configs. DLOP shares that
adapter in a separate observed-dispatch lane for two-affine ReLU/abs/square
workloads. Receipts count actual bound driver launch calls and require checked
completion plus copyback; they do not stand in for profiler kernel records.
Original and transformed artifacts remain distinct and numerically checked.
The older catalog's operation/decomposition estimates remain estimates.

Public SSD `vjp` now has independent float64 finite-difference comparisons for
three shapes, two cotangents and all five inputs, alongside the existing
asynchronous ownership test. Remaining: broader SuperBench GEMM/attention
adapters, actual DLOP catalog native mappings, profiler-correlated kernel
receipts, general frontend AD comparisons and clean bare-metal promotion.

## Broader adapters and kernel attribution — 2026-09-10

Both native SuperBench configs now execute four workloads: baseline ANN,
64×8 square ANN, serial SSD and cooperative SSD. The SSD adapter preserves
resident checked host-call and device-event window timing as separate fields.
All four pass on SM120 and gfx1151.

`attribute_ssd_cuda.py` supports the dedicated single-artifact SSD capture:
701 kernels must each correlate to one successful launch API record from the
recorded process/device. Wrong names, duplicate/missing correlations or failed
launches refuse. This is bounded process attribution, not a general mixed-workload
NVTX mapper. The current Nsight capture passes. Fresh ROCm tracing still emits
HIP API/agent records only; kernel and copy attribution remain unproven on WSL.
See `baselines/broader_adapters_20260910/`. GEMM/attention workload adapters,
mixed-artifact profiler ranges and clean performance promotion remain open.

## GEMM/attention and mixed artifact attribution — 2026-09-10

Scheduled fp16 GEMM and attention adapters now complement ANN/SSD. The input
Graph is lowered once; packaging consumes the resulting Schedule/Tile and
input layout is projected from its descriptor. CUDA GEMM and fp32 attention
pass independent NumPy oracles. ROCm GEMM and fp16 attention currently abort
inside LLVM GPUFuncOpLowering with duplicate DictionaryAttr names on the
installed assertions-enabled compiler; no executable ROCm result is claimed.

Mixed CUDA captures use unique run/artifact/image NVTX labels. The parser joins
range-owned successful launch APIs to correlated kernels, refusing missing,
ambiguous or asynchronous-outliving records. It supports sequential synchronous
calls across artifacts, not arbitrary concurrent streams. Warmups and unmarked
work are excluded explicitly. Remaining: resolve ROCm's compiler assertion,
validate those adapters on gfx1151, broaden dtype/shape/mask cases, and extend
attribution to asynchronous/multi-thread ownership. Promotion stays disabled.
