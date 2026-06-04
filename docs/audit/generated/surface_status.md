# Repo Surface Status (generated)

Consolidated status of the repo's audited surfaces — examples / benchmarks / research / tools / tests (formerly five separate `*_status.md` docs) plus operator-benchmark coverage. The canonical machine-readable artifact is `surface_status.csv` in this directory. Regenerate via `scripts/check_generated_docs.sh --write`.

## Aggregate

| Surface | Entries | Status breakdown |
|---|--:|---|
| examples | 19 | runnable=12, runnable_optional=1, scaffold=6 |
| benchmarks | 18 | archived=1, compile_only=6, runnable=11 |
| research | 2 | compile_only=1, runnable=1 |
| tools | 6 | archived=1, compile_only=2, runnable=3 |
| tests | 10 | archived=2, compile_only=3, runnable=1, scaffold=4 |

## examples

| Directory | Status | Entry point | Reason |
|---|---|---|---|
| `examples/advanced/Diffusion_LLM` | scaffold | `examples/advanced/Diffusion_LLM/tessera_diffusion_llm.py` | Research sketch — references non-existent APIs (``ts.compile(mode='training')``, ``ts.randint``, ``Tensor[]`` syntax) and the package modules require PyTorch.  Reimplement against the canonical Tessera surface or mark broken when that work starts. |
| `examples/advanced/Fast_dLLM_v2` | runnable | `examples/advanced/Fast_dLLM_v2/tests/smoke_random.py` |  |
| `examples/advanced/Jet_nemotron` | scaffold | `examples/advanced/Jet_nemotron/examples/e2e_infer.py` | Requires the upstream ``tessera.stdlib`` research stack which is not part of the standalone compiler surface.  Test ``tests/test_sanity.py`` locks the e2e_infer import block + skips honestly when stdlib is absent. |
| `examples/advanced/Nemotron_Nano_12B_v2` | runnable | `examples/advanced/Nemotron_Nano_12B_v2/tests/smoke_random.py` |  |
| `examples/advanced/Tessera_Empirical_Software_Agent` | scaffold | `examples/advanced/Tessera_Empirical_Software_Agent/src/agents/tree_search_runner.py` | End-to-end LLM + tree-search agent — requires a real LLM client, sandbox executor, and per-task harness.  DummyLLM only proposes ``print('hello from variant N')`` stubs; the orchestrator is not runnable as a CI smoke test. |
| `examples/advanced/gumiho` | runnable | `examples/advanced/gumiho/demo.py` |  |
| `examples/advanced/kv_cache_serving` | runnable | `examples/advanced/kv_cache_serving/demo.py` |  |
| `examples/advanced/long_context_attention` | runnable | `examples/advanced/long_context_attention/demo.py` |  |
| `examples/advanced/mla` | runnable | `examples/advanced/mla/tests/smoke_random.py` |  |
| `examples/advanced/power_retention` | scaffold | `examples/advanced/power_retention/examples/minimal_power_attn.py` | Placeholder — entry-point script currently just prints ``'example'``.  Real implementation lives in the ``python/tessera_power/`` subpackage (CUDA scaffolds, Retention op) which is not wired into the audit yet. |
| `examples/advanced/rlvr_reasoning_suite` | runnable | `examples/advanced/rlvr_reasoning_suite/run_demo.py` |  |
| `examples/compiler/dnas` | runnable | `examples/compiler/dnas/dnas_schedule_autotune.py` |  |
| `examples/compiler/ir_pipeline_tutorial` | runnable | `examples/compiler/ir_pipeline_tutorial/tessera_ir_pipeline_demo.py` |  |
| `examples/conformance` | runnable | `examples/conformance/apple_path_ga_ebm_demos.py` |  |
| `examples/getting_started` | runnable | `examples/getting_started/basic_tensor_ops.py` |  |
| `examples/getting_started` | runnable | `examples/getting_started/compile_and_explain.py` |  |
| `examples/getting_started/tessera_flash_attention_demo` | runnable_optional | `examples/getting_started/tessera_flash_attention_demo/examples/flash_attention_demo.py` |  |
| `examples/integration/HF_transformer` | scaffold | `examples/integration/HF_transformer/tessera_huggingface_transformers.py` | References non-existent Tessera APIs (``from tessera import function, Module``); needs a rewrite against the canonical surface (``@tessera.jit`` + ``tessera.nn.Module``). |
| `examples/optimization` | scaffold | `examples/optimization/README.md` | Top-level placeholder directory with only README.md and src/ stubs — no entry-point script exists yet. |

## benchmarks

| Directory | Status | Entry point | Reason |
|---|---|---|---|
| `archive/benchmarks/matrix_multiplication` | archived | `archive/benchmarks/matrix_multiplication/blackwell_matmul_tessera.py` | Pre-Phase-6 matmul benchmark. Superseded by ``benchmark_gemm.py`` + ``run_all.py``. Kept in-tree for historical replay; not part of the current performance story. |
| `benchmarks` | runnable | `benchmarks/run_all.py` |  |
| `benchmarks/DeepScholar-Bench` | runnable | `benchmarks/DeepScholar-Bench/tessera_deepscholar_model.py` |  |
| `benchmarks/Tessera_Operator_Benchmarks` | runnable | `benchmarks/Tessera_Operator_Benchmarks/scripts/opbench.py` |  |
| `benchmarks/Tessera_SuperBench` | compile_only | `benchmarks/Tessera_SuperBench/runner/bench_run.py` |  |
| `benchmarks/apple_cpu` | runnable | `benchmarks/apple_cpu/benchmark_execution_kind.py` |  |
| `benchmarks/apple_gpu` | runnable | `benchmarks/apple_gpu/benchmark_ga_ebm.py` |  |
| `benchmarks/apple_gpu` | runnable | `benchmarks/apple_gpu/benchmark_fusion.py` |  |
| `benchmarks/baselines` | runnable | `benchmarks/baselines/cpu_smoke.json` |  |
| `benchmarks/clifford_core` | compile_only | `benchmarks/clifford_core/core.py` |  |
| `benchmarks/common` | compile_only | `benchmarks/common/__init__.py` |  |
| `benchmarks/corrdiff` | runnable | `benchmarks/corrdiff/benchmark_corrdiff.py` |  |
| `benchmarks/energy_core` | compile_only | `benchmarks/energy_core/core.py` |  |
| `benchmarks/grid_ai_core` | compile_only | `benchmarks/grid_ai_core/core.py` |  |
| `benchmarks/linalg` | runnable | `benchmarks/linalg/linalg_bench.py` |  |
| `benchmarks/spectral` | runnable | `benchmarks/spectral/spectral_correctness.py` |  |
| `benchmarks/spectral` | runnable | `benchmarks/spectral/spectral_bench.py` |  |
| `benchmarks/visual_complex_core` | compile_only | `benchmarks/visual_complex_core/core.py` |  |

## research

| Directory | Status | Entry point | Reason |
|---|---|---|---|
| `research/pddl_instruct` | runnable | `research/pddl_instruct/tools/validator/validator.py` |  |
| `research/sandbox_compilers` | compile_only | `research/sandbox_compilers/tilec/driver.py` |  |

## tools

| Directory | Status | Entry point | Reason |
|---|---|---|---|
| `tools/CLI/Tessera_CLI_Starter_v0_1` | archived | `tools/CLI/Tessera_CLI_Starter_v0_1/CMakeLists.txt` | Historical standalone CLI starter suite.  It remains in-tree for reference, but the active compiler tools are the root ``tools/tessera-opt`` and ``tools/tessera-translate`` surfaces. |
| `tools/profiler` | compile_only | `tools/profiler/cli/tprof.cpp` |  |
| `tools/profiler/scripts` | runnable | `tools/profiler/scripts/tprof_report.py` |  |
| `tools/roofline_tools` | runnable | `tools/roofline_tools/tools/roofline/cli_v2.py` |  |
| `tools/tessera-opt` | compile_only | `tools/tessera-opt/tessera-opt.cpp` |  |
| `tools/tessera-translate` | runnable | `python/tessera/cli/translate.py` |  |

## tests

| Directory | Status | Entry point | Reason |
|---|---|---|---|
| `archive/tests` | archived | `archive/tests` | Historical tests preserved for reference; not run in any CI lane. |
| `tests/integration` | scaffold | `tests/integration` | Directory reserved for cross-component integration tests.  Currently empty (no test_*.py files).  Pytest skips it gracefully.  Status = scaffold so reviewers see it surface in the dashboard. |
| `tests/kernel_tests` | scaffold | `tests/kernel_tests/README_TESSERA_PERF.md` | C++ kernel-level scaffold (CUDA / HIP / ROCm).  Built via CMake when ``TESSERA_ENABLE_CUDA=ON`` or ``TESSERA_ENABLE_HIP=ON``.  Not exercised in the CPU validation spine.  Promotion to ``runnable`` is gated on Phase G (NVIDIA) / Phase H (ROCm) hardware bring-up. |
| `tests/performance` | compile_only | `tests/performance/test_compiler_performance_plan.py` |  |
| `tests/regression` | scaffold | `tests/regression` | Directory reserved for regression cases that lock in past bug fixes.  Currently empty (no test_*.py files).  Net-new regression tests should land under ``tests/unit/`` until the regression directory has its own ownership story. |
| `tests/tessera-ir` | compile_only | `tests/tessera-ir` |  |
| `tests/tessera_numerical_validation` | scaffold | `tests/tessera_numerical_validation/run_all.sh` | Numerical validation harness (reference-vs-runtime comparisons for compiled CPU + future hardware backends).  Today the directory contains ``README.md`` + ``requirements.txt`` + ``run_all.sh`` + a ``tessera_numerics/`` Python package, but **no test_*.py files** — pytest doesn't pick up any tests here.  Modernization onto current APIs (``ts.jit``, ``fn.explain()``, ``execution_kind``, fallback_reason) is deferred until a workload genuinely needs it. |
| `tests/tessera_tests/tessera_kernels_scaffold` | archived | `tests/tessera_tests/tessera_kernels_scaffold/README_TESSERA_PERF.md` | Structurally-similar scaffold to ``tests/kernel_tests/`` with the same README + ci/configs/scripts/tests layout.  Kept in-tree for reference until the kernel-tests lane is validated against real hardware (Phase G / H), at which point this directory becomes a candidate for merge or deletion. |
| `tests/unit` | runnable | `tests/unit` |  |
| `tests/unit/_slow_subset` | compile_only | `tests/unit (slow-marked tests)` |  |

<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.operator_benchmarks_coverage -->

# Operator Benchmark Coverage

This dashboard maps the current support-table families to the seven active ``Tessera_Operator_Benchmarks`` groups. It also records families that intentionally live in a different benchmark harness.

| Opbench group | Support-table family | Representative ops | Coverage | Notes |
|---------------|----------------------|--------------------|----------|-------|
| ``matmul`` | ``loop_nest`` | ``matmul``, ``gemm`` | ``direct`` | Direct C++ opbench group and direct support-table primitive family. |
| ``conv2d`` | ``stencil`` | ``conv2d`` | ``direct`` | Direct NHWC convolution group; native target runtime remains backend-dependent. |
| ``flash_attention`` | ``attention`` | ``flash_attn`` | ``direct`` | Direct attention group for the canonical flash-attention primitive. |
| ``reduce`` | ``stable_reduction`` | ``reduce``, ``sum`` | ``grouped`` | Covers sum-style reductions; broader stable reductions are represented separately. |
| ``elementwise`` | ``elementwise`` | ``tanh``, ``add``, ``mul``, ``relu``, ``sigmoid`` | ``grouped`` | One opbench group covers multiple scalar elementwise primitives. |
| ``softmax_layernorm`` | ``stable_reduction+normalization`` | ``softmax``, ``layer_norm`` | ``grouped`` | Composite group spanning softmax and layer normalization. |
| ``transpose_gather`` | ``layout_transform`` | ``transpose``, ``gather`` | ``grouped`` | Current smoke runs transpose; gather remains mapped but not separately timed. |
| ``spectral`` | ``spectral`` | ``fft``, ``dct``, ``spectral_conv`` | ``not-applicable`` | Covered by benchmarks/spectral rather than Tessera_Operator_Benchmarks. |
| ``ga_ebm`` | ``geometric_algebra+energy_based`` | ``clifford_geo_product``, ``ebm_inner_step`` | ``not-applicable`` | Covered by benchmarks/apple_gpu/benchmark_ga_ebm.py. |
