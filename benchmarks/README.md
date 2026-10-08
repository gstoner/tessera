# Tessera Benchmarks

This folder contains benchmark families at different maturity levels. The active
portable path is CPU-first and uses the current Python compiler surface where it
is available.

See the [review of math, linalg, energy, Clifford, autodiff and operator suites](COMPILER_ALIGNMENT.md) for current compiler boundaries, fixes and consolidation decisions.

## Current Compiler Support

| Benchmark | Status | Compiler fit |
|---|---|---|
| `benchmark_gemm.py` | Active | Can exercise `@tessera.jit` through the current CPU `matmul -> relu` lowering path with `--use-compiler` via `run_all.py`. |
| `benchmark_attention.py` | Active proxy | Roofline model; can emit current flash-attention Graph IR/lowering diagnostics with `--use-compiler`, but executable Tile/Target lowering is not wired yet. |
| `benchmark_collective.py` | Active proxy | Alpha-beta communication model; should connect to runtime collectives once C ABI/runtime hooks are ready. |
| `run_all.py` | Active | Orchestrates GEMM, attention, and collective suites; pass `--use-compiler` for current compiler artifact checks. |
| `common/` | Active contract | Shared row schema, correctness helpers, and compiler artifact hooks used by benchmark suites. |
| `Tessera_SuperBench/` | Active harness | GEMM uses the current JIT CPU path with telemetry/autotune artifacts; FlashAttention and Conv2D emit artifact-only compiler rows while running NumPy reference timing/correctness; collectives default to the Tessera mock facade. |
| `spectral/` | Active benchmark | NumPy/PyTorch FFT/DCT/convolution benchmark with `--backend tessera-artifact` for Graph IR artifact rows. Native FFT lowering remains artifact-only until Tile/Target runtime support lands. |
| `Tessera_Operator_Benchmarks/` | Active C++ harness | CPU reference timing for all 7 registered op groups, Graph IR artifact coverage, `tessera.telemetry.v1` JSON summaries, a CPU Python runtime bridge, and explicit `backend_unavailable` for native C ABI mode. |
| `../archive/benchmarks/matrix_multiplication/` | Archived | Blackwell concept sketch using non-existent APIs; future Blackwell work should land as Target IR tests/runtime kernels/operator cases. |
| `DeepScholar-Bench/` | Active CPU smoke | Current-API research-synthesis smoke using `@tessera.jit`, matmul, softmax, layer_norm, and NumPy text/source embeddings. LOTUS integration remains optional and guarded. |
| `lattice_reasoning_core/` | Active current-compiler probe | LDT-style lattice step plus MOPD, Mamba-2, GQA, and Latent MoE primitive microbenchmarks. Emits NumPy reference rows, public Tessera primitive rows, Apple GPU native rows when `metal_runtime` is observed, and an artifact-only integrated-step row for remaining LDT fusion work. |
| `apple_gpu/` | Active Apple GPU lane | Real Metal-dispatch benchmark drivers (fusion sweeps, package-vs-live MTL4 lane, GA/EBM stack walk, Gumiho spec-decode, MLA decode, grouped GEMM, MoE overlap, …). Skips cleanly off Darwin / without `clang++`. See [`apple_gpu/README.md`](apple_gpu/README.md). |
| `apple_cpu/` | Active Apple CPU probe | `benchmark_execution_kind.py` — empirically proves the `accelerate_native` vs `numpy_reference` execution-kind split (both link Accelerate on macOS, so the gate is "within a small factor", not "must beat numpy"). |
| `rocm/` | Hardware-gated | `benchmark_rocm_wmma_gemm.py` — device-timed WMMA GEMM ladder against the shipped `libtessera_rocm_gemm.so` via its C-ABI entry point. Honestly gated: with no AMD GPU it emits an empty result set (exit 0) and never fabricates numbers. |
| `linalg/` | Active CPU reference | `linalg_bench.py` — hardware-free cholesky/qr/svd/tri_solve reference path through `tessera.ops.*`, verified against numpy/scipy, in the canonical schema. |
| `rl/` | Active proxy / hardware-gated | `benchmark_policy_losses.py` (PPO/GRPO/CISPO loss rows split by proof level — python_reference, compiler_decomposed_reference, apple_gpu_value_target_ir) and `benchmark_glm52_serving_pressure.py` (CPU reference for the scaled GLM-5.2 DSA/MLA/MTP serving contract). |

## Native compiler integration recorders

- `nvidia/benchmark_native_producer_chain.py`: RTX 5070 static two/three-producer Graph→Schedule→Tile packages; independent numerical checks, per-stage CUDA events and prepared program wall time. Evidence: `baselines/nvidia_native_producer_chain_20261008/`.
- `nvidia/benchmark_prepared_attention_staging_ab.py`: alternating fresh-process control/candidate attention staging comparison, with correctness checks and separate device-event and host-wall measurements. Evidence: `baselines/nvidia_prepared_attention_staging_20261008/ab_packet.json`.

- `nvidia/benchmark_public_softmax_alias.py`: owning RTX 5070 public softmax/safe-alias JIT packages, FP16/BF16/FP32 correctness, matching native images, resident CUDA events and separate warm public wall timing. Evidence: `baselines/nvidia_public_softmax_alias_20261008/`.

## Quick Checks

```bash
PYTHONPATH=python python3 benchmarks/run_all.py --json-only --no-save --use-compiler --smoke
PYTHONPATH=python python3 benchmarks/Tessera_SuperBench/benches/kernel/gemm_tessera.py --m=64 --n=64 --k=64 --repeat=1
PYTHONPATH=python python3 benchmarks/Tessera_SuperBench/runner/bench_run.py --config benchmarks/Tessera_SuperBench/configs/compiler_smoke.yaml --out /tmp/tessera_superbench_smoke
cmake -S benchmarks/Tessera_Operator_Benchmarks -B /tmp/tessera_opbench_build && cmake --build /tmp/tessera_opbench_build -j
PYTHONPATH=python python3 benchmarks/Tessera_Operator_Benchmarks/scripts/opbench.py --config benchmarks/Tessera_Operator_Benchmarks/scripts/configs/quick_sweep.yaml --bin /tmp/tessera_opbench_build/opbench --out /tmp/tessera_opbench_quick
PYTHONPATH=python python3 benchmarks/DeepScholar-Bench/tessera_deepscholar_model.py --output /tmp/tessera_deepscholar_smoke.json
PYTHONPATH=python python3 benchmarks/spectral/spectral_bench.py --backend tessera-artifact --ops fft1d,dct2,conv1d_fft --sizes 64 --device cpu --repeats 1 --warmup 0 --outcsv /tmp/tessera_spectral.csv
PYTHONPATH=python python3 benchmarks/lattice_reasoning_core/benchmark_lattice_reasoning.py --smoke --json /tmp/tessera_lattice_reasoning_smoke.json
```

## Library-layer benchmarks (Phase 7)

A separate family from the operator/roofline benchmarks above.  Each
**category** is a generic compiler-surface label that captures the
primitive composition under test; each **proving workload** is a small,
domain-specific instantiation that anchors the category to a real paper
or canonical model.  The category label is what summaries, audit docs,
and talks should lead with — the workload name preserves the external
anchor for "yes, real surface, real reference."

| Category                | Proving workload(s)                                  | Source                                  |
|-------------------------|------------------------------------------------------|-----------------------------------------|
| Gridded-AI core         | (generic; no domain-specific anchor yet)             | `benchmarks/grid_ai_core/`              |
| **Diffusion grid core** | `corrdiff_core` (NVIDIA CorrDiff regional weather)   | `benchmarks/corrdiff/`                  |
| Clifford / GA core      | (generic; Cl(3, 0) rotor-sandwich chain)             | `benchmarks/clifford_core/`             |
| Energy / EBM core       | (generic; quadratic energy + annealed Langevin)      | `benchmarks/energy_core/`               |
| Cross-lane core         | `visual_complex_core` (M7 visual-complex milestone)  | `benchmarks/visual_complex_core/`       |
| Lattice reasoning core  | LDT step + MOPD/Mamba-2/GQA/Latent-MoE primitives    | `benchmarks/lattice_reasoning_core/`    |
| Long-memory core        | RULER / LongMemEval / MemoryArena resident-state     | `benchmarks/long_memory_core/`          |
| Long-tail fusion core   | DLOP-Bench-style composite fusion (attn/SwiGLU/…)    | `benchmarks/dlop_longtail_core/`        |

Each library-layer benchmark ships:
  * A small Python core (config + model + oracle + harness).
  * An IR-visible lit fixture in `tests/tessera-ir/phase7/`.
  * A Python guard exercising forward determinism + oracle parity +
    canonical Architecture-Decision-#12 JSON schema.

The audit doc
(`docs/audit/compiler/COMPILER_AUDIT.md`) tracks the
six-layer compiler-correctness coverage for each category.

## Support Files

| File | Purpose |
|---|---|
| `perf_gate.py` | Gates a deterministic `tessera.telemetry.v1` report against a small JSON baseline (schema, latency, event-count checks). |
| `baselines/` | Checked-in ratchet baselines — `cpu_smoke.json` (telemetry gate) and `apple_gpu_hot_paths.json` (`tessera.benchmark.ratchet.v1` median/max-latency rows; recorded by `apple_gpu/record_hot_path_baseline.py`). |
| `compiler_support.py` | Back-compat shim re-exporting the shared compiler contract (`CompilerRun`, `compiler_matmul_relu`, `compiler_*_ir`) from `common/compiler_contract.py`. |

## Recorders and their outputs

Every recorder under `benchmarks/` must be named by some other tracked file
(`tests/unit/test_benchmark_recorders_are_named.py`), and every sealed baseline
must be cited from outside `benchmarks/baselines/`
(`tests/unit/test_benchmark_baselines_are_cited.py`). Packet *directories* name
their recorder in their own `README.md`; the top-level baseline files below are
named here. Paths are relative to `benchmarks/`; outputs are under
`benchmarks/baselines/` unless stated.

| Recorder | Output |
|---|---|
| `rocm/benchmark_gfx1201_resident_norm_matmul.py` | `baselines/gfx1201_resident_frontend_20260930/` (exact-device Graph -> Schedule -> Tile resident RMSNorm/matmul packet) |
| `nvidia/benchmark_scheduled_rmsnorm_matmul_edge.py` | `baselines/sm120_rmsnorm_matmul_edge_20260929/` and `baselines/sm120_rmsnorm_matmul_edge_20260930/` (resident RMSNorm-to-matmul CUDA-event packets; 2026-09-30 includes bounded dynamic M) |
| `nvidia/record_attention_forward_schedule_matrix.py` | `nvidia_sm120_attention_forward_schedules.json` |
| `nvidia/record_autotune_reproducibility.py` | `nvidia_sm120_autotune_reproducibility.json` (reads `autotune_corpus.json` and every `nvidia*resource*.json`) |
| `nvidia/record_bf16_reduction_breadth.py` | `nvidia_sm120_bf16_reduction_breadth.json` |
| `nvidia/record_canonical_k_loop.py` | `nvidia_sm120_canonical_k_loop.json` |
| `nvidia/record_deltanet_backward_packet.py` | `nvidia_sm120_deltanet_backward_2026_07_31.json` |
| `nvidia/record_e2e_spine_attention.py` | `nvidia_sm120_e2e_spine_attention.json` |
| `nvidia/record_e2e_spine_comparative.py` | `nvidia_sm120_e2e_spine_comparative.json` |
| `nvidia/record_e2e_spine_epilogue.py` | `nvidia_sm120_e2e_spine_epilogue.json` |
| `nvidia/record_e2e_spine_paged_kv.py` | `nvidia_sm120_e2e_spine_paged_kv.json` |
| `nvidia/record_e2e_spine_reduction.py` | `nvidia_sm120_e2e_spine_reduction.json` |
| `nvidia/record_low_precision_native_resources.py` | `nvidia_sm120_low_precision_native_resources.json` |
| `nvidia/record_packed_storage_foundation.py` | `nvidia_sm120_packed_storage_foundation.json` |
| `nvidia/record_remaining_dtype_reduction.py` | `nvidia_sm120_remaining_dtype_reduction.json` (reads `nvidia_sm120_test5_route_resources.json`) |
| `nvidia/record_replay_parity.py` | `nvidia_sm120_replay_parity.json` (reads `nvidia_sm120_test5_route_resources.json`) |
| `nvidia/record_training_memory_foundation.py` | `nvidia_sm120_training_memory_foundation.json` |
| `nvidia/record_transport_parity.py` | `nvidia_sm120_transport_parity.json` (reads `nvidia_sm120_test5_route_resources.json`) |
| `nvidia/benchmark_scheduled_macro_matmul.py` | `nvidia_sm120_macro_cta_2026_08_24.json` (`tessera.nvidia.scheduled-macro-matmul.v3`) |
| `nvidia/benchmark_scheduled_rmsnorm_matmul_edge.py` | `baselines/sm120_rmsnorm_matmul_edge_20260929/` and `baselines/sm120_rmsnorm_matmul_edge_20260930/` (resident RMSNorm-to-matmul CUDA-event packets; 2026-09-30 includes bounded dynamic M) |
| `nvidia/profile_test5_routes.py` | Nsight launch target for the TEST-5 production-route capture; `nvidia/parse_ncu_resources.py` normalises the export into `nvidia_sm120_test5_resources.json` |
| `nvidia/profile_test5_emitted_gemm.py` | Nsight launch target for the `tessera_mma_gemm_f16` capture behind `nvidia_sm120_emitted_gemm_resources.json` |
| `nvidia/profile_gemm_schedule_candidates.py` | Nsight launch target for the `nvidia_generic_cuda` / `nvidia_mma_fused` rows of `nvidia_sm120_test5_route_resources.json` |
| `nvidia/profile_route_resources.py` + `nvidia/capture_route_resources.sh` | One-route-per-report Nsight capture of the native TF32/FP8, composed-FP8 and scalar arbiter routes added to `nvidia_sm120_test5_route_resources.json` (`build_test5_resource_manifest.py --base … --route NAME=payload.json`; sync `AUTOTUNE-LAUNCH-INTEGRITY-2026-09-27`) |
| `nvidia/prepare_test5_profile_artifacts.py` | Precompiles the MoE / resident-ops artifacts for `nvidia/profile_test5_transport_serving.py`, the launch target behind `nvidia_sm120_transport_serving_resources.json` |
| `record_native_nonlinear_ad.py` | `native_storage_nonlinear_nvidia.json`, `native_storage_nonlinear_rocm.json` |
| `rocm/benchmark_block_attnres_gfx1151.py` | `rocm_gfx1151_block_attnres_phase5.json`; also `runtime_source_maps_20260909/depth_{default,cooperative}.json` |
| `rocm/benchmark_rocm_es_low_rank.py` | `rocm_gfx1151_es_low_rank.json` |
| `rocm/benchmark_rocm_raster.py` | `rocm_gfx1151_raster_2026_07_29.json` |
| `rocm/record_deltanet_backward_selectors.py` | `rocm_gfx1151_deltanet_backward_selectors.json` |
| `spectral/benchmark_rocm_fft_plan_cache.py` | The per-N `results` rows of `rocm_fft_plan_cache_gfx1151_2026_08_05.json` (printed to stdout as `tessera.rocm_fft_plan_cache.v1`) |
| `x86/benchmark_x86_attention_lse.py` | `x86_avx512_attention_lse_2026_07_30.json` (printed to stdout; work item X86-LSE-1) |
| `x86/benchmark_x86_es_low_rank.py` | `x86_zen5_es_low_rank_2026_08_09.json` |
| `x86/benchmark_x86_fft_codelets.py` | `x86_zen5_fft_mixed_codelets_2026_08_09.json` |
| `x86/benchmark_x86_t1_cache_model.py` | `x86_zen5_t1_cache_model_2026_08_09.json` |
| `x86/record_deltanet_backward_selectors.py` | `x86_avx512_deltanet_backward_selectors.json` |

## Refactor Direction

- Promote compiler-backed benchmark kernels into small Python modules that expose
  Graph/Schedule/Tile/Target artifacts alongside timing rows.
- Keep analytical roofline/proxy benchmarks, but label them explicitly.
- Move purely speculative benchmark concepts to `archive/benchmarks/` once they
  are no longer feeding active compiler/runtime work.

## Compiler-slice recorders and their evidence

This inventory names the new compiler-slice recorder consumers. A source link
is not evidence of execution. Packet READMEs define the tested architecture,
envelope and timing domains; rows without a packet link remain diagnostic.

| Recorder | Packet reference | Evidence boundary |
| --- | --- | --- |
| [whole_copy_images](baselines/rocm_three_formats_20261003/whole_copy_images.py) | [rocm_three_formats_20261003](baselines/rocm_three_formats_20261003/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_native_compile_orchestration](benchmark_native_compile_orchestration.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_attention_argument_order](nvidia/benchmark_attention_argument_order.py) | [nvidia_attention_argument_order_20261002](baselines/nvidia_attention_argument_order_20261002/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_attention_gradient_activity](nvidia/benchmark_attention_gradient_activity.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_bias_attention_jvp](nvidia/benchmark_bias_attention_jvp.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_bounded_lhs_jit](nvidia/benchmark_bounded_lhs_jit.py) | [nvidia_bounded_lhs_jit_20261006](baselines/nvidia_bounded_lhs_jit_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_canonical_tensor_replay](nvidia/benchmark_canonical_tensor_replay.py) | [nvidia_sm120_registered_tensor_pipeline_20261002](baselines/nvidia_sm120_registered_tensor_pipeline_20261002/README.md); [nvidia_sm120_canonical_tensor_replay_20261002](baselines/nvidia_sm120_canonical_tensor_replay_20261002/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_checkpoint_bias_gradient](nvidia/benchmark_checkpoint_bias_gradient.py) | [nvidia_checkpoint_bias_gradient_20261002](baselines/nvidia_checkpoint_bias_gradient_20261002/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_checkpoint_broadcast_core](nvidia/benchmark_checkpoint_broadcast_core.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_checkpoint_broadcast_package](nvidia/benchmark_checkpoint_broadcast_package.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_compact_attention_gradients](nvidia/benchmark_compact_attention_gradients.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_cooperative_norm](nvidia/benchmark_cooperative_norm.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_dynamic_lhs_frontend](nvidia/benchmark_dynamic_lhs_frontend.py) | [nvidia_dynamic_lhs_owner_20261006](baselines/nvidia_dynamic_lhs_owner_20261006/README.md); [nvidia_dynamic_lhs_frontend_20261006](baselines/nvidia_dynamic_lhs_frontend_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_dynamic_row_lhs](nvidia/benchmark_dynamic_row_lhs.py) | [nvidia_dynamic_row_lhs_20261006](baselines/nvidia_dynamic_row_lhs_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_forward_attention_jvp](nvidia/benchmark_forward_attention_jvp.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_jit_attention_vjp](nvidia/benchmark_jit_attention_vjp.py) | [nvidia_jit_attention_vjp_20261002](baselines/nvidia_jit_attention_vjp_20261002/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_lhs_jit_dispatch](nvidia/benchmark_lhs_jit_dispatch.py) | [nvidia_prepared_lhs_20261006](baselines/nvidia_prepared_lhs_20261006/README.md); [nvidia_portable_lhs_owner_20261006](baselines/nvidia_portable_lhs_owner_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_native_norm_accuracy_program](nvidia/benchmark_native_norm_accuracy_program.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_native_norm_selection](nvidia/benchmark_native_norm_selection.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_norm_accuracy](nvidia/benchmark_norm_accuracy.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_norm_accuracy_performance](nvidia/benchmark_norm_accuracy_performance.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_ordinary_attention](nvidia/benchmark_ordinary_attention.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_prepared_attention_jvp](nvidia/benchmark_prepared_attention_jvp.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_prepared_attention_vjp](nvidia/benchmark_prepared_attention_vjp.py) | [nvidia_prepared_attention_vjp_20261006](baselines/nvidia_prepared_attention_vjp_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_public_attention_jvp](nvidia/benchmark_public_attention_jvp.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_public_attention_vjp](nvidia/benchmark_public_attention_vjp.py) | [nvidia_prepared_attention_vjp_20261006](baselines/nvidia_prepared_attention_vjp_20261006/README.md); [nvidia_public_attention_vjp_20261006](baselines/nvidia_public_attention_vjp_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_public_bias_attention_jvp](nvidia/benchmark_public_bias_attention_jvp.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_public_matmul_package](nvidia/benchmark_public_matmul_package.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_resident_tensor_epilogue](nvidia/benchmark_resident_tensor_epilogue.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_rhs_jit_dispatch](nvidia/benchmark_rhs_jit_dispatch.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_rhs_tensor_program](nvidia/benchmark_rhs_tensor_program.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_row_major_b_core](nvidia/benchmark_row_major_b_core.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_row_major_b_schedule](nvidia/benchmark_row_major_b_schedule.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_scheduled_attention_checkpoint_backward](nvidia/benchmark_scheduled_attention_checkpoint_backward.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_scheduled_softmax_matmul_edge](nvidia/benchmark_scheduled_softmax_matmul_edge.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_scheduled_typed_matmul](nvidia/benchmark_scheduled_typed_matmul.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [replay_attention_jvp_program](nvidia/replay_attention_jvp_program.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [replay_bias_attention_jvp](nvidia/replay_bias_attention_jvp.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [replay_public_attention_jvp](nvidia/replay_public_attention_jvp.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [replay_public_attention_vjp](nvidia/replay_public_attention_vjp.py) | [nvidia_public_attention_vjp_20261006](baselines/nvidia_public_attention_vjp_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [record_value_only_attention_jvp](record_value_only_attention_jvp.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_captured_movement](rocm/benchmark_captured_movement.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_gfx1201_checkpoint_formats](rocm/benchmark_gfx1201_checkpoint_formats.py) | [rocm_checkpoint_native_ingest_20261005](baselines/rocm_checkpoint_native_ingest_20261005/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_gfx1201_mxfp8_package](rocm/benchmark_gfx1201_mxfp8_package.py) | [rocm_mxfp8_exponent_scale_20261003](baselines/rocm_mxfp8_exponent_scale_20261003/README.md); [rocm_mxfp8_checked_package_20261003](baselines/rocm_mxfp8_checked_package_20261003/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_gfx1201_three_formats](rocm/benchmark_gfx1201_three_formats.py) | [rocm_three_formats_20261003](baselines/rocm_three_formats_20261003/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_graph_nvfp4_ingest](rocm/benchmark_graph_nvfp4_ingest.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_ingest_normalization_ab](rocm/benchmark_ingest_normalization_ab.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_jit_nvfp4_program](rocm/benchmark_jit_nvfp4_program.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_math_launch_attribution](rocm/benchmark_math_launch_attribution.py) | [rocm_native_math_20261006](baselines/rocm_native_math_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_movement_route_admission](rocm/benchmark_movement_route_admission.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_mxfp4_storage_jit](rocm/benchmark_mxfp4_storage_jit.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_native_math_package](rocm/benchmark_native_math_package.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_native_math_schedule](rocm/benchmark_native_math_schedule.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_native_math_staging](rocm/benchmark_native_math_staging.py) | [rocm_native_math_20261006](baselines/rocm_native_math_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_native_movement](rocm/benchmark_native_movement.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_native_nvfp4_allocation_reuse](rocm/benchmark_native_nvfp4_allocation_reuse.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_native_nvfp4_ingest_leaf](rocm/benchmark_native_nvfp4_ingest_leaf.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_native_nvfp4_owner](rocm/benchmark_native_nvfp4_owner.py) | [rocm_packed_image_identity_20261006](baselines/rocm_packed_image_identity_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_nvfp4_ingest_jit](rocm/benchmark_nvfp4_ingest_jit.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_nvfp4_manifest_cache](rocm/benchmark_nvfp4_manifest_cache.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_nvfp4_static_program_retention](rocm/benchmark_nvfp4_static_program_retention.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_packed_image_identity](rocm/benchmark_packed_image_identity.py) | [rocm_packed_image_identity_20261006](baselines/rocm_packed_image_identity_20261006/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [benchmark_paged_softmax_edge](rocm/benchmark_paged_softmax_edge.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_public_movement_jit](rocm/benchmark_public_movement_jit.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_resident_movement](rocm/benchmark_resident_movement.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [benchmark_resident_nvfp4](rocm/benchmark_resident_nvfp4.py) | No sealed packet link found in the current packet READMEs. | Development/diagnostic recorder; no execution or promotion claim from this inventory. |
| [folded_graph_windows](rocm/folded_graph_windows.py) | [rocm_folded_graph_windows_20261002](baselines/rocm_folded_graph_windows_20261002/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [record_gfx1201_folded_native_package](rocm/record_gfx1201_folded_native_package.py) | [rocm_folded_native_runtime_k_20261002](baselines/rocm_folded_native_runtime_k_20261002/README.md); [rocm_folded_native_fragment_retirement_20261002](baselines/rocm_folded_native_fragment_retirement_20261002/README.md); [rocm_folded_graph_windows_20261002](baselines/rocm_folded_graph_windows_20261002/README.md); [rocm_folded_native_runtime_mn_20261002](baselines/rocm_folded_native_runtime_mn_20261002/README.md); [rocm_folded_native_package_20261002](baselines/rocm_folded_native_package_20261002/README.md); [rocm_folded_cold_branch_20261002](baselines/rocm_folded_cold_branch_20261002/README.md); [rocm_folded_panel_group_20261002](baselines/rocm_folded_panel_group_20261002/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [record_gfx1201_folded_register_pressure](rocm/record_gfx1201_folded_register_pressure.py) | [rocm_folded_terminal_prefetch_20261002](baselines/rocm_folded_terminal_prefetch_20261002/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [record_gfx1201_matmul_shape_key](rocm/record_gfx1201_matmul_shape_key.py) | [rocm_gfx1201_shape_key_20261002](baselines/rocm_gfx1201_shape_key_20261002/README.md) | Receipt scope and limitations are recorded in the linked packet. |
| [record_gfx1201_mxfp8_native_boundary](rocm/record_gfx1201_mxfp8_native_boundary.py) | [rocm_mxfp8_schedule_20261003](baselines/rocm_mxfp8_schedule_20261003/README.md) | Receipt scope and limitations are recorded in the linked packet. |

- `benchmarks/nvidia/benchmark_nvfp4_shared_rhs_batch.py`: correctness-gated native shared-RHS NVFP4 batches versus serial native calls; [owning packet and separate timing domains](baselines/compiler_contract_revalidation_20261006/README.md). No generic batching or selector promotion.

## Named SM120 NVFP4 frontend recorders

- `benchmarks/nvidia/record_nvfp4_tensor_jit.py`: ordinary logical-storage JIT, static batch/vmap and typed constraint proofs; independent decoded oracle and separate public wall/native-event timing.
- `benchmarks/nvidia/record_nvfp4_transpose.py`: native Graph/Schedule/Tile operand orientation, packed storage and K16 scale checks; identical quantized inputs across orientation arms and exact-device source/tool receipts.

Packets and validation scope: [compiler contract revalidation](baselines/compiler_contract_revalidation_20261006/README.md). General batching and AD closure remain open.

## Public saved-LSE native recorder

`nvidia/record_public_saved_lse_attention.py` records ordinary SM120 saved-LSE
JIT, portable replay and resident CUDA-event evidence in
`baselines/nvidia_ordinary_attention_20261006/public-saved-lse-rtx5070.json`.
Independent FP64 correctness checks gate both outputs before timing.

`nvidia/record_lse_checkpoint.py` also records version-4 timed-output oracle
checks in `baselines/nvidia_checkpoint_event_readback_20261006/`. Native
readback follows the stop event and is excluded from CUDA-event latency.

Native ROCm paged-KV A/B recorder: benchmarks/rocm/record_paged_kv_index_ab.py; exact-architecture evidence: benchmarks/baselines/rocm_paged_kv_flat_index_20261007/README.md.

## Typed scaled-program recorders

These recorders bind the named gfx1201 envelopes below. Native event windows,
host wall costs and diagnostic member plans retain their separate meanings.

| Recorder | Evidence consumer |
| --- | --- |
| `baselines/scaled_matmul_native_members_20261007/validate_members.py` | Same packet: member images and independent numerical checks; diagnostic member launcher. |
| `baselines/scaled_matmul_native_package_20261007/validate_projected_package.py` | Same packet: compiler-projected package numerical and ABI checks. |
| `rocm/record_native_scaled_program.py` | Diagnostic native-owner plan admission and numerical/measurement output selected by --output. |
| `rocm/benchmark_public_scaled_jvp.py` | Public scale-JVP and native-owner cost packets under baselines/scaled_matmul_public_jvp_20261007. |
| `rocm/benchmark_native_program_format_staging.py` | baselines/scaled_program_three_format_staging_20261007: FP8/MXFP8/folded-MXFP4 diagnostic staging. |
| `rocm/benchmark_public_typed_scaled_primal.py` | baselines/rocm_typed_scaled_primal_20261007 and rocm_mxfp8_public_primal_20261007: public primal execution. |
| `rocm/benchmark_native_primal_projection.py` | baselines/rocm_primal_image_projection_20261007: paired image-projection attribution. |
| `rocm/benchmark_public_primal_transfers.py` | baselines/rocm_primal_transfer_attribution_20261007: public transfer attribution. |
| `rocm/benchmark_shared_scaled_batch.py` | baselines/rocm_shared_scaled_batch_20261007: native shared-RHS primal/JVP batch cost. |
| `rocm/benchmark_independent_scaled_batch.py` | baselines/rocm_independent_scaled_batch_20261007: independent-RHS/shared-LHS cost. |
| `rocm/benchmark_native_plan_binding.py` | baselines/rocm_native_plan_binding_20261007: complete-content ABI binding A/B. |
| `rocm/benchmark_public_typed_scaled_vmap.py` | baselines/rocm_public_typed_vmap_20261007: public leading-map cost. |
| `rocm/benchmark_public_mapped_scaled_jvp.py` | baselines/rocm_public_mapped_jvp_20261007: mapped native scale-JVP cost. |
| `rocm/benchmark_multidimensional_scaled_batch.py` | baselines/rocm_multidimensional_scaled_batch_20261007: static two-axis FP8/MXFP8 native numerics and separate public/prepared/HIP-event costs. |
| `rocm/benchmark_native_scaled_vjp.py` | baselines/rocm_native_scaled_vjp_20261007: native scale-gradient numerics and separate one/two-member HIP events plus prepared host costs. |
| `rocm/benchmark_nested_typed_scaled_vmap.py` | baselines/rocm_nested_typed_vmap_20261007: two public maps, native batch owner and separate public/prepared/HIP-event costs. |

The public scale-VJP regime recorder benchmarks/rocm/benchmark_native_scaled_vjp.py
is consumed by benchmarks/baselines/rocm_public_scaled_vjp_regimes_20261007/README.md,
including explicit --public-compare-wave --shape B0 B1 M N K execution.

| Archived recorder | Evidence consumer |
| --- | --- |
| baselines/rocm_nvfp4_short_m_20261007/recorder.py | Same packet: twelve correctness-gated packed resident shapes with native event, graph and prepared wall timings. |
| baselines/rocm_deep_leading_map_20261007/recorder.py | Same packet: equivalent two/three/four-map prefixes, alternating public serial/wave walls and separate native event windows; --output selects the receipt. |

| Native resident recorder | Evidence consumer |
| --- | --- |
| nvidia/record_resident_tensor_owner.py | Native resident producer/matmul image ownership: alternating same-image synchronized call walls, independent pre/post numerics, separate component CUDA event dispatch windows; TESSERA_RESIDENT_OWNER_PACKET selects the receipt. |
| nvidia/record_padded_resident_tensor_owner.py | Padded row/column offset RHS: same-image synchronized call A/B, pre/post numerics and separate component CUDA event dispatch; TESSERA_RESIDENT_OWNER_PACKET selects the receipt. |

### Native owner integration evidence — 2026-10-08

The component recorders below are frozen captures of the tested scratch
experiments; their original source/runtime identities are retained in the packets.
The integrated producer characterization is rerunnable through
benchmark_native_producer_chain.py --long-macro.

| Recorder | Evidence and scope |
| --- | --- |
| [benchmark_softmax_staging_ab.py](baselines/nvidia_native_owner_integration_20261008/softmax-staging/benchmark_softmax_staging_ab.py) | Five-window native staging A/B; public host wall and resident events remain separate. |
| [check_macro_edges.py](baselines/nvidia_native_owner_integration_20261008/macro-producer/check_macro_edges.py) | Frozen isolated macro/typed producer numerical and retained-output checks. |
| [check_softmax_staging_lifetime_20261008.py](baselines/nvidia_native_owner_integration_20261008/softmax-staging/check_softmax_staging_lifetime_20261008.py) | Frozen growing/shrinking staging and matmul-interleave lifetime check; captured filename normalized to identifier spelling. |

See baselines/nvidia_native_owner_integration_20261008/README.md for the
matching combined-build execution and stage timing receipts.

Cooperative softmax compiler-only proof captures:
[check_schedule_contract.py](baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/check_schedule_contract.py)
checks native hashing/replay and refusal boundaries;
[check_frontend_contract.py](baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/check_frontend_contract.py)
checks real frontend traces and caller-owned Graph preservation.
These are frozen prototype receipts; GPU execution and promotion remain pending.

Cooperative softmax package proof recorders: check_package_device,
check_softmax_ab and check_package_corruption are frozen in
benchmarks/baselines/nvidia_native_owner_integration_20261008/cooperative-softmax/.

The isolated cooperative regression fixture is
test_nvidia_scheduled_kernel_contract.py in the same frozen packet directory.

check_chain_device.py records the isolated cooperative softmax native
producer-chain execution and retained-output lifetime proof in that packet.

check_chain_ab.py records the matched alternating prepared full-chain
softmax policy comparison in the cooperative-softmax packet.

benchmark_cooperative_softmax_chain.py records matched prepared whole-chain
host timing for the explicit native softmax Schedule policy.

benchmark_composed_scaled_jvp.py records the two-product public scale-JVP native event and separate host timings in baselines/gfx1201_composed_scaled_jvp_20261008.

benchmark_composed_scaled_vjp.py records independent/shared scale-adjoint correctness, complete native event windows and separate public host costs; owning evidence: baselines/gfx1201_composed_scaled_vjp_20261008/README.md.

benchmark_composed_scaled_maps.py records gfx1201 mapped product/sum primal/JVP/VJP correctness and separate complete-native-event/public-host timings; evidence: baselines/gfx1201_composed_scaled_maps_20261008/README.md.

benchmark_attention_async_owner.py records SM120 saved-LSE native-owner submission, completed host and full owned-stream event costs with independent numerics and identical images; evidence: baselines/nvidia_attention_async_owner_20261008/README.md.

### Native bounded attention images (SM120)

Run benchmarks/nvidia/benchmark_bounded_attention_images.py with --output PATH
on the owning RTX 5070 after loading the matching compiler/CUDA environment.
This recorder proves bounded native checkpoint images across Sq/Sk with
independent numerical checks and separate event/host launch windows.
Public JIT/package ABI integration is explicitly pending.
Evidence: baselines/nvidia_bounded_attention_native_20261008/README.md.

### Checked bounded saved-LSE packages (SM120)

benchmarks/nvidia/benchmark_bounded_attention_packages.py --output PATH
records completed capture, backward submission, completed host and private
CUDA event windows. See baselines/nvidia_bounded_attention_packages_20261008.
The public JIT/automatic AD integration boundary remains explicit.
