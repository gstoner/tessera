# Tessera API Reference — Index

> **Start here for authoritative documentation.** This page is a navigation map
> into the normative specs and reference docs. The original four-volume API
> reference is **archived** (see the bottom of this page); use the canonical
> references below for all current work.
>
> Counts and section numbers drift — this index links to docs and named
> sections rather than copying numeric snapshots (per Architecture Decision
> #26, the generated dashboards under `docs/audit/generated/` are the count
> authority).

---

## Canonical References (start here)

| What you need | Authoritative document |
|---------------|----------------------|
| **Single naming authority** — wins all disputes | [`docs/CANONICAL_API.md`](../CANONICAL_API.md) |
| **Tensor attributes and dtypes** (`shape`, `dtype`, `layout`, `target`, `distribution`, `numeric_policy`) | [`docs/reference/tessera_tensor_attributes.md`](../reference/tessera_tensor_attributes.md) |
| **Complete Python API** (all public symbols; normative) | [`docs/spec/PYTHON_API_SPEC.md`](../spec/PYTHON_API_SPEC.md) |
| **Compiler structure** (IR layers, pass registry, named pipelines, production JIT lane) | [`docs/spec/COMPILER_REFERENCE.md`](../spec/COMPILER_REFERENCE.md) |
| **Lowering pipelines** (all named pipelines, input/output IR contracts, invariants) | [`docs/spec/LOWERING_PIPELINE_SPEC.md`](../spec/LOWERING_PIPELINE_SPEC.md) |
| **Graph IR ops** (`tessera.*` dialect ops + canonicalization patterns) | [`docs/spec/GRAPH_IR_SPEC.md`](../spec/GRAPH_IR_SPEC.md) |
| **Target IR dialects** (Schedule, Attn, Queue; NVIDIA / ROCm / Apple contracts; backend-neutral value lane) | [`docs/spec/TARGET_IR_SPEC.md`](../spec/TARGET_IR_SPEC.md) |
| **Runtime C ABI** (all `tsr*` functions, types, error model, G7 GPU launch bridge) | [`docs/spec/RUNTIME_ABI_SPEC.md`](../spec/RUNTIME_ABI_SPEC.md) |
| **Autodiff** (tape, transforms, custom rules) | [`docs/spec/AUTODIFF_SPEC.md`](../spec/AUTODIFF_SPEC.md) |
| **Per-primitive contract status** (count authority) | [`docs/audit/generated/s_series_status.md`](../audit/generated/s_series_status.md) |

---

## Quick Lookup by Topic

> Section numbers below point into [`PYTHON_API_SPEC.md`](../spec/PYTHON_API_SPEC.md)
> unless noted otherwise.

### Decorators

| Symbol | Spec location |
|--------|--------------|
| `@tessera.jit` | PYTHON_API_SPEC §2.1 |
| `@tessera.kernel` | PYTHON_API_SPEC §2.2 |

### Type Annotations

| Symbol | Spec location |
|--------|--------------|
| `tessera.Region["read"/"write"/"reduce_sum"]` | PYTHON_API_SPEC §3 |
| `tessera.f16[..., ...]`, `tessera.mut_f32[..., ...]`, `tessera.Tensor["B", "D"]` | PYTHON_API_SPEC §14 (Tensor Annotations) |
| Canonical dtype names, aliases, planned/gated dtypes, `tessera.dtype` helpers | PYTHON_API_SPEC §15 (Dtype Annotations) · [Tensor Attributes And Dtypes](../reference/tessera_tensor_attributes.md) |

### Distribution API

| Symbol | Spec location |
|--------|--------------|
| `tessera.domain.Rect` | PYTHON_API_SPEC §4 |
| `tessera.dist.Block`, `.Cyclic`, `.Replicated` | PYTHON_API_SPEC §5 |
| `tessera.array.from_domain` | PYTHON_API_SPEC §6 |
| `DistributedArray.parts(axis)` | PYTHON_API_SPEC §6.2 |
| `ShardSpec` | PYTHON_API_SPEC §7 |
| `tessera.index_launch` | PYTHON_API_SPEC §8 |

### Constraints and Effects

| Symbol | Spec location |
|--------|--------------|
| `tessera.constraint.Divisible`, `.Range`, `.Equal` | PYTHON_API_SPEC §9 |
| `tessera.compiler.effects.Effect` enum (`pure`/`random`/`movement`/`state`/`collective`) | PYTHON_API_SPEC §10 |

### GPU / Target

| Symbol | Spec location |
|--------|--------------|
| `GPUTargetProfile`, `ISA` enum | PYTHON_API_SPEC §11 |
| `FlashAttnLoweringConfig` | PYTHON_API_SPEC §12 |
| String targets (`"apple_cpu"`, `"apple_gpu"`, `"rocm"`, `"metalium"`) | [tessera-api-reference §Targeting](../reference/tessera-api-reference.md) |

### Operations

| Symbol | Spec location |
|--------|--------------|
| `tessera.ops.*` (the standard operator library) | PYTHON_API_SPEC §13 · [Tessera_Standard_Operations.md](../operations/Tessera_Standard_Operations.md) |

### Fusion Middle-End / Kernel Synthesis (`tessera.compiler.fusion`)

| Symbol | Spec location |
|--------|--------------|
| `FusedRegion`, `AttentionRegion` (region IR) | [tessera-api-reference §Fusion Middle-End](../reference/tessera-api-reference.md) |
| `discover_fusable_regions`, `discover_attention_regions` (discovery) | [tessera-api-reference §Fusion Middle-End](../reference/tessera-api-reference.md) |
| `synthesize_matmul_epilogue_msl`, `synthesize_matmul_epilogue_msl_tiled`, `synthesize_attention_msl` (synthesis) | [tessera-api-reference §Fusion Middle-End](../reference/tessera-api-reference.md) |
| `run_fused_region`, `run_fused_attention` (dispatch) | [tessera-api-reference §Fusion Middle-End](../reference/tessera-api-reference.md) |
| `fusion_cost`, `attention_cost`, `should_fuse_region`, `should_fuse_attention` (F3 cost model) | [tessera-api-reference §Fusion Middle-End](../reference/tessera-api-reference.md) |
| `verify_synthesized_region`, `verify_synthesized_attention` (F4 codegen-gated oracle) | [tessera-api-reference §Fusion Middle-End](../reference/tessera-api-reference.md) |
| `autotune_matmul_epilogue`, `best_variant_for` (F5 autotune) | [tessera-api-reference §Fusion Middle-End](../reference/tessera-api-reference.md) |
| `EPILOGUE_OPS`, `REDUCTION_OPS`, `SYNTH_VARIANTS`, `SYNTH_DTYPES`, `SYNTH_MAX_N`, `SYNTH_MAX_N_TILED` (vocabulary / caps) | [tessera-api-reference §Fusion Middle-End](../reference/tessera-api-reference.md) |
| Phased design + catalog retirement status | [OPTIMIZING_COMPILER_PLAN.md](../audit/compiler/OPTIMIZING_COMPILER_PLAN.md) |

### Error Types

| Symbol | Spec location |
|--------|--------------|
| `TesseraConstraintError`, `TesseraEffectError`, `TesseraJitError`, `TesseraTargetError`, `TesseraAttnConfigError`, `TesseraPrivilegeError`, `MockCollectiveError` | PYTHON_API_SPEC §16 |

### Testing

| Symbol | Spec location |
|--------|--------------|
| `MockRankGroup`, `MockRank` | PYTHON_API_SPEC §17 |

### Profiling and Model Analysis

| Symbol | Spec location |
|--------|--------------|
| `plan_profile`, `ProviderCapability`, `ModelAnalyzerSweep` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `IntraKernelProbe`, `plan_intra_kernel_probes` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `model_analyzer_manifest`, `ModelAnalyzerManifest` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `run_model_analyzer_manifest`, `AnalyzerTrial` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `annotate_target_ir_with_probes` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `AppleProfilerContext`, `classify_apple_profiler_context`, `apple_unified_memory_bandwidth_ceiling_gbs` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `AcceleratorProfilerContext`, `classify_accelerator_profiler_context`, `accelerator_profiler_context_contract` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `build_profiler_context_artifact`, `write_profiler_context_artifact`, `load_profiler_context_artifact`, `summarize_profiler_context` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `collect_profiler_context`, `mock_profiler_context`, `sample_nvidia_nvml_context`, `sample_rocm_amdsmi_context`, `sample_apple_system_context` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `ProviderTraceRecord`, `build_provider_trace_artifact`, `normalize_rocprofiler_api_record`, `normalize_rocprofiler_activity_record`, `normalize_rocprofiler_counter_record`, `normalize_rocprofiler_thread_trace_record` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `normalize_metal_command_buffer_record`, `normalize_metal_counter_record`, `normalize_cupti_callback_record`, `normalize_cupti_activity_record` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `merge_profiler_traces`, `write_merged_profiler_trace` | [Tessera_Profiling_And_Autotuning_Guide.md](../guides/Tessera_Profiling_And_Autotuning_Guide.md) |
| `tprof::attach_tessera_runtime_trace`, `tprof::detach_tessera_runtime_trace` | [`tools/profiler/README.md`](../../tools/profiler/README.md) |
| `tprof::provider_shell`, `tprof::provider_shells`, `tprof::native_system_context_init`, `tprof::heavy_provider_init` | [`tools/profiler/README.md`](../../tools/profiler/README.md) |
| `tprof::rocprofiler_adapter_init`, `tprof::metal_adapter_init`, `tprof::cupti_adapter_init` | [`tools/profiler/README.md`](../../tools/profiler/README.md) |

### Runtime C ABI

| Symbol | Spec location |
|--------|--------------|
| `tsrInit`, `tsrShutdown` | RUNTIME_ABI_SPEC §5.1 |
| `tsrGetDevice`, `tsrGetDeviceProps` | RUNTIME_ABI_SPEC §5.2 |
| `tsrCreateStream`, `tsrStreamSynchronize` | RUNTIME_ABI_SPEC §5.3 |
| `tsrMalloc`, `tsrFree`, `tsrMemcpy` | RUNTIME_ABI_SPEC §5.5 |
| `tsrLaunchHostTileKernel` (host fn-pointer kernel) | RUNTIME_ABI_SPEC §5.6 |
| `tsrLaunchKernel`, `tsrRegisterGpuLauncher` (G7 GPU launch bridge) | RUNTIME_ABI_SPEC §5.6.1 |
| `tsrSetProfileEventCallback`, `TsrProfileEventKind`, `tsrProfileEventFn` | RUNTIME_ABI_SPEC §5.7 |
| `TsrStatus` enum | RUNTIME_ABI_SPEC §4 |
| Python wrapper `tessera.runtime.TesseraRuntime`, `tessera.runtime.launch` | RUNTIME_ABI_SPEC §10 |

---

## Archived Pre-Canonical Volumes

The old four-volume API reference was moved to `archive/docs/pre_canonical/api/`.
Those files predate the normative spec and are retained for historical reference only.

| Volume | Contents | Status |
|--------|----------|--------|
| [Vol 1: Frontend & Type System](../../archive/docs/pre_canonical/api/Tessera_API_Vol1_Frontend_and_TypeSystem.md) | Python/Rust APIs, type system, numerical policies, effects | Archived |
| [Vol 2: Operations](../../archive/docs/pre_canonical/api/Tessera_API_Vol2_Operations.md) | Normalization, activations, attention, positional encodings, distributed ops | Archived |
| [Vol 3: IR & Target](../../archive/docs/pre_canonical/api/Tessera_API_Vol3_IR_and_Target.md) | Graph IR, Schedule IR, Tile IR, Target IR, passes | Archived |
| [Vol 4: Runtime & Deployment](../../archive/docs/pre_canonical/api/Tessera_API_Vol4_Runtime_and_Deployment.md) | Runtime engine, host-device, autotuning, profiling, deployment | Archived |

For all new work, use the **canonical references** in the table at the top of this document.
