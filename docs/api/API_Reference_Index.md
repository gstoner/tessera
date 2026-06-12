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

### Error Types

| Symbol | Spec location |
|--------|--------------|
| `TesseraConstraintError`, `TesseraEffectError`, `TesseraJitError`, `TesseraTargetError`, `TesseraAttnConfigError`, `TesseraPrivilegeError`, `MockCollectiveError` | PYTHON_API_SPEC §16 |

### Testing

| Symbol | Spec location |
|--------|--------------|
| `MockRankGroup`, `MockRank` | PYTHON_API_SPEC §17 |

### Runtime C ABI

| Symbol | Spec location |
|--------|--------------|
| `tsrInit`, `tsrShutdown` | RUNTIME_ABI_SPEC §5.1 |
| `tsrGetDevice`, `tsrGetDeviceProps` | RUNTIME_ABI_SPEC §5.2 |
| `tsrCreateStream`, `tsrStreamSynchronize` | RUNTIME_ABI_SPEC §5.3 |
| `tsrMalloc`, `tsrFree`, `tsrMemcpy` | RUNTIME_ABI_SPEC §5.5 |
| `tsrLaunchHostTileKernel` (host fn-pointer kernel) | RUNTIME_ABI_SPEC §5.6 |
| `tsrLaunchKernel`, `tsrRegisterGpuLauncher` (G7 GPU launch bridge) | RUNTIME_ABI_SPEC §5.6.1 |
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
