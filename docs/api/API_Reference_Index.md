# Tessera API Reference — Index

> **Start here for authoritative documentation.** The four volumes in this directory are
> pre-canonical references that predate the normative spec. Use the tables below to find
> the right document for your question.

---

## Canonical References (start here)

| What you need | Authoritative document |
|---------------|----------------------|
| **Single naming authority** — wins all disputes | [`docs/CANONICAL_API.md`](../CANONICAL_API.md) |
| **Complete Python API** (all public symbols, Phases 1–3) | [`docs/spec/PYTHON_API_SPEC.md`](../spec/PYTHON_API_SPEC.md) |
| **Compiler pass pipeline** (both named pipelines, all passes) | [`docs/spec/COMPILER_REFERENCE.md`](../spec/COMPILER_REFERENCE.md) |
| **Graph IR ops** (all 6 ops, 4 canonicalization patterns) | [`docs/spec/GRAPH_IR_SPEC.md`](../spec/GRAPH_IR_SPEC.md) |
| **Lowering passes** (input/output IR contracts, invariants) | [`docs/spec/LOWERING_PIPELINE_SPEC.md`](../spec/LOWERING_PIPELINE_SPEC.md) |
| **Target IR dialects** (Schedule, Attn, Queue, NVIDIA ops) | [`docs/spec/TARGET_IR_SPEC.md`](../spec/TARGET_IR_SPEC.md) |
| **Runtime C ABI** (all `tsr*` functions, types, error model) | [`docs/spec/RUNTIME_ABI_SPEC.md`](../spec/RUNTIME_ABI_SPEC.md) |

---

## Quick Lookup by Topic

### Decorators

| Symbol | Spec location |
|--------|--------------|
| `@tessera.jit` | [PYTHON_API_SPEC §2](../spec/PYTHON_API_SPEC.md) |
| `@tessera.kernel` | [PYTHON_API_SPEC §3](../spec/PYTHON_API_SPEC.md) |

### Type Annotations

| Symbol | Spec location |
|--------|--------------|
| `tessera.Region["read"/"write"/"reduce_sum"]` | [PYTHON_API_SPEC §4](../spec/PYTHON_API_SPEC.md) |
| `tessera.f16[..., ...]`, `tessera.mut_f32[..., ...]` | [PYTHON_API_SPEC §16](../spec/PYTHON_API_SPEC.md) |
| `tessera.Tensor["B", "D"]` | [PYTHON_API_SPEC §16](../spec/PYTHON_API_SPEC.md) |

### Distribution API

| Symbol | Spec location |
|--------|--------------|
| `tessera.domain.Rect` | [PYTHON_API_SPEC §5](../spec/PYTHON_API_SPEC.md) |
| `tessera.dist.Block`, `.Cyclic`, `.Replicated` | [PYTHON_API_SPEC §6](../spec/PYTHON_API_SPEC.md) |
| `tessera.array.from_domain` | [PYTHON_API_SPEC §7](../spec/PYTHON_API_SPEC.md) |
| `DistributedArray.parts(axis)` | [PYTHON_API_SPEC §8](../spec/PYTHON_API_SPEC.md) |
| `tessera.index_launch` | [PYTHON_API_SPEC §10](../spec/PYTHON_API_SPEC.md) |

### Constraints and Effects

| Symbol | Spec location |
|--------|--------------|
| `tessera.constraint.Divisible`, `.Range`, `.Equal` | [PYTHON_API_SPEC §11](../spec/PYTHON_API_SPEC.md) |
| `tessera.EffectLevel` enum | [PYTHON_API_SPEC §12](../spec/PYTHON_API_SPEC.md) |

### GPU Target

| Symbol | Spec location |
|--------|--------------|
| `GPUTargetProfile`, `ISA` enum | [PYTHON_API_SPEC §13](../spec/PYTHON_API_SPEC.md) |
| `FlashAttnLoweringConfig` | [PYTHON_API_SPEC §14](../spec/PYTHON_API_SPEC.md) |

### Operations

| Symbol | Spec location |
|--------|--------------|
| `tessera.ops.gemm`, `.flash_attn`, `.layer_norm`, … (15 ops) | [PYTHON_API_SPEC §15](../spec/PYTHON_API_SPEC.md) |

### Error Types

| Symbol | Spec location |
|--------|--------------|
| `TesseraConstraintError`, `TesseraPrivilegeError`, … (7 types) | [PYTHON_API_SPEC §17](../spec/PYTHON_API_SPEC.md) |

### Testing

| Symbol | Spec location |
|--------|--------------|
| `MockRankGroup`, `MockRank` | [PYTHON_API_SPEC §18](../spec/PYTHON_API_SPEC.md) |

### Runtime C ABI

| Symbol | Spec location |
|--------|--------------|
| `tsrInit`, `tsrShutdown` | [RUNTIME_ABI_SPEC §5.1](../spec/RUNTIME_ABI_SPEC.md) |
| `tsrGetDevice`, `tsrGetDeviceProps` | [RUNTIME_ABI_SPEC §5.2](../spec/RUNTIME_ABI_SPEC.md) |
| `tsrCreateStream`, `tsrStreamSynchronize` | [RUNTIME_ABI_SPEC §5.3](../spec/RUNTIME_ABI_SPEC.md) |
| `tsrMalloc`, `tsrFree`, `tsrMemcpy` | [RUNTIME_ABI_SPEC §5.5](../spec/RUNTIME_ABI_SPEC.md) |
| `tsrLaunchHostTileKernel` | [RUNTIME_ABI_SPEC §5.6](../spec/RUNTIME_ABI_SPEC.md) |
| `TsrStatus` enum | [RUNTIME_ABI_SPEC §4](../spec/RUNTIME_ABI_SPEC.md) |

---

## Volume Index (Pre-Canonical)

These volumes predate the normative spec. They contain useful conceptual material but use
outdated API names. Each volume has a banner noting the specific corrections needed.

| Volume | Contents | Status |
|--------|----------|--------|
| [Vol 1: Frontend & Type System](Tessera_API_Vol1_Frontend_and_TypeSystem.md) | Python/Rust APIs, type system, numerical policies, effects | Pre-canonical — `@ts.function` → `@tessera.jit` |
| [Vol 2: Operations](Tessera_API_Vol2_Operations.md) | Normalization, activations, attention, positional encodings, distributed ops | Pre-canonical — `@tessera.function` → `@tessera.jit` |
| [Vol 3: IR & Target](Tessera_API_Vol3_IR_and_Target.md) | Graph IR, Schedule IR, Tile IR, Target IR, passes | Pre-canonical — stub level |
| [Vol 4: Runtime & Deployment](Tessera_API_Vol4_Runtime_and_Deployment.md) | Runtime engine, host-device, autotuning, profiling, deployment | Pre-canonical — see `RUNTIME_ABI_SPEC.md` |

For all new work, use the **canonical references** in the table at the top of this document.
