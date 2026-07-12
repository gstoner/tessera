<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.conformance_matrix --render -->

# Op×Target Conformance Matrix

This dashboard reports, per (op, target), where the op is on the seven-step proof ladder:

  `graph_emitted` → `schedule_legal` → `tile_legal` → `target_legal` → `backend_compile` → `runtime_execute` → `numerical_check`

A cell is **complete** only when every proof column is `complete`. Its `first_failing_gate` is then empty (`—`); otherwise that field names the first incomplete proof rung. Rows use exact manifest target grain. `cpu` is the portable host reference lane; `x86` is the native x86 lane; NVIDIA architectures are separate rows.

The four IR columns are derived by compiling each curated program and running the typed Graph/Schedule/Tile/Target verifiers. Backend and runtime columns join exact-target `backend_manifest` evidence to an executable `execution_matrix` target row. Numerical completion requires an exact-target execute-and-compare fixture.

Audit response to [docs/audit/compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) recommendation **A**: the gap between *architecture-implied capability* and *executable capability* is now drift-gated rather than implicit.

## Status legend

| Symbol | Status | Meaning |
|--------|--------|---------|
| ✅ | `complete` | Real path lit up end-to-end on this target. |
| 🧪 | `reference` | Correct reference execution; no target-native compile claim. |
| 🔧 | `compileable` | Pinned backend compiler accepts the artifact; execution unproven. |
| ⚙️ | `partial` | Evidence exists but does not satisfy the rung's full contract. |
| ⚠️ | `artifact_only` | Target artifact emits; concrete backend compilation is absent. |
| 📋 | `planned` | Declared in the registry / manifest, not yet implemented. |
| ❌ | `missing` | The evidence required by this rung is absent. |
| ➖ | `not_applicable` | Concept does not apply to this target. |

## Derived family rollup

| Family | Exact-target cells | Status counts |
|---|---:|---|
| `host_reference` | 7 | reference=7 |
| `x86` | 7 | complete=7 |
| `apple` | 14 | complete=8, reference=6 |
| `rocm` | 7 | complete=7 |
| `nvidia` | 28 | complete=1, missing=27 |

## Overall counts

| Overall (weakest column wins) | Count |
|---|---:|
| ✅ `complete` | 23 |
| 🧪 `reference` | 13 |
| 🔧 `compileable` | 0 |
| ⚙️ `partial` | 0 |
| ⚠️ `artifact_only` | 0 |
| 📋 `planned` | 0 |
| ❌ `missing` | 27 |
| **total cells** | **63** |

## `matmul`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=matmul |  |
| `x86` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `apple_cpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `rocm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `nvidia_sm80` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=matmul |  |
| `nvidia_sm90` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=matmul |  |
| `nvidia_sm100` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=matmul |  |
| `nvidia_sm120` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |

## `matmul_relu`

_composes from primitives; no fused single-kernel today_

**Composition:** `matmul`, `relu`.  Fused-single-kernel targets: —.

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=matmul,relu | composes from per-op kernels (no fusion pass on this target) |
| `x86` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | composes from per-op kernels (no fusion pass on this target) |
| `apple_cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=matmul,relu | composes from per-op kernels (no fusion pass on this target) |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | composes from per-op kernels (no fusion pass on this target) |
| `rocm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | composes from per-op kernels (no fusion pass on this target) |
| `nvidia_sm80` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | `backend_compile` — backend_compile=missing; components=matmul,relu | composes from per-op kernels (no fusion pass on this target) |
| `nvidia_sm90` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | `backend_compile` — backend_compile=missing; components=matmul,relu | composes from per-op kernels (no fusion pass on this target) |
| `nvidia_sm100` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | `backend_compile` — backend_compile=missing; components=matmul,relu | composes from per-op kernels (no fusion pass on this target) |
| `nvidia_sm120` | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | `backend_compile` — backend_compile=missing; components=matmul,relu | composes from per-op kernels (no fusion pass on this target) |

## `softmax`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=softmax |  |
| `x86` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `apple_cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=softmax |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `rocm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `nvidia_sm80` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=softmax |  |
| `nvidia_sm90` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=softmax |  |
| `nvidia_sm100` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=softmax |  |
| `nvidia_sm120` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=softmax |  |

## `matmul_softmax`

_fused MSL kernel on apple_gpu (single-kernel scores); compose elsewhere_

**Composition:** `matmul`, `softmax`.  Fused-single-kernel targets: apple_gpu.

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=matmul,softmax | composes from per-op kernels (no fusion pass on this target) |
| `x86` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | composes from per-op kernels (no fusion pass on this target) |
| `apple_cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=matmul,softmax | composes from per-op kernels (no fusion pass on this target) |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | fused single-kernel on this target |
| `rocm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | composes from per-op kernels (no fusion pass on this target) |
| `nvidia_sm80` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=matmul,softmax | composes from per-op kernels (no fusion pass on this target) |
| `nvidia_sm90` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=matmul,softmax | composes from per-op kernels (no fusion pass on this target) |
| `nvidia_sm100` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=matmul,softmax | composes from per-op kernels (no fusion pass on this target) |
| `nvidia_sm120` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=matmul,softmax | composes from per-op kernels (no fusion pass on this target) |

## `conv2d`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=conv2d |  |
| `x86` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `apple_cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=conv2d |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `rocm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `nvidia_sm80` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=conv2d |  |
| `nvidia_sm90` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=conv2d |  |
| `nvidia_sm100` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=conv2d |  |
| `nvidia_sm120` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=conv2d |  |

## `flash_attn`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=flash_attn |  |
| `x86` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `apple_cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=flash_attn |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `rocm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `nvidia_sm80` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=flash_attn |  |
| `nvidia_sm90` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=flash_attn |  |
| `nvidia_sm100` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=flash_attn |  |
| `nvidia_sm120` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=flash_attn |  |

## `kv_cache_read`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=kv_cache_read |  |
| `x86` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `apple_cpu` | 🧪 | ✅ | ✅ | ✅ | ✅ | 🧪 | 🧪 | ✅ | `backend_compile` — backend_compile=reference; components=kv_cache_read |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `rocm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |  |
| `nvidia_sm80` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=kv_cache_read |  |
| `nvidia_sm90` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=kv_cache_read |  |
| `nvidia_sm100` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=kv_cache_read |  |
| `nvidia_sm120` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | `backend_compile` — backend_compile=artifact_only; components=kv_cache_read |  |

