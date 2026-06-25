<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.conformance_matrix --render -->

# Op×Target Conformance Matrix

This dashboard reports, per (op, target), where the op is on the seven-step proof ladder:

  `graph_emitted` → `schedule_legal` → `tile_legal` → `target_legal` → `backend_compile` → `runtime_execute` → `numerical_check`

The matrix is a **pure aggregator** over `primitive_coverage` (12-axis contracts), `backend_manifest` (per-target kernel status), `execution_matrix` (runtime executors), and the Apple-GPU runtime envelope sets. No proof column has its own private truth source — change the upstream status and the matrix regenerates.

Audit response to [docs/audit/compiler/COMPILER_AUDIT.md](compiler/COMPILER_AUDIT.md) recommendation **A**: the gap between *architecture-implied capability* and *executable capability* is now drift-gated rather than implicit.

## Status legend

| Symbol | Status | Meaning |
|--------|--------|---------|
| ✅ | `complete` | Real path lit up end-to-end on this target. |
| ⚙️ | `partial` | Works but with a known caveat (reference / composes / contract axis partial). |
| ⚠️ | `artifact_only` | IR emits a target artifact; no native compile / link / launch path yet (hardware-gated). |
| 📋 | `planned` | Declared in the registry / manifest, not yet implemented. |
| ❌ | `missing` | Not declared on this target. |
| ➖ | `not_applicable` | Concept does not apply to this target. |

## Overall counts

| Overall (weakest column wins) | Count |
|---|---:|
| ✅ `complete` | 6 |
| ⚙️ `partial` | 14 |
| ⚠️ `artifact_only` | 0 |
| 📋 `planned` | 0 |
| ❌ `missing` | 15 |
| **total cells** | **35** |

## `matmul`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ | — |  |
| `apple_cpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `hardware_smoke` — Apple silicon required for native execution |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `hardware_smoke` — Apple silicon required for native execution |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | `toolchain` — nvcc not on PATH (CUDA Toolkit 13.3 not installed) |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | `toolchain` — hipcc not on PATH (ROCm 7.2.4 not installed) |  |

## `matmul_relu`

_composes from primitives; no fused single-kernel today_

**Composition:** `matmul`, `relu`.  Fused-single-kernel targets: —.

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ⚙️ | — | composes from per-op kernels (no fusion pass on this target) |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ⚙️ | `hardware_smoke` — Apple silicon required for native execution | composes from per-op kernels (no fusion pass on this target) |
| `apple_gpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ | ⚙️ | `hardware_smoke` — Apple silicon required for native execution | composes from per-op kernels (no fusion pass on this target) |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚙️ | `toolchain` — nvcc not on PATH (CUDA Toolkit 13.3 not installed) | composes from per-op kernels (no fusion pass on this target) |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚙️ | `toolchain` — hipcc not on PATH (ROCm 7.2.4 not installed) | composes from per-op kernels (no fusion pass on this target) |

## `softmax`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | — |  |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | `hardware_smoke` — Apple silicon required for native execution |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `hardware_smoke` — Apple silicon required for native execution |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⚙️ | `toolchain` — nvcc not on PATH (CUDA Toolkit 13.3 not installed) |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⚙️ | `toolchain` — hipcc not on PATH (ROCm 7.2.4 not installed) |  |

## `matmul_softmax`

_fused MSL kernel on apple_gpu (single-kernel scores); compose elsewhere_

**Composition:** `matmul`, `softmax`.  Fused-single-kernel targets: apple_gpu.

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ⚙️ | — | composes from per-op kernels (no fusion pass on this target) |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ⚙️ | `hardware_smoke` — Apple silicon required for native execution | composes from per-op kernels (no fusion pass on this target) |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `hardware_smoke` — Apple silicon required for native execution | fused single-kernel on this target |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⚙️ | `toolchain` — nvcc not on PATH (CUDA Toolkit 13.3 not installed) | composes from per-op kernels (no fusion pass on this target) |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⚙️ | `toolchain` — hipcc not on PATH (ROCm 7.2.4 not installed) | composes from per-op kernels (no fusion pass on this target) |

## `conv2d`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ | — |  |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | `hardware_smoke` — Apple silicon required for native execution |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `hardware_smoke` — Apple silicon required for native execution |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚙️ | `codegen` — no backend_manifest entry for 'conv2d' on 'nvidia' |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚙️ | `codegen` — no backend_manifest entry for 'conv2d' on 'rocm' |  |

## `flash_attn`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | — |  |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | `hardware_smoke` — Apple silicon required for native execution |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | `hardware_smoke` — Apple silicon required for native execution |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⚙️ | `toolchain` — nvcc not on PATH (CUDA Toolkit 13.3 not installed) |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | `toolchain` — hipcc not on PATH (ROCm 7.2.4 not installed) |  |

## `kv_cache_read`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | first failing gate (B) | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|------------------------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ | — |  |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | `hardware_smoke` — Apple silicon required for native execution |  |
| `apple_gpu` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ⚙️ | `hardware_smoke` — Apple silicon required for native execution |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚙️ | `codegen` — no backend_manifest entry for 'kv_cache_read' on 'nvidia' |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚙️ | `codegen` — no backend_manifest entry for 'kv_cache_read' on 'rocm' |  |

