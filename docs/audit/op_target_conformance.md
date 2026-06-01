<!-- AUTO-GENERATED — DO NOT EDIT BY HAND. -->
<!-- Regenerate via: python -m tessera.cli.conformance_matrix --render -->

# Op×Target Conformance Matrix

This dashboard reports, per (op, target), where the op is on the seven-step proof ladder:

  `graph_emitted` → `schedule_legal` → `tile_legal` → `target_legal` → `backend_compile` → `runtime_execute` → `numerical_check`

The matrix is a **pure aggregator** over `primitive_coverage` (12-axis contracts), `backend_manifest` (per-target kernel status), `execution_matrix` (runtime executors), and the Apple-GPU runtime envelope sets. No proof column has its own private truth source — change the upstream status and the matrix regenerates.

Audit response to [docs/audit/compiler_layer_gap_remediation.md](compiler_layer_gap_remediation.md) recommendation **A**: the gap between *architecture-implied capability* and *executable capability* is now drift-gated rather than implicit.

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
| ✅ `complete` | 5 |
| ⚙️ `partial` | 12 |
| ⚠️ `artifact_only` | 0 |
| 📋 `planned` | 0 |
| ❌ `missing` | 25 |
| **total cells** | **42** |

## Surfaced upstream gaps

These cells are `missing` because the upstream truth source is incomplete, not because the path doesn't exist. Each row is an actionable follow-up: fix the upstream entry and the matrix regenerates cleanly.

| Op | Target | Upstream source | Fix |
|----|--------|-----------------|-----|
| `matmul_relu` | `apple_gpu` | `backend_manifest entry for 'relu'` | add an `apple_gpu` `BackendKernelEntry` for 'relu' in `backend_manifest.py` (runtime envelope already dispatches it) |

## `matmul`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ |  |
| `apple_cpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |  |
| `metalium` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |  |

## `matmul_relu`

_composes from primitives; no fused single-kernel today_

**Composition:** `matmul`, `relu`.  Fused-single-kernel targets: —.

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `apple_gpu` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `metalium` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | composes from per-op kernels (no fusion pass on this target) |

## `softmax`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ |  |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |  |
| `metalium` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |  |

## `matmul_softmax`

_fused MSL kernel on apple_gpu (single-kernel scores); compose elsewhere_

**Composition:** `matmul`, `softmax`.  Fused-single-kernel targets: apple_gpu.

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ⚙️ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | fused single-kernel on this target |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | composes from per-op kernels (no fusion pass on this target) |
| `metalium` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | composes from per-op kernels (no fusion pass on this target) |

## `conv2d`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ |  |
| `apple_cpu` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ❌ |  |
| `apple_gpu` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |
| `metalium` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |

## `flash_attn`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ |  |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ |  |
| `apple_gpu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ |  |
| `metalium` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |

## `kv_cache_read`

| target | overall | graph | schedule | tile | target_legal | backend_compile | runtime | numerical | notes |
|--------|---------|-------|----------|------|--------------|-----------------|---------|-----------|-------|
| `cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ |  |
| `apple_cpu` | ⚙️ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚙️ | ✅ |  |
| `apple_gpu` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |
| `nvidia` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |
| `rocm` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |
| `metalium` | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |  |

