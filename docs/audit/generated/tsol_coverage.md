# TSOL Coverage Dashboard

Generated from `python/tessera/compiler/tsol_coverage.py`.  Don't edit by hand — regenerate via `python -m tessera.compiler.generated_docs --write tsol_coverage`, which writes this file *and* its CSV companion.  Drift gated by `tests/unit/test_tsol_coverage.py` and `scripts/check_generated_docs.sh`.

Spec: `docs/operations/Tessera_Standard_Operations.md`.  Full primitive registry: `docs/audit/standalone_primitive_coverage.md`.

## Headline

- **51** canonical TSOL ops in the spec catalog.
- **51** of those have a matching row in `primitive_coverage.py`.

## Per-axis status counts (TSOL slice only)

Counts below are restricted to the 51 TSOL canonical names.  The full 491-primitive registry is summarised in `docs/audit/standalone_primitive_coverage.md`.

| Axis | complete | partial | planned | by-design | other |
|------|----------|---------|---------|-----|-------|
| `math_semantics` |  51 |   0 |   0 |   0 |   0 |
| `shape_rule` |  51 |   0 |   0 |   0 |   0 |
| `dtype_layout_rule` |  51 |   0 |   0 |   0 |   0 |
| `vjp` |  43 |   0 |   0 |   8 |   0 |
| `jvp` |  42 |   0 |   0 |   9 |   0 |
| `lowering_rule` |  51 |   0 |   0 |   0 |   0 |
| `sharding_rule` |  33 |  18 |   0 |   0 |   0 |
| `backend_kernel` |   0 |  50 |   0 |   1 |   0 |

## Per-op coverage

Status legend: ✅ `complete`  • ◐ `partial`  • ◯ `planned`  • — explicit by-design disposition  • ? `unknown` / missing registry entry.

### Linear Algebra

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `gemm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `matmul` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `batched_gemm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `einsum` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `factorized_matmul` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `tri_solve` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `cholesky` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `qr` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `svd` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |

### Neural Network Primitives

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `conv2d` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `conv3d` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `layer_norm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `rmsnorm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `softmax` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `gelu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `relu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `silu` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `dropout` | ✅ | ✅ | ✅ | ✅ | — `non_differentiable` | ✅ | ✅ | ◐ |
| `qkv_projection` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `flash_attn` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `rope` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `moe` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `moe_dispatch` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `moe_combine` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |

### Spectral Operators

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `fft` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `ifft` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `rfft` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `irfft` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `dct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `stft` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `istft` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `spectral_conv` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `spectral_filter` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |

### Sparse, Segment, And Graph Operators

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `spmm_coo` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `spmm_csr` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `sddmm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `bsmm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `segment_reduce` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |

### RNG And Initialization

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `rng_uniform` | ✅ | ✅ | ✅ | — `non_differentiable` | — `non_differentiable` | ✅ | ✅ | ◐ |
| `rng_normal` | ✅ | ✅ | ✅ | — `non_differentiable` | — `non_differentiable` | ✅ | ✅ | ◐ |
| `rng_philox_uniform` | ✅ | ✅ | ✅ | — `non_differentiable` | — `non_differentiable` | ✅ | ✅ | ◐ |
| `rng_philox_normal` | ✅ | ✅ | ✅ | — `non_differentiable` | — `non_differentiable` | ✅ | ✅ | ◐ |

### Collectives

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `all_reduce` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `reduce_scatter` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `all_gather` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `all_to_all` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |

### Layout And Packing

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `transpose` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `rearrange` | ✅ | ✅ | ✅ | — `non_differentiable` | — `non_differentiable` | ✅ | ✅ | ◐ |
| `pack` | ✅ | ✅ | ✅ | — `non_differentiable` | — `non_differentiable` | ✅ | ✅ | ◐ |
| `unpack` | ✅ | ✅ | ✅ | — `non_differentiable` | — `non_differentiable` | ✅ | ✅ | ◐ |
| `tile_view` | ✅ | ✅ | ✅ | — `non_differentiable` | — `non_differentiable` | ✅ | ✅ | — `no_kernel_required` |

## Notable gaps

_None today — every TSOL canonical op has a registry entry and an explicit VJP/JVP implementation or by-design disposition._

## Backend kernel honest baseline

Per the registry's `backend_kernel` gating rule (see the "backend_kernel stays partial until each backend ships a real" note in `primitive_coverage.py`), `backend_kernel = complete` requires every declared target to ship a real hardware kernel with numerical proof.  Today **zero** of the 51 TSOL entries can claim that all-target aggregate.  Per-target native proof is reported separately and may exist even while this aggregate remains incomplete.  See `docs/audit/backend/BACKEND_AUDIT.md` and its target maps for the exact-target evidence and remaining punch list.
