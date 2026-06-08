# TSOL Coverage Dashboard

Generated from `python/tessera/compiler/tsol_coverage.py`.  Don't edit by hand — regenerate via `python -c "from tessera.compiler.tsol_coverage import render_dashboard; open('docs/audit/generated/tsol_coverage.md', 'w').write(render_dashboard())"`.  Drift gated by `tests/unit/test_tsol_coverage.py`.

Spec: `docs/operations/Tessera_Standard_Operations.md`.  Full primitive registry: `docs/audit/standalone_primitive_coverage.md`.

## Headline

- **47** canonical TSOL ops in the spec catalog.
- **47** of those have a matching row in `primitive_coverage.py`.

## Per-axis status counts (TSOL slice only)

Counts below are restricted to the TSOL canonical names.  The full 432-primitive registry is summarised in `docs/audit/standalone_primitive_coverage.md`.

| Axis | complete | partial | planned | N/A | other |
|------|----------|---------|---------|-----|-------|
| `math_semantics` |  47 |   0 |   0 |   0 |   0 |
| `shape_rule` |  47 |   0 |   0 |   0 |   0 |
| `dtype_layout_rule` |  47 |   0 |   0 |   0 |   0 |
| `vjp` |  41 |   0 |   0 |   6 |   0 |
| `jvp` |  40 |   0 |   0 |   7 |   0 |
| `lowering_rule` |  47 |   0 |   0 |   0 |   0 |
| `sharding_rule` |  31 |  16 |   0 |   0 |   0 |
| `backend_kernel` |   0 |  47 |   0 |   0 |   0 |

## Per-op coverage

Status legend: ✅ `complete`  • ◐ `partial`  • ◯ `planned`  • – `not_applicable`  • ? `unknown` / missing registry entry.

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
| `dropout` | ✅ | ✅ | ✅ | ✅ | – | ✅ | ✅ | ◐ |
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
| `stft` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `istft` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `spectral_filter` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |

### Sparse, Segment, and Graph Operators

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `spmm_coo` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ | ◐ |
| `spmm_csr` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `sddmm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `bsmm` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `segment_reduce` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |

### RNG and Initialization

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `rng_uniform` | ✅ | ✅ | ✅ | – | – | ✅ | ✅ | ◐ |
| `rng_normal` | ✅ | ✅ | ✅ | – | – | ✅ | ✅ | ◐ |

### Collectives

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `all_reduce` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `reduce_scatter` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `all_gather` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `all_to_all` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |

### Layout and Packing

| Op | math | shape | dtype | vjp | jvp | lowering | sharding | backend |
|----|------|-------|-------|-----|-----|----------|----------|---------|
| `transpose` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ◐ |
| `rearrange` | ✅ | ✅ | ✅ | – | – | ✅ | ✅ | ◐ |
| `pack` | ✅ | ✅ | ✅ | – | – | ✅ | ✅ | ◐ |
| `unpack` | ✅ | ✅ | ✅ | – | – | ✅ | ✅ | ◐ |
| `tile_view` | ✅ | ✅ | ✅ | – | – | ✅ | ✅ | ◐ |

## Notable gaps

_None today — every TSOL canonical op has a registry entry, a VJP (or N/A), and a JVP (or N/A)._

## Backend kernel honest baseline

Per the registry's gating rule (`primitive_coverage.py` line 351-352), `backend_kernel = complete` requires every declared target to ship a real hardware kernel with numerical proof.  Today **zero** TSOL entries can claim `complete` because NVIDIA / ROCm / NVIDIA / ROCm proofs aren't available on this Mac.  See `docs/audit/backend/BACKEND_AUDIT.md` for the full hardware-gated punch list.
