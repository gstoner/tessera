# Bootstrap Prune — Mainline Coverage Gap

**Generated. Do not hand-edit.** Regenerate with
`python -m tessera.compiler.generated_docs --write`.

The Python per-backend `package_*` inventory includes bootstrap and
artifact packagers; the architecture is core MLIR/LLVM (Graph → Schedule →
Tile → Target via `tessera-opt`). This dashboard answers what must be
settled before any of it is deleted: **which families does the
mainline compiler already cover, and which would lose their only
lowering?** Decision #31's ordering caveat is the rule — a duplicate
authority is removed only after the survivor is proven to carry what
it carried.

A `gap` row is *not* a defect. It is scope: work the mainline
compiler must absorb, or a fast path that must be re-expressed
through a declared Target IR boundary (Decision #28 Tier 3) before
the bootstrap row can go.

## Summary

| Metric | Count |
|---|---|
| Backends with a bootstrap module | 5 |
| `package_*` functions total | 72 |
| — Graph-input boundaries (including scheduled wrappers) | 45 |
|   ·  of the bootstrap, construct Tile IR then run `tessera-opt` | 11 |
|   ·  of the bootstrap, **delegate** (runtime compiler / library / object) | 2 |
|   ·  of the bootstrap, both | 1 |
|   ·  of the bootstrap, other (wrapper / dispatcher) | 31 |
| — typed scheduled-artifact inputs (consumption needs verification) | 14 |
| — unclassified/raw inputs (not assumed compiled) | 13 |
| Lines in those modules | 10461 |
| Classified family/target candidates (shape admission not implied) | 54 |
| — covered by a compiled route | 6 |
| — **gap (no declared family route)** | 48 |
| Packagers matching no family | 9 |

Graph input alone does not prove reconstruction: wrappers may call the
scheduled producer for some or all envelopes. Review direct calls and
exact artifacts before treating a row as a constructor deletion target.

## Per-backend bootstrap surface

| Target | Module | Graph input | Typed artifact input | Unknown/raw input | Family candidates | Lines |
|---|---|---|---|---|---|---|
| `nvidia_sm120` | `nvidia_native.py` | 19 | 1 | 11 | 12 | 3828 |
| `rocm_gfx1151` | `rocm_native.py` | 7 | 5 | 0 | 5 | 2922 |
| `x86` | `x86_native.py` | 7 | 4 | 1 | 8 | 1684 |
| `apple_cpu` | `apple_cpu_native.py` | 1 | 0 | 0 | 10 | 215 |
| `apple_gpu` | `apple_native.py` | 11 | 4 | 1 | 19 | 1812 |

## Census limits

Input annotations are inventory evidence, not proof of semantic authority.
Any, unannotated and raw-IR inputs remain unclassified; inspect their producers and consumers.
Known Apple computed returns are derived from their producer tables; other computed returns remain unresolved.
A missing family mapping is not proof that no generic scheduled route accepts it.

| Target | Unresolved classifier return |
|---|---|

## Family coverage

`compiled` means a family has a declared admission-predicate mapping.
The target module must also define the corresponding package consumer.
Actual driver paths, shapes and policies require separate checks.
It does **not** assert the compiled route reaches parity on
every shape and dtype — that is per-family evidence the backend
queues own.

| Target | Family | Compiled route | Status |
|---|---|---|---|
| `nvidia_sm120` | `attention_backward_lse` | — | 🔴 **gap** |
| `nvidia_sm120` | `attention_lse` | — | 🔴 **gap** |
| `nvidia_sm120` | `attention_backward` | — | 🔴 **gap** |
| `nvidia_sm120` | `paged_kv` | — | 🔴 **gap** |
| `nvidia_sm120` | `attention` | `scheduled_attention.supports_scheduled_attention` | ✅ compiled |
| `nvidia_sm120` | `softmax` | — | 🔴 **gap** |
| `nvidia_sm120` | `norm` | — | 🔴 **gap** |
| `nvidia_sm120` | `reduction` | — | 🔴 **gap** |
| `nvidia_sm120` | `nvfp4_matmul` | — | 🔴 **gap** |
| `nvidia_sm120` | `int4_matmul` | — | 🔴 **gap** |
| `nvidia_sm120` | `mx_matmul` | — | 🔴 **gap** |
| `nvidia_sm120` | `matmul` | `scheduled_matmul.supports_scheduled_matmul` | ✅ compiled |
| `rocm_gfx1151` | `softmax` | — | 🔴 **gap** |
| `rocm_gfx1151` | `reduction` | — | 🔴 **gap** |
| `rocm_gfx1151` | `paged_kv` | — | 🔴 **gap** |
| `rocm_gfx1151` | `attention` | `scheduled_attention.supports_scheduled_attention` | ✅ compiled |
| `rocm_gfx1151` | `moe_dispatch` | — | 🔴 **gap** |
| `x86` | `softmax` | — | 🔴 **gap** |
| `x86` | `reduction` | — | 🔴 **gap** |
| `x86` | `matmul` | `scheduled_matmul.supports_scheduled_matmul` | ✅ compiled |
| `x86` | `attention` | `scheduled_attention.supports_scheduled_attention` | ✅ compiled |
| `x86` | `attention_backward` | `scheduled_attention_backward.supports_scheduled_attention_backward` | ✅ compiled |
| `x86` | `cohort2` | — | 🔴 **gap** |
| `x86` | `breadth` | — | 🔴 **gap** |
| `x86` | `elementwise` | — | 🔴 **gap** |
| `apple_cpu` | `batched_gemm` | — | 🔴 **gap** |
| `apple_cpu` | `cholesky` | — | 🔴 **gap** |
| `apple_cpu` | `cholesky_solve` | — | 🔴 **gap** |
| `apple_cpu` | `gemm` | — | 🔴 **gap** |
| `apple_cpu` | `lu` | — | 🔴 **gap** |
| `apple_cpu` | `matmul` | — | 🔴 **gap** |
| `apple_cpu` | `qr` | — | 🔴 **gap** |
| `apple_cpu` | `softmax` | — | 🔴 **gap** |
| `apple_cpu` | `svd` | — | 🔴 **gap** |
| `apple_cpu` | `tri_solve` | — | 🔴 **gap** |
| `apple_gpu` | `batched_gemm` | — | 🔴 **gap** |
| `apple_gpu` | `softmax` | — | 🔴 **gap** |
| `apple_gpu` | `dynamic_softmax` | — | 🔴 **gap** |
| `apple_gpu` | `transpose` | — | 🔴 **gap** |
| `apple_gpu` | `gelu` | — | 🔴 **gap** |
| `apple_gpu` | `dynamic_gelu` | — | 🔴 **gap** |
| `apple_gpu` | `dynamic_popcount` | — | 🔴 **gap** |
| `apple_gpu` | `dynamic_count_nonzero` | — | 🔴 **gap** |
| `apple_gpu` | `dynamic_topk` | — | 🔴 **gap** |
| `apple_gpu` | `svd` | — | 🔴 **gap** |
| `apple_gpu` | `value_cholesky` | — | 🔴 **gap** |
| `apple_gpu` | `value_cholesky_solve` | — | 🔴 **gap** |
| `apple_gpu` | `value_clifford_geometric_product` | — | 🔴 **gap** |
| `apple_gpu` | `value_ebm_energy_quadratic` | — | 🔴 **gap** |
| `apple_gpu` | `value_ebm_langevin_step` | — | 🔴 **gap** |
| `apple_gpu` | `value_ebm_partition_exact` | — | 🔴 **gap** |
| `apple_gpu` | `value_ebm_refinement` | — | 🔴 **gap** |
| `apple_gpu` | `value_rl_ppo_policy_loss` | — | 🔴 **gap** |
| `apple_gpu` | `value_tri_solve` | — | 🔴 **gap** |

## Packagers matching no classified family

`package_<family>` is the convention, so these are reached by some
other entry point — a sibling call site, a dtype specialisation, or
dead code. Each needs its own disposition; none may be assumed
covered because a same-named family is compiled.

| Target | Packager |
|---|---|
| `nvidia_sm120` | `package_attention_checkpoint_pair` |
| `nvidia_sm120` | `package_bf16_matmul` |
| `nvidia_sm120` | `package_bf16_softmax` |
| `nvidia_sm120` | `package_f16_matmul` |
| `nvidia_sm120` | `package_f16_softmax` |
| `nvidia_sm120` | `package_f32_softmax` |
| `nvidia_sm120` | `package_paged_kv_read` |
| `rocm_gfx1151` | `package_attention_backward` |
| `rocm_gfx1151` | `package_paged_kv_read` |

## How to read a closing gap

**Measured, and it redirects the work:** the bootstrap surface is
overwhelmingly *IR-constructing*, not delegating. Most packagers build
Tile IR in Python and then compile it through `tessera-opt`, so the
MLIR pipeline already runs from Tile onward and what bypasses it is
Graph → Schedule → Tile. Those retire by **absorption**, and there is
no fast path in them to preserve. The genuine delegation surface —
the one the Target IR boundary exists for — is elsewhere
(`ptx_emit.py`, `emit/nvidia_cuda.py`, `runtime.py`). A plan that
treats the whole bootstrap surface as fast paths to re-express scopes
the wrong work; this table exists partly to stop that.

A family leaves this table one of two ways, and only these two:

1. **Absorbed** — the mainline compiler grows an admission predicate
   and lowering for it, proven against the bootstrap row it replaces.
2. **Re-expressed** — it stays hand-written or library-backed, but is
   reached through a declared Target IR boundary
   (`tessera_x86.abi_call` and its per-backend equivalents) so the
   Decision #28 arbiter can score it. Chosen, never defaulted into.

Deleting a `gap` row without one of those is capability loss, which
is the failure mode Decision #31's ordering caveat exists to prevent.
