# Bootstrap Prune — Mainline Coverage Gap

**Generated. Do not hand-edit.** Regenerate with
`python -m tessera.compiler.generated_docs --write`.

The Python per-backend `package_*` families are the **bootstrap
compiler**; the architecture is core MLIR/LLVM (Graph → Schedule →
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
| Backends with a bootstrap module | 4 |
| `package_*` functions total | 49 |
| — **bootstrap** (re-enter Graph IR; prune target) | 34 |
| — compiled-route packagers (consume a lowered artifact) | 15 |
| Lines in those modules | 8738 |
| Classified families | 24 |
| — covered by a compiled route | 6 |
| — **gap (no compiled route)** | 18 |
| Packagers matching no family | 9 |

## Per-backend bootstrap surface

| Target | Module | bootstrap | compiled-route | Families | Lines |
|---|---|---|---|---|---|
| `nvidia_sm120` | `nvidia_native.py` | 19 | 5 | 12 | 3783 |
| `rocm_gfx1151` | `rocm_native.py` | 7 | 5 | 5 | 2894 |
| `x86` | `x86_native.py` | 7 | 5 | 7 | 1846 |
| `apple_cpu` | `apple_cpu_native.py` | 1 | 0 | 0 | 215 |

## Family coverage

`compiled` means a compiled-route admission predicate serves that
family. It does **not** assert the compiled route reaches parity on
every shape and dtype — that is per-family evidence the backend
queues own.

| Target | Family | Compiled route | Status |
|---|---|---|---|
| `nvidia_sm120` | `attention_backward_lse` | — | 🔴 **gap** |
| `nvidia_sm120` | `attention_lse` | — | 🔴 **gap** |
| `nvidia_sm120` | `attention_backward` | `scheduled_attention_backward.supports_scheduled_attention_backward` | ✅ compiled |
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
| `x86` | `cohort2` | — | 🔴 **gap** |
| `x86` | `breadth` | — | 🔴 **gap** |
| `x86` | `elementwise` | — | 🔴 **gap** |

## Packagers matching no classified family

`package_<family>` is the convention, so these are reached by some
other entry point — a sibling call site, a dtype specialisation, or
dead code. Each needs its own disposition; none may be assumed
covered because a same-named family is compiled.

| Target | Packager |
|---|---|
| `nvidia_sm120` | `package_bf16_matmul` |
| `nvidia_sm120` | `package_bf16_softmax` |
| `nvidia_sm120` | `package_f16_matmul` |
| `nvidia_sm120` | `package_f16_softmax` |
| `nvidia_sm120` | `package_f32_softmax` |
| `nvidia_sm120` | `package_paged_kv_read` |
| `nvidia_sm120` | `package_scheduled_matmul` |
| `rocm_gfx1151` | `package_attention_backward` |
| `rocm_gfx1151` | `package_paged_kv_read` |

## How to read a closing gap

A family leaves this table one of two ways, and only these two:

1. **Absorbed** — the mainline compiler grows an admission predicate
   and lowering for it, proven against the bootstrap row it replaces.
2. **Re-expressed** — it stays hand-written or library-backed, but is
   reached through a declared Target IR boundary
   (`tessera_x86.abi_call` and its per-backend equivalents) so the
   Decision #28 arbiter can score it. Chosen, never defaulted into.

Deleting a `gap` row without one of those is capability loss, which
is the failure mode Decision #31's ordering caveat exists to prevent.
