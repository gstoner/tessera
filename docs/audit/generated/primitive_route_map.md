# Primitive Route Map — which compiler serves each primitive

**Generated. Do not hand-edit.** Regenerate with
`python -m tessera.compiler.generated_docs --write primitive_route_map`.

`standalone_primitive_coverage.md` tracks per-primitive *contracts*;
`bootstrap_prune_gap.md` tracks which *families* the mainline compiler
covers. This is the join. It exists because `lowering_rule` is
**route-blind** -- it asks whether a lowering rule is *specified*, so a
primitive served only by a Python packager scores the same as one that
descends Graph -> Schedule -> Tile -> Target through `tessera-opt`.

**This is not a kernel-quality claim and not a device claim.** It says
which compiler produces the code, nothing about whether that code was
proven on silicon -- that is `runtime_execution_matrix.md`
(Decision #26). It is deliberately NOT folded into the
`backend_kernel` axis: that axis means *hardware proofs across every
declared target*, and its zero is an honest statement that nothing has
met that bar, not an axis carrying no information.

## Summary

| Metric | Count |
|---|---:|
| Primitives reached by any route | 37 |
| — served by a **compiled** route on some target | 7 |
| — **bootstrap only** (deleting the packager strands them) | 30 |
| Families a backend classifies | 16 |
| Families with a compiled route | 4 |

## Per primitive

| Primitive | nvidia_sm120 | rocm_gfx1151 | x86 |
|---|---|---|---|
| `abs` | — | — | 🟡 bootstrap |
| `add` | — | — | 🟡 bootstrap |
| `amax` | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `amin` | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `attn_sliding_window` | ✅ compiled | ✅ compiled | ✅ compiled |
| `attn_with_stats` | 🟡 bootstrap | — | — |
| `depth_attn` | ✅ compiled | ✅ compiled | ✅ compiled |
| `div` | — | — | 🟡 bootstrap |
| `erf` | — | — | 🟡 bootstrap |
| `exp` | — | — | 🟡 bootstrap |
| `flash_attn` | ✅ compiled | ✅ compiled | ✅ compiled |
| `gelu` | — | — | 🟡 bootstrap |
| `gqa_attention` | ✅ compiled | ✅ compiled | ✅ compiled |
| `layer_norm` | 🟡 bootstrap | — | — |
| `log` | — | — | 🟡 bootstrap |
| `matmul` | ✅ compiled | — | ✅ compiled |
| `max` | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `maximum` | — | — | 🟡 bootstrap |
| `mean` | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `min` | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `minimum` | — | — | 🟡 bootstrap |
| `moe_dispatch` | — | 🟡 bootstrap | — |
| `mqa_attention` | ✅ compiled | ✅ compiled | ✅ compiled |
| `mul` | — | — | 🟡 bootstrap |
| `multi_head_attention` | ✅ compiled | ✅ compiled | ✅ compiled |
| `pow` | — | — | 🟡 bootstrap |
| `rmsnorm` | 🟡 bootstrap | — | — |
| `rsqrt` | — | — | 🟡 bootstrap |
| `sigmoid` | — | — | 🟡 bootstrap |
| `sign` | — | — | 🟡 bootstrap |
| `silu` | — | — | 🟡 bootstrap |
| `softmax` | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `sqrt` | — | — | 🟡 bootstrap |
| `sub` | — | — | 🟡 bootstrap |
| `sum` | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `tanh` | — | — | 🟡 bootstrap |
| `where` | — | — | 🟡 bootstrap |

## Family membership

Declared, and verified in both directions against
`bootstrap_prune_audit`: a member must be a real coverage
primitive with a Graph IR name, and a family any backend
classifies must be declared here. Either half going stale
raises rather than degrading quietly.

| Family | Compiled route | Classified by | Members |
|---|---|---|---|
| `attention` | ✅ | nvidia_sm120, rocm_gfx1151, x86 | `flash_attn`, `multi_head_attention`, `gqa_attention`, `mqa_attention`, `attn_sliding_window` |
| `attention_backward` | ✅ | nvidia_sm120 | `flash_attn` |
| `attention_backward_lse` | — | nvidia_sm120 | `flash_attn`, `attn_with_stats` |
| `attention_lse` | — | nvidia_sm120 | `flash_attn`, `attn_with_stats` |
| `breadth` | — | x86 | *benchmark cohort, not an op pattern* |
| `cohort2` | — | x86 | *benchmark cohort, not an op pattern* |
| `depth_attention` | ✅ | — | `depth_attn` |
| `elementwise` | — | x86 | `add`, `sub`, `mul`, `div`, `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `gelu`, `silu`, `erf`, `abs`, `sign`, `where`, `pow`, `maximum`, `minimum` |
| `int4_matmul` | — | nvidia_sm120 | `matmul` |
| `matmul` | ✅ | nvidia_sm120, x86 | `matmul` |
| `moe_dispatch` | — | rocm_gfx1151 | `moe_dispatch` |
| `mx_matmul` | — | nvidia_sm120 | `matmul` |
| `norm` | — | nvidia_sm120 | `layer_norm`, `rmsnorm` |
| `nvfp4_matmul` | — | nvidia_sm120 | `matmul` |
| `paged_kv` | — | nvidia_sm120, rocm_gfx1151 | `flash_attn` |
| `reduction` | — | nvidia_sm120, rocm_gfx1151, x86 | `sum`, `mean`, `max`, `min`, `amax`, `amin` |
| `softmax` | — | nvidia_sm120, rocm_gfx1151, x86 | `softmax` |
