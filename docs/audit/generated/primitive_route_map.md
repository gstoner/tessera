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
| Primitives reached by any route | 32 |
| — served by a **compiled** route on some target | 31 |
| — **bootstrap only** (deleting the packager strands them) | 1 |
| Families a backend classifies | 18 |
| Families with a compiled route | 12 |

**`❔ unclassified` is not `—`.** A dash means measured and nothing
serves it; `unclassified` means this map cannot see that target's
families at all, so no claim is possible. Collapsing the two is how a
whole backend reads as unserved.

| Target | Why unclassified |
|---|---|
| `apple_cpu` | computed families are inventoried; primitive membership and shape-policy joins remain unverified |
| `apple_gpu` | computed value families are inventoried; primitive membership and shape-policy joins remain unverified |

## Per primitive

| Primitive | apple_cpu | apple_gpu | nvidia_sm120 | rocm_gfx1151 | x86 |
|---|---|---|---|---|---|
| `add` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `amax` | ❔ unclassified | ❔ unclassified | ✅ compiled | ✅ compiled | ✅ compiled |
| `amin` | ❔ unclassified | ❔ unclassified | ✅ compiled | — | — |
| `depth_attn` | ❔ unclassified | ❔ unclassified | — | ✅ compiled | — |
| `div` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `erf` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `exp` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `flash_attn` | ❔ unclassified | ❔ unclassified | ✅ compiled | ✅ compiled | ✅ compiled |
| `gelu` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `layer_norm` | ❔ unclassified | ❔ unclassified | ✅ compiled | — | — |
| `log` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `matmul` | ❔ unclassified | ❔ unclassified | ✅ compiled | — | ✅ compiled |
| `max` | ❔ unclassified | ❔ unclassified | ✅ compiled | ✅ compiled | ✅ compiled |
| `maximum` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `mean` | ❔ unclassified | ❔ unclassified | ✅ compiled | ✅ compiled | ✅ compiled |
| `min` | ❔ unclassified | ❔ unclassified | ✅ compiled | — | — |
| `minimum` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `moe_dispatch` | ❔ unclassified | ❔ unclassified | — | 🟡 bootstrap | — |
| `mul` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `pow` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `relu` | ❔ unclassified | ❔ unclassified | ✅ compiled | — | — |
| `rmsnorm` | ❔ unclassified | ❔ unclassified | ✅ compiled | — | — |
| `rsqrt` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `sigmoid` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `sign` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `silu` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `softmax` | ❔ unclassified | ❔ unclassified | ✅ compiled | ✅ compiled | ✅ compiled |
| `sqrt` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `sub` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `sum` | ❔ unclassified | ❔ unclassified | ✅ compiled | ✅ compiled | ✅ compiled |
| `tanh` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |
| `where` | ❔ unclassified | ❔ unclassified | — | — | ✅ compiled |

## Family membership

Declared, and verified in both directions against
`bootstrap_prune_audit`: a member must be a real coverage
primitive with a Graph IR name, and a family any backend
classifies must be declared here. Either half going stale
raises rather than degrading quietly.

| Family | Compiled route | Classified by | Members |
|---|---|---|---|
| `activation` | ✅ nvidia_sm120 | nvidia_sm120 | **nvidia_sm120**: `relu` |
| `attention` | ✅ | nvidia_sm120, rocm_gfx1151, x86 | `flash_attn` |
| `attention_backward` | ✅ | nvidia_sm120, x86 | `flash_attn` |
| `attention_backward_lse` | — | nvidia_sm120 | `flash_attn` |
| `attention_lse` | — | nvidia_sm120 | `flash_attn` |
| `breadth` | ✅ | x86 | *benchmark cohort, not an op pattern* |
| `cohort2` | ✅ | x86 | *benchmark cohort, not an op pattern* |
| `depth_attention` | ✅ rocm_gfx1151 | — | `depth_attn` |
| `elementwise` | ✅ | x86 | `add`, `sub`, `mul`, `div`, `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `gelu`, `silu`, `erf`, `sign`, `where`, `pow`, `maximum`, `minimum` |
| `int4_matmul` | — | nvidia_sm120 | `matmul` |
| `matmul` | ✅ | nvidia_sm120, x86 | `matmul` |
| `moe_dispatch` | — | rocm_gfx1151 | `moe_dispatch` |
| `mx_matmul` | — | nvidia_sm120 | `matmul` |
| `norm` | ✅ | nvidia_sm120 | *****: `layer_norm`, `rmsnorm`; **nvidia_sm120**: `layer_norm`, `rmsnorm`; **x86**: `layer_norm`, `rmsnorm` |
| `nvfp4_matmul` | — | nvidia_sm120 | `matmul` |
| `paged_kv` | — | nvidia_sm120, rocm_gfx1151 | `flash_attn` |
| `reduction` | ✅ | nvidia_sm120, rocm_gfx1151, x86 | **nvidia_sm120**: `sum`, `mean`, `max`, `min`, `amax`, `amin`; **rocm_gfx1151**: `sum`, `mean`, `max`, `amax`; **x86**: `sum`, `mean`, `max`, `amax` |
| `scalar_matmul` | ✅ nvidia_sm120 | nvidia_sm120 | **nvidia_sm120**: `matmul` |
| `softmax` | ✅ | nvidia_sm120, rocm_gfx1151, x86 | `softmax` |
