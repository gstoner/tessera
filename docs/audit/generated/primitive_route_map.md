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
| Primitives reached by any route | 31 |
| — served by a **compiled** route on some target | 3 |
| — **bootstrap only** (deleting the packager strands them) | 28 |
| Families a backend classifies | 16 |
| Families with a compiled route | 4 |

**`❔ unclassified` is not `—`.** A dash means measured and nothing
serves it; `unclassified` means this map cannot see that target's
families at all, so no claim is possible. Collapsing the two is how a
whole backend reads as unserved.

| Target | Why unclassified |
|---|---|
| `apple_cpu` | native_package_kind returns a computed expression, not string literals, so the AST walker derives no families |
| `apple_gpu` | not in bootstrap_prune_audit._BACKEND_MODULES, though driver.py has a live scheduled dispatch for it |

## Per primitive

| Primitive | apple_cpu | apple_gpu | nvidia_sm120 | rocm_gfx1151 | x86 |
|---|---|---|---|---|---|
| `add` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `amax` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `amin` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | — | — |
| `depth_attn` | ❔ unclassified | ❔ unclassified | — | ✅ compiled | — |
| `div` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `erf` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `exp` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `flash_attn` | ❔ unclassified | ❔ unclassified | ✅ compiled | ✅ compiled | ✅ compiled |
| `gelu` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `layer_norm` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | — | — |
| `log` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `matmul` | ❔ unclassified | ❔ unclassified | ✅ compiled | — | ✅ compiled |
| `max` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `maximum` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `mean` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `min` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | — | — |
| `minimum` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `moe_dispatch` | ❔ unclassified | ❔ unclassified | — | 🟡 bootstrap | — |
| `mul` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `pow` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `rmsnorm` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | — | — |
| `rsqrt` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `sigmoid` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `sign` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `silu` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `softmax` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `sqrt` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `sub` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `sum` | ❔ unclassified | ❔ unclassified | 🟡 bootstrap | 🟡 bootstrap | 🟡 bootstrap |
| `tanh` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |
| `where` | ❔ unclassified | ❔ unclassified | — | — | 🟡 bootstrap |

## Family membership

Declared, and verified in both directions against
`bootstrap_prune_audit`: a member must be a real coverage
primitive with a Graph IR name, and a family any backend
classifies must be declared here. Either half going stale
raises rather than degrading quietly.

| Family | Compiled route | Classified by | Members |
|---|---|---|---|
| `attention` | ✅ | nvidia_sm120, rocm_gfx1151, x86 | `flash_attn` |
| `attention_backward` | ✅ | nvidia_sm120 | `flash_attn` |
| `attention_backward_lse` | — | nvidia_sm120 | `flash_attn` |
| `attention_lse` | — | nvidia_sm120 | `flash_attn` |
| `breadth` | — | x86 | *benchmark cohort, not an op pattern* |
| `cohort2` | — | x86 | *benchmark cohort, not an op pattern* |
| `depth_attention` | ✅ rocm_gfx1151 | — | `depth_attn` |
| `elementwise` | — | x86 | `add`, `sub`, `mul`, `div`, `exp`, `log`, `sqrt`, `rsqrt`, `tanh`, `sigmoid`, `gelu`, `silu`, `erf`, `sign`, `where`, `pow`, `maximum`, `minimum` |
| `int4_matmul` | — | nvidia_sm120 | `matmul` |
| `matmul` | ✅ | nvidia_sm120, x86 | `matmul` |
| `moe_dispatch` | — | rocm_gfx1151 | `moe_dispatch` |
| `mx_matmul` | — | nvidia_sm120 | `matmul` |
| `norm` | — | nvidia_sm120 | *****: `layer_norm`, `rmsnorm`; **nvidia_sm120**: `layer_norm`, `rmsnorm`; **x86**: `layer_norm`, `rmsnorm` |
| `nvfp4_matmul` | — | nvidia_sm120 | `matmul` |
| `paged_kv` | — | nvidia_sm120, rocm_gfx1151 | `flash_attn` |
| `reduction` | — | nvidia_sm120, rocm_gfx1151, x86 | **nvidia_sm120**: `sum`, `mean`, `max`, `min`, `amax`, `amin`; **rocm_gfx1151**: `sum`, `mean`, `max`, `amax`; **x86**: `sum`, `mean`, `max`, `amax` |
| `softmax` | — | nvidia_sm120, rocm_gfx1151, x86 | `softmax` |
