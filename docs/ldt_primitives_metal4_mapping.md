# LDT primitives → Metal 4 capability mapping

> First LDT-family PR: `count_nonzero`, `popcount`, `asymmetric_bce`,
> `masked_categorical`. This note records how each maps onto Metal 4 /
> MetalPerformanceShadersGraph / MSL — *functional today, with the real-kernel
> path each would take as a perf follow-on*. Grounded in the on-machine SDK
> headers per Decision #27 (not from memory).

## Execution status today

All four are **functional on `@jit(target="apple_gpu")`** via the numpy-fallback
chain: the Apple GPU runtime executes Metal-envelope ops on Metal and falls
every other registered op through to its numpy reference, transparently. So
"Apple GPU dispatch" is satisfied (correct results on the apple_gpu target);
none yet runs a dedicated Metal kernel. The table below is the *functional →
perf* upgrade path, not a current claim.

| Op | Metal 4 path | SDK-header evidence | Status |
|---|---|---|---|
| `count_nonzero` | MPSGraph: `notEqualWithPrimaryTensor:secondaryTensor:` (vs 0) → `castTensor:toType:` (bool→i32/f32) → `reductionSumWithTensor:axes:` | `MPSGraphArithmeticOps.h` (`notEqualWithPrimaryTensor`), `MPSGraphMemoryOps.h`/cast, `MPSGraphReductionOps.h` (`reductionSumWithTensor`) — all present | **functional** (fallback); clean MPSGraph compose, **0 new MSL** |
| `masked_categorical` (greedy) | MPSGraph: `selectWithPredicateTensor:truePredicateTensor:falsePredicateTensor:` (mask→logits/−inf) → `reductionArgMaximumWithTensor:axis:` | `MPSGraphTensorShapeOps.h`/select, `argMax` present; plus our shipped `tessera_apple_gpu_mpsgraph_argreduce_f32` | **functional** (fallback); composes from a **shipped** kernel |
| `masked_categorical` (sample) | Gumbel-max over masked logits → argmax | our shipped `tessera_apple_gpu_gumbel_argmax_f32` already implements the Gumbel-max trick | **functional** (fallback); shipped kernel + a mask pre-step |
| `asymmetric_bce` | MPSGraph elementwise: `log1p`/`exp`/`abs`/`maximum`/`multiply`/`subtract` + `reductionSumWithTensor` — or compose from our shipped `softplus`/`sigmoid` MPSGraph unary lane | `MPSGraphArithmeticOps.h` unary/binary set present; our MPSGraph lane already ships `softplus`, `sigmoid`, `exp`, `log` | **functional** (fallback); MPSGraph epilogue, **0 new MSL** |
| `popcount` | MSL kernel: `popcount(x)` is a Metal Shading Language integer intrinsic (one-line elementwise kernel over a `uint` buffer) | MSL Spec, integer functions (`popcount`/`clz`/`ctz`). **Not verifiable from the CommandLineTools SDK**, which omits the `metal_stdlib` headers — needs full Xcode to grep `metal_integer` | **functional** (fallback); needs **1 new MSL kernel** (the only true codegen of the four) |

## What Metal 4 specifically buys here

- **MPSGraph compose covers 3 of 4 with zero new shader source.** `count_nonzero`,
  `masked_categorical`, and `asymmetric_bce` are all expressible as MPSGraph node
  graphs over ops already linked in our runtime (`notEqual`/`cast`/`reductionSum`,
  `select`/`argMax`, the `softplus`/`sigmoid`/`exp`/`log` unary lane). They would
  cache like the rest of the MPSGraph lane (keyed on shape-class + opcode + dtype),
  so a future promotion is additive — no new C ABI surface beyond a dispatcher.
- **`popcount` is the one genuine MSL codegen.** Metal exposes `popcount` as an
  integer built-in, so it's a trivial elementwise kernel — but it *is* a new MSL
  symbol (bump `_SENTINEL_SYMBOL`, regen `runtime_abi`). It's also the only op
  here that is purely integer/bit-level, i.e. the one with no MPSGraph float path.
- **No Metal-4-only feature is required.** None of these need MTL4 tensor ops,
  ML-pass encoders, or the residency/encode-session machinery — they're plain
  MPSGraph + one MSL elementwise kernel. The Metal-4 ML surface (inline tensor
  ops in shaders) would only matter if we wanted to *fuse* `count_nonzero` /
  `popcount` into an adjacent matmul/attention epilogue, which is a later,
  model-shaped optimization, not a per-op win.

## Recommendation

Keep all four on the numpy-fallback path for this PR (functional, correct,
exercised by the differential harness). When a real model exercises an LDT
hot-loop, promote in this order — cheapest first:

1. `asymmetric_bce` → MPSGraph epilogue (reuses the shipped unary lane).
2. `count_nonzero` → MPSGraph `notEqual→cast→reductionSum`.
3. `masked_categorical` → `select→argMax` (greedy) / shipped `gumbel_argmax` (sample).
4. `popcount` → one new MSL `popcount` kernel (only after a bitmask-encoded
   lattice workload actually lands on-device).
