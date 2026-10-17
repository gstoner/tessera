# Python-authored NVFP4 / BF16 store / resident activation

Owner: ROCM-NVFP4-INGEST-1 / FRONTEND-IR-MEDIUM-1 / LAYOUT-ALG-1 / E2E-REAL-6.
Sync: ROCM-NVFP4-BF16-STORE-20261017.

The authored Graph retains expansion, FP32 scaled matmul, an explicit BF16
product cast, two complementary views, two FP32 widenings, SiLU and BF16 output.
Native partitioning groups the single-use product cast with matmul. Native
Graph-to-Schedule folds it into the BF16 store, with FP32 accumulation unchanged.
There are three native members and nine physical buffers; the BF16 parent is
private, with no FP32 parent allocation or host arithmetic.

Recorded by [record_bf16_store_source.py](record_bf16_store_source.py):
synthetic three-member correctness-gated timing characterization.

## Evidence

- source-identity.json binds 54 source/dependency files and three matching
  compiler/library files. Independent gfx1201 mirror parity has zero mismatches.
- Final exact RX9070XT/gfx1201 gate: 56 passed, zero skips. This includes both
  SiLU operand roles, changed roots, retained outputs, compiler/eager-disabled
  fresh-process replay, v1/v2 schema separation and existing prepared/bounded
  compatibility cases.
- Final native/portable plus exact RTX5070 sibling gate: 81 passed, zero skips,
  including twelve existing source-active producer cases. This does not
  establish a new CUDA physical route.
- Shared registry/dtype/pass/diagnostic/audit gates: 477 passed. These groups
  overlap; counts must not be summed into a unique-test total.
- Focused mypy passes four changed modules; Ruff passes changed Python.
- bf16-store-red.txt retains the original unsupported explicit-store failure.

## Timing scope

Three synthetic M/I/K cases: 3/17/48, 16/64/80 and 256/128/256. Each
captured member window is independently correctness-checked, calibrated and
required to last at least 10 ms. Actual minimum windows are 20.12–23.09 ms.
The native API already returns average per-iteration ms; no second division.

Captured expansion/product/activation measurements include device graph
dispatch and exclude capture, instantiation and CPU enqueue. The ordinary
resident product+activation event range may include CPU enqueue gaps.
Completed retained-owner host medians are approximately 0.485, 0.490 and
0.565 ms respectively, including activation update/upload, invocation and
output readback, excluding compilation/preparation.

These are characterization results, not matched speedups, pure-kernel timing,
default promotion, representative full-model quality or goal completion.
The prior textual Graph packet remains separate historical evidence.

## Remaining

Three-member bounded runtime expressions, K16 activation quantization/down
projection, original NVFP4 MLP and representative model quality/performance
acceptance remain open. W1.1, saved-LSE forward/backward and broader ROCm
route/performance obligations retain their own acceptance gates.


## Original-checkpoint follow-through

See original_checkpoint/README.md for hash-checked Qwen gate/up weights and
captured activations at M19/M256, N24576/K4096. Both authored native
three-member programs pass independent numerical intervals and compiler-disabled
portable replay, with separate device-member and completed-host timings.
This closes the representative gate/up activation connection; native activation
quantization/down projection and full-model acceptance remain open.
