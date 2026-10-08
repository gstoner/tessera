# gfx1201 MXFP8 checked package and image identity

Owner ROCM-FP8-BLOCKSCALE-1; sync ROCM-MXFP8-PACKAGE-2026-10-03.

The typed textual frontend states FP8 E4M3FN operands and raw E8M0 scale
bytes. Native MLIR passes own Graph -> Schedule -> Tile -> ROCm Target ->
ROCDL/LLVM -> HSACO. Python checks the native Target/Tile contract and binds
a distinct checked wide-scale ABI; runtime.launch validates capacities,
shapes, storage and policy before staging five buffers and executing on HIP.
No Python kernel/Tile constructor or execution fallback is introduced.

Native image identity removes M/N/K from only the verified 16x16 one-wave
MXFP8 K32/per-column contract. It retains byte-scale semantics, wide arithmetic,
weight layout and output storage, and cannot collide with FP8 fp32-scale
images. Shape-specific launch guards and compiler ancestry remain descriptor-owned.
This uses the existing native image and leased HIP module-cache services.

## Validation

- 432 focused tests passed on RX 9070 XT/gfx1201: checked MXFP8 device cases,
  profile refusals, existing FP8 device cases and diagnostic/pass metadata gates.
- 169 cache/package tests passed; 9 exact-gfx1151 cases skipped on gfx1201.
- 179 host dtype/op/profile gates passed on Super-Bear.
- 32 audit/runtime-ABI checks passed, 32 generated documents were in sync,
  compiler ownership/log links and diff whitespace checks passed.
- Focused lint passed. Graphify update could not run because the command is
  absent from the WSL engineering checkout; no graph freshness proof is claimed.
- 86 existing scheduled NVIDIA matmul/attention device cases passed on RTX 5070,
  with six existing causal-reference warnings. This is sibling regression
  evidence, not NVIDIA MXFP8 support.

Checked numerical cases include KN/NK, f32/BF16, ragged M/N, multiple
K32 groups, reciprocal code-0/code-254 pairs, code-255 NaNs and code-0 finite
subnormal results. Exact small-integer FP8 operands let the independent
ml_dtypes/f64 oracle check partial rounding and ascending f32 joins.
BF16 rounds only at store. Image reuse executes across M/N/K shapes.
The FP8 scale profile produces a distinct image. Conflicting
scale/numerical/geometry descriptor fields are refused before HIP access.

## Matched timing baseline

Two independent processes compare identical quantized operands, per-column
K32 scales and BF16 outputs. FP8 receives decoded fp32 scale values; MXFP8
receives their E8M0 codes. Timing scales are codes 125..129; extreme/NaN/
subnormal semantics are separate numerical tests.

Every arm passes bitwise BF16 checks through runtime.launch and through the
resident adapter before timing and after graph replay. Every image is
disassembled and must contain RDNA4 FP8 WMMA. Packets retain compiler,
source, image, descriptor and ISA identities plus static instruction counts.

Device-clock markers bracket graph-replayed resident kernels and HIP events
witness each window. Windows are at least 20 ms, arm order alternates, and
clock/event disagreement over 5% rejects the run. Device timing includes GPU
graph dispatch; it does not isolate ISA phase cost. Host end-to-end timing
separately measures checked runtime staging/transfers/submission/completion;
graph construction and compilation are excluded from steady measurements.

A 5 ms timing run failed the agreement guard and was rejected. Final 20 ms
process packets passed. An initial cache invocation omitted device opt-in and
the hollow-lane gate rejected it; the final run actually executes gfx1201 cases.

| M/N/K | Weight | MXFP8 / FP8 device ratio, process 1 | Process 2 |
| --- | --- | ---: | ---: |
| 17/19/64 | kn | 1.101 | 1.098 |
| 17/19/64 | nk | 1.162 | 1.159 |
| 200/256/128 | kn | 1.434 | 1.430 |
| 200/256/128 | nk | 0.536 | 0.536 |
| 256/512/1024 | kn | 2.933 | 2.936 |
| 256/512/1024 | nk | 4.475 | 4.415 |

The long-K MXFP8 seed is substantially slower than the independently
selected FP8 schedule. Short-K results depend on layout. Static ISA counts
identify emitted f64 scale instructions but are not dynamic phase attribution
or proof that arithmetic alone causes the gap. No selector promotion follows.

Remaining: attribute scaling/schedule costs, test a numerically equivalent
optimized scale consumer and architecture-owned larger schedules, and evaluate
MXFP4 under its different quantization/folded numerical contract.
FP8, MXFP8 and MXFP4 are required before short/long or persistent selection.
Canonical MXFP8 storage promotion, partial trailing K32 groups, general dynamic
frontend admission, resident checked HIP buffer APIs, and sibling MXFP8
consumers remain open.

Reproduce on the matching gfx1201 environment:

    python benchmarks/rocm/benchmark_gfx1201_mxfp8_package.py --compiler "$TESSERA_OPT" --llvm-bin "$TESSERA_LLVM_BIN" --output PATH/timing.json

Follow-on: [native exponent-scale consumer](../rocm_mxfp8_exponent_scale_20261003/README.md) replaces the f64 scale implementation with numerically equivalent LLVM ldexp and retains its fresh compiler/ISA/timing receipts. This packet remains the original functional baseline.
